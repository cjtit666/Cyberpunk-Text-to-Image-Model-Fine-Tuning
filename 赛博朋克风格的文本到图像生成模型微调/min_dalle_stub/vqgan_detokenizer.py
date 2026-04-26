import torch
from torch import nn
from torch import FloatTensor, LongTensor
from math import sqrt


class ResnetBlock(nn.Module):
    """
    ResNet 残差块。
    结构：GroupNorm -> Sigmoid -> Conv -> GroupNorm -> Sigmoid -> Conv -> Residual
    """
    def __init__(self, log2_count_in: int, log2_count_out: int):
        """
        初始化残差块。

        参数:
            log2_count_in (int): 输入通道数的对数 (2^n)。
            log2_count_out (int): 输出通道数的对数 (2^m)。
        """
        super().__init__()
        m, n = 2 ** log2_count_in, 2 ** log2_count_out
        self.is_middle = m == n
        # GroupNorm，分组数为 32
        self.norm1 = nn.GroupNorm(2 ** 5, m)
        self.conv1 = nn.Conv2d(m, n, 3, padding=1)
        self.norm2 = nn.GroupNorm(2 ** 5, n)
        self.conv2 = nn.Conv2d(n, n, 3, padding=1)
        # 如果输入输出通道数不同，需要 1x1 卷积调整残差分支的维度
        if not self.is_middle:
            self.nin_shortcut = nn.Conv2d(m, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        h = x
        h = self.norm1.forward(h)
        # Swish 激活函数的变体：x * sigmoid(x)
        h *= torch.sigmoid(h)
        h = self.conv1.forward(h)
        h = self.norm2.forward(h)
        h *= torch.sigmoid(h)
        h = self.conv2(h)
        if not self.is_middle:
            x = self.nin_shortcut.forward(x)
        return x + h


class AttentionBlock(nn.Module):
    """
    自注意力模块，用于处理空间特征。
    """
    def __init__(self):
        super().__init__()
        n = 2 ** 9  # 512 channels
        self.norm = nn.GroupNorm(2 ** 5, n)
        self.q = nn.Conv2d(n, n, 1)
        self.k = nn.Conv2d(n, n, 1)
        self.v = nn.Conv2d(n, n, 1)
        self.proj_out = nn.Conv2d(n, n, 1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        n, m = 2 ** 9, x.shape[0]
        h = x
        h = self.norm(h)
        k = self.k.forward(h)
        v = self.v.forward(h)
        q = self.q.forward(h)
        
        # 重塑并转置以计算注意力矩阵
        k = k.reshape(m, n, -1)
        v = v.reshape(m, n, -1)
        q = q.reshape(m, n, -1)
        q = q.permute(0, 2, 1)
        
        # 计算注意力权重: Q * K^T
        w = torch.bmm(q, k)
        w /= n ** 0.5
        w = torch.softmax(w, dim=2)
        
        # 应用注意力权重: Attention * V
        w = w.permute(0, 2, 1)
        h = torch.bmm(v, w)
        
        # 恢复空间维度 (H, W)
        token_count = int(sqrt(h.shape[-1]))
        h = h.reshape(m, n, token_count, token_count)
        h = self.proj_out.forward(h)
        return x + h


class MiddleLayer(nn.Module):
    """
    中间层，包含 ResnetBlock -> AttentionBlock -> ResnetBlock。
    """
    def __init__(self):
        super().__init__()
        self.block_1 = ResnetBlock(9, 9)
        self.attn_1 = AttentionBlock()
        self.block_2 = ResnetBlock(9, 9)
    
    def forward(self, h: FloatTensor) -> FloatTensor:
        h = self.block_1.forward(h)
        h = self.attn_1.forward(h)
        h = self.block_2.forward(h)
        return h


class Upsample(nn.Module):
    """
    上采样层。
    结构：Upsample(nearest) -> Conv
    """
    def __init__(self, log2_count):
        super().__init__()
        n = 2 ** log2_count
        self.upsample = torch.nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = nn.Conv2d(n, n, 3, padding=1)

    def forward(self, x: FloatTensor) -> FloatTensor:
        x = self.upsample.forward(x.to(torch.float32))
        x = self.conv.forward(x)
        return x


class UpsampleBlock(nn.Module):
    """
    上采样块，包含多个 ResNet 块，可选的注意力块和上采样层。
    """
    def __init__(
        self, 
        log2_count_in: int, 
        log2_count_out: int, 
        has_attention: bool, 
        has_upsample: bool
    ):
        super().__init__()
        self.has_attention = has_attention
        self.has_upsample = has_upsample
        
        self.block = nn.ModuleList([
            ResnetBlock(log2_count_in, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out),
            ResnetBlock(log2_count_out, log2_count_out)
        ])

        if has_attention:
            self.attn = nn.ModuleList([
                AttentionBlock(),
                AttentionBlock(),
                AttentionBlock()
            ])

        if has_upsample:
            self.upsample = Upsample(log2_count_out)


    def forward(self, h: FloatTensor) -> FloatTensor:
        for j in range(3):
            h = self.block[j].forward(h)
            if self.has_attention:
                h = self.attn[j].forward(h)
        if self.has_upsample:
            h = self.upsample.forward(h)
        return h


class Decoder(nn.Module):
    """
    VQGAN 解码器核心网络。
    结构：ConvIn -> MiddleLayer -> UpsampleBlocks -> Norm -> ConvOut
    """
    def __init__(self):
        super().__init__()

        self.conv_in = nn.Conv2d(2 ** 8, 2 ** 9, 3, padding=1)
        self.mid = MiddleLayer()

        # 逐步上采样，从低分辨率特征图恢复到高分辨率图像
        self.up = nn.ModuleList([
            UpsampleBlock(7, 7, False, False),
            UpsampleBlock(8, 7, False, True),
            UpsampleBlock(8, 8, False, True),
            UpsampleBlock(9, 8, False, True),
            UpsampleBlock(9, 9, True, True)
        ])

        self.norm_out = nn.GroupNorm(2 ** 5, 2 ** 7)
        self.conv_out = nn.Conv2d(2 ** 7, 3, 3, padding=1)

    def forward(self, z: FloatTensor) -> FloatTensor:
        z = self.conv_in.forward(z)
        z = self.mid.forward(z)

        for i in reversed(range(5)):
            z = self.up[i].forward(z)

        z = self.norm_out.forward(z)
        z *= torch.sigmoid(z)
        z = self.conv_out.forward(z)
        return z


class VQGanDetokenizer(nn.Module):
    """
    VQGAN 反标记化器（图像生成器）。
    将离散的图像 token 索引转换为 RGB 图像。
    """
    def __init__(self):
        super().__init__()
        vocab_count, embed_count = 2 ** 14, 2 ** 8
        self.vocab_count = vocab_count
        # Codebook 嵌入层
        self.embedding = nn.Embedding(vocab_count, embed_count)
        self.post_quant_conv = nn.Conv2d(embed_count, embed_count, 1)
        self.decoder = Decoder()

    def forward(self, is_seamless: bool, z: LongTensor) -> FloatTensor:
        """
        前向传播：Token -> Image。

        参数:
            is_seamless (bool): 是否为平铺模式（生成无缝纹理）。
            z (LongTensor): 图像 token 索引。

        返回:
            FloatTensor: RGB 图像张量，值域 [0, 255]。
        """
        grid_size = int(sqrt(z.shape[0]))
        token_count = grid_size * 2 ** 4
        
        if is_seamless:
            # 平铺模式处理逻辑
            z = z.view([grid_size, grid_size, 2 ** 4, 2 ** 4])
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)
            z = z.flatten().unsqueeze(1)
            z = self.embedding.forward(z)
            z = z.view((1, token_count, token_count, 2 ** 8))
        else:
            # 标准网格模式
            z = self.embedding.forward(z)
            z = z.view((z.shape[0], 2 ** 4, 2 ** 4, 2 ** 8))

        # 调整维度顺序为 (N, C, H, W) 以适配 PyTorch 卷积
        z = z.permute(0, 3, 1, 2).contiguous()
        z = self.post_quant_conv.forward(z)
        z = self.decoder.forward(z)
        
        # 调整回 (N, H, W, C)
        z = z.permute(0, 2, 3, 1)
        # 裁剪值域并缩放到 [0, 255]
        z = z.clip(0.0, 1.0) * 255

        if is_seamless:
            z = z[0]
        else:
            # 恢复网格结构
            z = z.view([grid_size, grid_size, 2 ** 8, 2 ** 8, 3])
            z = z.flatten(1, 2).transpose(1, 0).flatten(1, 2)

        return z
