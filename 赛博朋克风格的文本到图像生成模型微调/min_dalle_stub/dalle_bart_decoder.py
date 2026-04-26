from typing import Tuple, List
import torch
from torch import nn, LongTensor, FloatTensor, BoolTensor
from .dalle_bart_encoder import GLU, AttentionBase

# 图像 token 序列长度 (16x16 = 256)
IMAGE_TOKEN_COUNT = 256


class DecoderCrossAttention(AttentionBase):
    """
    解码器交叉注意力层，继承自 AttentionBase。
    Q 来自解码器状态，K, V 来自编码器状态。
    """
    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_mask: BoolTensor
    ) -> FloatTensor:
        """
        前向传播。
        
        参数:
            decoder_state (FloatTensor): 解码器当前状态 (Query)。
            encoder_state (FloatTensor): 编码器输出状态 (Key, Value)。
            attention_mask (BoolTensor): 编码器 padding mask。
        """
        keys = self.k_proj.forward(encoder_state)
        values = self.v_proj.forward(encoder_state)
        queries = self.q_proj.forward(decoder_state)
        return super().forward(keys, values, queries, attention_mask)


class DecoderSelfAttention(AttentionBase):
    """
    解码器自注意力层，支持增量式生成 (Caching)。
    """
    def __init__(self, head_count: int, embed_count: int):
        super().__init__(head_count, embed_count)

    def forward(
        self, 
        decoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        """
        前向传播，包含 KV Cache 更新逻辑。

        参数:
            decoder_state (FloatTensor): 当前步的解码器状态。
            attention_state (FloatTensor): 缓存的历史 KV 状态，形状 (2*batch, seq_len, embed_count)。
            attention_mask (BoolTensor): 自注意力 mask (因果 mask)。
            token_index (LongTensor): 当前生成的 token 索引。
            
        返回:
            decoder_state (FloatTensor): 注意力输出。
            attention_state (FloatTensor): 更新后的 KV Cache。
        """
        keys = self.k_proj.forward(decoder_state)
        values = self.v_proj.forward(decoder_state)
        queries = self.q_proj.forward(decoder_state)
        
        token_count = token_index.shape[1]
        
        # 增量生成模式：仅处理最新生成的 token
        if token_count == 1:
            batch_count = decoder_state.shape[0]
            # 将新的 K, V 拼接到缓存中
            # 注意：attention_state 存储结构为 [Keys; Values]
            attn_state_new = torch.cat([keys, values]).to(attention_state.dtype)
            attention_state[:, token_index[0]] = attn_state_new
            
            # 从缓存中取出完整的 K, V 序列用于计算注意力
            keys = attention_state[:batch_count]
            values = attention_state[batch_count:]
        
        # 否则为全量处理模式（例如训练时），不需要更新 cache
        
        decoder_state = super().forward(keys, values, queries, attention_mask)
        return decoder_state, attention_state


class DecoderLayer(nn.Module):
    """
    解码器层。
    结构：SelfAttention -> CrossAttention -> GLU
    """
    def __init__(
        self, 
        head_count: int, 
        embed_count: int,
        glu_embed_count: int,
        device: str
    ):
        super().__init__()
        # 自注意力子层
        self.pre_self_attn_layer_norm = nn.LayerNorm(embed_count)
        self.self_attn = DecoderSelfAttention(head_count, embed_count)
        self.self_attn_layer_norm = nn.LayerNorm(embed_count)
        
        # 交叉注意力子层
        self.pre_encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        self.encoder_attn = DecoderCrossAttention(head_count, embed_count)
        self.encoder_attn_layer_norm = nn.LayerNorm(embed_count)
        
        # 前馈网络子层
        self.glu = GLU(embed_count, glu_embed_count)
        
        # 预生成位置索引，用于构建因果 mask
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)


    def forward(
        self,
        decoder_state: FloatTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        attention_mask: BoolTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        """
        解码器层前向传播。
        """
        # --- Self Attention ---
        token_count = token_index.shape[1]
        
        # 构建因果 mask (Causal Mask)，防止关注到未来的 token
        if token_count == 1:
            # 增量生成时，mask 取决于当前位置
            self_attn_mask = self.token_indices <= token_index
            self_attn_mask = self_attn_mask[:, None, None, :]
        else:
            # 全量处理时，构建下三角 mask
            self_attn_mask = (
                self.token_indices[None, None, :token_count] <= 
                token_index[:, :, None]
            )
            self_attn_mask = self_attn_mask[:, None, :, :]
        
        residual = decoder_state
        decoder_state = self.pre_self_attn_layer_norm.forward(decoder_state)
        
        # 执行自注意力，并更新状态
        decoder_state, attention_state = self.self_attn.forward(
            decoder_state=decoder_state,
            attention_state=attention_state,
            attention_mask=self_attn_mask,
            token_index=token_index
        )
        decoder_state = self.self_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # --- Cross Attention ---
        residual = decoder_state
        decoder_state = self.pre_encoder_attn_layer_norm.forward(decoder_state)
        # 执行交叉注意力，关注编码器输出
        decoder_state = self.encoder_attn.forward(
            decoder_state=decoder_state,
            encoder_state=encoder_state,
            attention_mask=attention_mask
        )
        decoder_state = self.encoder_attn_layer_norm.forward(decoder_state)
        decoder_state = residual + decoder_state

        # --- Feed Forward ---
        residual = decoder_state
        decoder_state = self.glu.forward(decoder_state)
        decoder_state = residual + decoder_state

        return decoder_state, attention_state


class DalleBartDecoder(nn.Module):
    """
    DALL-E BART 解码器主类。
    负责根据文本编码特征生成图像 token。
    """
    def __init__(
        self,
        image_vocab_count: int,
        embed_count: int,
        attention_head_count: int,
        glu_embed_count: int,
        layer_count: int,
        device: str
    ):
        """
        初始化 BART 解码器。

        参数:
            image_vocab_count (int): 图像词汇表大小 (Codebook size)。
            embed_count (int): 嵌入维度。
            attention_head_count (int): 注意力头数。
            glu_embed_count (int): GLU 中间层维度。
            layer_count (int): 解码器层数。
            device (str): 运行设备。
        """
        super().__init__()
        self.layer_count = layer_count
        self.embed_count = embed_count
        self.image_vocab_count = image_vocab_count
        
        # 图像 token 嵌入层，+1 用于 BOS/EOS 等特殊 token
        self.embed_tokens = nn.Embedding(image_vocab_count + 1, embed_count)
        # 位置嵌入层
        self.embed_positions = nn.Embedding(IMAGE_TOKEN_COUNT, embed_count)
        
        # 堆叠解码器层
        self.layers: List[DecoderLayer] = nn.ModuleList([
            DecoderLayer(
                head_count=attention_head_count,
                embed_count=embed_count,
                glu_embed_count=glu_embed_count,
                device=device
            ) 
            for _ in range(layer_count)
        ])
        
        self.layernorm_embedding = nn.LayerNorm(embed_count)
        self.final_ln = nn.LayerNorm(embed_count)
        # 输出层，映射回词汇表大小
        self.lm_head = nn.Linear(embed_count, image_vocab_count + 1, bias=False)
        self.token_indices = torch.arange(IMAGE_TOKEN_COUNT, device=device)


    def forward(
        self,
        attention_mask: BoolTensor,
        encoder_state: FloatTensor,
        attention_state: FloatTensor,
        prev_tokens: LongTensor,
        token_index: LongTensor
    ) -> Tuple[FloatTensor, FloatTensor]:
        """
        解码器前向传播。
        
        参数:
            attention_mask (BoolTensor): 编码器 padding mask。
            encoder_state (FloatTensor): 编码器输出特征。
            attention_state (FloatTensor): KV Cache 状态。
            prev_tokens (LongTensor): 上一步生成的 token。
            token_index (LongTensor): 当前生成的位置索引。
            
        返回:
            logits (FloatTensor): 输出概率分布 logits。
            attention_state (FloatTensor): 更新后的 KV Cache。
        """
        # 确保输入维度匹配 (batch size 处理)
        image_count = encoder_state.shape[0] // 2
        token_index = token_index.unsqueeze(0).repeat(image_count * 2, 1)
        prev_tokens = prev_tokens.repeat(2, 1)
        
        # 词嵌入 + 位置嵌入
        decoder_state = self.embed_tokens.forward(prev_tokens)
        decoder_state += self.embed_positions.forward(token_index)
        decoder_state = self.layernorm_embedding.forward(decoder_state)
        
        # 逐层传递，并更新 attention_state
        for i in range(self.layer_count):
            decoder_state, attention_state[i] = self.layers[i].forward(
                decoder_state,
                encoder_state,
                attention_state[i],
                attention_mask,
                token_index
            )
            
        decoder_state = self.final_ln(decoder_state)
        logits = self.lm_head(decoder_state)
        return logits, attention_state
    

    def sample_tokens(self, settings, **kwargs) -> Tuple[LongTensor, FloatTensor]:
        """
        执行单步 token 采样。
        
        参数:
            settings (Tensor): 包含 temperature, top_k, supercondition_factor。
            **kwargs: 传递给 forward 的其他参数。
            
        返回:
            image_tokens (LongTensor): 采样得到的下一个 token。
            attention_state (FloatTensor): 更新后的状态。
        """
        # 计算 Logits
        logits, attention_state = self.forward(**kwargs)
        
        # 解析采样设置
        image_count = logits.shape[0] // 2
        temperature = settings[[0]]
        top_k = settings[[1]].to(torch.long)
        supercondition_factor = settings[[2]]
        
        # 截取 logits，只取最后一个时间步，并忽略特殊 token
        logits = logits[:, -1, : 2 ** 14]
        
        # 应用 Classifier-Free Guidance (CFG)
        # 公式: logits = unconditional + (conditional - unconditional) * factor
        #      = unconditional * (1 - factor) + conditional * factor
        logits: FloatTensor = (
            logits[:image_count] * (1 - supercondition_factor) + 
            logits[image_count:] * supercondition_factor
        )
        
        # Top-K 过滤
        logits_sorted, _ = logits.sort(descending=True)
        is_kept = logits >= logits_sorted[:, top_k - 1]
        
        # 数值稳定性处理：减去最大值
        logits -= logits_sorted[:, [0]]
        # 应用 Temperature
        logits /= temperature
        # Softmax (未归一化，直接 exp)
        logits.exp_()
        # 应用过滤 mask
        logits *= is_kept.to(torch.float32)
        
        # 多项式采样
        image_tokens = torch.multinomial(logits, 1)[:, 0]
        return image_tokens, attention_state
