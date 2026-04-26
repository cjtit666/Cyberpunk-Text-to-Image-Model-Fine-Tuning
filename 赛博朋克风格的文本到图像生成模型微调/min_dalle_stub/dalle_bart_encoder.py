from typing import List
import torch
from torch import nn, BoolTensor, FloatTensor, LongTensor
import math

# 定义了使用GeLU激活函数的GLU模型层（替代前馈神经网络层）
class GLU(nn.Module):
    def __init__(self, d_model, middle_dim): # d_model：输入和输出特征维度
                                             # middle_dim: 中间隐藏层特征维度
        super().__init__()

        # GeLU激活函数
        self.gelu = nn.GELU()
        
        # 层归一化（这个层归一化对应的是transformer原始结构中 前馈神经网络之前的layernorm）
        self.ln0 = nn.LayerNorm(d_model)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(middle_dim)
        
        # 线性层0（d_model -> middle_dim）
        self.fc0 = nn.Linear(d_model, middle_dim, bias=False)
        
        # 线性层1（d_model -> middle_dim）
        self.fc1 = nn.Linear(d_model, middle_dim, bias=False)
        
        # 线性层2（middle_dim -> d_model）
        self.fc2 = nn.Linear(middle_dim, d_model, bias=False)
    
    def forward(self, z): # z: 输入张量，形状为 [batch_size, seq_len, d_model]
        # 输入层归一化
        z = self.ln0.forward(z)

        # 分支 0：线性变换 -> GELU 激活
        w = self.fc0.forward(z)
        
        w = self.gelu.forward(w)
        
        # 分支 1：线性变换
        v = self.fc1.forward(z)
        
        # 门控机制：逐元素相乘 -> 层归一化
        z = self.ln1.forward(w * v)
        
        # 输出投影
        z = self.fc2.forward(z)
        
        return z


# 多头注意力层的基类（可用于编码器，也可用于解码器）
class AttentionBase(nn.Module):
    def __init__(self, num_heads, d_model): # num_heads: 注意力头数
                                            # d_model：词向量维度
        super().__init__()
        
        self.num_heads = num_heads
        
        self.d_model = d_model

        self.d_k = d_model // num_heads

        # q_proj矩阵
        self.q_proj = nn.Linear(d_model, d_model, bias=False)

        # k_proj矩阵
        self.k_proj = nn.Linear(d_model, d_model, bias=False)

        # v_proj矩阵
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        
        # out_proj矩阵
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
    
    def forward(self, k, v, q, mask): # q的形状: [batch_size, len_q, d_model]；k, v的形状: [batch_size, len_k, d_model]（交叉注意力时，len_q和len_k可能不同）
                                                # 这里传进来的q, k, v是已经经过q_proj, k_proj, v_proj处理后的张量！
                                                # mask: 用于遮挡的掩码
       
        # 提取当前数据的 batch_size 大小
        batch_size = q.size(0)
        
        Q = q.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # 改变形状并转置，拆分出多头维度，最终形状为: [batch_size, num_heads, len_q, d_k]
        
        K = k.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # 改变形状并转置，拆分出多头维度，最终形状为: [batch_size, num_heads, len_k, d_k]
        
        V = v.view(batch_size, -1, self.num_heads, self.d_k).transpose(1, 2) # 改变形状并转置，拆分出多头维度，最终形状为: [batch_size, num_heads, len_k, d_k]
        
        # Q乘K的转置 (后两维转置)，并除以缩放因子sqrt(d_k)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k) # scores形状: [batch_size, num_heads, len_q, len_k]
        
        # 如果传入了有效的掩码mask
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
        # 在最后一个维度 (序列长度方向) 应用 Softmax，计算注意力权重。
        attention_weights = torch.softmax(scores, dim=-1) # attention_weights的形状: [batch_size, num_heads, len_q, len_k]

        # 将注意力权重矩阵乘以V
        context = torch.matmul(attention_weights, V) # context的形状为: [batch_size, num_heads, len_q, d_k]
        
        # 将多头的输出进行拼接
        context = context.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model) # context.transpose(1, 2)：形状转置成[batch_size, len_q, num_heads, d_k]
                                                                                            # contiguous() 确保内存连续
                                                                                            # view(batch_size, -1, self.d_model)：重塑回原始形状: [batch_size, len_q, d_model]
        
        # 将重塑后的张量通过最后一层线性层进行映射，输出形状为: [batch_size, len_q, d_model]
        output = self.out_proj(context)
        
        # 返回多头注意力的计算结果
        return output


# 编码器中的自注意力层的类
class EncoderSelfAttention(AttentionBase): # 继承AttentionBase类
    def forward(self, encoder_state, mask):
        # 生成 Q, K, V
        q = self.q_proj.forward(encoder_state)
        
        k = self.k_proj.forward(encoder_state)

        v = self.v_proj.forward(encoder_state)
        
        # 调用基类，计算注意力的结果
        return super().forward(k, v, q, mask)


# （单层）编码器的类
class EncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, glu_embed_dim):
        super().__init__()

        # layernorm层
        self.pre_self_attn_layer_norm = nn.LayerNorm(d_model)
        
        # 多头注意力层
        self.self_attn = EncoderSelfAttention(num_heads, d_model)
        
        # layernorm层
        self.self_attn_layer_norm = nn.LayerNorm(d_model)
        
        # GLU模块
        self.glu = GLU(d_model, glu_embed_dim)
    
    def forward(
        self,
        encoder_state, # 词向量；形状：[2, 64, d_model]
        mask # 标记哪些词不为<pad>的矩阵；形状：[2, 1, 1, 64]
    ):
        # 先保存一下词向量（为了注意力层之后的残差连接使用）
        residual = encoder_state
        
        # 层归一化
        encoder_state = self.pre_self_attn_layer_norm.forward(encoder_state)
        
        # 经过多头注意力机制处理
        encoder_state = self.self_attn.forward(encoder_state, mask)
        
        # 层归一化
        encoder_state = self.self_attn_layer_norm.forward(encoder_state)
        
        # 残差连接
        encoder_state = residual + encoder_state

        
        # 保存当前的词向量（为了GLU之后的残差连接使用）
        residual = encoder_state
        
        # 传入GLU处理
        encoder_state = self.glu.forward(encoder_state)
        
        # 残差连接
        encoder_state = residual + encoder_state
        
        # 当前单层编码器的输出
        return encoder_state


# BART编码器的主类（包含token embedding、加入位置编码、多层编码器）
class DalleBartEncoder(nn.Module):
    def __init__(
        self,
        layer_count, # 编码器层数
        d_model, # 词向量维度
        attention_num_heads, # 注意力头数
        text_vocab_count, # 词表的大小
        text_token_count, # 输入文本序列的最大长度
        glu_embed_dim, # GLU的中间层维度
        device # CPU/cuda
    ):
        super().__init__()

        self.text_vocab_count = text_vocab_count
        
        # token embedding矩阵
        self.embed_tokens = nn.Embedding(text_vocab_count, d_model)
        
        # 位置编码矩阵
        self.embed_positions = nn.Embedding(text_token_count, d_model)
        
        # 堆叠编码器层
        self.layers: List[EncoderLayer] = nn.ModuleList([
            EncoderLayer(
                d_model = d_model,
                num_heads = attention_num_heads,
                glu_embed_dim = glu_embed_dim
            ) 
            for _ in range(layer_count)
        ])
        
        # layernorm层（输入编码器前进行）
        self.layernorm_embedding = nn.LayerNorm(d_model)

        # layernorm层（在多层编码器处理完成后进行）
        self.final_ln = nn.LayerNorm(d_model)
        
        # 生成位置索引张量
        token_indices = torch.arange(text_token_count, device = device)
        # token_indices：[0, 1, 2, ..., 63]（text_token_count = 64）

        # 堆叠两次，以匹配min-DALLE推理时的固定输入格式（可能用于条件和无条件生成）
        pose_tokens = torch.stack([token_indices] * 2)
        # pose_tokens的形状：[2, 64]

        self.register_buffer("pose_tokens", pose_tokens)

    def forward(self, text_tokens): # text_tokens：token id组成的张量，形状为[2, 64]
        # 寻找text_tokens中不为1（<pad>的id）的元素，据此构造一个新的bool矩阵
        mask = text_tokens.not_equal(1)[:, None, None, :] # text_tokens.not_equal(1)：创建一个bool矩阵
                                                                    # [:, None, None, :]：在中间扩展2维
        # mask的形状：[2, 1, 1, 64]
        
        # 词向量 + 位置编码向量
        encoder_state = (self.embed_tokens.forward(text_tokens) + self.embed_positions.forward(self.pose_tokens))
        # encoder_state的形状：[2, 64, d_model]
        
        # 进行层归一化
        encoder_state = self.layernorm_embedding.forward(encoder_state)
        
        # 传入编码器中进行处理
        for layer in self.layers:
            encoder_state = layer.forward(encoder_state, mask)
            
        # 最终的层归一化
        encoder_state = self.final_ln.forward(encoder_state)

        return encoder_state # 形状：[2, 64, d_model]
