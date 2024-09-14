import numpy as np

np.random.seed(114514)


# 测试数据的三维度
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    1. 需要完成调整 K 的转置来匹配 Q 的最后一个维度，
    2. 计算attn_score并缩放，
    3. softmax 应用于最后一个轴计算attn_weight，
    4. 应用attn_weights输出output
    5. 带掩码mask的的注意力可以不用实现,但请记住encoder和decoder的transformer块是不一样的，很大一部分都在就在mask上
    """
    # 计算attn_score
    # Q(batch_size,num_heads,sequence_length,embed_size)
    # K.T(batch_size,num_heads,embed_size,sequence_length)
    # Q@K.T(batch_size,num_heads,sequence_length,sequence_length)
    embed_size = Q.shape[-1]  # 获取每个头的embed_size
    k_T = np.transpose(K, (0, 1, 3, 2))
    attn_score = np.matmul(Q, k_T) / np.sqrt(embed_size)  # (batch_size,num_heads,sequence_length,sequence_length)
    # 针对每一行softmax
    attention_weights = np.exp(attn_score)
    attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
    # (batch_size,num_heads,sequence_length,sequence_length)@(batch_size,num_heads,sequence_length,embed_size)
    output = np.matmul(attention_weights, V)  # (batch_size,sequence_length,embed_size)
    return output, attention_weights


def multi_head_attention(embed_size, num_heads, input, mask=None):
    """
   1. embed_size 确保可以等分 num_heads 份， 否则输出错误
   2. 随机初始化Wq,Wk,Wv,Wo矩阵，并对input做线性变换
   3. 利用scaled_dot_product_attention()输出的attn_output计算O
   4. 返回output, attN_weights
   """
    # 初始化W_q,W_k,W_v
    w_q = np.random.randn(embed_size, embed_size)
    w_k = np.random.randn(embed_size, embed_size)
    w_v = np.random.randn(embed_size, embed_size)
    w_o = np.random.randn(embed_size, embed_size)  # 用于线性变换
    # 生成对应的q,k,v(input(batch_size,sequence_length,embed_size))
    q = np.matmul(input, w_q)  # (batch_size,sequence_length,embed_size)
    k = np.matmul(input, w_k)
    v = np.matmul(input, w_v)
    # 针对多头注意力机制划分
    batch_size = input.shape[0]
    sequence_length = input.shape[1]
    embed_sized_small = embed_size // num_heads
    q = q.reshape(batch_size, sequence_length, num_heads, embed_sized_small).transpose(0, 2, 1, 3)
    k = k.reshape(batch_size, sequence_length, num_heads, embed_sized_small).transpose(0, 2, 1, 3)
    v = v.reshape(batch_size, sequence_length, num_heads, embed_sized_small).transpose(0, 2, 1, 3)
    # 先reshape再transpose的目的是：先将embed_size划分为num_heads*embed_sized_small
    # 再将num_heads维度前置
    # 进入scaled_dot_product_attention计算
    (output, attention_weights) = scaled_dot_product_attention(q, k, v, mask=None)
    # 将多个头的输出结果合并
    output = np.transpose(output, (0, 2, 1, 3)).reshape(batch_size, sequence_length, embed_size)
    # 进行线性变换
    output = np.matmul(output, w_o)
    return output, attention_weights


# test e.g.
embed_size = 128
num_heads = 8
input = np.random.randn(10, 20, embed_size)
output, weights = multi_head_attention(embed_size, num_heads, input)
print(output.shape)
print(weights.shape)
print(output[0][0][:10])
print(weights[0][0][0][:10])
