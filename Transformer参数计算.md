代码运行结果:
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20250615162300.png)
# 编码器的计算;
### Embedding:
(word_embeddings): Embedding(30522, 768, padding_idx=0)->$30522\times768=23.44M$
(position_embeddings): Embedding(512, 768)->$512\times768=0.393M$
(token_type_embeddings): Embedding(2, 768)->$2\times768=0.02M$
$23.853M$
### 12 x BertLayer
(query,key,value): Linear(in_features=768, out_features=768, bias=True)->$769\times768\times3=1.772M$
BertSelfOutput:Linear(in_features=768, out_features=768, bias=True)->$769\times768=0.591M$
BertIntermediate: Linear(in_features=768, out_features=3072, bias=True)->$3072\times769=2.362M$
BertOutput:Linear(in_features=3072, out_features=768, bias=True)->$768\times3073=2.359M$
$85.008M$
### BertPooler
Linear(in_features=768, out_features=768, bias=True)->$769\times768=0.591M$

### Encoder_sum
$Encoder=109.452M$
计算结果与运行结果基本一致，误差由计算时的对于数据末位的截断近似有关
# 解码器计算

### Embedding:
(wte): Embedding(50257, 768)->$50257\times768=38.257M$
(wpe): Embedding(1024, 768)->$1024\times768=0.787M$
$39.044M$
### 12 x GPT2Block
这部分GPT2的参数不透明，难以估计，认为与编码器部分参数保持一致
$85.008M$
### GPT2OUTPUT
(lm_head): Linear(in_features=768, out_features=50257, bias=False)->$768\times50257=38.597M$
### Decoder_sum
$Decoder=162.649M$

### sum
$sum=272.093M$

### 总结
- 本次计算与统计结果尤其是解码器部分存在一定差异，主要原因为GPT2的部分参数细节难以获取。
- 使用的并非标准的transformer的模型而是选取Bert作为解码器，GPT2作为编码器，因此参数量会多于标准版的transformer，主要原因为：两种模型的tokenizer的不同，需要计算两次Embedding过程的参数。