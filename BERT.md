## 大模型的三种架构

大模型通常采用三种主要架构：Encoder-Only, Decoder-Only, 和 Encoder-Decoder。

### 1. Encoder-Only 架构

**特点**：Encoder-Only 架构仅包含编码器部分。它通常用于处理那些只需要理解输入数据而不需要生成新数据的任务。这种架构通过堆叠多层编码器（通常是自注意力层和前馈神经网络层）来处理和理解输入。

**应用场景**：**文本分类**：如情感分析、意图识别等。**实体识别**：从文本中识别和分类命名实体。**特征提取**：为下游任务提取有用的特征，比如在更复杂的模型中使用。

**典型模型**：

- BERT（Bidirectional Encoder Representations from Transformers）是最著名的 Encoder-Only 架构的例子，广泛用于各种文本理解任务。

## BERT论文部分
- BERT是双向的model，GPT是单向的model
- GPT由左到右的预测，而BERT是利用左右的信息同时训练（双向的$Transformer$模型）。
- 预训练（Pre-training）是深度学习和自然语言处理（NLP）中的一种常见技术。它指的是在一个大规模的数据集上先训练一个模型，使其能够学习到广泛的基础知识和特征，然后再将这个模型应用到特定任务中，通过微调（fine-tuning）来进一步提升性能。
### BERT的训练
#### 预训练
- 预训练时使用的是：unlabeled sentences pair
#### 微调
- 使用的是预训练的参数来针对不同的任务微调。
- ![Pasted image 20240826123229.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240826123229.png)

### BERT的架构
- ![Pasted image 20240826123137.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240826123137.png)

#### 输入：$sequenece$可以是一个句子或者是一个句子对
- ![Pasted image 20240826152528.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240826152528.png)

- 利用词根词缀提取句子的公共部分可以减小嵌入矩阵的维度
- ![Pasted image 20240826183531.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240826183531.png)

- $flightless$在原文中为一个，分为两个词。##表示提示$flight$和$less$的联系
- [CLS]表示$classification$，放在每个句子的头部。利用$transformer$来将这个信息融入整个句子。
- [SEP]表示标记一个序列中的句子的分界线。
- ![Pasted image 20240826153040.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240826153040.png)
- ![Pasted image 20240826153507.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240826153507.png)

- 一个token的嵌入向量=词本身的嵌入向量$E_{word}$+此所在句子的信息$E_{segement}$+词的位置信息$E_{index}$
### 掩码的构造、
- $[mask]$表示用于替换掉需要猜的词
- ![Pasted image 20240826183012.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240826183012.png)
