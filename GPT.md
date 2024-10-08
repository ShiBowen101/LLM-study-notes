### Decoder-Only 架构

**特点**：Decoder-Only 架构仅包含解码器部分。它设计用于生成任务，通过自回归方式逐个生成输出序列的元素。每个解码器层通常包含掩蔽的自注意力层，确保预测当前元素时只使用之前的元素，从而保持生成过程的因果关系。

**应用场景**：**文本生成**：如语言模型、机器翻译、文本摘要。**代码生成**：自动编写程序代码。**对话系统**：自动生成用户交互响应。

**典型模型**：

- GPT（Generative Pre-trained Transformer）系列是 Decoder-Only 架构的代表，广泛用于各种生成任务。
- Llama,llama2,llama3
- Qwen,Qwen2
- ....
# GPT1
## 技术特点
- 采用大量unlabelled数据进行预训练，然后根据任务进行微调
- 微调时，调整输入不需要对于模型架构的改变，使用少量的标记数据
- ### **难点**
- 1.对于无标签数据损失函数的设定
- 2.由文本的输出到其他子任务的转化
## 模型架构
- ![Pasted image 20240903151027.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240903151027.png)
- 目标函数为使得长为$K$的序列用最大的概率与训练给出的文本尽可能的相同
- ### 编码器与解码器
- 编码器无mask部分，目的是语义的融合->BERT->完形填空
- 解码器由mask部分，目的是预测输出->GPT->读后续写
- 预训练部分
- ![Pasted image 20240903151941.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240903151941.png)
- 微调部分
- ![Pasted image 20240903152254.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240903152254.png)
- $L_1$目标函数的目的是预测后面序列的概率最大化
- $L_2$目标函数的目的是预测句子标号概率的最大化
- $L_3$目标函数的目的是前两者的结合
- **微调时根据下游子任务的处理方式**
- ![Pasted image 20240903153031.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240903153031.png)
- [start],[delim],[extract]分别为开始，分割，截止符。
# GPT2
## 不同点
- zero-shot（类似于多模态）：在预训练结束后不需要根据不同的子任务使用标号数据训练
- 去除预训练时没有的分割符等。
- prompt（提示）：替代分割符
- ![Pasted image 20240904152434.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240904152434.png)
- 效果一般，但是揭示多模态的潜力
- ![Pasted image 20240904153518.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240904153518.png)
- 制作数据集时，通过爬取网页信息，并且通过读者反馈来筛选
# GPT3
## 不同点
- 去除了微调的步骤，模型过大不好计算梯度
- GPT3中的Meta-Learning和In-Context Learning
- # ？
- GPT-3（Generative Pre-trained Transformer 3）中的 **Meta-Learning** 和 **In-Context Learning** 是理解其强大功能的关键概念。
- #### 1. Meta-Learning
- **Meta-Learning** 是一种学习如何学习的能力。在GPT-3的上下文中，Meta-Learning意味着模型在训练过程中不仅仅是学习如何解决具体任务，还学习如何从有限的提示和上下文中快速适应新任务。GPT-3 在训练时接受了大量的通用文本数据，这使得它能够在看到新任务的少量示例时，快速地推断出解决该任务的策略。
- Meta-Learning的过程可以分为三个主要阶段：

- **预训练**：模型在大量的文本数据上进行训练，以学习广泛的语言模式和知识。
- **微调（可选）**：对于一些特定任务，模型可以通过微调来进一步优化其能力。
- **应用**：模型在实际使用时，可以根据输入的上下文迅速适应新任务，而不需要进一步的训练。这就是GPT-3的少样本学习（few-shot learning）能力。
- ### 2. In-Context Learning
- **In-Context Learning** 是Meta-Learning的一种应用形式。在GPT-3中，这意味着模型可以根据输入的上下文（也就是提示中的示例）来推断出如何完成一项新任务，而不需要显式地调整模型的权重。简单来说，GPT-3能够从你给出的示例中“学习”并推断出你想要解决的问题。
- 例如，当你给GPT-3提供一些例子时，模型会“理解”这些例子，并在给定上下文的基础上生成相应的输出。这种能力允许GPT-3在不同任务之间无缝切换，并通过少量的上下文信息来推断出新任务的解法。
- **In-Context Learning的步骤：**

- **提供上下文**：用户在输入中提供一些示例或问题描述，作为模型的上下文。
- **模型推断**：GPT-3根据输入的上下文，利用其在训练期间学习到的广泛知识和模式，来推断出如何生成与上下文一致的输出。
- **生成输出**：模型生成符合上下文的答案或响应。
- **总结**：

- **Meta-Learning** 是GPT-3在训练期间所获得的一种学习如何学习的能力，使得它能够从有限的上下文中快速适应新任务。
- **In-Context Learning** 是这种能力的具体体现，GPT-3能够根据输入的示例和上下文，立即推断并解决新问题，而无需进一步训练。
- ![Pasted image 20240904155558.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240904155558.png)
- 不同提示条件下的效果
- ![Pasted image 20240904155919.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240904155919.png)
- ![Pasted image 20240904160501.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240904160501.png)
- 说明替代微调的过程，zero-shot，one-shot等时不更新参数的，而是将训练数据嵌入到提示词当中。
- Fine-tuning是需要更新参数的。
- 制作数据集时，将GPT-2中的数据集作为正类训练一个分类器，来对劣质数据进行分类->然后对于相似度过高的文章过滤
- ![Pasted image 20240905164500.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240905164500.png)
- 作者认为Common Crawl的质量不高，所以权重较低
- ![Pasted image 20240905165200.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/Pasted%20image%2020240905165200.png)
- 根据这张图的可以大致预测出模型随着计算量的增加，损失值的拐点
## 局限性
- 语言生成仍然较弱：上下文联系过短
- 不能向BERT一样兼顾下文
- 在训练无法抓住重点的部分，均匀的去学习每一个词
- 不确定的点：在对子任务的处理时，模型时通过提示词从头学习（泛化性更强）还是调取预训练的数据来回答
- 黑盒效应


