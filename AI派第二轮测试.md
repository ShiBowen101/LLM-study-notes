## Step 1. Transformer
见代码
## Step 2. 大模型的三种架构
本节主要考察你对一些基础知识的理解，请解决如下问题：

### 1.为什么bert不能像gpt一样做文本生成？
- bert是基于编码器的架构。而gpt是基于生成式解码器的架构。bert是需要同时获取上下文的信息的双向架构，而gpt是通过获取上文信息完成下文预测的单向架构（在训练时有掩码的处理）。
### 2. 对于decoder-only的模型，它是由tokenizer，embedding层， transformer block，lm_head，请你简单说明一下这4个部分分别在做什么？token是一个什么东西？
    1.tokenizer：将待处理的文本进行token的划分。
	2.embedding层：将划分好的token完成嵌入向量编码，将文本信息转化为数字信息。
	3.transformer block：将整个文本的信息进行语义的融合，使得每个token形成符合当前语境下的语义。
	4.lm_head：将处理过的语义信息，通过变换形成预测下一个token的概率分布。
### 3. 为什么decoder-only的模型的数量远远多于Encoder-Decoder模型？明明二者都可以做文本生成
	1.decoder-only的模型只需要训练解码器的部分，而Encoder-Decoder模型需要训练编码器解码器两部分，训练的成本更高
	2.在文本生成的过程中decoder-only的模型的生成速度更快
### 1. 使用预训练好的bert/gpt以及它们对应的tokenizer在imdb任务上finetune，计算这两种模型在IMDB分类任务在测试集上的准确率
    
    见代码
## Step 3. 一个decoder-only的Generative LLM的前世今生
### 1. Pre-training

请你阅读参考文章的第三节，回答以下问题：

- 训练时有一个参数max_length，它是做什么的
##### max_length用于限制模型在训练时，文本输入的最大token数目，用于控制模型的参数规模与存储规模
#### 
- 在真正开始训练时有一个warm up，它是用来做什么的
##### 在训练的初期，将学习率调小再逐渐放大，使得模型的训练更加平稳，提升训练的效率。
### 2. Post-training

在经过预训练后，我们得到的模型还不能直接应用于对话，它本质上还是一个next token predictor，为了让其具有良好的qa的属性，往往还需要一下额外的步骤才能得到一个商业可用的llm

请你阅读参考文章的第四节

#### 2.1 Instruction Tuning

请你自行搜索资料，回答一下问题：

- 什么是instruction tuning？为什么需要instruction tuning?
##### instruction tuning可以将输入的自然语言指令转化为针对不同任务的标准指令
##### 理解用户输入自然语言表达的任务类型，可以降低使用者的使用门槛，同时可以使得用户输入自然语言却应用于多种任务
##### 
- Llama3 的instruction tuning的格式是怎么样的？
```structured text

<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{{ system_prompt }}<|eot_id|><|start_header_id|>user<|end_header_id|>

{{ user_message }}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
```


提示，llama2的instruction tuning的格式如下：

```structured text
<s>[INST] <<SYS>>
{system_message}
<</SYS>>
{user_message} [/INST]
{assistant_message} 
```

#### 2.2 SFT

在预训练结束后，我们会用高质量的qa对去sft llm，让llm初步具有qa的能力，但仅仅这样是不够的，我们还要将llm与人类偏好对齐

问题：

- 为什么要将llm与人类偏好对齐？不这么做会出现什么问题？
##### 使得大模型的回答内容更加接近人的预期与人类的价值观保持一致。
##### 如果不进行对齐，可能会导致大模型生成不良内容违反法律与道德

#### 2.3 RLHF/PPO

PPO是非常经典的强化学习算法，可以用于将llm与人类价值观对齐，由于这不是我们需要深入考虑的问题，所以我希望你解决以下几个小问题：

- rlhf的偏好数据集是如何构造的？
##### 为注释者提示抽取两个来自不同模型的响应，要求注释者根据他们更喜欢选择的响应的程度将其归类为四个级别之一：明显更好，更好，略好或略好。并且鼓励注释者进一步改进首选响应。注释者可以直接编辑首选响应，或者提示模型用反馈来细化自己的响应。
- reward model是做什么的？它是如何被训练的？
##### reward model用于将大模型的输出根据人类的偏好进行打分，从而优化大模型的输出与人类的价值观对齐。
##### reward model在预训练模型的基础上利用偏好数据，由极大似然估计使reward model的打分人类偏好数据保持一致。

#### 2.4 [DPO](https://arxiv.org/abs/2305.18290)

DPO是去年Stanford提出的新的rlhf算法，在数学上可以证明与ppo等价，我仍然不需要你深入了解，但我希望你能理解DPO解决了什么问题：

- DPO和PPO相比优势在哪里？请你详细阐述一下
1.DPO相比于PPO的计算效率更高
2.DPO是不需要真正的reward model的，而是用隐式的reward model进行替换，节省了训练reward model的时间
3.DPO直接利用人类的偏好数据，训练时更加直接稳定，避免了训练reward model时的引入的噪声


一个视频供你参考[DPO (Direct Preference Optimization) 算法讲解](https://www.bilibili.com/video/BV1GF4m1L7Nt/?vd_source=827e9d926cec44ef6817b376d985aae5)
## Step 4. llm实战演练1
- 见代码
- 结果：
##### zero-shot
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240927152707.png)
##### 2-shot
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240927154549.png)
##### 4-shot
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240928231333.png)
## Step 5. llm实战演练2
- 见代码
- 结果：
##### zero-shot:
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240926200831.png)
##### 2-shot:
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240926201459.png)
##### 4-shot
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240926202709.png)
## Step 6. Research 

恭喜你，如果你做到这里，说明你对llm已经有了一个基本的理解了，下面我会向你介绍几篇工作，浅浅接触一些前沿的工作（only 我了解的领域hhhh）吧。

1. 一篇关于越狱攻击：http://arxiv.org/abs/2403.07865

    见代码CodeStack

2. 一篇关于越狱攻击的防御：https://arxiv.org/abs/2407.09121 

   讲清楚几个问题：他研究了/发现了一个什么有意思的问题，用什么样的方式解决了这个问题（这也是research的一般过程）
   ##### 他们发现了像Codestack这类利用位置偏见诱导LLM生成不安全内容的攻击方式。他们提出了一种新的安全调整方法，称为解耦拒绝训练（DeRTa），明确训练LLMs在任何响应位置上拒绝遵从通过嵌入构建的有害响应。该方法有两个开创性的组成部分：有害响应前缀的最大似然估计（MLE）和强化转换优化（RTO）。


