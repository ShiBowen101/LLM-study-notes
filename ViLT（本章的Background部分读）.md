## 自身卖点
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241019225417.png)
- 大幅减少img部分的训练时间
- 卷积提取+目标检测->卷积提取->划分patch，embedding
- 运行时间大幅度缩短，但是性能还算不错
- 数据增强：Also, for the first time, we empirically show that whole word masking and image augmentations that were unprecedented in VLP training schemes further drive downstream performance.？
## 前面工作的痛点
- 1.在img预训练是抽取特征消耗大量时间
- 2.运用目标检测的方式来完成img和text间的转化，难以实现端到端的任务目标（缺失细节）
- 3.不能将像素直接扔给transformer，由于序列过长，必须将img转化为离散的带有语义特征的序列
- 4.较为省时间的做法是保存预训练的特征提取数据部分，改进语义融合与匹配的部分，但是在实时应用时仍然较慢
#### 常用损失函数
- 1.image text matching
- 2.masked language modeling
#### 本文中的小综述
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241020001844.png)
- 不同VLT模型在文本处理，图像处理和二者融合上模型投入的精力
- 属于（b）类的CLIP表现出在下游子任务迁移性不强的缺陷，引发作者对模式交互需要加强的想法
- ###### 图文匹配（交互模式）的方式：
- 1.将文本特征，图片特征拼接直接输入，单输入
- 2.将文本和图像分别处理
- 本文选择单流输入，减少参数
## 架构
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241023192531.png)
- 由于文本和图片同时输入，增加了type_embedding部分
- 嵌入向量的组成=类别信息+位置信息+token词义
## 训练方式
### Image Text Matching
输入匹配与不匹配的文本图片对，进行二分类的判断

### Word Patch Alignment
减小文本和图像的分布距离

### Mashed Language Modeling‘
进行文本的完形填空
## whole word masking
- mask的对象把token转换为word，目的防止model在预测时，利用文本信息如单词构造而不使用图像信息
- 在不影响文本和图片对应的前提下进行数据增强
## 数据集
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241023194019.png)
- 数据集文本和描述文字不一定是一一对应的
## Retrieval Tasks
## Ablation Study
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241023195253.png)
- 采用了whole word masking和图片的数据增强
- 没有采用图片patch的完型填空->图片patch的生成
- 图片的数据增强的那点在于文本和图片要始终对应。
### 最终表现
- 训练时长也不短，运行速度快且精度不差，但精度并不领先
- 未来展望图像重建来训练以及把图像打为Patch后的数据增强
