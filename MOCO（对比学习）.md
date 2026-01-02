## 背景
- 采用无监督的方式完成计算机视觉的任务，通过完成区分正负样本的任务驱动，来使得模型学习样本中的特征。
- MoCO的本质是训练了一个基于instance discrimination任务的正负样本的编码器，从而构成正负样本编码特征的字典。
### 正样本的定义
定义方式很灵活：例如同一物体的不同角度，同一照片的处理后的衍生照片等等
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20250209172951.png)
正样本形成query

