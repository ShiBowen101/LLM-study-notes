## 背景
- 1.在传统的图像识别任务中，都是将一张图片进行单一类别的标注后训练，这样一方面回到图片信息的提取不够高效，另一方面导致模型的泛化能力不强。例如：在imagenet中有1000个类，但是传统的网络训练后无法识别出1001个类别。可以做到zero-shot推理。
- 思路来源：LLM在执行不同的下游任务时，不需要对对模型架构进行改变，并且LLM的训练为大量无标签的数据不同于传统CV任务。


## 对比学习+混精度训练+linear probe和微调


## 预训练
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20250204120844.png)
- 1.通过对像imagenet这样的object label 套用句子模板形成图片文本对，并且不需要做分类头，将文本特征与图片特征提取后计算相似度来分类，导致最终的迁移能力很强
- 2.根据图片来预测其对应的文本。->（使用对比学习提高训练效率）可以解决同一图片的不同文本描述。
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20250207161711.png)
- zero-shot的动机：通过语言的引导来替代的大模型微调的过程。
## 架构
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20250207163640.png)
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20250207162228.png)
（具体参考对比学习）
- 对于图片和文本分别使用特征提取的编码器
- [How to Train Really Large Models on Many GPUs? | Lil'Log](https://lilianweng.github.io/posts/2021-09-25-train-large/)分布式训练的介绍
### prompt engineering
- 解决的问题：
- 1.多义词产生的歧义问题
- 2.在推理时只输入一个词作为文本无法和预训练时的句子一样很好的进行特征提取
- 解决方式：1.通过固定句子模板将类别进行填充，2.将已知的信息放入提示信息中。
- 例如：a photo of XX
## 实验
![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20250207165445.png)
few-shot的结果不一定优于zero-shot
## 局限性
- 在细分领域的性能仍然有限
- 应对复杂的任务（尤其是非分类任务）性能明显不足
- few-shot与zero-shot相比性能仍然的不足
### 读论文：如何去除固定的标签类别
文本与图像的对应