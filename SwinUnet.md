## 针对Unet的缺点
- 由于卷积核的限制难以对大范围的特征进行提取
- 而Transformer则缺少对于细节特征的提取，造成了分辨率的缺失
- it cannot learn global and long-range semantic information interaction well due to the locality of convolution operation.
## SwinUnet的特点
- 在编码器中使用Transformer
- 在不使用卷积的情况下完成上采样
- 仍然保留跳跃连接层
## SwinUnet的架构
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240925210100.png)
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241021185859.png)
- [Swin Transformer](Swin%20Transformer.md)
1. $$ \hat{z}_l = \text{W-MSA}(\text{LN}(z_{l-1})) + z_{l-1} $$
2. $$ z_l = \text{MLP}(\text{LN}(\hat{z}_l)) + \hat{z}_l $$
3. $$ \hat{z}_{l+1} = \text{SW-MSA}(\text{LN}(z_l)) + z_l $$
4. $$ z_{l+1} = \text{MLP}(\text{LN}(\hat{z}_{l+1})) + \hat{z}_{l+1} $$
- Bottleneck?
- Patch expanding layer?
- data augmentations such as flips and rotations?

### Hausdorff Distance 的定义
- 在深度学习中，**Hausdorff Distance（Hausdorff 距离）是一种常用的距离度量，特别是在图像分割和目标检测任务中用来评估预测结果与真实标注（ground truth）之间的相似性。它测量两个点集之间的最大距离，反映了两个集合的几何差异

给定两个点集 $A = \{a_1, a_2, \dots, a_n\}$ 和 $B = \{b_1, b_2, \dots, b_m\}$，Hausdorff 距离定义为：

$$
H(A, B) = \max\left\{ \sup_{a \in A} \inf_{b \in B} \|a - b\|, \sup_{b \in B} \inf_{a \in A} \|b - a\| \right\}
$$

- **$\sup$** 表示取集合中最大值。
- **$\inf$** 表示取集合中最小值。
- **$\|a - b\|$** 表示 $a$ 和 $b$ 两点之间的距离，通常是欧氏距离。

这个定义中的两个部分可以理解为：
1. 对于每个点 $a \in A$，计算到 $B$ 中最近点的距离 $\inf_{b \in B} \|a - b\|$，然后取 $A$ 中最大的最小距离。
2. 对于每个点 $b \in B$，计算到 $A$ 中最近点的距离 $\inf_{a \in A} \|b - a\|$，然后取 $B$ 中最大的最小距离。

Hausdorff 距离是这两个最大最小距离的最大值。
对mask图片计算HD可以用于度量分割的相似度。

## 实验
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241021193223.png)
- we achieved accuracy improvement of about 4% and 10% on the HD evaluation metric, which indicates that our approach can achieve better edge predictions.
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20241021193717.png)
- have over-segmentation problems,？
- 说明Patch expand的下采样方式更加适合Swin-Unet的架构
- By using the image data of MR mode as input？