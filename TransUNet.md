# Unet的缺陷
- Therefore, these architectures generally yield weak performances especially for target structures that show large inter-patient variation in terms of texture, shape and size.？
- 由于卷积核的限制难以对大范围的特征进行提取
- 而Transformer则缺少对于细节特征的提取，造成了分辨率的缺失
# CNN
-  CNN 特征提供的详细高分辨率空间信息
-  更密集地纳入低层次特征通常会提高分割精度
# 网络架构
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240910205833.png)
## Encoder
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240923232700.png)
- $images$($H\times W\times c$)->$n\times vectors$($p^2\cdot c$)->$n\times vectors$($D$)->$matric$($n\times D$)->$tensor$($\frac{H}{P}\times\frac{H}{P}\times D$)
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240924002027.png)
- 编码时按照每一个块形成的向量为一个token进行编码
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240924003758.png)
## Decoder
- cascaded upsampler (CUP)
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240924005126.png)
## Experiments
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240924005217.png)
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240924005715.png)
## Analytical Study
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240925194845.png)
- 跳跃层的增加可以改善性能
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240925195453.png)
- 分辨率提高->patchsize不变,提高tranformer的sequence处理长度->性能得到提升
- ![image.png](https://raw.githubusercontent.com/ShiBowen101/PicGo_imgs/main/obsidian/20240925195909.png)
- patchsize减小->sequence_length增加->性能提升
