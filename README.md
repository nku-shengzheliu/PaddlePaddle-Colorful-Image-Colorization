# PaddlePaddle-Colorful Image Colorization

## 一、简介

本项目基于paddlepaddle框架复现"Colorful image colorization."，它的主体是作者设计搭建的卷积神经网络。网络的pipeline如下图所示：

<img src="https://github.com/nku-shengzheliu/PaddlePaddle-Colorful-Image-Colorization/blob/main/colornet.JPG" width = 80% height = 80% align=center/>

此外，为了处理自然图像中存在的低饱和度现象，作者还设计了Lab色彩空间的量化策略，使用分类的形式处理图像着色问题。并统计了imagenet数据集中每种量化级别的先验概率信息，作为计算损失时的权重。

**论文:**

- [1] Zhang, Richard, Phillip Isola, and Alexei A. Efros. "Colorful image colorization." *European conference on computer vision*. Springer, Cham, 2016.

**参考项目：**

- [official project](https://github.com/richzhang/colorization)


## 二、复现精度





**模型下载**
模型地址：[待上传]

## 三、数据集

[ImageNet数据集](https://image-net.org/download)。

- 数据集大小：
  - 训练集：1281167张
  - 验证集：10000张，取自imagenet验证集
  - 测试集：10000张，取自imagenet验证集，来源：论文[Learning Representations for Automatic Colorization](http://people.cs.uchicago.edu/~larsson/colorization/)

## 四、快速开始

### step1: clone

```
# clone this repo
git clone https://github.com/nku-shengzheliu/PaddlePaddle-Colorful-Image-Colorization.git
cd PaddlePaddle-Colorful-Image-Colorization
```

**安装依赖**

```
pip install -r requirements.txt
```

### step2: 训练

```
python train.py
```

如果训练中断通过 --resume 参数恢复，此时设定 --resume 为模型上次保存权重文件。

### step3: 测试

```
python train.py --test --chechpoint {XXX}
```

## 六、代码结构与详细说明

### 6.1 代码结构

```
├─saved_models                    # 保存模型权重
├─logs                            # 训练日志
├─dataloader                      # 数据集加载
├─models                          # 模型
├─results                         # 可视化结果
├─utils                           # 工具代码
│  README.md                      # 中文readme
│  requirement.txt                # 依赖
│  train.py                       # 训练&测试&验证
│  demo.py                        # 测试&生成可视化结果
```


