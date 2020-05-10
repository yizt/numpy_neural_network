[TOC]

​        网友 [**msnh2012**](https://github.com/msnh2012) 提出是否在经典网络上验证过numpy实现的有效性；由于numpy构建的卷积层、池化层确实非常慢; 因此选择一个较小的网络vgg11，在cifar10数据集(分辨率32*32,其它数据集分辨率太大,耗时太长)上做验证。

​      cifar10 包含5w张训练图像，1w张测试图像；训练1w步，batch size设置为32，训练一共耗时**6天半**$ \color {#F00}{实在是太慢}$

​      预测1w张测试集耗时约**100分钟**；1w步,测试集准确率**0.576**, 由于耗时太长没有训练更多步数，步数更多步数精度会进一步提升。如果仅仅想看是否有效，训练500步，测试集准确率**0.283**。

## VGG网络定义

​         vgg网络定义如下,详见[vgg.py](nn/vgg.py)

```python
from modules import *

cfgs = {'vgg11': [1, 1, 2, 2, 2],
        'vgg13': [2, 2, 2, 2, 2],
        'vgg16': [2, 2, 3, 3, 3],
        'vgg19': [2, 2, 4, 4, 4]}


class VGG(Model):
    """
    VGG 模型
    """

    def __init__(self, image_size=224, in_channels=3, num_classes=10, name='', **kwargs):
        """

        :param image_size: 图像尺寸，假定长宽一致，且尺寸能够被32整除
        :param in_channels: 图像通道数
        :param num_classes: 类别数
        :param name:
        :param kwargs:
        """
        self.image_size = image_size
        self.num_block_layers = cfgs[name]
        self.num_classes = num_classes
        self.in_channels = in_channels
        # block 1~5的通道数分布为64，128，256，512，512
        self.block_channel_list = [64, 128, 256, 512, 512]
        layers = self.make_layers()
        super(VGG, self).__init__(layers, name=name, **kwargs)

    def make_layers(self):
        layers = []
        in_filters = self.in_channels
        # 卷积池化层
        for b_idx, num_layers in enumerate(self.num_block_layers):
            out_filters = self.block_channel_list[b_idx]
            for l_idx in range(num_layers):
                layers.append(Conv2D(in_filters, out_filters,
                                     name='Conv_{}_{}'.format(b_idx + 1, l_idx + 1)))
                layers.append(ReLU(name='ReLU_{}_{}'.format(b_idx + 1, l_idx + 1)))
                # 输出通道数是下一层的输入通道数
                in_filters = out_filters
            # 每个block以max pooling结尾
            layers.append(MaxPooling2D(kernel=(2, 2),
                                       stride=(2, 2),
                                       name='MaxPooling_{}'.format(b_idx + 1)))
        # 全局平均池化
        layers.append(GlobalAvgPooling2D(name='Global_Avg_Pooling'))
        # 两层全连接
        layers.append(Linear(out_filters, 4096, name='fc_6'))
        layers.append(ReLU(name='ReLU_6'))
        layers.append(Linear(4096, 4096, name='fc_7'))
        layers.append(ReLU(name='ReLU_7'))
        # 分类
        layers.append(Linear(4096, self.num_classes, name='cls_logit'))
        return layers
```





## 训练

```shell
cd nn
python cnn.py -d /sdb/tmp/open_dataset/cifar-10-batches-py/
```

主要参数说明:

- `-d,--cifar-root` : cifar数据集路径 
- `-c,--checkpoint` : 模型权重参数路径
- `-b,--batch-size` : 默认32

- `-s,--steps` ：训练步数，默认10000

- `--lr` : 学习率，默认0.01
- `--decay` : 学习率衰减，默认人1e-3
- `--eval-only` : 仅仅评估



## 预测

a) 1w步,测试集准确率0.576

```shell
cd nn
python cnn.py -d /sdb/tmp/open_dataset/cifar-10-batches-py/ -c /tmp/weights-9900.pkl --eval-only --eval-num 10000   
load weights done
Fri May  8 11:03:05 2020 start evaluate
Fri May  8 12:41:11 2020 acc on test dataset is :0.576
```



b) 500步,测试集准确率0.283

```shell
cd nn
python cnn.py -d /sdb/tmp/open_dataset/cifar-10-batches-py/ -c /tmp/weights-500.pkl --eval-only --eval-num 10000 
load weights done
Thu May  7 13:39:13 2020 start evaluate
Thu May  7 15:13:11 2020 acc on test dataset is :0.283
```

