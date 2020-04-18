# -*- coding: utf-8 -*-
"""
 @File    : vgg.py
 @Time    : 2020/4/18 上午11:01
 @Author  : yizuotian
 @Description    :
"""
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


def test():
    import time
    vgg16 = VGG(name='vgg11')
    start = time.time()
    y = vgg16.forward(np.random.randn(6, 3, 32, 32))
    loss, dy = cross_entropy_loss(y, np.abs(np.random.randn(6, 10)))
    vgg16.backward(dy)
    print('耗时:{}'.format(time.time() - start))


if __name__ == '__main__':
    test()
