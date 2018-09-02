

[TOC]

## 依赖知识

a) 熟悉[全连接层、损失函数的反向传播](0_1-全连接层、损失函数的反向传播.md)

b) 熟悉ReLU、LeakyReLU、PReLU、ELU、SELU



## ReLU

​          整流线性单元Rectified Linear Unit

### 前向过程

$$
ReLU(z)=\begin{cases}
z &  z>0 \\
0 & z<=0    \tag 1
\end{cases}
$$

### 后向过程

a) 我们将激活函数也看做一层, 设第$l$层输出为$z^l$, 经过激活函数后的输出为$z^{l+1}$

b) 记损失函数L关于第$l$ 层的输出$z^l$ 的偏导为$\delta^l = \frac {\partial L} {\partial z^l}  $ 

​        则损失函数L关于关于第l层的偏导如下：
$$
\begin{align}
&\delta^l = \frac {\partial L} {\partial z^{l+1}}   \frac {\partial z^{l+1}} {\partial z^{l}}  \\
&=\delta^{l+1} \frac {\partial ReLU(z^l)} {\partial z^{l}} \\
&=\delta^{l+1} \begin{cases}
1    & z^l>0 \\
0    & z^l<=0   
\end{cases} \\
&= \begin{cases}
\delta^{l+1}    & z^l>0 \\
0    & z^l<=0    \tag 2
\end{cases}
\end{align}
$$


## LeakyReLU

​           ReLU在取值小于零部分没有梯度，LeakyReLU在取值小于0部分给一个很小的梯度

### 前向过程

$$
LeakyReLU(z)=\begin{cases}
z &  z>0 \\
\alpha z & z<=0, \alpha=0.1    \tag 3
\end{cases}
$$

### 后向过程

同Relu可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha\delta^{l+1}    & z^l<=0, \alpha=0.1    \tag 4
\end{cases}
\end{align}
$$


## PReLU

​           参数化ReLU，形式同LeakyRelu,不过$\alpha$ 不是固定的常量而是根据数据学习到的。

论文地址：https://arxiv.org/pdf/1502.01852.pdf

### 前向过程

$$
PReLU(z)=\begin{cases}
z &  z>0 \\
\alpha z & z<=0, \alpha是与z相同形状的变量    \tag 5
\end{cases}
$$

### 后向过程

a) 同LeakyRelu可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha\delta^{l+1}    & z^l<=0,\alpha是需要学习的参数    \tag 6
\end{cases}
\end{align}
$$


b) 损失函数L关于关于参数$\alpha$的偏导为:
$$
\begin{align}
&\frac {\partial L} {\partial \alpha} = \frac {\partial L} {\partial z^{l+1}}   \frac {\partial z^{l+1}} {\partial \alpha} \\
&=\delta^{l+1} \frac {\partial PReLU(z^l)} {\partial \alpha} \\
&=\delta^{l+1} \begin{cases}
0  & z^l >0 \\
z^l & z^l<=0
\end{cases} \\
&= \begin{cases}
0  & z^l >0 \\
\delta^{l+1}z^l & z^l<=0  \tag 7
\end{cases} 
\end{align}
$$


## ELU

​           指数化ReLU，在取值小于0的部分使用指数

论文地址: https://arxiv.org/pdf/1511.07289.pdf

### 前向过程

$$
ELU(z)=\begin{cases}
z &  z>0 \\
\alpha(\exp(z)-1) & z<=0, \alpha=0.1    \tag 8
\end{cases}
$$

### 后向过程

同LeakyRelu可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha \delta^{l+1} \exp(z^l)    & z^l<=0    \tag 9
\end{cases}
\end{align}
$$

## SELU

​           缩放指数型线性单元, 就是对ELU加上一个缩放因子$\lambda$

论文地址: https://arxiv.org/pdf/1706.02515.pdf

### 前向过程

$$
RELU(z)=\lambda\begin{cases}
z &  z>0 \\
\alpha(\exp(z)-1) & z<=0    \tag {10}
\end{cases}
$$

​             其中$\lambda \approx 1.0507 , \alpha \approx  1.673$ (论文中有大段证明)

### 后向过程

同ELU可知损失函数L关于关于第l层的偏导为:
$$
\begin{align}&\delta^l = \lambda \begin{cases}
\delta^{l+1}    & z^l>0 \\
\alpha \delta^{l+1} \exp(z^l)    & z^l<=0    \tag {11}
\end{cases}
\end{align}
$$
