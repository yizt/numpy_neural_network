[TOC]



## 依赖知识

a) 熟悉[全连接层、损失函数的反向传播](0_1-全连接层、损失函数的反向传播.md)

b) 熟悉[卷积层的反向传播-单通道、无padding、步长1](0_2_1-卷积层的反向传播-单通道、无padding、步长1.md)

c) 熟悉[卷积层的反向传播-多通道、无padding、步长1](0_2_2-卷积层的反向传播-多通道、无padding、步长1.md)

d) 熟悉[卷积层的反向传播-多通道、无padding、步长不为1](0_2_3-卷积层的反向传播-多通道、无padding、步长不为1.md)

e) 熟悉以上4点的依赖知识



## 约定说明

a) $l$ 代表网络的第$l$ 层, $z^l$ 代表第$l$ 层卷积，$z_{d,i,j}^l$ 代表第$l$ 层卷积第$d$ 通道$(i,j)$ 位置的值; $z^l$ 的通道数为$C^l$, 高度和宽度分别为$H^l,\hat W^l$ ($\color{red}{避免与权重相同}$) 

b) $W^{l-1},b^{l-1}$ 代表连接第$l-1$ 层和第$l$ 层的卷积核权重和偏置; 卷积核的维度为$(k_1^{l-1},k_2^{l-1})$ ; 卷积核的步长为$(s_1^{l-1},s_2^{l-1})$ ,padding 为$(p_1^{l-1},p_2^{l-1})$ , 记$pz^{l-1}$ 为$z^{l-1}$ 高度和宽度这两个维度填充padding后的多维向量。

c) 记损失函数L关于第$l$ 层卷积的输出$z^l$ 的偏导为$\delta^l = \frac {\partial L} {\partial z^l}  \ \ \ (3)$   



## 前向传播

​        有padding和没padding的前向传播过程完全一样，只需要将 [无padding](0_2_3-卷积层的反向传播-多通道、无padding、步长不为1.md) 中所有公式中的$z^{l-1}$ 替换为$pz^{l-1}$ 即可。

​        因此根据以上约定，卷积核权重$W^{l-1} \in \Bbb R^{k_1^{l-1} \times k_2^{l-1} \times C^{l-1} \times C^{l}}$ ,偏置$b^{l-1} \in \Bbb R^{C^l}$ ,每个输出通道一个偏置。 则有第$l$ 层卷积层,第$d$个通道输出为:
$$
\begin{align}
&z^l_{d,i,j} = \sum_{c=1}^{C^{l-1}}\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1} W_{m,n,c,d}^{l-1} pz_{c,i \cdot s_1^{l-1}+m,j \cdot s_2^{l-1}+n}^{l-1} + b^{l-1}_d  & i \in [0,H^l-1], j\in [0,\hat W^l-1]\tag 4
\end{align}
$$

​       其中：$H^l = (H^{l-1} - k_1^{l-1} + 2p_1^{l-1})/s_1^{l-1}+ 1;\ \ \ \ \  \hat W^l = (\hat W^{l-1} - k_2^{l-1} +2p_2^{l-1})/s_2^{l-1}+ 1 $ ;



## 反向传播

​         同样反向传播过程中，也只需要将 [无padding](0_2_3-卷积层的反向传播-多通道、无padding、步长不为1.md) 中所有公式中的$z^{l-1}$ 替换为$pz^{l-1}$ 即可。

### 权重梯度

a) 首先来看损失函数$L​$关于第$l-1$层权重$W^{l-1}$和偏置$b^{l-1}$的梯度：
$$
\begin{align}
&\frac {\partial L} {\partial W_{m,n,c,d}^{l-1}} 
= \sum_i \sum_j \frac {\partial L} {\partial z^l_{d,i,j}} * \frac {\partial z^l_{d,i,j}} {\partial W_{m,n,c,d}^{l-1}} &//l层的d通道每个神经元都有梯度传给权重W^{l-1}_{m,n,c,d}\\
&=\sum_i \sum_j \delta^l_{d,i,j} * \frac {\partial ( \sum_{c=1}^{C^{l-1}}\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1} W_{m,n,c,d}^{l-1} pz_{c,i \cdot s_1^{l-1}+m,j \cdot s_2^{l-1}+n}^{l-1} + b^{l-1}_d  )} {\partial W^{l-1}_{m,n,c,d}}  \\
&=\sum_i \sum_j \delta^l_{d,i,j} * pz_{c,i \cdot s_1^{l-1}+m,j \cdot s_2^{l-1}+n}^{l-1} \tag 5
\end{align} \\
$$




b) 损失函数$L$关于第$l-1$层偏置$b^{l-1}$的梯度同
$$
\begin{align}
\frac {\partial L} {\partial b^{l-1}_d} =\sum_i \sum_j \delta^l_{d,i,j}  \tag 6
\end{align}
$$

### l-1层梯度​

​         根据 [无padding](0_2_3-卷积层的反向传播-多通道、无padding、步长不为1.md) 中公式(8)有
$$
\frac {\partial L} {\partial pz^{l-1}_{c,i,j}}=\sum_{d=1}^{C^l}\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}rot_{180^\circ} W^{l-1}_{m,n,c,d}p\delta^{l_{padding}}_{d,i+m,j+n} \tag 8
$$

​           其中$p\delta^{l_{padding}}_{d,i,j}​$ 是$\delta^l​$ 在行列直接插入$(s_1^{l-1}-1,s_2^{l-1}-1)​$ 行列零元素后(即$\delta^{l_{padding}}​$)，再在元素外围填充高度和宽度为 $(k_1^{l-1}-1,k_2^{l-1}-1)​$ 的零元素后的梯度矩阵。

​           我们知道$\delta^{l-1}_c$ 就是$(\frac {\partial L} {\partial pz^{l-1}_{c,i,j}})_{h \times w}$ 去除高度和宽度的填充后的矩阵，因此有
$$
\begin{align}
&\delta^{l-1}_c=(\frac {\partial L} {\partial pz^{l-1}_{c,i,j}})_{p_1^{l-1} \le i < H^{l-1} + p_1^{l-1},p^{l-1}_2 \le j < \hat W^{l-1}+p_2^{l-1}} \\
&=(\sum_{d=1}^{C^l}\sum_{m=0}^{k_1^{l-1}-1} \sum_{n=0}^{k_2^{l-1}-1}rot_{180^\circ} W^{l-1}_{m,n,c,d}p\delta^{l_{padding}}_{d,i+m,j+n})_{p_1^{l-1} \le i < H^{l-1} + p_1^{l-1},p^{l-1}_2 \le j < \hat W^{l-1}+p_2^{l-1}} \tag 9
\end{align}
$$




​          