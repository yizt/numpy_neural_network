[TOC]



## 依赖知识

a) 熟悉[全连接层、损失函数的反向传播](0_1-全连接层、损失函数的反向传播.md)

b) 熟悉[卷积层的反向传播-多通道、无padding、步长1](0_2_2-卷积层的反向传播-多通道、无padding、步长1.md)

c) 熟悉池化层Max Pooling、Average Pooling、Global Average Pooling、Global Max Pooling

d) 熟悉以上三点的依赖知识



## 约定说明

a) $l$ 代表网络的第$l$ 层, $z^l$ 代表第$l$ 层输出，$z_{d,i,j}^l$ 代表第$l$ 层第$d$ 通道$(i,j)$ 位置的值; $z^l$ 的通道数为$C^l$, 高度和宽度分别为$H^l,\hat W^l$ ($\color{red}{避免与权重相同}$) 

b)  池化单元的大小为$(k_1^{l-1},k_2^{l-1})$ ; 池化单元的步长为$(s_1^{l-1},s_2^{l-1})$ ; padding为$(p^{l-1}_1,p^{l-1}_2)$

c) 记$pz^{l-1}$ 为l-1层增加零填充后的张量

d) 记损失函数L关于第$l$ 层输出$z^l$ 的偏导为$\delta^l = \frac {\partial L} {\partial z^l} $   



## 前向传播

### Max Pooling和Average Pooling

​        最大池化和平均池化，最大池化和平均池化和平均池化前向过程完全一样，只是使用的函数不同。

​        根据以上约定，经过池化层后输出的高度$H^l$和宽度$\hat W^l$分别为$(H^{l-1}+2\cdot p_1^{l-1}-k_1^{l-1})/s_1^{l-1}+1$ 和$(\hat W^{l-1}+2\cdot p_2^{l-1}-k_2^{l-1})/s_2^{l-1}+1$

​         因此最大池化的前向公式为：
$$
\begin{align}
&z^l_{c,i,j} = \max_{i\cdot s_1^{l-1} \le m <i\cdot s_1^{l-1}+k_1^{l-1}; j \cdot s_2^{l-1} \le n< j \cdot s_2^{l-1} + k_2^{l-1}} (pz_{c,i,j}^{l-1})  & i \in [0,H^l-1], j\in [0,\hat W^l-1]   \tag 1
\end{align}
$$

​	平均池化的前向公式为：

$$
\begin{align}
&z^l_{c,i,j} = \sum_{m=i\cdot s_1^{l-1}}^{i\cdot s_1^{l-1}+k_1^{l-1}-1} \sum_{n=j \cdot s_2^{l-1}}^{j \cdot s_2^{l-1} + k_2^{l-1}-1} (pz_{c,i,j}^{l-1})/(k_1^{l-1}\cdot k_2^{l-1})  & i \in [0,H^l-1], j\in [0,\hat W^l-1]   \tag 2
\end{align}
$$


### Global Max Pooling和Global Average Pooling

​         全局最大池化和全局平均池化更加简单,是对单个通道上所有的元素求最大值和均值。所以经过全局平均池化后输出就是一维的了。

​         因此全局最大池化的前向公式为：
$$
z_c^l=\max_{0 \le m<H^{l-1}; 0 \le n < \hat W^{l-1}} (z_{c,m,n}^{l-1})   \tag3
$$
​         全局平均池化的前向公式为：
$$
z^l_{c} = \sum_{m=0}^{H^{l-1}-1} \sum_{n=0}^{\hat W^{l-1}-1} (z_{c,i,j}^{l-1})/(H^{l-1}\cdot \hat W^{l-1})   \tag 4
$$




## 反向传播

### Max Pooling

​       设 $I(c,a,b)=\{(i,j)|\mathop{\arg\max}_{m,n}(pz_{c,i,j}^{l-1})_ {i\cdot s_1^{l-1} \le m <i\cdot s_1^{l-1}+k_1^{l-1}; j \cdot s_2^{l-1} \le n< j \cdot s_2^{l-1} + k_2^{l-1}}=(a,b)\}$ 代表最大池化过程中c通道上所有在l-1层最大值坐标在$(a,b)$位置的坐标$(i,j)$ (l层)的集合; 

​       则损失函数L关于最大池化层的偏导如下：
$$
\begin{align}
& \frac {\partial L} {\partial pz_{c,a,b}^{l-1} } = \sum_{(i,j) \in I(c,a,b)}
\frac {\partial L} {\partial z^l_{c,i,j}} \cdot \frac {\partial z^l_{c,i,j}} {\partial pz^{l-1}_{z,a,b}} \\
&=\sum_{(i,j) \in I(c,a,b)} \delta_{c,i,j}^l  \tag 5
\end{align}
$$

$$
\begin{align}
&\delta_{c}^{l-1}=( \frac {\partial L} {\partial pz_{c,a,b}^{l-1} })_{p_1^{l-1} \le a < H^{l-1}+p_1^{l-1};\ p_2^{l-1} \le b<\hat W^{l-1}+p_2^{l-1}} \\
&=(\sum_{(i,j) \in I(c,a,b)} \delta_{c,i,j}^l)_{p_1^{l-1} \le a < H^{l-1}+p_1^{l-1};\ p_2^{l-1} \le b<\hat W^{l-1}+p_2^{l-1}} \tag 6
\end{align}
$$

​           注：设矩阵$A=(a_{i,j})_{m \times n}$ 则$(a_{i,j})_{2 \le i <5; 4 \le j <8}$ 代表高度为第2行到第5行，宽度为第4列到第8列组成的矩阵



### Average Pooling

​           由公式(2)可知l层在高度i和宽度j上接收l-1层坐标范围分别是$[{i\cdot s_1^{l-1}},{i\cdot s_1^{l-1}+k_1^{l-1}-1}]  \text{和}[{j\cdot s_2^{l-1}},{j\cdot s_2^{l-1}+k_2^{l-1}-1}]$ ; 即
$$
\begin{cases}
{i\cdot s_1^{l-1}} \le m \le {i\cdot s_1^{l-1}+k_1^{l-1}-1}  \\
{j\cdot s_2^{l-1}} \le n \le {j\cdot s_2^{l-1}+k_2^{l-1}-1}  \tag 7
\end{cases}
$$
​           可以推知l-1层坐标(m,n)对应l层坐标范围是：
$$
\begin{cases}
\lfloor \frac {m-k_1^{l-1}+1} {s_1^{l-1}} \rfloor \le i \le \lfloor \frac {m} {s_1^{l-1}} \rfloor \\
\lfloor \frac {n-k_2^{l-1}+1} {s_2^{l-1}} \rfloor \le j \le \lfloor \frac {n} {s_2^{l-1}} \rfloor  \tag 8
\end{cases}
$$
​           故则损失函数L关于平均池化层的偏导如下：
$$
\begin{align}
&\delta_{c}^{l-1}=(\frac {\partial L} {\partial pz_{c,m,n}^{l-1} } )_{p_1^{l-1} \le m < H^{l-1}+p_1^{l-1};\ p_2^{l-1} \le n<\hat W^{l-1}+p_2^{l-1}} \\
&=(\sum_i\sum_j\delta_{c,i,j}^l/(k_1^{l-1}\cdot k_2^{l-1}) )_{p_1^{l-1} \le m < H^{l-1}+p_1^{l-1};\ p_2^{l-1} \le n<\hat W^{l-1}+p_2^{l-1}}   \tag 9
\end{align}
$$
​           其中(i,j)满足公式(8)的条件，并且大于等于0



### Global Max Pooling

​           全局最大池化的反向公式如下
$$
\begin{align}
&\delta_{c,i,j}^{l-1}=\begin{cases}
\delta^l_c ;  &如果(i,j)=\arg\max_{m,n}(z_{c,m,n}) \\
0 ; &其它  \tag {10}
\end{cases}
\end{align}
$$
​           注意第l层是一维的

### Global Average Pooling

​        全局平均池化就是后一层梯度平均的分给前一层所有的神经元，反向公式如下：
$$
\delta_{c,i,j}^{l-1}=\delta^l_c/(H^l \cdot \hat W^{l-1})  \tag {12}
$$


