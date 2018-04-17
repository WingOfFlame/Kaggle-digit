 - take a sample of train data 
 - grid search can be used to find the best values of parameters C and gamma.
 - After achieving a decent score, I increased the sample size and again ran the algorithm for different combination of parameters. The values that yielded best scores in my case were: kernel - rbf, C - 7, gamma - 0.009

- Data Augmentation
    - flip, crop, shift, color, lighting
- Dropout: avoid overfit
- ReLU
    - \+ ReLU本质上是分段线性模型，前向计算非常简单，无需指数之类操作；
    + \+ ReLU的偏导也很简单，反向传播梯度，无需指数或者除法之类操作；
    + \+ ReLU不容易发生梯度发散问题，Tanh和Logistic激活函数在两端的时候导数容易趋近于零，多级连乘后梯度更加约等于0；
    + \+ ReLU关闭了左边，从而会使得很多的隐层输出为0，即网络变得稀疏，起到了类似L1的正则化作用，可以在一定程度上缓解过拟合。
    - \- 左边全部关了很容易导致某些隐藏节点永无翻身之日，
    - \* pReLU、random ReLU等改进，
    - \* 而且ReLU会很容易改变数据的分布，因此ReLU后加Batch Normalization也是常用的改进的方法。
