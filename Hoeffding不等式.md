#### 1.Hoeffding不等式：

对与$$m$$个独立的随机变量$$X_1,X_2,...,X_m$$，且$$\forall X_i X_i\subset[a_i,b_i]， (b_i>a_i)$$。令$$\overline{x}=\frac{1}{m}\sum_{i=1}^m X_i$$,则如下不等式成立：

​                                      $$P(\overline{x}-E[\overline{x}\ge t]\le e^{-\frac{2t^2m^2}{\sum_{i=1}^{m}(b_i-a_i)^2}} $$





那么，怎么证明呢？哈哈哈，就不难为自己了...



我们可以看看这个公式的用途：

在统计推断中，我们可以利用样本的统计量来推断总体的参数，譬如使用样本均值来估计总体期望（可以思考一下之前说过的bagging的想法）。从hoeffding不等式可以看出，当n逐渐变大时，不等式的UpperBound越来越接近0，所以样本期望越来越接近总体期望。