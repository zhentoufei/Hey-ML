**前言：**在介绍Bagging和Boosting算法之前，首先要简单了解什么是集成学习？

集成学习（Ensemble Learning）是目前模式识别与机器学习中常用的一种学习算法，是使用一系列的学习器（分类器）通过某种规则（投票法、加权投票等）将各分类器的学习结果进行融合，达到比单学习器识别效果更好地目的。

可以打一个简单的比喻，如果我们将“学习器”看做是一个“人”，现在我们需要进行的任务是识别汉字。一个人的识别内容终归是有限的，但是如果我们现在利用三个人来识别，同一个字当A识别错误时，B、C识别正确，最终以少数服从多数的原则取BC的识别结果，那么相比较只用A一个人来识别汉字的情况，我们的准确率会大大提升。当然也许这里有人会存在疑问：万一A本身是一个很有文化的人，B是一个不识字的人，那么叫他们两个一起识别汉字岂不是会拉低整体识别率么？这里也就牵扯到了学习器与学习器之间[差异性度量](http://www.infocool.net/kb/Other/201608/170553.html)的话题，这个话题在本文中不详细讲述，有兴趣的朋友可以自己查询了解，最白话的解释就是：参与集成学习的学习器们需要具有一定的差异值，既不能完全相同，也不能差异的太大。

​                                                                     ----这段话来自[模式识别与机器学习—bagging与boosting](https://zhuanlan.zhihu.com/p/26112517)

------



#### Bagging

> **Bagging算法** （英语：**B**ootstrap **agg**regat**ing**，引导聚集算法），又称**装袋算法**，是[机器学习](https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)领域的一种[团体学习](https://zh.wikipedia.org/w/index.php?title=%E5%9B%A2%E4%BD%93%E5%AD%A6%E4%B9%A0&action=edit&redlink=1)[算法](https://zh.wikipedia.org/wiki/%E7%AE%97%E6%B3%95)。最初由[Leo Breiman](https://zh.wikipedia.org/w/index.php?title=Leo_Breiman&action=edit&redlink=1)于1994年提出。Bagging算法可与其他[分类](https://zh.wikipedia.org/wiki/%E7%BB%9F%E8%AE%A1%E5%88%86%E7%B1%BB)、[回归](https://zh.wikipedia.org/wiki/%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90)算法结合，提高其准确率、稳定性的同时，通过降低结果的[方差](https://zh.wikipedia.org/wiki/%E6%96%B9%E5%B7%AE)，避免[过拟合](https://zh.wikipedia.org/wiki/%E8%BF%87%E6%8B%9F%E5%90%88)的发生。
>
> ​															-----来自[维基百科](https://zh.wikipedia.org/wiki/Bagging%E7%AE%97%E6%B3%95)

**1. 算法步骤**：

[1] **抽取**：每一轮从哦那个原始样本集合中使用bootstrap方法抽取n个训练样本（*存在这样的状况：有的样本被多次抽取到，个别样本一次没有被抽中*）。共进行N轮抽取，得到N个训练集合（*N个训练集合中间是相互独立的*）

[2] **训练**：每次使用一个训练集合得到一个模型，N个训练集合共得到N个模型（*具体的模型可以采用不同的分类或者回归方法，如决策树，感知机等*）

[3] **输出**：对分类问题将N个模型的输出结果采用投票的方式得到分类结果

​                  对回归问题，计算上诉模型的均值作为最后的结果（*所有模型的重要性相同*）

**2. 关于随机森林特征总要行的选择?**



**3. 为什么说bagging是减少variance？**

*这是上次在[知乎](https://www.zhihu.com/question/26760839)上看到的一个问题，关于这个问题的回答从[过拟合](https://www.zhihu.com/people/guo-ni-he/answers)的回答中学习到了不少*

从上面的定义中可以看到，bootstrap的采样一般会产生具有一定相**似度的训练集合**，而且在实际的过程中我们一般会使用**相同的子模型**(就是针对每个训练集合的训练模型啦~)，所以可以做一个这样的结论：

各个模型具有近似相等的bias和variance(事实上，各个模型的分布也几乎相同，但是不独立，毕竟，有放回重采样得到的两个子集训得两个子模型，这两个子集可能有重叠，甚至完全相同，这样模型不会独立)。

由于$$E[\frac{\sum_{i=1}^n a_i}{n}]=E[X_i]$$，所以bagging后的bias和单个子模型接近，一般来说，不能显著降低bias。另外一方面，若各个子模型独立，则$$Var(\frac{\sum X_i }{n})=\frac{Var(X_i)}{n}$$，此时可以显著降低variance。若各个子模型完全相同，则$$Var(\frac{\sum X_i}{n})=Var(X_i)$$，此时不会降低variance。bagging方法得到的各个子模型具有一定的相关性，属于上述两个极端状况的中间态，因此可以一定程度降低variance。为了进一步降低varicance。RandomForest通过随机选取变量子集做你和的方式de-correlated了各子模型(tree)，时得variance进一步降低。

为了进一步说明，我们可以看看Elements of Statistical Learning 2nd edition这本书的588页：

假设又i.d的n个随机变量，方差$$\sigma^2$$，两两变量之间的相关性为$$\rho$$，则$$\frac{\sum X_i}{n}$$的方差为

​					$$\rho*\sigma^2+(1-\rho)*\sigma^2/n$$

bagging降低了第二项，random forest是同时降低两项。

*注*：关于上面相关公式的推到可以参见[Variance](https://en.wikipedia.org/wiki/Variance#Sum_of_correlated_variables)

**4. 关于并行计算**

随机森林的并行计算显而易见，只要选择出相应的数据子集，就可进行训练了一棵子树了

------



#### Boosting

> **提升方法**（Boosting），是一种可以用来减小[监督式学习](https://zh.wikipedia.org/wiki/%E7%9B%A3%E7%9D%A3%E5%BC%8F%E5%AD%B8%E7%BF%92)中[偏差](https://zh.wikipedia.org/wiki/%E5%81%8F%E5%B7%AE)的[机器学习](https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)元算法。面对的问题是迈可·肯斯（Michael Kearns）提出的：[[1\]](https://zh.wikipedia.org/wiki/%E6%8F%90%E5%8D%87%E6%96%B9%E6%B3%95#cite_note-Kearns88-1)一组“弱学习者”的集合能否生成一个“强学习者”？弱学习者一般是指一个分类器，它的结果只比随机分类好一点点；强学习者指分类器的结果非常接近真值。
>
> ​															-----来自[维基百科](https://zh.wikipedia.org/wiki/%E6%8F%90%E5%8D%87%E6%96%B9%E6%B3%95)



**为什么说boosting是降低bias？**

从优化角度来看boosting是使用forward-stagewise这种贪心算法去最小化损失函数$$L(y_i, \sum_i a_if_i(x))$$例如，常见的Adaboost就是等价于用这种方法最小化exponential loss：$$L(y, f(x))=exp(-yf(x))$$。这里的forward-stagewise，是指在迭代的第n步，求解子模型$$f(x)$$以及步长$$a$$（或者叫组合系数），来最小化$$L(y, f_{n-1}+af(x))$$，这里的$$f_{n-1}(x)$$是前面$$n-1$$步得到的子模型的和。因此，boosting是在序列化（sequential）的最小化损失函数，期bias自然逐步下降，但是由于采取这种sequential， adaptive的策略，各子模型之间是强相关的，于是子模型之和并不能显著降低variance。所以说boosting主要还是靠降低bias来提升预测精度。

**关于并行计算**

boosting有强力工具stochastic gradient boosting，其本质等价于sgd，并行化方法参考[async sgd](http://tech.it168.com/a2013/0621/1498/000001498141.shtml)之类的业界常用方法即可。

应用：boosting思想的一种改进型adaboost方法在邮件过滤，文本分类中有很好的性能。

#### Bagging&Boosting的区别

1 **取样方式不同**：bagging采用均匀取样，而boosting根据错误率来采样

2 **样本关系不同**：baging的训练集选择是随机的，各轮训练集之前互相独立，而boosting的各轮训练集的选择与前面各轮的学习结果相关

3 **权重关系**：bagging的各个预测函数没有权重，而boost有权重

4 **函数生成**：bagging的各个函数可以并行生成，而boosting的各个预测函数只能顺序生成

5 **关于应用**：对于象神经网络这样极为消耗时间的算法，bagging可通过并行节省大量的时间开销

6 **其他**：baging和boosting都可以有效地提高分类的准确性。在大多数数据集中，boosting的准确

性要比bagging高。有一些数据集中国，boosting会overfit。



------

#### 参考

1 [ 为什么说bagging是减少variance，而boosting是减少bias?](https://www.zhihu.com/question/26760839)

2 [ 快速理解bootstrap,bagging,boosting-三个概念](http://blog.csdn.net/wangqi880/article/details/49765673)

3 [模式识别与机器学习—bagging与boosting](https://zhuanlan.zhihu.com/p/26112517)