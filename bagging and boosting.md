#### Bagging

> **Bagging算法** （英语：**B**ootstrap **agg**regat**ing**，引导聚集算法），又称**装袋算法**，是[机器学习](https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)领域的一种[团体学习](https://zh.wikipedia.org/w/index.php?title=%E5%9B%A2%E4%BD%93%E5%AD%A6%E4%B9%A0&action=edit&redlink=1)[算法](https://zh.wikipedia.org/wiki/%E7%AE%97%E6%B3%95)。最初由[Leo Breiman](https://zh.wikipedia.org/w/index.php?title=Leo_Breiman&action=edit&redlink=1)于1994年提出。Bagging算法可与其他[分类](https://zh.wikipedia.org/wiki/%E7%BB%9F%E8%AE%A1%E5%88%86%E7%B1%BB)、[回归](https://zh.wikipedia.org/wiki/%E5%9B%9E%E5%BD%92%E5%88%86%E6%9E%90)算法结合，提高其准确率、稳定性的同时，通过降低结果的[方差](https://zh.wikipedia.org/wiki/%E6%96%B9%E5%B7%AE)，避免[过拟合](https://zh.wikipedia.org/wiki/%E8%BF%87%E6%8B%9F%E5%90%88)的发生。
>
> ​															-----来自[维基百科](https://zh.wikipedia.org/wiki/Bagging%E7%AE%97%E6%B3%95)

**算法步骤**：

[1] **抽取**：每一轮从哦那个原始样本集合中使用bootstrap方法抽取n个训练样本（*存在这样的状况：有的样本被多次抽取到，个别样本一次没有被抽中*）。共进行N轮抽取，得到N个训练集合（*N个训练集合中间是相互独立的*）

[2] **训练**：每次使用一个训练集合得到一个模型，N个训练集合共得到N个模型（*具体的模型可以采用不同的分类或者回归方法，如决策树，感知机等*）

[3] **输出**：对分类问题将N个模型的输出结果采用投票的方式得到分类结果

​		  对回归问题，计算上诉模型的均值作为最后的结果（*所有模型的重要性相同*）

**为什么说bagging是减少variance？**

*这是上次在[知乎](https://www.zhihu.com/question/26760839)上看到的一个问题，关于这个问题的回答从[过拟合](https://www.zhihu.com/people/guo-ni-he/answers)的回答中学习到了不少*

从上面的定义中可以看到，bootstrap的采样一般会产生具有一定相**似度的训练集合**，而且在实际的过程中我们一般会使用**相同的子模型**(就是针对每个训练集合的训练模型啦~)，所以可以做一个这样的结论：

各个模型具有近似相等的bias和variance(事实上，各个模型的分布也几乎相同，但是不独立)









#### Boosting

> **提升方法**（Boosting），是一种可以用来减小[监督式学习](https://zh.wikipedia.org/wiki/%E7%9B%A3%E7%9D%A3%E5%BC%8F%E5%AD%B8%E7%BF%92)中[偏差](https://zh.wikipedia.org/wiki/%E5%81%8F%E5%B7%AE)的[机器学习](https://zh.wikipedia.org/wiki/%E6%9C%BA%E5%99%A8%E5%AD%A6%E4%B9%A0)元算法。面对的问题是迈可·肯斯（Michael Kearns）提出的：[[1\]](https://zh.wikipedia.org/wiki/%E6%8F%90%E5%8D%87%E6%96%B9%E6%B3%95#cite_note-Kearns88-1)一组“弱学习者”的集合能否生成一个“强学习者”？弱学习者一般是指一个分类器，它的结果只比随机分类好一点点；强学习者指分类器的结果非常接近真值。
>
> ​															-----来自[维基百科](https://zh.wikipedia.org/wiki/%E6%8F%90%E5%8D%87%E6%96%B9%E6%B3%95)