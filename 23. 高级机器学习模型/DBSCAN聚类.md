# DBSCAN聚类算法

DBSCAN(Density-Based Spatial Clustering of Applications with Noise，具有噪声的基于密度的聚类方法)是一种很典型的密度聚类算法，

DBSCAN的主要优点有：

- 1） 可以对任意形状的稠密数据集进行聚类，相对的，K-Means之类的聚类算法一般只适用于凸数据集。

- 2） 可以在聚类的同时发现异常点，对数据集中的异常点不敏感。

- 3） 聚类结果没有偏倚，相对的，K-Means之类的聚类算法初始值对聚类结果有很大影响。



DBSCAN的主要缺点有：

- 1）如果样本集的密度不均匀、聚类间距差相差很大时，聚类质量较差，这时用DBSCAN聚类一般不适合。

- 2） 如果样本集较大时，聚类收敛时间较长，此时可以对搜索最近邻时建立的KD树或者球树进行规模限制来改进。

- 3） 调参相对于传统的K-Means之类的聚类算法稍复杂，主要需要对距离阈值ϵϵ，邻域样本数阈值MinPts联合调参，不同的参数组合对最后的聚类效果有较大影响。
- 4）不是稳定算法。DBSCAN采用先来后到原则，如果有样本可以划分到多个类中，则先进行聚类的类别簇会标记这个样本为它的类别。



## 补充说明

参考文献的原理推导已经很明确，此处仅作一些补充说明：

### 1. 凸数据集

在凸几何中，凸集(convex set)是在凸组合下闭合的仿射空间的子集。更具体地说，在欧氏空间中，凸集是对于集合内的每一对点，连接该对点的直线段上的每个点也在该集合内。



判断一个数据集是否为凸集的一个简单方法就是使用不同的k值在数据集上进行kmeans聚类，如果效果都不好，则考虑该数据集是凸集

## 参考文献：

原理推导：https://www.cnblogs.com/pinard/p/6208966.html

