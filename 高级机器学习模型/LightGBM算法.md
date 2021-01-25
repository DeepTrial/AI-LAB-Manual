# LightGBM算法

基于梯度提升方法的集成决策树模型在机器学习中应用十分广泛。梯度提升方法在构建决策树时，需要遍历所有特征的所有取值，查找最优切分点。但是这种基于pre-order排序的方法十分耗时，在处理高维的大数据集时十分困难。LightGBM算法在查找特征的最优切分点时，没有使用基于pre-order的方法，而是使用了基于histogram的方法，大大加快运行速度，



## 参数注释

##  参考文献

1. https://blog.csdn.net/qq_24519677/article/details/82811215
2. https://blog.csdn.net/maqunfi/article/details/82219999

