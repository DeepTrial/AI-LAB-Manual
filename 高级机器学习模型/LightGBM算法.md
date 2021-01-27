# LightGBM算法

基于梯度提升方法的集成决策树模型在机器学习中应用十分广泛。梯度提升方法在构建决策树时，需要遍历所有特征的所有取值，查找最优切分点。但是这种基于pre-sorted的方法十分耗时，在处理高维的大数据集时十分困难。LightGBM算法在查找特征的最优切分点时，没有使用基于pre-order的方法，而是使用了基于histogram的方法，大大加快运行速度，降低了内存消耗。

若将lightgbm与XGBoost相比，lightgbm算法的改进之处包括：

- 采用基于直方图（histogram）的决策树算法。XGBoost使用基于pre-sorted方法查找最有特征的最优切分点，但面对高维大数据的处理速度较慢（相比基于直方图的方法）。
- 面对大量数据，设计了GOSS（基于梯度的one-side）采样方法提高训练速度。机器学习算法面对大数据量时候都会使用采样的方式（根据样本权值）来提高训练速度。
- 使用EFB（互斥特征捆绑）提高基于直方图的算法对稀疏数据的处理能力。基于直方图的方法在处理稀疏数据时效率较低，因此lightgbm设计了EFB方法处理稀疏数据。



## 1.基于直方图的方法

### 1.1直方图算法的思想

直方图算法的基本思想是将连续的特征离散化为 k 个离散特征，同时构造一个宽度为 k 的直方图用于统计信息（含有 k 个 bin）。利用直方图算法我们无需遍历数据，只需要遍历 k 个 bin 即可找到最佳分裂点。

我们知道特征离散化的具有很多优点，如存储方便、运算更快、鲁棒性强、模型更加稳定等等。对于直方图算法来说最直接的有以下两个优点（以 k=256 为例）：

- **内存占用更小：**XGBoost 需要用 32 位的浮点数去存储特征值，并用 32 位的整形去存储索引，而 LightGBM 只需要用 8 位去存储直方图，相当于减少了 1/8；
- **计算代价更小：**计算特征分裂增益时，XGBoost 需要遍历一次数据找到最佳分裂点，而 LightGBM 只需要遍历一次 k 次，直接将时间复杂度从 ![[公式]](https://www.zhihu.com/equation?tex=+O%28%5C%23data++%2A+%5C%23feature%29+) 降低到 ![[公式]](https://www.zhihu.com/equation?tex=+O%28k++%2A+%5C%23feature%29+) ，而我们知道 ![[公式]](https://www.zhihu.com/equation?tex=%5C%23data+%3E%3E+k) 。





### GOSS采样方法





## 参数注释

##  参考文献

1. https://blog.csdn.net/qq_24519677/article/details/82811215
2. https://blog.csdn.net/maqunfi/article/details/82219999
3. https://zhuanlan.zhihu.com/p/87885678
4. https://blog.csdn.net/anshuai_aw1/article/details/83040541

