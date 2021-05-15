# numpy学习教程

## numpy数据存储顺序（order）

numpy在创建数组时可以指定数组数据在内存中存储的顺序，可选参数为‘C’即C语言方式，行存储，‘F’即Fortran语言方式，列存储。对于矩阵运算而言，列存储会有一定的性能优势。

```
a=np.zeros(data,dtype,order='C')
```

两种存储方式的区别：[(1 条消息) numpy中 C order与F order的区别是什么？ - 知乎 (zhihu.com)](https://www.zhihu.com/question/23798415)

## numpy广播规则（broadcast）

**广播的规则:**

- 让所有输入数组都向其中形状最长的数组看齐，形状中不足的部分都通过在**前面加 1 补齐**。
- 输出数组的形状是输入数组形状的各个维度上的最大值。
- 如果输入数组的某个维度和输出数组的对应维度的长度相同或者其长度为 1 时，这个数组能够用来计算，否则出错。
- 当输入数组的某个维度的长度为 1 时，沿着此维度运算时都用此维度上的第一组值。

**简单理解：**对两个数组，分别比较他们的每一个维度（若其中一个数组没有当前维度则忽略），满足：

- 数组拥有相同形状。
- 当前维度的值相等。
- 当前维度的值有一个是 1。

若条件不满足，抛出 **"ValueError: frames are not aligned"** 异常。

例如：4x3的数组与一维数组（3个元素）相加时会触发广播机制，但4x3的数组与3x1的数组相加时会抛出异常



## numpy数学运算

numpy内建的数学运算包括：

- 三角函数 

  ```
  np.sin() np.cos() np.tan()
  np.arcsin() np.arccos() np.arctan()
  ```

- 舍入函数

  ```
  np.around() #四舍五入
  np.floor() #向下取整
  np.ceil() #向上取整
  ```

- 加减乘除

  ```
  np.add(a,b) np.subtract(a,b) np.multiply(a,b) np.divide(a,b)
  ```

- 倒数

  ```
  np.reciprocal()
  ```

- 乘方

  ```
  np.power(a,b)
  np.exp(a)     #e的次方
  ```

- 取余

  ```
  np.mod()
  np.reminder() #与mod功能相同
  ```

## numpy位运算

- 按位与 np.bitwise_and(7,13) np.logical_and()
- 按位或 np.bitwise_or(7,13) np.logical_or()
- 异或 np.logical_xor(7,13)
- 翻转 np.invert
- 二进制表示 np.binary_repr()
- 左移 np.left_shift() 右移 np.right_shift()