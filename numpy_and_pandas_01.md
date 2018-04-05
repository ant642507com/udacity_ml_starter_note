# 课程2:Numpy & Pandas - 第一部分

## 2.4 Numpy & Pandas中的一维数据

### 2.4.1 读取CSV文件中的内容，并返回数据集list
```python
import unicodecsv

def read_csv(filename):
    with open(filename, 'rb') as f:
        reader = unicodecsv.DictReader(f)
        return list(reader)

daily_management = read_csv('daily_management_full.csv')
```

### 2.4.2 从传入的数据集合中，对数据元素进行去重，并返回

```python
def get_unique_students(data):
    unique_students = set()
    for data_point in data:
        unique_students.add(data_point['acct'])
    return unique_students

unique_engagement_students = get_unique_students()
len(unique_engagement_students)
```
### 2.4.3 使用pandas内置的函数，读取CSV文件并去重
使用Pandas 实现了 2.4.1 和2.4.2 中代码的功能。
```python
import pandas as pd
daily_engagement = pd.read_csv('daily_management_full.csv')
len(daily_engagement['acct'].unique())
```
## 2.5 Numpy数组
一维数据结构:Pandas中的Series， NumPy (ie. Numerical Python) 中的Array。
Pandas的Series是建立在Numpy的array之上的，功能更丰富。

创建numpy array
```python
import numpy as np
countries = np.array(["Albania", "Angola",...])
```

### 2.5.1 Numpy array 与python list的用法相同点

### 2.5.1.1 通过下标访问Numpy元素
```python
a[0] = 'AL'
```
### 2.5.1.1 通过切片来访问某一范围元素
```python
a[1:3] = ["AA", "BB"]
print(a[:])
```
### 2.5.1.2 通过循环访问元素
```python
for x in a:
    ....
```
### 2.5.2 Numpy array 与python list的用法不同点
1. Numpy array每个元素都必须是相同的数据类型(string,boolean,int,etc...)
2. Numpy 提供了方便的函数。 mean(), std(), max(), sum()
3. Numpy array 可以是多维的

### 2.5.3 通过numpy.argmax返回最大值的索引
numpy.argmax 可以返回最大值的位置
使用数组的实现：
```python
def max_employment(countries, employment):
    max_country = None
    max_employment = 0
    for i in range(len(countries)):
        country = countries[i]
        country_employment = employment[i]
        
        if country_employment > max_employment:
            max_country = country
            max_employment = country_employment
        
    return (max_country, max_employment)
```
使用numpy的实现
```python
def max_employment(countries, employment):
    max_employment_index = employment.argmax()
    
    max_country = countries[max_employment_index]
    max_employment = employment[max_employment_index]
    
    return (max_country, max_employment)
```

### 2.5.4 numpy.dtype 查看numpy中的数据类型

## 2.5 Numpy 向量化运算
```python
1. 数学运算：+，-，*，/
2. 逻辑运算：&， |， ~ （and， or， not）
3. 比较运算：>, >=, <, <=, ==, !=

在 NumPy 中，a & b 执行 a 和 b 的“按位与”。这不一定要与执行“逻辑与”（“与”没有对应的向量版）的 a 和 b 相同。但是，如果 a 和 b 都是布尔型而非整数型数组，“按位与”和“逻辑与”的作用是一样的。

如果你想要对整数型向量进行“逻辑与”计算，你可以使用 NumPy 函数 np.logical_and(a,b)，或者先把它们转换为布尔型向量。

类似地，a | b 执行“按位或”，而 ~a 执行“按位非”。但是，如果数组包含布尔值，它们与执行“逻辑或”和“逻辑非”的效果是一样的。

NumPy 也有类似的函数：逻辑或，逻辑非，用于对含整数型数值的数组进行逻辑运算。

在答案中，我们要用 / 2.，而不是 / 2。注意 2 后面有一个句点。这是因为在 Python 2 中，将一个整数除以另一个整数 (2)，会舍去分数。所以如果输入是整数值，就会丢失信息。因此使用浮点数值 (2.)，我们就能保留结果小数点后的值了。
```

## 2.6 标准化数据
标准化数据定义：将各数据点转换为相对于平均值的标准偏差值。
用来描述某个数据点相比于其他数据点有什么区别

```python
"""对数据进行标准化求值"""
def standarize_data(values):
    return (values - values.mean()) / values.std()
```
## 2.7 Numpy索引数组

```python
a = np.array([1,2,3,4,5])
b = np.array([False, False, True, True, True])
# b作为a的index array
a[b] = [3,4,5]

a[a>3] = [4,5]
```
应用实例：求7天之后放弃的学员的学习平均时长
```python
def mean_time_for_paid_students(time_spent, days_to_cancel):
    return time_spent[days_to_cancel >= 7].mean()
```

## 2.8 +与+=的区别

Code Snap1:
```python
import numpy as np
a = np.array([1, 2, 3, 4])
b = a
"""虽然，a和b使用相同的内存空间，但有等号的加法运算不会创建新的数组"""
a += np.array([1, 1, 1, 1])
"""运算后，a与b地址空间相同"""
"""output value is array([2,3,4,5])"""
print(b)
```
Code Snap2: 
```python
import numpy as np
a = np.array([1, 2, 3, 4])
b = a
"""虽然，a和b使用相同的内存空间，但没有等号的加法运算会先创建一个新数组"""
a = a + np.array([1, 1, 1, 1])
"""运算后，a与b地址空间不同"""
"""output value isarray([1, 2, 3, 4])"""
print(b)
```

## 2.9 原地(in-place)与非原地
+=运算是原位运算，加法运算不是原位运算
原位运算是将新值存储在原来数据存放位置，覆盖原值

```python
import numpy as np
a = np.array([1, 2, 3, 4, 5, 6])
"""此处并没有创建新的数组，而是原数组的另一个视图（如果改变它，原数组也会变）"""
slice = a[:3]
slice[0] = 100
""" output value is array([100, 2, 3, 4, 5]) """
print(a)
```

Numpy数组行为与Python列表的行为有所不同
更改Numpy切片数据时，要格外谨慎

## 2.10 Pandas Series

Pandas Series 和 Numpy array 类似，但提供了额外的功能。
例如：s.describe()
Numpy array 的运算 同样适用于 Pandas Series。
（元素访问，切片，循环，以及提供的方便使用的函数mean,max,...）

```python
# 求两个Series中，两个都大于或都小于平均值的数量
# 相同方向数量：都大于数量+都小于数量
# 不同方向数量：
def variable_correlation(variable1, variable2):
    
    both_above = (variable1>variable1.mean()) & (variable2>variable2.mean())
    both_below = (variable1<variable1.mean()) & (variable2<variable2.mean())
    is_same_direction = both_above | both_below
    num_same_direction = is_same_direction.sum()
    num_different_direction = len(variable1) - num_same_direction
    
    return (num_same_direction, num_different_direction)
```
## 2.11 Pandas Series 索引
numpy array 和 pandas series 的区别在于， pandas series有索引值
pandas series 就像字典和列表的合集
pandas series 在不指定索引的情况下，将以从0开始的数字作为索引值
```python
life_expectancy = pd.Series([74.7, 75., 83.4, 57.6],
                            index=['Albania', 'Algeria', 'Andorra', 'Angola'])
'''通过下标或者索引访问元素,不够明确'''
life_expectancy[0]
'''通过 索引 访问元素'''
life_expectancy.loc['Angola']
'''通过下标来获取元素，访问方式更明确'''
life_expectancy.iloc[0]
```

pandas series argmax和loc的使用
```python
"""使用argmax获取最大值的索引"""
def max_employment(employment):
    max_country = employment.argmax()
    max_value = employment.loc[max_country]
    return (max_country, max_value)
```

## 2.12 向量化运算和Series索引，填充缺失值

将两个索引不同的Series相加，值的匹配是根据索引(index)进行，而不是位置
默认情况下 index会取并集，value中index相同的部分会相加，
index在两个Series不同的部分，值为NaN。

解决不出现NaN值的方法：
1. 使用pandas.Series.dropna() 来移除NaN。
2. 使用pandas.Series..add(s2, fill_value=0), 提前给出默认值。

## 2.13 通过 Pandas Series apply() 使用自定义函数

pandas.Series.apply() 和 Python中map.apply() 是一样的。

```python
import pandas as pd
s = pd.Series([1, 2, 3, 4, 5])
def add_one(x):
    return x + 1
print s.apply(add_one)
```

## 2.14 在Pandas中画图
如果变量 data 是一个 NumPy 数组或 Pandas Series，就像它是一个列表一样，代码
```python
import matplotlib.pyplot as plt
plt.hist(data)
plt.show()
```
将创建数据的直方图.

Pandas 库实际上已经内置了 matplotlib 库的绘图函数。

使用 data.plot() 创建 Series 的线条图。Series 索引被用于 x 轴，值被用于 y 轴。

使用pandas的一个绘图例子:
```python
import pandas as pd
import seaborn as sns

path = '/datasets/ud170/gapminder/'

employment = pd.read_csv(path + 'employment_above_15.csv',
                     index_col='Country')

employment_us = employment.loc['United States']

print(employment.index.values)

employment_us.plot()
```

如果想在IPython notebook中出现图表，加上以下代码
```python
%pylab inline
employment_us.plot()
```





