# 课程3:Numpy & Pandas - 第二部分

## 3.1 二维NumPy数组
* 二维数组的不同实现
    1. Python : List of Lists
    1. NumPy : 2D array
    1. Pandas : DataFrame
    
* NumPy的运行方式使得创建一个二维数组更节约内存

[此页面](http://docs.scipy.org/doc/numpy/reference/arrays.ndarray.html#internal-memory-layout-of-an-ndarray)
介绍了 2D NumPy 数组的内存布局。

* 获取元素的方法a[1, 3]而不是a[1][3]
* mean(), std(), etc...等函数将在整个数组上运行。

```python
"""
各个地铁站每天的人流量，数据格式
第1天[车站1，车站2，。。。]，
第2天[车站1，车站2，。。。]，
。。。
"""
ridership = np.array([
    [1478, 3877, 3674, 2328, 2539],
    [1613, 4088, 3991, 6461, 2691]
])
# 选取第二行第四列元素
print(ridership[1, 3])
# 选择第二行和第三行元素中的第四列和第五列元素
print(ridership[1:3, 3:5])
# 选择第二行的整行元素
print(ridership[1, :])
# 第一行元素 与 第二行元素的和
print(ridership[0, :] + ridership[1, :])

a = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
b = np.array([[1, 1, 1], [2, 2, 2], [3, 3, 3]])
print(a + b)

"""计算第一天中流量最大的车站的人流量的平均值"""
def mean_riders_for_max_station(ridership):
    max_station = ridership[0,:].argmax()
    # 第一天车流量最大的车站的平均值
    overall_mean = ridership[:,max_station].mean()
    # 所有车站的平均值
    mean_for_max = ridership.mean()
    
    return (overall_mean, mean_for_max)
```

## 3.2 NumPy轴
* 按照行或者按照列计算平均值
    1. axis=0, 计算每一列的平均值
    1. axis=1, 计算每一行的平均值
```python
print(ridership.mean(axis=0))
print(ridership.sum(axis=1))
ridership.min(axis=0) 
ridership.max(axis=0) 
```

## 3.3 NumPy和Pandas数据类型

```python
import numpy as np
np.array([1,2,3,4,5]).dtype
# dtype('int64')
```
* numpy中所有数据是不同的数据类型时，会出现 ***dtype='<U5'***
* numpy中出现不同的数据类型时，将不能使用mean(),max(),etc...等函数求值。
  会出现 ***TypeError: cannot perform reduce with flexible type***

* pandas.DataFrame 也是二维数据结构，但每一列可以是不同的数据类型
* pandas.DataFrame 拥有类似pandas.Series的索引。
    1. 每一行都有一个索引值
    2. 每一列都有一个名称

```python
import pandas as pd
enrollments_df = pd.DataFrame({
    "account_key":[1, 2, 3, 4, 5]
})
```

* pandas.DataFrame.mean() 默认计算每一列的平均值，并忽略掉其他值(ie.NaN)
* pandas.DataFrame功能性函数默认计算每一列的数值，而并不是整个DataFrame的，这样做是因为不同的列可能是不同的数据类型，而这样做似乎更合理些。
* 利用axis计算各行的平均值
```python
enrollments_df.mean(axis=1)
```

## 3.4 访问DataFrame中的元素

创建如下DataFrame
```python
ridership_df = pd.DataFrame(
    data=[[   0,    0,    2,    5,    0],
          [1478, 3877, 3674, 2328, 2539],
          [1613, 4088, 3991, 6461, 2691],
          [1560, 3392, 3826, 4787, 2613],
          [1608, 4802, 3932, 4477, 2705],
          [1576, 3933, 3909, 4979, 2685],
          [  95,  229,  255,  496,  201],
          [   2,    0,    1,   27,    0],
          [1438, 3785, 3589, 4174, 2215],
          [1342, 4043, 4009, 4665, 3033]],
    index=['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
           '05-06-11', '05-07-11', '05-08-11', '05-09-11', '05-10-11'],
    columns=['R003', 'R004', 'R005', 'R006', 'R007']
)
```

* 使用.loc按照索引来获取相应的元素(Label-location)

```
Purely label-location based indexer for selection by label.

.loc[] is primarily label based, but may also be used with a boolean array.

Allowed inputs are:

A single label, e.g. 5 or 'a', (note that 5 is interpreted as a label of the index, and never as an integer position along the index).
A list or array of labels, e.g. ['a', 'b', 'c'].
A slice object with labels, e.g. 'a':'f' (note that contrary to usual python slices, both the start and the stop are included!).
A boolean array.
A callable function with one argument (the calling Series, DataFrame or Panel) and that returns valid output for indexing (one of the above)
.loc will raise a KeyError when the items are not found.

See more at Selection by Label

```

```python
"""按照索引获取元素"""
ridership_df.loc['05-02-11']
ridership_df.loc['05-02-11', 'R003']
ridership_df.loc[['05-05-11','05-08-11'],['R003', 'R005']]
ridership_df.loc[:,'R003']
```

* 使用.iloc按照位置来获取相对应的元素(Integer-Location)

```
Purely integer-location based indexing for selection by position.

.iloc[] is primarily integer position based (from 0 to length-1 of the axis), but may also be used with a boolean array.

Allowed inputs are:

An integer, e.g. 5.
A list or array of integers, e.g. [4, 3, 0].
A slice object with ints, e.g. 1:7.
A boolean array.
A callable function with one argument (the calling Series, DataFrame or Panel) and that returns valid output for indexing (one of the above)
.iloc will raise IndexError if a requested indexer is out-of-bounds, except slice indexers which allow out-of-bounds indexing (this conforms with python/numpy slice semantics).
```

```python
"""根据位置来获取第二行元素"""
ridership_df.iloc[1]
ridership_df.iloc[1:4]
ridership_df.iloc[1:4, [0,2]]
ridership_df.iloc[1:4, 1:2]
""根据位置来获取元素，第二行，第三列元素""""
ridership_df.iloc[1][2]
```

DataFrame无法计算所有数据的mean(),min(),max(),...，但是numpy可以

```python
"""按照列名来取元素"""
ridership_df['R006']
ridership_df[['R003', 'R005']]
"""输出DataFrame中所有的数据值"""
print(ridership_df.values)
"""计算DataFrame中所有值的平均值"""
print(ridership_df.values.mean())
```

* 应用实例
```python
"""计算第一天人流量最大的车站的平均值，以及整体数据的平均值"""
def mean_riders_for_max_station(ridership):
    
    max_station = ridership.iloc[0].idxmax()
    overall_mean = ridership.values.mean() 
    mean_for_max = ridership[max_station].mean()
    
    return (overall_mean, mean_for_max)
```

## 3.5 将数据加载到DataFrame中

```python
import pandas as pd
subway_df = pd.read_csv("subway_weather.csv")
"""返回前五条数据"""
subway_df.head()
subway_df.describe()
```


## 3.6 计算相关性（皮尔逊积矩相关系数）

* 皮尔逊积矩相关系数（[Pearson's r](https://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient)）仅测量线性相关系数！
* NumPy 的 corrcoef() 函数可用来计算皮尔逊积矩相关系数，也简称为“相关系数”。
* 算法：
    1. 将各变量标准化，也就是将其转换为高于或低于平均值的标准偏差值，然后将每一对值相乘并计算乘积的平均值
    2. correlation = average of (x in standard units) times (y in standard units)
    3. correlation等于标准单位的x乘以标准单位的y的平均值
* 规律：
    1. 通过标准化，可将两个变量都转换为相同的比例。
    2. 若皮尔逊积矩相关系数为正数，那么一个变量会随着另一个变量的增加而增加。
    3. 若皮尔逊积矩相关系数为负数，则一个变量会随着另一个变量的增加而减少。
    4. 皮尔逊积矩相关系数位于-1和+1之间，若它接近0则意味着变量的相关度较低
    
* 默认情况下，Pandas 的 std() 函数使用
[贝塞耳校正系数](https://en.wikipedia.org/wiki/Bessel%27s_correction)
来计算标准偏差。调用 std(ddof=0) 可以禁止使用贝塞耳校正系数。

* 多数情况下，标准化变量时，取已更正还是未更正的标准偏差并没有区别。
    **但在计算皮尔逊积矩相关系数时，必须要使用未更正标准偏差，因此一定要设置ddof=0**
    
* 应用实例
**Remember to pass the argument "ddof=0" to the Pandas std() function!**
```python
import pandas as pd

filename = '/datasets/ud170/subway/nyc_subway_weather.csv'
subway_df = pd.read_csv(filename)

def correlation(x, y):
    '''
    correlation = average of (x in standard units) times (y in standard units)
    '''
    std_x = (x-x.mean())/x.std(ddof=0)
    std_y = (y-y.mean())/y.std(ddof=0)
    return  (std_x * std_y).mean()

entries = subway_df['ENTRIESn_hourly']
cum_entries = subway_df['ENTRIESn']
rain = subway_df['meanprecipi']
temp = subway_df['meantempi']

print correlation(entries, rain)
print correlation(entries, temp)
print correlation(rain, temp)
print correlation(entries, cum_entries)
```

## 3.7 Pandas 轴名
axis=0 -----> axis='index'
axis=1 -----> axis='column'

## 3.8 DataFrame向量化运算
DataFrame与pandas.Series类似，它根据索引和列名而不是根据位置将元素匹配。

* pandas.DataFrame.shift() 
    Shift index by desired number of periods with an optional time freq
* pandas.DataFrame.diff()
    1st discrete difference of object

* DataFrame的向量化运算(相同列名和索引的进行计算)

```python
def get_hourly_entries_and_exits(entries_and_exits):
    # 或 return entries_and_exits - entries_and_exits.shift(1)
    return entries_and_exits.diff()
```


## 3.9 DataFrame.applymap()
* 在没有内置函数可用时,可以使用applymap调用自定义函数，实现一些逻辑。
    ***applymap 是针对单个数值，逐个进行处理。***
    
* 应用实例
```python
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grade(grade):
    '''
    The conversion rule is:
        90-100 -> A
        80-89  -> B
        70-79  -> C
        60-69  -> D
        0-59   -> F
    '''
    val = None
    if 90<=grade<=100:
        val = 'A'
    elif 80<=grade<=89:
        val = 'B'
    elif 70<=grade<=79:
        val = 'C'
    elif 60<=grade<=69:
        val = 'D'
    else:
        val = 'F'
    return val

def convert_grades(grades):
    return grades.applymap(convert_grade)

print(convert_grades(grades_df))
```


## 3.10 DataFrame.apply()
***apply 是针对数值，按整行或整列进行处理。***
默认是按照整列处理,因为要处理的值，一般都取决于整列的值。
DataFrame可以看做是多个Series的组合

* DataFrame.std() 与 Series.std()
    ***注意，计算得出的默认标准偏差类型在 numpy 的 .std() 和 pandas 的 .std() 函数之间是不同的。默认情况下，numpy 计算的是总体标准偏差，ddof = 0。另一方面，pandas 计算的是样本标准偏差，ddof = 1。如果我们知道所有的分数，那么我们就有了总体——因此，要使用 pandas 进行归一化处理，我们需要将“ddof”设置为 0。***

```python
import pandas as pd

grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def convert_grades_curve(exam_grades):
    # Pandas has a bult-in function that will perform this calculation
    # This will give the bottom 0% to 10% of students the grade 'F',
    # 10% to 20% the grade 'D', and so on. You can read more about
    # the qcut() function here:
    # http://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html
    return pd.qcut(exam_grades,
                   [0, 0.1, 0.2, 0.5, 0.8, 1],
                   labels=['F', 'D', 'C', 'B', 'A'])
    
# qcut() operates on a list, array, or Series. This is the
# result of running the function on a single column of the
# DataFrame.
print convert_grades_curve(grades_df['exam1'])

# qcut() does not work on DataFrames, but we can use apply()
# to call the function on each column separately
print grades_df.apply(convert_grades_curve)

def standarize_column(column):
    return (column - column.mean())/column.std(ddof=0)
    
def standardize(df):
    return df.apply(standarize_column)
    
print(standardize(grades_df))
```

* 通过DataFrame将某列数据转换为单个值
在这种情况下，apply函数不会创建新的DataFrame而是创建一个新的Series
DataFrame的每一列都被转换为一个单值,例如找到每列的最大值
```python
df.apply(np.max)
```
* 找出每一列的第二大值
   Series.sort_values()
   
```python
import numpy as np
import pandas as pd

df = pd.DataFrame({
    'a': [4, 5, 3, 1, 2],
    'b': [20, 10, 40, 50, 30],
    'c': [25, 20, 5, 15, 10]
})

# Change False to True for this block of code to see what it does

# DataFrame apply() - use case 2
if False:   
    print df.apply(np.mean)
    print df.apply(np.max)
    
def second_largest_column(column):
    #return list(sorted(column))[-2]
    sorted_column = column.sort_values(ascending=False)
    return sorted_column.iloc[1]

def second_largest(df):
    '''
    Fill in this function to return the second-largest value of each 
    column of the input DataFrame.
    '''
    return df.apply(second_largest_column)
```

## 3.11 向 Series 添加 DataFrame


* df.add()可以指定轴参数。 axis='columns' (列,默认)，或axis='index'(行)
    如果设置 axis='columns'，则得到的结果与使用"+"相同
    默认按照index进行匹配。不匹配的index，值将得到NaN。可以设置fill_value来解决。
* 应用实例
```python
import pandas as pd

# Change False to True for each block of code to see what it does

# Adding a Series to a square DataFrame
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print df
    print '' # Create a blank line between outputs
    print df + s
    
# Adding a Series to a one-row DataFrame 
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10], 1: [20], 2: [30], 3: [40]})
    
    print df
    print '' # Create a blank line between outputs
    print df + s

# Adding a Series to a one-column DataFrame
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({0: [10, 20, 30, 40]})
    
    print df
    print '' # Create a blank line between outputs
    print df + s
    

    
# Adding when DataFrame column names match Series index
if False:
    s = pd.Series([1, 2, 3, 4], index=['a', 'b', 'c', 'd'])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })
    
    print df
    print '' # Create a blank line between outputs
    print df + s
    
# Adding when DataFrame column names don't match Series index
if False:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        'a': [10, 20, 30, 40],
        'b': [50, 60, 70, 80],
        'c': [90, 100, 110, 120],
        'd': [130, 140, 150, 160]
    })
    
    print df
    print '' # Create a blank line between outputs
    print df + s
```
## 3.12 再一次归一化每一列
* 使用add(),div() 按行进行归一化

```python

import pandas as pd

# Adding using +
if 0:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df + s)
    
# Adding with axis='index'
if 0:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df.add(s, axis='index'))
    # The functions sub(), mul(), and div() work similarly to add()
    
# Adding with axis='columns'
if 0:
    s = pd.Series([1, 2, 3, 4])
    df = pd.DataFrame({
        0: [10, 20, 30, 40],
        1: [50, 60, 70, 80],
        2: [90, 100, 110, 120],
        3: [130, 140, 150, 160]
    })
    
    print(df)
    print('') # Create a blank line between outputs
    print(df.add(s, axis='columns'))
    # The functions sub(), mul(), and div() work similarly to add()
    
grades_df = pd.DataFrame(
    data={'exam1': [43, 81, 78, 75, 89, 70, 91, 65, 98, 87],
          'exam2': [24, 63, 56, 56, 67, 51, 79, 46, 72, 60]},
    index=['Andre', 'Barry', 'Chris', 'Dan', 'Emilio', 
           'Fred', 'Greta', 'Humbert', 'Ivan', 'James']
)

def standardize_column(column):
    #print(column)
    return (column-column.mean()) / column.std(ddof=0)

def standardize(df):
    '''
    Fill in this function to standardize each column of the given
    DataFrame. To standardize a variable, convert each value to the
    number of standard deviations it is above or below the mean.
    
    This time, try to use vectorized operations instead of apply().
    You should get the same results as you did before.
    '''
    
    #print(df.apply(standardize_column))
    df["exam1"] = standardize_column(df["exam1"])
    df["exam2"] = standardize_column(df["exam2"])
    print(df)
    
    return df
    
    #return None

#print(standardize(grades_df))
#standardize(grades_df)

def standarize_row(row):
    return (row-row.mean()) / row.std(ddof=1)

def standardize_rows(df):
    '''
    Optional: Fill in this function to standardize each row of the given
    DataFrame. Again, try not to use apply().
    
    This one is more challenging than standardizing each column!
    
    print(df.apply(standarize_row, axis=1))
    '''
    for idx in df.index:
        df.loc[idx] = standarize_row(df.loc[idx])
    
    print(df)
    
    """ 另一种答案
    print((df.sub(df.mean(axis="columns"), axis="index"))
    .div(df.std(axis="columns",ddof=1), axis="index"))
    """
    return None

standardize_rows(grades_df)
```

## 3.13 Pandas.groupby() 

类似于关系型数据库中对表数据的分组。

* Python 参数传入：
    *args可以传入列表，元组。**kwargs可以传入字典作为参数。

* groupby对象也有很多内置的函数mean(),max(),etc... describe()，
* 如果没有找到合适的，可以通过apply实现调用自定义函数。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for each block of code to see what it does

# Examine DataFrame
if False:
    print example_df
    
# Examine groups
if False:
    grouped_data = example_df.groupby('even')
    # The groups attribute is a dictionary mapping keys to lists of row indexes
    print grouped_data.groups
    
# Group by multiple columns
if False:
    grouped_data = example_df.groupby(['even', 'above_three'])
    print grouped_data.groups
    
# Get sum of each group
if False:
    grouped_data = example_df.groupby('even')
    print grouped_data.sum()
    
# Limit columns in result
if False:
    grouped_data = example_df.groupby('even')
    
    # You can take one or more columns from the result DataFrame
    print grouped_data.sum()['value']
    
    print '\n' # Blank line to separate results
    
    # You can also take a subset of columns from the grouped data before 
    # collapsing to a DataFrame. In this case, the result is the same.
    print grouped_data['value'].sum()
```

```python
ridership_by_day = subway_df.groupby('day_week').mean()['ENTRIESN_hourly']

%pylab inline
import seaborn as sns
ridership_by_day.plot()
```

* Examples：
DataFrame results

>>> data.groupby(func, axis=0).mean()
>>> data.groupby(['col1', 'col2'])['col3'].mean()
DataFrame with hierarchical index

>>> data.groupby(['col1', 'col2']).mean()

* Pandas.groupby 函数的使用
    参考[这里](http://pandas.pydata.org/pandas-docs/stable/groupby.html)

* 应用实例

```python
import numpy as np
import pandas as pd

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])
#print(example_df)
# Change False to True for each block of code to see what it does

# Standardize each group
if 0:
    def standardize(xs):
        return (xs - xs.mean()) / xs.std()
    grouped_data = example_df.groupby('even')
    print(grouped_data['value'].apply(standardize))
    
# Find second largest value in each group
if 0:
    def second_largest(xs):
        sorted_xs = xs.sort_values(inplace=False, ascending=False)
        return sorted_xs.iloc[1]
    grouped_data = example_df.groupby('even')
    print(grouped_data['value'].apply(second_largest))

# --- Quiz ---
# DataFrame with cumulative entries and exits for multiple stations
ridership_df = pd.DataFrame({
    'UNIT': ['R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051', 'R079', 'R051'],
    'TIMEn': ['00:00:00', '02:00:00', '04:00:00', '06:00:00', '08:00:00', '10:00:00', '12:00:00', '14:00:00', '16:00:00'],
    'ENTRIESn': [3144312, 8936644, 3144335, 8936658, 3144353, 8936687, 3144424, 8936819, 3144594],
    'EXITSn': [1088151, 13755385,  1088159, 13755393,  1088177, 13755598, 1088231, 13756191,  1088275]
})

def hourly_for_group(entries_and_exits):
    return entries_and_exits - entries_and_exits.shift(1)
    
def get_hourly_entries_and_exits(entries_and_exits):
    '''
    Fill in this function to take a DataFrame with cumulative entries
    and exits and return a DataFrame with hourly entries and exits.
    The hourly entries and exits should be calculated separately for
    each station (the 'UNIT' column).
    
    Hint: Take a look at the `get_hourly_entries_and_exits()` function
    you wrote in a previous quiz, DataFrame Vectorized Operations. If
    you copy it here and rename it, you can use it and the `.apply()`
    function to help solve this problem.
    '''
    #print(list(entries_and_exits.groupby("UNIT")))
    grouped_data = entries_and_exits.groupby("UNIT")["ENTRIESn", "EXITSn"].apply(hourly_for_group)
    print(list(entries_and_exits.groupby("UNIT")["ENTRIESn", "EXITSn"]))
    
    print(grouped_data)
    return None

get_hourly_entries_and_exits(ridership_df)
```

## 3.14 合并 Pandas.DataFrame

* 类似于关系型数据库的表连接。
    1. inner join, 
    1. left join, right join, 
    1. left outer join, right outer join, 
    1. full join
    
* pandas.merge
  可以将两个表格合并为一个表格，从而创建一个含有这两个表格所有列的新表格。

* 在数据合并前，先对数据去重，删除表中重复的数目

```python
submissions.merge(enrollments, on='account_key', how='left')
# on 规定如何将不同表格的行进行匹配
# how 决定的是 数据将以哪个表为主，非主表的数据将会被忽略
```

* 应用实例
```python
# -*- coding: utf-8 -*-
"""
Created on Thu Apr  5 10:12:08 2018

@author: admin
"""

import pandas as pd

subway_df = pd.DataFrame({
    'UNIT': ['R003', 'R003', 'R003', 'R003', 'R003', 'R004', 'R004', 'R004',
             'R004', 'R004'],
    'DATEn': ['05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11',
              '05-01-11', '05-02-11', '05-03-11', '05-04-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'ENTRIESn': [ 4388333,  4388348,  4389885,  4391507,  4393043, 14656120,
                 14656174, 14660126, 14664247, 14668301],
    'EXITSn': [ 2911002,  2911036,  2912127,  2913223,  2914284, 14451774,
               14451851, 14454734, 14457780, 14460818],
    'latitude': [ 40.689945,  40.689945,  40.689945,  40.689945,  40.689945,
                  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ,  40.69132 ],
    'longitude': [-73.872564, -73.872564, -73.872564, -73.872564, -73.872564,
                  -73.867135, -73.867135, -73.867135, -73.867135, -73.867135]
})

weather_df = pd.DataFrame({
    'DATEn': ['05-01-11', '05-01-11', '05-02-11', '05-02-11', '05-03-11',
              '05-03-11', '05-04-11', '05-04-11', '05-05-11', '05-05-11'],
    'hour': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'latitude': [ 40.689945,  40.69132 ,  40.689945,  40.69132 ,  40.689945,
                  40.69132 ,  40.689945,  40.69132 ,  40.689945,  40.69132 ],
    'longitude': [-73.872564, -73.867135, -73.872564, -73.867135, -73.872564,
                  -73.867135, -73.872564, -73.867135, -73.872564, -73.867135],
    'pressurei': [ 30.24,  30.24,  30.32,  30.32,  30.14,  30.14,  29.98,  29.98,
                   30.01,  30.01],
    'fog': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'rain': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    'tempi': [ 52. ,  52. ,  48.9,  48.9,  54. ,  54. ,  57.2,  57.2,  48.9,  48.9],
    'wspdi': [  8.1,   8.1,   6.9,   6.9,   3.5,   3.5,  15. ,  15. ,  15. ,  15. ]
})

def combine_dfs(subway_df, weather_df):
    '''
    Fill in this function to take 2 DataFrames, one with subway data and one with weather data,
    and return a single dataframe with one row for each date, hour, and location. Only include
    times and locations that have both subway data and weather data available.
    '''
    print(subway_df.head(3))
    print(weather_df.head(3))
    return subway_df.merge(weather_df, on=["DATEn", "hour", "longitude", "latitude"], how="inner")

print(combine_dfs(subway_df, weather_df))
```


* 如果左表和右表的列名不同，使用left_on, right_on
```python
subway_df.merge(weather_df, 
                left_on=["DATEn", "hour", "longitude", "latitude"],
                right_on=["date", "hour", "longitude", "latitude"],
                how='inner')
```

## 3.15 pandas.DataFrame 绘制图形

* 使用DataFrame绘制图形
    DataFrame 也像 Pandas Series 一样拥有 plot() 方法。如果 df 是 DataFrame，那么 df.plot() 将生成线条图，其中不同颜色的每条线代表 DataFrame 中的一个变量。这种方法能使你方便快速地查看数据，特别是对于小型 DataFrame 而言，但是对于更复杂的图形，你通常会希望直接使用 matplotlib
    
* 使用matplotlib绘图执行
  [参考文档](http://matplotlib.org/api/pyplot_api.html)
  
* 应用实例

* group by 使用的列，在后续处理中，还要使用的情况下，使用as_index=False来解决。

```python
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

values = np.array([1, 3, 2, 4, 1, 6, 4])
example_df = pd.DataFrame({
    'value': values,
    'even': values % 2 == 0,
    'above_three': values > 3 
}, index=['a', 'b', 'c', 'd', 'e', 'f', 'g'])

# Change False to True for this block of code to see what it does

# groupby() without as_index
if False:
    first_even = example_df.groupby('even').first()
    print first_even
    print first_even['even'] # Causes an error. 'even' is no longer a column in the DataFrame
    
# groupby() with as_index=False
if False:
    first_even = example_df.groupby('even', as_index=False).first()
    print first_even
    print first_even['even'] # Now 'even' is still a column in the DataFrame

subway_df = pd.read_csv('subway_weather.csv')
# 绘制地铁站的散点图， 并将纬度和精度分别设为x轴和y轴，将客流量设为气泡大小

# 1. 按照经纬度对数据进行分组,计算出各车站的平均客流量
data_by_location = subway_df.groupby(['latitude', 'longitude']).mean()

# 这里latitude已经不是DataFrame的列，变成了DataFrame的行索引值
# 可以通过在group_by中设置as_index=False 来解决该问题。
data_by_location.head()['latitude']

%pylab inline

import matplotlib.pyplot as plt
import seaborn as sns

# 绘制出散点图
plt.scatter(data_by_location['latitude'], data_by_location['longitude'])

# 绘制出带气泡大小的散点图
# matplotlib scatterplot bubble size
plt.scatter(data_by_location['latitude'], 
            data_by_location['longitude'],
            s=data_by_location['ENTRIESn_hourly'])

# 设置气泡大小
plt.scatter(data_by_location['latitude'], 
            data_by_location['longitude'],
            s=(data_by_location['ENTRIESn_hourly'] / data_by_location['ENTRIESn_hourly'].std())
)

```

## 3.16 三维数据

* NumPy 数组可以具有任意多的维度。
* pandas.Panel 可以支持三维数据。
    Pandas 有一个叫作 Panel 的数据结构，类似 DataFrame 或 Series，只不过用于 3D 数据。
    不过当前已经Deprecated了。因为出现了更强大的库xarray

* 多维数据结构的处理 xarray。
    [xarray参考资料](http://xarray.pydata.org/en/stable/)
    





