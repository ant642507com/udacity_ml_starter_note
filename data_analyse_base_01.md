
# 3. 数据分析基础

## 3.1 数据分析过程

### 3.1.1 数据分析概述

* NumPy, Pandas, Matplotlib 这三个包，数据分析必不可少。
    - NumPy 提供了特别好的数据结构，即数组，能使数字数据的处理变得特别简单。NumPy是Python数据栈的基础，也是Python精于数据分析的根本所在。
    - Pandas 中，使用DataFrame，用于数据操作和分析。
    - Matplotlib 是进行数据可视化操作的包，只需要少量代码，就能创建各种简单的图形，让数据以很专业的形式展示出来。

* 数据分析师就是用数据来回答问题的人
    - 有什么特征会增加肿瘤发展成癌的可能性？
    - 自2008年来，汽车车型改进了多少？
    - 什么交通工具改进幅度最大？
    - 葡萄酒质量受种类，酒精度，糖分，酸度等因素影响，这些影响有什么不同？
* 课程目的，介绍工具，能够自信地用Python分析数据，体验真实数据集的每步分析操作。

### 3.1.2 数据分析师解决的问题

* 利用机器学习，更好地做出决策
    - [Facebook 博文](https://research.fb.com/exposure-to-diverse-information-on-facebook-2/) 与[论文](https://research.fb.com/publications/exposure-to-ideologically-diverse-information-on-facebook/)
     包含不同意识形态的信息
    - [OkCupid 博文](http://blog.okcupid.com/index.php/the-best-questions-for-first-dates/)，解释关于第一次约会时可提的最佳问题
    - [文章](http://www.dezyre.com/article/how-big-data-analysis-helped-increase-walmart-s-sales-turnover/109)，分析沃尔玛如何使用大数据分析来增加销量
    - [维基百科页面](https://en.wikipedia.org/wiki/Bill_James)，解释 Bill James 如何将数据分析应用于棒球
    - [Numerate 博文](http://www.numerate.com/numerates-ranking-technology-pharmaceutical-rd-gains-u-s-patent/)，解释如何使用数据分析来设计药物
    
### 3.1.3 数据分析过程概述

* 数据分析过程(5步)
    - 提出问题(Question)
    - 整理数据(Wrangle)
    - 探索数据(Explore)
    - 得出结论(Draw Conclusions)
    - 交流结果(Communicate)
    
    这个过程可以帮助你理解，探索并巧妙地运用数据，从而最大限度地利用得到的信息，无论是要制作仪表盘报表、分析A/B测试结果，还是用机器学习和人工智能进行更深入的分析，都需要这个过程。

* 第 1 步：提问
你要么获取一批数据，然后根据它提问，要么先提问，然后根据问题收集数据。在这两种情况下，好的问题可以帮助你将精力集中在数据的相关部分，并帮助你得出有洞察力的分析。

* 第 2 步：整理数据
你通过三步来获得所需的数据：收集，评估，清理。你收集所需的数据来回答你的问题，评估你的数据来识别数据质量或结构中的任何问题，并通过修改、替换或删除数据来清理数据，以确保你的数据集具有最高质量和尽可能结构化。

* 第 3 步：执行 EDA（探索性数据分析）
你可以探索并扩充数据，以最大限度地发挥你的数据分析、可视化和模型构建的潜力。探索数据涉及在数据中查找模式，可视化数据中的关系，并对你正在使用的数据建立直觉。经过探索后，你可以删除异常值，并从数据中创建更好的特征，这称为特征工程。

* 第 4 步：得出结论（或甚至是做出预测）
这一步通常使用机器学习或推理性统计来完成，不在本课程范围内，本课的重点是使用描述性统计得出结论。

* 第 5 步：传达结果
你通常需要证明你发现的见解及传达意义。或者，如果你的最终目标是构建系统，则通常需要分享构建的结果，解释你得出设计结论的方式，并报告该系统的性能。传达结果的方法有多种：报告、幻灯片、博客帖子、电子邮件、演示文稿，甚至对话。数据可视化总会给你呈现很大的价值。
    - 
### 3.1.4 读取数据集

* 读取数据集
```python
# import pandas
import pandas as pd

# load cancer data into data frame
df = pd.read_csv('cancer_data.csv')

#df = pd.read_csv('student_scores.csv', sep=':')

#df = pd.read_csv('student_scores.csv', header=2)

#labels = ['id', 'name', 'attendance', 'hw', 'test1', 'project1', 'test2', 'project2', 'final']
#df = pd.read_csv('student_scores.csv', names=labels)

#df = pd.read_csv('student_scores.csv', index_col='Name')
#df = pd.read_csv('student_scores.csv', index_col=['Name', 'ID'])

#默认情况下，read_csv 使用 header=0，使用第一行作为列标签。
#如果文件中不包括列标签，可以使用 header=None 防止数据的第一行被误当做列标签。
#df = pd.read_csv('student_scores.csv', header=None)

# display first five rows of data
df.head()

# print the column labels in the dataframe
for i, v in enumerated(df.columns):
    print(i, v)


# 将结果保存为CSV文件
#df_powerplant.to_csv('powerplant_data_edited.csv')

```
### 3.1.5 数据整理和EDA

整理数据：收集，评估和整理数据，不过在清理数据集时做的修改并不能优化你的分析，可视化界面或模型，仅仅是让他们可用而已。
EDA: EDA意味着要探索并增加数据，从而将分析、可视化界面和模型的潜力最大化。

### 3.1.6 阅读和理解数据

* 阅读和理解数据基础

```python
import pandas as pd

df = pd.read_csv('cancer_data.csv')
df.head()

# 返回数据框维度的元组
df.shape

# 返回列的数据类型
df.dtypes

# 虽然供诊断的数据类型是对象，但进一步的
# 调查显示，它是字符串
type(df['diagnosis'][0])

# 显示数据框的简明摘要，
# 包括每列非空值的数量
df.info()

# 返回每列数据的有效描述性统计
df.describe()

# 返回数据框中的前几行
# 默认返回前五行
df.head()

# 但是也可以指定你希望返回的行数
df.head(20)

# `.tail()` 返回最后几行，但是也可以指定你希望返回的行数
df.tail(2)

# 查看每列的索引号和标签
for i, v in enumerate(df.columns):
    print(i, v)
    
# 选择从 'id' 到最后一个均值列的所有列
df_means = df.loc[:,'id':'fractal_dimension_mean']
df_means.head()

# 用索引号重复以上步骤
df_means = df.iloc[:,:11]
df_means.head()

#保存均值数据框，以便稍后使用。
df_means.to_csv('cancer_data_edited.csv', index=False)

```

* 如何在 Pandas 中选择多个范围

```python
np.r_[1:10, 15, 17, 50:100]

array([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 15, 17, 50, 51, 52, 53, 54, 55,
       56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72,
       73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89,
       90, 91, 92, 93, 94, 95, 96, 97, 98, 99])
       
df.iloc[:, np.r_[1:10, 15, 17, 50:100]]
```

### 3.1.7 数据清理

```python
import pandas as pd
# 读入 `cancer_data_edited.csv`

cancer_data_df = pd.read_csv('cancer_data_edited.csv')

# 用 info() 检查哪些列有缺失值
cancer_data_df.info()

# 用均值填充缺失值
cancer_data_df['texture_mean'].fillna(cancer_data_df['texture_mean'].mean(), inplace=True)
cancer_data_df['smoothness_mean'].fillna(cancer_data_df['smoothness_mean'].mean(), inplace=True)
cancer_data_df['symmetry_mean'].fillna(cancer_data_df['symmetry_mean'].mean(), inplace=True)

# 检查数据中的重复
cancer_data_df.duplicated()

# 丢弃重复
cancer_data_df.drop_duplicates(inplace=True)

# 再次检查数据中的重复，确认修改
cancer_data_df.duplicated()


# 从列名称中移除 "_mean"
new_labels = []
for col in cancer_data_df.columns:
    if '_mean' in col:
        new_labels.append(col[:-5])  # 不包括最后 6 个字符
    else:
        new_labels.append(col)

# 列的新标签
print(new_labels)

# 为数据框中的列分配新标签
df.columns = new_labels

# 显示数据框的前几行，确认更改
df.head()

# 将其保存，供稍后使用
df.to_csv('cancer_data_edited.csv', index=False)


```

### 3.1.8 用可视化探索数据，使用Pandas绘图

* 直方图 Histograms
* 散点图 Scatterplots

* 箱线图 

```python
import pandas as pd
% matplotlib inline

df_census = pd.read_csv('census_income_data.csv')
df_census.info()

df_census.hist()
df_census.hist(figsize=(8, 8))
df_census['age'].hist()
df_census['age'].plot(kind='hist')
df_census['education'].value_counts().plot(kind='bar')
df_census['education'].value_counts().plot(kind='pie', figsize=(8, 8))

df_cancer = pd.read_csv('cancer_data_edited.csv')
df_cancer.info()
"""迅速了解变量间的关系，还可以展示每个变量的直方图"""
pd.plotting.scatter_matrix(df_cancer, figsize=(15, 15))

df_cancer.plot(x='compactness', y='concavity', kind='scatter')

df_cancer['concave_points'].plot(kind='box')

#每个变量的箱线图
df_cancer.plot(kind='box')


```


```python
import pandas as pd

df = pd.read_csv('cancer_data_edited.csv')
df.head()

# 过滤出所有恶性肿瘤数据
df_m = df[df['diagnosis'] == 'M']
# 过滤出所有良性肿瘤数据
df_b = df[df['diagnosis'] == 'B']

df_m.head()

df_m['area'].describe()


```

相同的数据指标，分类对比
```python
import matplotlib.pyplot as plt

% matplotlib inline

fig, ax = plt.subplots(figsize=(8, 6))

ax.hist(df_b['area'], alpha=0.5, label='benign')
ax.hist(df_m['area'], alpha=0.5, label='malignant')

ax.set_title('Distributions of Benign and Malignant Tumor Areas')
ax.set_xlabel('Area')
ax.set_ylabel('Count')
ax.legend(loc='upper right')

plt.show()

```

### 3.1.9 传达结果示例

```python
import pandas as pd

% matplotlib inline

df_census = pd.read_csv('census_income_data.csv')

# 绘制柱状图，对比收入与教育程度的关系

# 将收入分开
df_a = pd_census[df_census['income'] == '>50K']
df_b = pd_census[df_census['income'] == '<=50K']
# 生成比较收入与教育程度的关系图
df_a['education'].value_counts().plot(kind='bar')
df_b['education'].value_counts().plot(kind='bar')
# value_cournts 返回的值是按照大小来排序的

# 使用相同的索引，保证两个图的横坐标使用相同的顺序
ind = df_a['education'].value_counts().index
df_a['education'].value_counts()[ind].plot(kind='bar')
df_b['education'].value_counts()[ind].plot(kind='bar')

# 绘制饼状图，对比两组的主要工薪阶层

ind = df_a['workclass'].value_counts().index
df_a['workclass'].value_counts()[ind].plot(kind='pie', figsize=(8, 8))
df_b['workclass'].value_counts()[ind].plot(kind='pie', figsize=(8, 8))

# 用直方图，绘制两组的年龄分布
df_a['age'].hist()
df_b['age'].hist()

df_a['age'].describe()
df_b['age'].describe()


```



