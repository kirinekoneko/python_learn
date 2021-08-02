# -*- coding: utf-8 -*-

#%% Pandas基础和KNN模型

# 引入pandas库
import pandas as pd
# 引入KNN模型
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# iris.csv文件路径
path = "./iris.csv"
# 读取csv文件
data = pd.read_csv(path, index_col=0)

# 数据的列名
data.columns
# 获取数据值，请尝试观察data_v的数据类型
data_v = data.values

# data的第0行第0列数据
data.iloc[0, 0]
# data的第1行所有列数据
data.iloc[1, :]
# data的所有行最后一列数据
data.iloc[:, -1]

# KNN分类器, 设置neighbors个数为5, 距离为1
model = KNeighborsClassifier(n_neighbors=5, p=1)
# X是所有行，第0列到第3列的数据
X = data.iloc[:, 0:4]
# y是所有行，第四列的数据
y = data.iloc[:, 4]

# 训练数据
model.fit(X, y)
# 模型准确率评估
model.score(X, y)

# 计算面积1：第0列乘以第1列
area1 = data.iloc[:, 0] * data.iloc[:, 1]
# 计算面积2：第2列乘以第3列
area2 = data.iloc[:, 2] * data.iloc[:, 3]
# data新增一列area1
data['area1'] = area1
# data新增一列area2
data['area2'] = area2
# 构造新的数据，新数据为data的第0-1-2-3-5-6-4列
data_final = data.iloc[:, [0, 1, 2, 3, 5, 6, 4]]

# 新的路径
save_path = "./iris_new.csv"
# 将data_final保存成一个csv文件
data_final.to_csv(save_path)

#%% 划分训练集和测试集

# X是data_final的所有行，第0列到最后一列(不包含最后一列)
X = data_final.iloc[:, :-1]
# y是data_final的所有行，最后一列数据
# 请注意观察X和y的值
y = data_final.iloc[:, -1]

# train_test_split划分训练集和测试集，通过test_size调整测试集数据占比
# 请思考这个函数的参数X和y都是什么类型
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.4)

# KNN分类器
model = KNeighborsClassifier(n_neighbors=5)
# 使用模型对训练集进行训练
model.fit(train_X, train_y)
# 评估模型对训练集的准确性
print(model.score(train_X, train_y))
# 评估模型对测试集的准确性
print(model.score(test_X, test_y))

#%% 利用梯度下降法寻找最小值

import numpy as np
import matplotlib.pyplot as plt

# 函数f(x) = x^2 - 2x -5
def f(x):
    return x * x - 2 * x - 5

# 函数f(x)的导函数 = 2x - 2
def fderiv(x):
    return 2 * x - 2

# 学习率参数
learning_rate = 1e-1
# 迭代次数
n_iter = 100

# 构造xs数组，均为0
xs = np.zeros(n_iter + 1)
# 设置xs的初值为100
xs[0] = 100

# 利用梯度下降法寻找最小值
# 需要理解梯度下降的算法思想
for i in range(n_iter):
    xs[i + 1] = xs[i] - learning_rate * fderiv(xs[i])

# 打印xs，注意观察xs后面的数据是不是函数fx的最小值
plt.plot(xs)

#%%  最优化

from scipy.optimize import minimize

# 使用scipy封装好的minimize寻找f(x)的最小值，设置初值为100
a = minimize(f, x0=100).x
print(a)

# 换一个函数f2(x) = x^ 2  * e ^ (-x^2)
def f2(x):
    return np.exp(-x*x) * (x**2)

# 计算f2(1)
f2(1)

# 使用scipy封装好的minimize寻找f2(x)的最小值，设置初值为5
a = minimize(f2, x0=5).x
# 使用scipy封装好的minimize寻找f2(x)的最小值，设置初值为-5
b = minimize(f2, x0=-5).x

# 查看a和b的值
# 尝试画出f2(x)在-10到10之间的图形

# %% 决策树原理——信息熵

# 一种信息熵计算函数，熵越小确定性越高，可分别带入x=0.5和x=0.99分析
def E(x):
    a = - x * np.log(x) - (1-x) * np.log(1-x)
    return a

# 另一种信息熵计算函数，熵越小确定性越高，可分别带入x=1和x=0.5分析
def E2(x):
    a = 1 - x ** 2 - (1-x) ** 2
    return a

# 定义x在0.01到0.99之间
x = np.linspace(0.01, 0.99, 100)

# 画出E(x)函数图像
y = E(x)
plt.plot(x, y)

# 画出E2(x)函数图像
y2 = E2(x)
plt.plot(x, y2)

# %% 决策树模型

# 读取西瓜数据集
wm = pd.read_csv('./wm_1.csv', index_col=0)
# X是西瓜数据集的0-6列，不包含第6列
X = wm.iloc[:, :6]
# y是西瓜数据集的最后一列
y = wm.iloc[:, -1]

# 引入决策树
import sklearn.tree as tree

# 决策树模型，信息熵函数为entropy, 也就是E(x)
model = tree.DecisionTreeClassifier(criterion='entropy')
# 训练决策树模型
model.fit(X, y)

# 画出整个决策树过程
plt.figure(figsize=(10, 10))
tree.plot_tree(model, filled=True)

# %% 使用决策树模型对breast_cancer数据集进行预测

# 引入breast_cancer数据集
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

# 引入数据集
bc = load_breast_cancer()
# X为样本，请注意观察X的值
X = bc.data
# y为label
y = bc.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

# model是决策树模型
model = tree.DecisionTreeClassifier(max_depth=10)
model.fit(X_train, y_train)

# 计算模型得分
print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

plt.figure(figsize=(20, 20))
tree.plot_tree(model, filled=True)

# %% 使用三种分类模型对breast_cancer数据集进行预测

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression

bc = load_breast_cancer()
X = bc.data
y = bc.target

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

# model为决策树模型
model = tree.DecisionTreeClassifier(max_depth=10)
model.fit(X_train, y_train)

print(model.score(X_train, y_train))
print(model.score(X_test, y_test))

# model2为KNN模型
model2 = KNeighborsClassifier()
model2.fit(X_train, y_train)

print(model2.score(X_train, y_train))
print(model2.score(X_test, y_test))

# model3为Logistic模型
model3 = LogisticRegression(max_iter=10000)
model3.fit(X_train, y_train)

print(model3.score(X_train, y_train))
print(model3.score(X_test, y_test))


# %% 画圆——生成点

import numpy as np

# 生成（1000 * 2）个服从-10~10的均匀分布的数据
# 生成1000个点，点的坐标范围在[-10~10, -10~10]之间
sample_size = 1000
X = np.random.uniform(-10, 10, (sample_size, 2))

# 画散点图
plt.scatter(X[:, 0], X[:, 1])

# %% 画圆——标记

# 设置圆的半径
radius = 8

# 初始化1000个点的标记
labels = np.zeros(sample_size)

# 计算X中所有点到(0,0)点的距离，将距离小于半径的点标记为1，将距离大于半径的点标记为0
for i in range(sample_size):
    if np.sqrt(X[i, 0] ** 2 ++ X[i, 1] ** 2) <= radius:
        labels[i] = 1
    else:
        labels[i] = 0
        
# plt.scatter(X[labels == 1, 0], X[labels == 1, 1])
# 默认的x轴和y轴不是1：1等比例的，可以利用figsize=(10, 10)调整
plt.figure(figsize=(10, 10))
# 绘制距离大于半径的点的散点图，距离大于半径的点labels == 0
plt.scatter(X[labels == 0, 0], X[labels == 0, 1])