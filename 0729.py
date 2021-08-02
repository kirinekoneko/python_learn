#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jul 30 08:56:32 2021

@author: kirineko
"""
# %% 上午内容
# LinearRegression线性回归

# %% 引入库
import numpy as np
import matplotlib.pyplot as plt 
from sklearn.linear_model import LinearRegression

# 定义样本容量（生成的点的数量）
sample_size = 11

# 从0-10开始均匀生成11个点
x_array = np.linspace(0,10,sample_size)

# 定义线性方程 y = 2x + 1
slope = 2
intercept = 1
y_array = x_array * slope + intercept

std = 1
# 生成服从均值为0，标准差为1的正态分布的11个点
epsilon = np.random.normal(0, std, sample_size)

# 定义线性方程 y2 = y + epsilon(小误差)
y_array_2 = y_array + epsilon

# 先画 y = 2x + 1的散点图
plt.figure()
plt.scatter(x_array, y_array)

# 再画 y2 = 2x + 1 + epsilon的散点图
plt.figure()
plt.scatter(x_array, y_array_2)

# 注意观察上面两个图的细微差别

# %% 使用线性回归模型拟合曲线（直线）

# 实例化模型对象
model = LinearRegression()
# 使用模型进行训练
# 第一个参数为x数据，需要是一个矩阵（二维数据）
# 第二个参数为y数据，需要是一个数组
# 使用x_array.reshape(sample_size,1)函数将x_array转换为一个矩阵
# 使用y_array_2.reshape(-1)函数将y_array_2转换为一个数组
# model.fit表示训练
model.fit(x_array.reshape(sample_size,1), y_array_2.reshape(-1))

# coef_为线性方程的斜率，intercept_为线性方程的截距
a = model.coef_
b = model.intercept_

# 从0-1生成100个均匀的点
z_array = np.linspace(0,1,100)
# 使用model对这100个点的y值进行预测
# 思考为什么不直接使用z_array,而是z_array.reshape(100,1)？
z_predict = model.predict(z_array.reshape(100,1))

# 画z_array【x】和z_predict【y】的值
plt.figure()
plt.scatter(z_array, z_predict)

# %% 使用线性回归模型对iris问题进行预测
# 请自己思考以下代码内容

from sklearn.datasets import load_iris

# 取数据
iris = load_iris()
x = iris.data
y = iris.target

# 训练模型
model_iris = LinearRegression()
model_iris.fit(x, y)

# 预测新的值
z = np.array([[5.6, 3, 6, 2], [4.6, 3.1, 2, 0.5]])
z_predict = model_iris.predict(z)


# %% reshape

# 观察x和x_1, 理解reshape函数的作用
x = np.zeros(100)
x_1 = x.reshape(100, -1)


#%% 下午内容

# LogisticRegression分类器

#%% 导入库
import numpy as np
import matplotlib.pyplot as plt

#%% 定义sigmod函数

# sigmod(x) = 1 / (1 + e^(-x))
def sigmod(x):
    y = 1 / (1 + np.exp(-x))
    return y

#%% 画出sigmod函数图像

# x, y均为数组
# 请在右侧的Variable Explorer观察x和y的数据
x = np.linspace(-5, 5, 100)
y = sigmod(x)

# 请在右侧plots观察函数图像
plt.plot(x, y)

#%% 随机生成100个点，点坐标在(0, 0)到(10, 10)之间

# 定义样本容量
sample_size = 100

# 生成100行2列的数据，数据在0-10之间
# 第一列是100个x坐标，第二列是100个y坐标
# uniform表示生成的点服从均匀分布
X = np.random.uniform(0, 10, (sample_size, 2))

# x1取第一列的所有值，共100个，请注意在右侧观察结果
# 逗号左边用于选取行，右边用于选取列
# 左边只写一个`:`表示选取所有行
x1 = X[:, 0]
# x2取第二列的所有值，共100个，请注意在右侧观察结果
x2 = X[:, 1]

# 请在右侧plot观察函数图像
plt.scatter(x1, x2)

#%% 对100个点进行划分
# 将左上角的点标记为1，右下角的点标记为0

# 定义100个元素的数组，值均为0
y = np.zeros(sample_size)

# 噪声noise, 增加样本分类的不确定性
# normal表示正态分布，生成100个值服从均值为0，标准差为2的正态分布
# 请注意查看z的值
z = np.random.normal(0, 2, sample_size)

# 通过for循环标记点，i从0取到99
# 在每次循环中取点的横坐标x1[i]和纵坐标x2[i]
# 如果x2[i] > x1[i]，则将该点标记为1，否则标记为0
# z[i]为随机噪声，对总体结果影响不大
for i in range(sample_size):
    if x2[i] > x1[i] + z[i]:
        y[i] = 1
    else:
        y[i] = 0
        
# 请注意对比观察y, y1_index, y0_index的值
# y1_index将y==1的位置标记为True
y1_index = (y == 1)
# y2_index将y==0的位置标记为True
y0_index = (y == 0)

# 不同批次画的点颜色不一样，也可以使用color参数设定颜色
# 先画y==0的点
plt.scatter(x1[y0_index], x2[y0_index])
# 再画y==1的点
plt.scatter(x1[y1_index], x2[y1_index])

# 请思考为什么有一些橙色的点跑到蓝色点的区域了，一些蓝色的点跑到橙色区域了

# %% 使用Logistic分类器进行建模

# 引入Logistic分类器
from sklearn.linear_model import LogisticRegression

# 生成模型对象
model = LogisticRegression()
# 给模型提供数据，模型会根据数据推断模型的参数
# fit就是模型在推断参数的过程
model.fit(X, y)

# 打印模型推断出来的参数
# intercept_: 表示截距参数
# coef_: 表示斜率参数
print(model.intercept_)
print(model.coef_)
# 计算模型的准确性【得分】
print(model.score(X, y))

# 定义3个点用于测试
X_test = np.array([
    [10, 0],
    [0, 10],
    [5, 2]
])

# 使用模型预测【标记】三个点的结果
# 注意在右侧variable explorer查看y_test_pred的结果
y_test_pred = model.predict(X_test)
print(y_test_pred)

# %% 产生10404个点铺满整个区域

# 从-0.1到10.0生成102个数据
xt1 = np.arange(-0.1, 10.1, 0.1)
xt2 = np.arange(-0.1, 10.1, 0.1)

# 使用meshgrid函数扩展，请注意观察xxt1和xxt2的值
# xxt1是按行扩展数据，xxt2是按列扩展数据
xxt1, xxt2 = np.meshgrid(xt1, xt2)

# xxt1.reshape(-1)将xxt1一维化(平铺)
# xxt2.reshape(-1)将xxt2一维化(平铺)
# 以xxt1.reshape(-1)为横坐标，以xxt2.reshape(-1)为纵坐标画散点图
# 共102*102 = 10404个点（请思考为什么有这么多点）
plt.scatter(xxt1.reshape(-1), xxt2.reshape(-1))

# %% 使用前面的分类器对10404个点进行分类

# 先使用xxt1.reshape(-1, 1)将xxt1转换为10404 * 1的矩阵（列向量）
# 再使用xxt2.reshape(-1, 1)将xxt2转换为10404 * 1的矩阵（列向量）
# 此时使用hstack拼接上面的两个列向量
# xt是一个矩阵，矩阵的第一列是10404个点的横坐标，矩阵的第二列是10404个点的纵坐标
# 请思考为什么
xt = np.hstack([xxt1.reshape(-1, 1), xxt2.reshape(-1, 1)])

# 使用刚才的模型对xt(10404个点)进行预测
# 将左上角的点标记为1，右下角的点标记为0
yt = model.predict(xt)

# 先绘制yt==1的散点图（左上角）
# yt==1为布尔索引，此处用于选出xt中yt==1的行
plt.scatter(xt[yt == 1, 0], xt[yt == 1, 1])
# 再绘制yt==0的散点图（右下角）
plt.scatter(xt[yt == 0, 0], xt[yt == 0, 1])

# %% 使用Logistic分类器对iris问题进行预测
# 请自己思考以下代码内容

from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
import numpy as np

iris = load_iris()
X = iris.data
y = iris.target

model = LogisticRegression(max_iter=1000)
model.fit(X, y)
z = np.array([
    [5.6, 3, 6, 2],
    [4.6, 3.1, 2, 0.5]
])

y_prodict = model.predict(z)
