"""
函数定义脚本 for LogisticRegression.py
"""

import pandas as pd 
import numpy as np 
from pathlib import Path
from matplotlib import pyplot as plt


def LoadDataset():
    """读取测试训练集"""
    path = Path("./data/Ch05-Logistic/testSet.txt")
    data = pd.read_table(path, header=None, names=['Feature1','Feature2','Label'])
    return data[['Feature1','Feature2']].values, data['Label'].values

def Sigmoid(x):
    return 1 / (1 + np.exp(-x))

def GradientAscent(
    dataSetIn :np.ndarray,      # m * n
    dataSetLabel :np.ndarray,   # 1 * m
    alpha :np.float64 = 0.001,          # 学习率
    maxRound :np.int16 = 500,         # 迭代次数
):  
    dataSetIn = np.insert(dataSetIn, 0, values=1, axis=1)
    numSam, numFeat = dataSetIn.shape
    dataSetLabel = dataSetLabel.reshape(numSam, 1)  # m * 1
    # 初始化系数都是 1
    weight = np.ones(numFeat).reshape(numFeat,1)    # n * 1
    traceWeights = []
    # 开始迭代
    for r in range(maxRound):
        # 记录w
        traceWeights.append([i[0] for i in weight])
        # 计算当前系数下, 每个样本的线性和
        linearPred = dataSetIn.dot(weight)          # m * 1 == m * n · n * 1
        # 计算偏差
        error = dataSetLabel - Sigmoid(linearPred)  # m * 1 == m * 1 - m * 1
        # 计算梯度: 步长 * 方向
        gradVect = alpha * dataSetIn.T.dot(error)   # n * 1 == n * m · m * 1
        # 梯度上升
        weight += gradVect
    
    return weight, traceWeights

def RandGradientAscent(
    dataSetIn :np.ndarray,      # m * n
    dataSetLabel :np.ndarray,   # 1 * m
    alpha :np.float64 = 0.001,          # 学习率
    maxRound :np.float64 = 500,
):
    """随机梯度上升"""
    dataSetIn = np.insert(dataSetIn, 0, values=1, axis=1)   # 增加一列 0, 其系数对应 y = Wi·Xi + w0 中的 w0
    numSam, numFeat = dataSetIn.shape
    dataSetLabel = dataSetLabel.reshape(numSam, 1)  # m * 1
    # 初始化系数
    weight = np.ones(numFeat) # 1 * n
    traceWeights = []
    # 开始迭代, 每次迭代一个样本
    for r in range(maxRound):
        for samIdx in range(numSam):
            # 动态学习率
            alpha = 4 / (1 + r + samIdx) + 0.01
            traceWeights.append([w for w in weight])
            # 线性和预测
            linearPred = dataSetIn[samIdx].dot(weight)  # 标量 = 1 * n · 1 * n
            # 计算偏差
            error = dataSetLabel[samIdx] - linearPred   # 标量 = 标量 - 标量
            # 计算梯度
            gradWeight = alpha * error * dataSetIn[samIdx]  # 1 * n
            # 更新系数
            weight += gradWeight
    return weight, traceWeights


def PlotWeights(traceWeights :list):
    """绘图参数收敛"""
    # 参数量
    numWeight = len(traceWeights[0])
    # 迭代次数
    numRound = len(traceWeights)
    
    fig, axs = plt.subplots(numWeight, 1, figsize=(4*numWeight, 3*numWeight))
    for idx, ax in enumerate(axs):
        ax.plot(range(numRound), [w[idx] for w in traceWeights], linewidth=0.5)
    
    plt.show()
    return

