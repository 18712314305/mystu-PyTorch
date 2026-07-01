import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    import Utils as ut 

    from matplotlib import pyplot as plt
    from importlib import reload
    reload(ut)
    return mo, np, pd, plt, ut


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. 测试
    """)
    return


@app.cell
def _(ut):
    # 数据集
    dataSet, dataLabel = ut.LoadDataset()

    # 获得系数
    # weight, wtrace = ut.GradientAscent(dataSetIn=dataSet, dataSetLabel=dataLabel, maxRound=1000, alpha=0.001)

    # 随机梯度
    weight, wtrace = ut.RandGradientAscent(dataSetIn=dataSet, dataSetLabel=dataLabel, alpha=0.001, maxRound=500)
    return dataLabel, dataSet, weight, wtrace


@app.cell
def _(ut, wtrace):
    ut.PlotWeights(wtrace)
    return


@app.cell
def _(dataLabel, dataSet, np, plt, weight):
    # 绘图
    def plot(dataSet, dataLabel):
        fig, ax = plt.subplots(1,1, figsize=(5,5))

        # 1 类点
        ax.scatter(
            x=dataSet[dataLabel==1][:,0],
            y=dataSet[dataLabel==1][:,1],
            label="Class 1",
            color='orange',
            s = 10
        )

        # 0 类点
        ax.scatter(
            x=dataSet[dataLabel==0][:,0], 
            y=dataSet[dataLabel==0][:,1],
            label="Class 0",
            color="grey",
            s=10,
            marker='s'
        )

        # 回归线
        xLinear = np.linspace(-5,5,100)
        yLinear = (weight[0] + weight[1] * xLinear) / (-weight[2])
        ax.plot(xLinear, yLinear)

        plt.show()

    plot(dataSet=dataSet, dataLabel=dataLabel)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. HorseDataSet
    """)
    return


@app.cell
def _(pd):
    # 读取数据
    horseData = pd.read_table("./data/Ch05-Logistic/horseColicTraining.txt")
    horseData
    return


@app.cell
def _(np):
    np.random.uniform(0, 11)
    return


if __name__ == "__main__":
    app.run()
