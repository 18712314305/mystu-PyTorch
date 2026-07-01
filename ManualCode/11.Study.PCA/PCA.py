import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd
    from matplotlib import pyplot as plt
    from sklearn.datasets import load_iris
    from sklearn.decomposition import PCA
    return PCA, load_iris, mo, np, plt


@app.cell
def _(mo):
    mo.md(r"""
    # 手写 PCA 实现
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 标准PCA实现
    """)
    return


@app.cell
def _(PCA, load_iris):
    irisData = load_iris()
    targetDim = 2 # 目标维度
    pcaModel = PCA(n_components=targetDim)
    stdRes = pcaModel.fit(irisData['data'])
    return irisData, stdRes


@app.cell
def _(irisData, plt, stdRes):
    fig, ax = plt.subplots(1,1,figsize=(4,2),dpi=300)
    data = stdRes.transform(irisData['data'])
    ax.scatter([i[0] for i in data], [i[1] for i in data])
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 手写 PCA
    """)
    return


@app.cell
def _(irisData, np):
    def ManualPCA(X, targetDim):
        # 标准化
        stdX = X - X.mean(0)
        # 协方差矩阵
        covX = (stdX.T @ stdX) / (X.shape[0] - 1) 
        # 特征值分解
        feaValue, feaVector = np.linalg.eigh(covX)
        # 前 k 个
        queryVector = feaVector[:,-targetDim:][:, ::-1]
        queryValue  = feaValue[-targetDim:][::-1]
        # 投影
        mapX = stdX @ queryVector
        return mapX, queryValue, queryVector

    mapX, queryValue, queryVector = ManualPCA(irisData['data'], targetDim=2)
    return (mapX,)


@app.cell
def _(mapX, plt):
    fig_, ax_ = plt.subplots(1,1,figsize=(4,2),dpi=300)
    # data_ = stdRes.transform(irisData['data'])
    ax_.scatter([i[0] for i in mapX], [i[1] for i in mapX])
    plt.show()
    return


if __name__ == "__main__":
    app.run()
