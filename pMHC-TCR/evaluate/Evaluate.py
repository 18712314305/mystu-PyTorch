import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    import numpy  as np

    import torch
    import torch.nn as nn

    from pathlib import Path
    from typing  import List, Tuple
    from matplotlib import pyplot as plt

    fileDir = Path(__file__).absolute().parent
    modelDir = fileDir / "../model/"
    return List, Tuple, mo, modelDir, nn, np, plt, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 定义作图函数
    """)
    return


@app.cell
def _(List, Tuple, np):
    def PlotLossByEpoch(
        ax, # 要作图的 axes 对象
        data :List[Tuple[np.ndarray, np.ndarray]], # 作图数据, List[tuple]
        colors

    ):
        for lidx, line in enumerate(data):
            epochArray = np.arange(1, 1+len(line[0]))
            ax.plot(epochArray, line[0], color = colors[lidx], ls='dashed')
            ax.plot(epochArray, line[1], color = colors[lidx], ls='solid')

        ax.set_xlabel("epoch")
        ax.set_ylabel("Loss")

        return

    def PlotAccByEpoch(
        ax, 
        data :List[np.ndarray],
        colors
    ):
        for lidx, line in enumerate(data):
            epochArray = np.arange(1, 1+len(line))
            ax.plot(epochArray, line, color=colors[lidx])

        ax.set_xlabel("epoch")
        ax.set_ylabel("Accuracy")

        return

    def PlotRecByEpoch(
        ax, 
        data :List[np.ndarray],
        colors
    ):
        for lidx, line in enumerate(data):
            epochArray = np.arange(1, 1+len(line))
            ax.plot(epochArray, line, color=colors[lidx])

        ax.set_xlabel("epoch")
        ax.set_ylabel("Recall")

        return

    def PlotLossByBatch(
        ax, 
        data, 
    ):
        ax.imshow(data, cmap='viridis', origin='upper')
        ax.set_xlabel("batch")
        ax.set_ylabel("epoch")

        return

    def PlotRocByEpoch(
        ax, 
        data :List[np.ndarray],
        colors,
        dtype
    ):
        for lidx, line in enumerate(data):
            epochArray = np.arange(1, 1+len(line))
            ax.plot(epochArray, line, color=colors[lidx])

        ax.set_xlabel("epoch")
        ax.set_ylabel(f"ROC-AUC-{dtype}")

        return

    def PlotPrByEpoch(
        ax, 
        data :List[np.ndarray],
        colors,
        dtype
    ):
        for lidx, line in enumerate(data):
            epochArray = np.arange(1, 1+len(line))
            ax.plot(epochArray, line, color=colors[lidx])

        ax.set_xlabel("epoch")
        ax.set_ylabel(f"PR-AUC-{dtype}")

        return





    return (
        PlotAccByEpoch,
        PlotLossByEpoch,
        PlotPrByEpoch,
        PlotRecByEpoch,
        PlotRocByEpoch,
    )


@app.cell
def _(modelDir, torch):
    logData = torch.load(modelDir / "v0_2.withoutWeight.Log.epoch60.pt")
    return (logData,)


@app.cell
def _(logData):
    logData[1].keys()
    return


@app.cell
def _(
    PlotAccByEpoch,
    PlotLossByEpoch,
    PlotPrByEpoch,
    PlotRecByEpoch,
    PlotRocByEpoch,
    logData,
    np,
    plt,
):
    # 总览图
    fig, axs = plt.subplots(3,3, figsize=(12,12), dpi=200)

    # 全局参数
    foldColors = ['grey','black','red']

    # loss 曲线
    epochArray = np.arange(1, 1+len(logData[1]['trainEpochLoss']))
    PlotLossByEpoch(
        ax=axs[0][0], 
        data=[(np.array(v['trainEpochLoss']), np.array(v['testEpochLoss'])) for k,v in logData.items()], 
        colors = foldColors
    )

    # accuracy
    PlotAccByEpoch(
        ax=axs[0][1],
        data=[np.array(v['accuracy']) for k,v in logData.items()],
        colors = foldColors

    )

    # recall
    PlotRecByEpoch(
        ax=axs[0][2],
        data=[np.array(v['recall']) for k,v in logData.items()],
        colors = foldColors
    )


    # batch loss
    # PlotLossByBatch(ax=axs[1][0], data=np.array(logData[1]['trainBatchLoss']))

    # ROC
    PlotRocByEpoch(ax=axs[1][0], data=[np.array(v['ROC-AUC-train']) for k,v in logData.items()], colors=foldColors, dtype='train')
    # PR
    PlotPrByEpoch(ax=axs[1][1], data=[np.array(v['PR-AUC-train']) for k,v in logData.items()], colors=foldColors, dtype='train')
    # ROC
    PlotRocByEpoch(ax=axs[2][0], data=[np.array(v['ROC-AUC-test']) for k,v in logData.items()], colors=foldColors, dtype='test')
    # PR
    PlotPrByEpoch(ax=axs[2][1], data=[np.array(v['PR-AUC-test']) for k,v in logData.items()], colors=foldColors, dtype='test')

    plt.show()
    return


@app.cell
def _(logData):
    logData[3]['accuracy'][:30]
    return


@app.cell
def _(logData):
    logData[3]['testEpochLoss'][:30]
    return


@app.cell
def _(nn, torch):
    qq = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([49]))
    lab = torch.tensor([1.0]*100+[0.0]*4900)
    pred = torch.tensor([0.02]*5000)
    qq(lab, pred)
    return


if __name__ == "__main__":
    app.run()
