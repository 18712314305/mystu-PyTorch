import marimo

__generated_with = "0.18.4"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import pandas as pd 
    import Utils as ut 
    from importlib import reload

    reload(ut)
    np.random.seed(42)
    return mo, pd, ut


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 数据集预处理
    """)
    return


@app.cell
def _(pd):
    # 读文件
    raw_data = pd.read_table(
        "./data/Ch03-DecisionTree/lenses.txt", 
        header=None, 
        names=['Age','Prescript','Astigmatic','TearRate', 'Label']
    )

    return (raw_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 训练决策树
    """)
    return


@app.cell
def _(raw_data, ut):
    mytree = ut.CreateTree(data=raw_data)
    return


if __name__ == "__main__":
    app.run()
