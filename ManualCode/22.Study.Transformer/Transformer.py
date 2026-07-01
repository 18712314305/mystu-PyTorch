import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn
    import numpy as np

    device = torch.accelerator.current_accelerator()
    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    return mo, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 手写 Transformer
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 01. 位置编码
    $PE(pos,2i)=sin(\frac{pos}{10000^{2i/d}})$
    $PE(pos,2i+1)=cos(\frac{pos}{10000^{2i/d}})$
    """)
    return


@app.cell
def _(torch):
    def PosEncoding(pos, feaDim):
        # 先计算三角函数以内的部分
        idxTsn = torch.tensor(range(feaDim),dtype=torch.float32) // 2
        idxTsn = pos / (10000 ** (2 * idxTsn / feaDim))
        # 加 pi/2 的偏移, 统一成 sin 函数
        shiftTsn = torch.zeros(feaDim)
        shiftTsn[1::2] = torch.pi / 2 # 用三角函数转换公式将 cos 统一成 sin
        mergeTsn = idxTsn + shiftTsn
        mergeTsn = torch.sin(mergeTsn)
        return mergeTsn
    return (PosEncoding,)


@app.cell
def _(PosEncoding):
    PosEncoding(pos=1, feaDim=4)
    return


@app.cell
def _(torch):
    tt = torch.ones(5) + torch.rand(5)
    tt
    return (tt,)


@app.cell
def _(torch, tt):
    torch.sin(tt)
    return


@app.cell
def _():
    4.4//2
    return


@app.cell
def _(torch, tt):
    ((torch.tensor(range(5)) + tt)//2)/100
    return


@app.cell
def _(torch):
    # 测试QKt
    X = torch.randn(20,512)

    Q = torch.randn(200,484)
    K = torch.randn(200,484)
    # QX = X @ Q
    # KX = X @ K
    QK = Q @ K.T
    return K, Q, QK


@app.cell
def _(K, Q, QK):
    print(Q.mean(), Q.var())
    print(K.mean(), K.var())
    # print(QX.mean(), QX.var())
    # print(KX.mean(), KX.var())
    print(QK[0].mean(), QK[0].var())
    return


@app.cell
def _(QK, torch):
    QKD = QK / torch.tensor([484]).sqrt()
    print(QKD.mean(), QKD.var())
    return


if __name__ == "__main__":
    app.run()
