import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import numpy as np

    return mo, np, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 1. 初始化一个张量
    """)
    return


@app.cell
def _(np, torch):
    ## 直接从数据创建
    tenFromData = torch.tensor([[[1,2,3],[2,3,4],[3,4,5]],[[0,0,1],[1,1,0],[0,1,1]]])
    tenFromData.shape

    ## 从数组创建
        # 从数组创建 tensor 且储存在 CPU 时, 其在内存上是同一个位置(本质同一变量)
    tenFromArray = torch.from_numpy(np.random.randint(0,10,size=[3,2])) # randint 左开右闭
    tenFromArray

    ## 从张量创建
    tenFromTensor_one = torch.ones_like(tenFromData) # 创建形状完全相同的数据全是 1 的张量
    tenFromTensor_one 

    tenFromTensor_ran = torch.rand_like(tenFromData, dtype=torch.float) 
        # 创建形状相同的随机数张量, 默认数据类型会跟输入一致
        # 因为只能输出0-1之间的小数类型, 支持多种小数精度, 因此, 如果输入张量是整数, 这里必须指定输出类型
    tenFromTensor_ran

    ## 根据形状定义
    tenFromShape_ran = torch.rand([2,3,4]) # 深度 2, 行 3, 列 4
    tenFromShape_one = torch.ones([2,3,4])
    tenFromShape_zero = torch.zeros([2,3,4])
    tenFromShape_zero
    return tenFromArray, tenFromData, tenFromShape_one


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 张量的属性
    """)
    return


@app.cell
def _(torch):
    tenAttr = torch.rand(2,3,4, device='cuda') # 创建一个 深2行3列4 的随机小数张量, 等同于 rand([2,3,4])
    tenAttr

    ## 主要是三种属性
    tenAttr.shape # 形状
    tenAttr.device # 存储设备
    tenAttr.dtype # 数据类型
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 3. 张量的运算
    """)
    return


@app.cell
def _(tenFromArray, tenFromData, tenFromShape_one, torch):
    ## 运算前先将张量转移到 GPU
    for tensor in [tenFromData, tenFromArray, tenFromShape_one]:
        print(tensor.device)

    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'

    for tensor in [tenFromData, tenFromArray, tenFromShape_one]:
        tensor = tensor.to(device)
            # 这里只是将新的 tensor 变量移动, 原变量并没有修改
        print(tensor.device)
    return (device,)


@app.cell
def _(device, torch):
    ## 创建一个运算用 tensor
    tenOperate = torch.rand(2,3,4, device=device, dtype=torch.float)
    tenOperate

    return (tenOperate,)


@app.cell
def _(device, tenOperate, torch):
    ## 索引
    print("索引: ", tenOperate[:,:,3])

    ## 切片
    tenOperate[:,:,3] = 0.6666
    print(tenOperate)

    ## 拼接
    torch.cat([tenOperate, tenOperate], dim=1)
        # dim 是维度, 在 3 阶 tensor 中, 0-2 分别是 深度,高度,宽度 / 层,行,列

    ## 乘法
    tenHarfOne = torch.ones(2,3,4, device=device) * 0.5
    tenOperateHarf = tenOperate.mul(tenHarfOne)
    print(tenOperateHarf)
    return


@app.cell
def _(device, torch):
    torch.ones(2,3,4, device=device) * 0.5

    return


@app.cell
def _(device, torch):
    tt = torch.tensor(
        [[1,0,1,1],[1,0,1,1],[1,0,1,1],[1,0,1,1]],
        device=device, dtype=torch.float
    )
    tt @ tt
    return


if __name__ == "__main__":
    app.run()
