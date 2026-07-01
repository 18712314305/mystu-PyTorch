import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import torch
    import torch.nn as nn

    device = torch.accelerator.current_accelerator()

    torch.manual_seed(42)
    torch.mps.manual_seed(42)

    Xtrain = torch.rand(2,4,3,dtype=torch.float32).to(device) * 10
    Xtest = torch.rand(2,4,3,dtype=torch.float32).to(device) * 10
    Y = torch.rand(2,4,1,dtype=torch.float32).to(device) * 10
    return Xtest, Xtrain, Y, device, mo, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 手写批次归一化和层归一化
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Batch Normalization
    """)
    return


@app.cell
def _(Xtest, Xtrain, Y, device, nn, torch):
    ## pytorch 标准实现
    def StandardBatchNorm(Xtrain, Xtest, Y, device):
        # 对比两个模型
        modelNorm = nn.Sequential(
            nn.BatchNorm1d(num_features=3, affine=False, eps=1e-5),
            nn.Linear(3,1,bias=False)
        ).to(device)
        modelUNorm = nn.Linear(3,1,bias=False).to(device)

        # init
        with torch.no_grad():
            modelNorm[1].weight.fill_(0.5)
            modelUNorm.weight.fill_(0.5)
        # optim
        optimiMN = torch.optim.SGD(modelNorm.parameters(), lr=0.01)
        optimiMU = torch.optim.SGD(modelUNorm.parameters(), lr=0.01)
        # loss
        lossFunc = nn.MSELoss()
        # train
        # xBN = Xtrain[0]
        for bidx, batch in enumerate(Xtrain):
            # clear
            modelUNorm.zero_grad()
            modelNorm.zero_grad()
            # forward
            # predMN = modelNorm(batch)
            xBN = modelNorm[0](batch)
            predMN = modelNorm[1](xBN)
            predMU = modelUNorm(batch)

            # print
            print(f"without Norm, in this batch, weight is: ", modelUNorm.weight.clone().detach())
            print(f"with Norm, in this batch, weight is: ", modelNorm[1].weight.clone().detach())
            print(f"without Norm, in this batch, mean is: ", batch.mean(dim=0))
            print(f"with Norm, in this batch, mean is: ", xBN.mean(dim=0))
            print(f"with Norm, in this batch, momentum mean is: ", modelNorm[0].running_mean)
            print(f"without Norm, in this batch, var is: ", batch.var(dim=0))
            print(f"with Norm, in this batch, var is: ", xBN.var(dim=0, unbiased=False))
            print("……"*40)

            # loss & backward
            lossMN = lossFunc(predMN, Y[bidx])
            lossMU = lossFunc(predMU, Y[bidx])
            lossMN.backward()
            lossMU.backward()
            # update
            optimiMN.step()
            optimiMU.step()

        # test
        modelNorm.eval()
        for bidx, batch in enumerate(Xtest):
            # forward
            # predMN = modelNorm(batch)
            xBN = modelNorm[0](batch)
            predMN = modelNorm[1](xBN)
            predMU = modelUNorm(batch)
            # print
            print(f"in test, without Norm, in this batch, mean is: ", batch.mean(dim=0))
            print(f"in test, with Norm, in this batch, mean is: ", xBN.mean(dim=0))
            print(f"in test, with Norm, in this batch, momentum mean is: ", modelNorm[0].running_mean)
            print(f"in test, without Norm, in this batch, var is: ", batch.var(dim=0))
            print(f"in test, with Norm, in this batch, var is: ", xBN.var(dim=0, unbiased=False))
            print("……"*40)
    StandardBatchNorm(Xtrain, Xtest, Y, device)
    return


@app.cell
def _(Xtest, Xtrain, Y, device, nn, torch):
    ## 手搓 batch norm
    def ManualBatchNorm(Xtrain, Xtest, Y, device):
        def batchNorm(xBatch):
            bmean = xBatch.mean(dim=0)
            bvar = xBatch.var(dim=0, unbiased=False)
            bnorm = (xBatch - bmean) / (bvar + 1e-5).sqrt()
            return bnorm

        modelMU = nn.Linear(3,1, bias=False).to(device)
        modelMN = nn.Linear(3,1, bias=False).to(device)
        # init
        with torch.no_grad():
            modelMN.weight.fill_(0.5)
            modelMU.weight.fill_(0.5)
        # loss
        lossFunc = nn.MSELoss()
        # optim
        optimMU = torch.optim.SGD(modelMU.parameters(), lr=0.01)
        optimMN = torch.optim.SGD(modelMN.parameters(), lr=0.01)

        # 训练
        for bidx, batch in enumerate(Xtrain):
            # clear
            modelMU.zero_grad()
            modelMN.zero_grad()

            # forward
            predMU = modelMU(batch)
                # norm
            normBatch = batchNorm(batch)
            predMN = modelMN(normBatch)

            # print
            print(f"without Norm, in this batch, weight is: ", modelMU.weight.clone().detach())
            print(f"with Norm, in this batch, weight is: ", modelMN.weight.clone().detach())
            print(f"without Norm, in this batch, mean is: ", batch.mean(dim=0))
            print(f"with Norm, in this batch, mean is: ", normBatch.mean(dim=0))
            print(f"without Norm, in this batch, var is: ", batch.var(dim=0))
            print(f"with Norm, in this batch, var is: ", normBatch.var(dim=0, unbiased=False))
            print("……"*40)

            # loss & backward
            lossMU = lossFunc(predMU, Y[bidx])
            lossMU.backward()
            lossMN = lossFunc(predMN, Y[bidx])
            lossMN.backward()
            # update
            optimMU.step()
            optimMN.step()

        # 测试


    ManualBatchNorm(Xtrain, Xtest, Y, device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## Layer Normaliztion
    """)
    return


@app.cell
def _(device, torch):
    In = torch.randn(10, 16, device=device, dtype=torch.float32) * 10
    Out = torch.randn(10, 1, device=device, dtype=torch.float32) * 100

    return In, Out


@app.cell
def _(In, Out, device, nn, torch):
    # 标准
    def StandardLayerNorm(X,Y,device):
        # model and init
        modelMN = nn.Sequential(
            nn.Linear(16, 32, bias=False), 
            nn.LayerNorm(32, elementwise_affine=False,bias=False, eps=1e-5), 
            nn.Linear(32,1, bias=False)
        ).to(device)
        modelMU = nn.Sequential(
            nn.Linear(16,32, bias=False), 
            nn.Linear(32,1, bias=False)
        ).to(device)
        with torch.no_grad():
            modelMU[0].weight.fill_(0.5)
            modelMU[1].weight.fill_(0.5)
        # 输出modelMN的参数给手搓用, 因为不能全相同权重, 不然输出的方差为0, 导致layernorm输出全是0
        StdWeight = {
            0: modelMN[0].weight.data.clone().detach(), 
            2: modelMN[2].weight.data.clone().detach(), 
        }
        print(StdWeight[0][0])
        # loss
        lossFunc = nn.MSELoss()

        # optim
        optimMN = torch.optim.SGD(modelMN.parameters(), lr=0.001)
        optimMU = torch.optim.SGD(modelMU.parameters(), lr=0.001)

        RESwightNorm = {"weight": []}
        RESwithoutNorm = {"weight": []}
        # train
        modelMN.train()
        modelMU.train()
        for e in range(5):
            # clear grad
            modelMN.zero_grad()
            modelMU.zero_grad()
            # forward
            predMN = modelMN(X)
            predMU = modelMU(X)
            # collect
            RESwightNorm["weight"].append(modelMN[0].weight.data.clone().detach().cpu().numpy()[0])
            RESwithoutNorm["weight"].append(modelMU[0].weight.data.clone().detach().cpu().numpy()[0])
            # loss and back
            lossMN = lossFunc(predMN, Y)
            lossMN.backward()
            # print(lossMN.item())
            lossMU = lossFunc(predMU, Y)
            lossMU.backward()
            # print(lossMU.item())
            # update weight
            optimMN.step()
            optimMU.step()

        return RESwightNorm, RESwithoutNorm, StdWeight
    
    StdNorm, StdUnorm, StdWeight = StandardLayerNorm(In, Out, device)
    return StdNorm, StdWeight


@app.cell
def _(StdNorm):
    sum(StdNorm['weight'][0])
    return


@app.cell(hide_code=True)
def _():
    # StdUnorm
    return


@app.cell
def _(In, Out, StdWeight, device, nn, torch):
    ## 手写 layernorm
    def ManualLayerNorm(X, Y, device, weight):
        def layerNorm(inTen):
            layerMean = inTen.mean(dim=1)
            layerVar = inTen.var(dim=1, unbiased=False)
            Norm = (inTen - layerMean.unsqueeze_(dim=1)) / (layerVar.unsqueeze_(dim=1).sqrt() + 1e-5)
            return Norm
        
        # model
        model = nn.Sequential(
            nn.Linear(16,32,bias=False), 
            nn.Linear(32,1, bias=False)
        ).to(device)
        # init
        with torch.no_grad():
            model[0].weight.data = weight[0]
            model[1].weight.data = weight[2]
        # loss
        lossFunc = nn.MSELoss()
        # optim
        optim = torch.optim.SGD(model.parameters(), lr=0.001)
        ResWeights = []
        # train
        for e in range(5):
            # clear
            model.zero_grad()
            # forward
            m1out = model[0](X)
            # print(m1out.shape)
            normOut = layerNorm(m1out)
            pred = model[1](normOut)
            # print
            ResWeights.append(model[0].weight.data.clone().detach().cpu().numpy()[0])
            # loss 
            loss = lossFunc(pred, Y)
            loss.backward()
            # update
            optim.step()
        return ResWeights
    maunalNorm = ManualLayerNorm(In, Out, device, StdWeight)
    return (maunalNorm,)


@app.cell
def _(maunalNorm):
    sum(maunalNorm[0])
    return


@app.cell
def _(StdWeight):
    StdWeight[0][0]
    return


if __name__ == "__main__":
    app.run()
