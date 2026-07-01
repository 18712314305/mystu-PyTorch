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

    X = torch.tensor([[1,2],[3,4],[4,5]],dtype=torch.float32).to(device)
    Y = torch.tensor([[11],[12],[13]],dtype=torch.float32).to(device)
    return X, Y, device, mo, nn, torch


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 手搓优化器: 惯性分支
    - momentum
    - nesterov momentum
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 标准SGD
    用 torch 内置的 SGD
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    def StandardSGD(
        x :torch.tensor, # 3个样本, 2个特征, 3*2
        y :torch.tensor, # 3个样本, 1个输出, 3*1
        device
    ):
        # 构建最简单模型
        model = nn.Linear(2,1,bias=False).to(device)

        # 初始化固定参数
        with torch.no_grad():
            model.weight.fill_(0.5)

        # 定义损失函数
        lossFunc = nn.MSELoss()

        # 优化器
        optimizer = torch.optim.SGD(model.parameters(),lr=0.01, momentum=0.9, nesterov=True)

        # 3个循环
        for epoch in range(1,11):
            optimizer.zero_grad()
            pred = model(x)
            loss = lossFunc(y, pred)
            loss.backward()
            print(f"in epoch {epoch}, weight is: ", model.weight.clone().detach())
            # print(f"in epoch {epoch}, grad is: ", model.weight.grad.clone().detach())
            optimizer.step()

    StandardSGD(x=X, y=Y, device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 手搓无动量的 SGD
    """)
    return


@app.cell(hide_code=True)
def _(X, Y, device, nn, torch):
    def ManualSGD(x, y, device):
        # 梯度下降函数
        def GetGrad(X, Y, Ypred):
            error = Ypred - Y
            grad = (2/X.shape[0]) * (error.T @ X)
            return grad

        # 构建最简单模型
        model = nn.Linear(2,1,bias=False).to(device)
        # 初始化固定参数
        with torch.no_grad():
            model.weight.fill_(0.5)
        # 定义损失函数
        lossFunc = nn.MSELoss()

        # 无优化器
        with torch.no_grad():
            for epoch in range(1,5):
                pred = model(X)
                manualGrad = GetGrad(X, Y, pred)
                # 更新参数
                print(f"in epoch {epoch}, weight is: ", model.weight.clone().detach())
                print(f"in epoch {epoch}, grad is: ", manualGrad)
                model.weight.data -= 0.01 * manualGrad

    ManualSGD(x=X, y=Y, device=device)  
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 有动量的SGD
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    def ManualSGDwithMomentum(x, y, device):
        # 梯度下降函数
        def GetGrad(X, Y, Ypred):
            error = Ypred - Y
            grad = (2/X.shape[0]) * (error.T @ X)
            return grad

        # 构建最简单模型
        model = nn.Linear(2,1,bias=False).to(device)
        # 初始化固定参数
        with torch.no_grad():
            model.weight.fill_(0.5)
        # 定义损失函数
        lossFunc = nn.MSELoss()

        # 无优化器
        with torch.no_grad():
            vt = torch.zeros_like(model.weight)
            for epoch in range(1,11):
                pred = model(X)
                manualGrad = GetGrad(X, Y, pred)
                # 更新参数
                print(f"in epoch {epoch}, weight is: ", model.weight.clone().detach())
                # print(f"in epoch {epoch}, grad is: ", manualGrad)
                vt = 0.9 * vt + manualGrad
                model.weight.data -= 0.01 * vt

    ManualSGDwithMomentum(x=X, y=Y, device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## nesterov momentum
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    def ManualSGDwithNesterov(X, Y, device):
        # 梯度下降函数
        def GetGrad(X, Y, Ypred):
            error = Ypred - Y
            grad = (2/X.shape[0]) * (error.T @ X)
            return grad

        # 构建最简单模型
        model = nn.Linear(2,1,bias=False).to(device)
        # 初始化固定参数
        with torch.no_grad():
            model.weight.fill_(0.5)
        # 定义损失函数
        lossFunc = nn.MSELoss()

        # 无优化器
        with torch.no_grad():
            vt = torch.zeros_like(model.weight)
            for epoch in range(1,11):
                pred = model(X)
                futureGrad = GetGrad(X, Y, pred)
                # 先保存当前梯度
                print(f"in epoch {epoch}, weight is: ", model.weight.clone().detach())
                vt = 0.9 * vt + futureGrad
                model.weight.data -= 0.01 * (futureGrad + 0.9 * vt)

    ManualSGDwithNesterov(X=X, Y=Y, device=device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 手搓优化器: 步长分支
    - Adagrad
    - RMSprop
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pytorch 标准 AdaGrad
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    def StandardAda(X,Y,device):
        # 极简模型
        model = nn.Linear(2,1,bias=False).to(device)
        # 初始化参数
        with torch.no_grad():
            model.weight.fill_(0.5)
        # 优化器
        optimizer = torch.optim.Adagrad(model.parameters(), lr=0.1, eps=1e-10)
        # loss
        lossFunc = nn.MSELoss()
        # 训练
        model.train()
        for e in range(1,11):
            optimizer.zero_grad()
            print(f"before epoch {e}, the weight is: ", model.weight.clone().detach())
            # forward
            pred = model(X)
            # loss
            loss = lossFunc(Y, pred)
            # backward
            loss.backward()
            # update
            optimizer.step()
        return 

    StandardAda(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## AdaGrad 算法: 仅调整偏导的一次项, 即, 对每个参数动态调整学习率
    $S_t=S_{t-1}+g_t\odot{g_t}$
    其中, $g_t$ 是当前梯度, $S_t$ 是与权重同形状的矩阵, 初始化为 0, 每次梯度更新时, $S_t$ 累加梯度的平方

    $\theta_t=\theta_{t-1}-{\frac{\eta}{\sqrt{S_t}+\epsilon}}.g_t$
    其中, $\theta$ 是权重, $\epsilon$ 是极小量, 防止分母为 0, $\eta$ 是全局学习率
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    def ManualAda(X,Y,device):
        # 极简模型
        model = nn.Linear(2,1,bias=False).to(device)
        # 初始化参数
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss
        lossFunc = nn.MSELoss()
        # 训练
        model.train()
        st = torch.zeros_like(model.weight.clone().detach()) # 这里不需要 to device, 因为 zeroslike 复制时也会复制设备
        for e in range(1,11):
            currWeight = model.weight.clone().detach()
            model.zero_grad()
            print(f"before epoch {e}, the weight is: ", currWeight)
            # forward
            pred = model(X)
            # loss
            loss = lossFunc(pred, Y)
            # backward
            loss.backward()
            gt = model.weight.grad.detach() 
            # update
            st = st + gt * gt
            adaG = (0.1 / (st.sqrt() + 1e-10)) * gt
            model.weight.data -= adaG

    ManualAda(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pytorch 标准 RMSprop
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    def StandardRMSprop(X,Y,device):
        # 极简模型
        model = nn.Linear(2,1,bias=False).to(device)
        # 初始化参数
        with torch.no_grad():
            model.weight.fill_(0.5)
        # 优化器
        optimizer = torch.optim.RMSprop(model.parameters(), lr=0.5, eps=1e-10, alpha=0.1)
        # loss
        lossFunc = nn.MSELoss()
        # 训练
        model.train()
        for e in range(1,11):
            optimizer.zero_grad()
            print(f"before epoch {e}, the weight is: ", model.weight.clone().detach())
            # forward
            pred = model(X)
            # loss
            loss = lossFunc(Y, pred)
            # backward
            loss.backward()
            # update
            optimizer.step()
        return 

    StandardRMSprop(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## RMSprop 算法: 指数移动平均, 防止 Adagrad 过快衰减

    $S_t=\alpha*S_{t-1}+(1-\alpha)*g_t\odot{g_t}$.
    其中, $g_t$ 是当前梯度, $S_t$ 是与权重同形状的矩阵, 初始化为 0, 每次梯度更新时, $\alpha$ 是平滑常数
    $\theta_t=\theta_{t-1}-{\frac{\eta}{\sqrt{S_t}+\epsilon}}.g_t$
    其中, $\theta$ 是权重, $\epsilon$ 是极小量, 防止分母为 0, $\eta$ 是全局学习率
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    def ManualRMSprot(X,Y,device):
        # 模型并初始化
        model = nn.Linear(2,1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss
        lossFunc = nn.MSELoss()
        # train
        st = torch.zeros_like(model.weight)
        for e in range(1,11):
            print(f"before epoch {e}, the weight is: ", model.weight.clone().detach())
            # refresh
            model.zero_grad()
            # forward
            pred = model(X)
            # loss and backward
            loss = lossFunc(pred, Y)
            loss.backward()
            # update
            gt = model.weight.grad.clone().detach()
            st = 0.1 * st + 0.9 * gt.square()
            prop = (0.5/(st.sqrt() + 1e-10)) * gt
            model.weight.data -= prop

    ManualRMSprot(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 手搓优化器: 合并分支
    - Adam
    - Nadam
    - AdamW
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pytorch 标准 adam
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    # pytorch 标准 SGD
    def StandardAdam(X,Y,device):
        # model and init
        model = nn.Linear(2,1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss and optim
        lossFunc = nn.MSELoss()
        optim = torch.optim.Adam(model.parameters(), lr=0.1, betas=(0.9,0.999), eps=1e-5)

        # train
        for e in range(1,6):
            # clear
            model.zero_grad()
            # print
            print(f"in epoch {e}, weight: ", model.weight.data.clone().detach())
            # forward
            pred = model(X)
            # loss & backward
            loss = lossFunc(pred, Y)
            loss.backward()
            # update
            optim.step()

    StandardAdam(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 手搓 adam

    $m_t=\beta_1*m_{t-1}+(1-\beta_1)*g_t$
    $v_t=\beta_2*v_{t-1}+(1-\beta_2)*g_t^2$
    $\hat{m_t}=\frac{m_t}{1-\beta_1^t}$
    $\hat{v_t}=\frac{v_t}{1-\beta_2^t}$
    $\theta_t=\theta_{t-1}-\frac{\eta}{\epsilon + \sqrt{\hat{v_t}}}*\hat{m_t}$

    说明:
    $m_t$ 即动量项, $v_t$ 即动态学习率项,
    一般 $\beta_1$ 和 $\beta_2$ 取值 0.99 和 0.999, 这导致训练早期, $m_0=0$ 时, $m_1$ 更新幅度太小, 因此, $\hat{m_t}$ 通过 $t$ 的幂次项, 在早期小分母放大变化, 在后期分母$\approx$1, 消除缩放
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    ## manual adam
    def ManualAdam(X,Y,device):
        # model and init
        model = nn.Linear(2,1,bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss and private optim
        lossFunc = nn.MSELoss()
        mt = torch.zeros_like(model.weight)
        vt = torch.zeros_like(model.weight)
        b1 = 0.9
        b2=0.999
        # train
        for e in range(1,6):
            # clear
            model.zero_grad()
            # print
            print(f"in epoch {e}, weight: ", model.weight.data.clone().detach())
            # forward
            pred = model(X)
            # loss and backward
            loss = lossFunc(pred, Y)
            loss.backward()
            # private param
            grad = model.weight.grad.detach()
            mt = b1 * mt + (1-b1) * grad
            vt = b2 * vt + (1-b2) * grad.square()
            mtHat = mt / (1 - b1**e)
            vtHat = vt / (1 - b2**e)
            # update
            model.weight.data -= (0.1 / (1e-5 + vtHat.sqrt())) * mtHat

    ManualAdam(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pytorch 标准 nadam
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    # pytorch nadam
    def StandardNadam(X,Y,device):
        # model and init
        model = nn.Linear(2,1,bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss and optim
        lossFunc = nn.MSELoss()
        optim = torch.optim.NAdam(model.parameters(), betas=(0.9, 0.999), eps=1e-8, momentum_decay=0.004, lr=0.1)

        # train
        for e in range(1,6):
            # clear
            model.zero_grad()
            # print
            print(f"in epoch {e}, weight: ", model.weight.data.clone().detach())
            # forward
            pred = model(X)
            # loss and backward
            loss = lossFunc(pred, Y)
            loss.backward()
            # update
            optim.step()

    StandardNadam(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 手搓 nadam

    $m_t=\beta_1*m_{t-1}+(1-\beta_1)*g_t$
    $v_t=\beta_2*v_{t-1}+(1-\beta_2)*g_t^2$
    $\hat{v_t}=\frac{v_t}{1-\beta_2^t}$
    $\mu_t=\beta_1*(1-0.5*0.96^{t*\phi})$
    $\prod_{t}=\prod_{i}^{t}\mu_i$
    $\hat{m_t}=\frac{(1-\mu_t)gt}{1-\prod_{t}}+\frac{\mu_{t+1}*m_t}{1-\prod_{t+1}}$
    $\theta_t=\theta_{t-1}-\frac{\eta}{\epsilon+\sqrt{\hat{v_t}}}*\hat{m_t}$
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    # 手搓 nadam
    def ManualNadam(X, Y, device):
        # model and init
        model = nn.Linear(2,1,bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss
        lossFunc = nn.MSELoss()
        # private optim param
        b1, b2, phi, eps, lr = 0.9, 0.999, 0.004, 1e-8, 0.1
        mT = torch.zeros_like(model.weight)
        vT = torch.zeros_like(model.weight)
        prodT = 1
        for e in range(1,6):
            # clear
            model.zero_grad()
            # print
            print(f"in epoch {e}, weight: ", model.weight.data.detach())
            # forward
            pred = model(X)
            # loss and back
            loss = lossFunc(pred, Y)
            loss.backward()
            # update
            grad = model.weight.grad.detach()
            mT = b1 * mT + (1-b1) * grad
            vT = b2 * vT + (1-b2) * grad.square()
            vThat = vT / (1-b2**e)
            muT = b1 * (1 - 0.5*0.96**(e*phi))
            muTp1 = b1 * (1 - 0.5*0.96**((e+1)*phi))
            prodT *= muT
            prodTp1 = prodT * muTp1
            mThat = ((1-muT)*grad / (1-prodT)) + ((muTp1 * mT) / (1-prodTp1))
            model.weight.data -= (lr / (eps + vThat.sqrt())) * mThat

    ManualNadam(X, Y, device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## pytorch 标准 adamw
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    # pytorch 标准 SGD
    def StandardAdamw(X,Y,device):
        # model and init
        model = nn.Linear(2,1, bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss and optim
        lossFunc = nn.MSELoss()
        optim = torch.optim.AdamW(model.parameters(), lr=0.1, betas=(0.9,0.999), eps=1e-5, weight_decay=0.1)

        # train
        for e in range(1,6):
            # clear
            model.zero_grad()
            # print
            print(f"in epoch {e}, weight: ", model.weight.data.clone().detach())
            # forward
            pred = model(X)
            # loss & backward
            loss = lossFunc(pred, Y)
            loss.backward()
            # update
            optim.step()

    StandardAdamw(X,Y,device)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 手搓 adamw

    $m_t=\beta_1*m_{t-1}+(1-\beta_1)*g_t$
    $v_t=\beta_2*v_{t-1}+(1-\beta_2)*g_t^2$
    $\hat{m_t}=\frac{m_t}{1-\beta_1^t}$
    $\hat{v_t}=\frac{v_t}{1-\beta_2^t}$
    $\theta_{t-1} = \theta_{t-1}*(1-\eta\lambda)$
    $\theta_t=\theta_{t-1}-\frac{\eta}{\epsilon + \sqrt{\hat{v_t}}}*\hat{m_t}$

    说明:
    $m_t$ 即动量项, $v_t$ 即动态学习率项,
    一般 $\beta_1$ 和 $\beta_2$ 取值 0.99 和 0.999, 这导致训练早期, $m_0=0$ 时, $m_1$ 更新幅度太小, 因此, $\hat{m_t}$ 通过 $t$ 的幂次项, 在早期小分母放大变化, 在后期分母$\approx$1, 消除缩放
    解耦本质上就是先把权重整体缩放, 再更新纯来自 loss 的梯度
    """)
    return


@app.cell
def _(X, Y, device, nn, torch):
    ## manual adamw
    def ManualAdamw(X,Y,device):
        # model and init
        model = nn.Linear(2,1,bias=False).to(device)
        with torch.no_grad():
            model.weight.fill_(0.5)
        # loss and private optim
        lossFunc = nn.MSELoss()
        mt = torch.zeros_like(model.weight)
        vt = torch.zeros_like(model.weight)
        b1 = 0.9
        b2=0.999
        weight_decoy = 0.1
        # train
        for e in range(1,6):
            # clear
            model.zero_grad()
            # print
            print(f"in epoch {e}, weight: ", model.weight.data.clone().detach())
            # forward
            pred = model(X)
            # loss and backward
            loss = lossFunc(pred, Y)
            loss.backward()
            # private param
            grad = model.weight.grad.detach()
            mt = b1 * mt + (1-b1) * grad
            vt = b2 * vt + (1-b2) * grad.square()
            mtHat = mt / (1 - b1**e)
            vtHat = vt / (1 - b2**e)
            # update
            model.weight.data *= (1-weight_decoy*0.1)
            model.weight.data -= (0.1 / (1e-5 + vtHat.sqrt())) * mtHat

    ManualAdamw(X,Y,device)
    return


if __name__ == "__main__":
    app.run()
