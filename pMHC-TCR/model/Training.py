import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    import torch
    import torch.nn as nn
    from torch.utils.data import Dataset, Subset, DataLoader
    from tqdm import tqdm
    import pandas as pd 
    import numpy as np
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, recall_score, roc_auc_score, precision_recall_curve, auc
    import gc

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    fileDir = Path(__file__).resolve().parent
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    return (
        DataLoader,
        Dataset,
        StratifiedGroupKFold,
        Subset,
        accuracy_score,
        auc,
        device,
        fileDir,
        mo,
        nn,
        np,
        pd,
        precision_recall_curve,
        recall_score,
        roc_auc_score,
        torch,
        tqdm,
    )


@app.cell
def global_func(DataLoader, Dataset, Subset, fileDir, pd, torch):
    ## embedding 读取器
    class ThreeTowerDataset(Dataset):
        def __init__(self, dataTrain, tcrPtPath, hlaPtPath, epitopePtPath):
            # 读取各文件
            self.data = dataTrain # data 传入读取过的 dataframe
            self.tcrFeat = torch.load(tcrPtPath, weights_only=False)
            self.hlaFeat = torch.load(hlaPtPath, weights_only=False)
            self.epitopeFeat = torch.load(epitopePtPath, weights_only=False)
            self.labels = self.data['label'].values

        # 强制两个魔术方法
        def __len__(self):
            return self.data.shape[0]

        def __getitem__(self, queryIdx):
            # 使用时会由 dataloader 向数据集实例请求数据, 传入是一个索引整数
            row = self.data.iloc[queryIdx]
            tcrVec = torch.tensor(data = self.tcrFeat[row['tcr']], dtype=torch.bfloat16)
            hlaVec = torch.tensor(data = self.hlaFeat[row['hla_pseudo']], dtype=torch.bfloat16)
            epitopeVec = torch.tensor(data = self.epitopeFeat[row['epitope']], dtype=torch.bfloat16)
            label = torch.tensor(data=self.labels[queryIdx], dtype=torch.bfloat16)
            return tcrVec, hlaVec, epitopeVec, label

    def TrainDataLoad():
        # 读取训练数据
        trainFile = fileDir / "../data/Train.DownSampling.csv"
        trainData = pd.read_csv(trainFile, header=0, sep="\t")
        trainDataset = ThreeTowerDataset(
            dataTrain = trainData, 
            tcrPtPath = fileDir / "../feature/tcr.GlobalEmbedding.pt", 
            hlaPtPath = fileDir / "../feature/hla_pseudo.GlobalEmbedding.pt", 
            epitopePtPath = fileDir / "../feature/epitope.GlobalEmbedding.pt"
        )
        return trainDataset

    def FoldSplit(
        trainDataset, # 数据集
        trainFoldIdx, # 训练 fold 索引
        testFoldIdx, # 测试 fold 索引
        trainShuffle = True, # 训练集加载时是否需要随机打乱
        testShuffle = False, # 测试集加载时是否需要随机打乱
    ):
        # 每个 fold 拆分训练数据集, 并返回数据集加载
        trainSub = Subset(trainDataset, trainFoldIdx)
        testSub  = Subset(trainDataset, testFoldIdx)
        trainLoader = DataLoader(trainSub, batch_size=1600, shuffle=trainShuffle)
        testLoader = DataLoader(testSub, batch_size=1600, shuffle=testShuffle)

        return trainSub, testSub, trainLoader, testLoader


    # 日志模板
    trainLogTemp = [
        ("testAntigenFrac", 0), 
        ("trainEpochLoss", []), ("trainBatchLoss", []), 
        ("testEpochLoss", []), ("testBatchLoss", []), 
        ("accuracy", []), ("recall", []), 
        ("ROC-AUC", []), ("PR-AUC", [])
    ] # 每个元素是 key,value 的tuple
        # trainBatchLoss 和 testBatchLoss 形状是 [epoh_num * batch_num], 其他形状是 [1 * epoch_num]
    return FoldSplit, TrainDataLoad


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## baseline: 无任何正则化的全连接MLP
    三塔+一个分类器的方式, 三塔输出特征融合采样横向拼接,
    tcr 塔: 1280 -> 256[W:256*1280] -> 128[W:128*256]
    hla 塔: 1280 -> 128 -> 128
    antigen 塔: 1280 -> 256 -> 128
    分类器塔 : 128*3 -> 32 -> 1 -> sigmoid
    """)
    return


@app.cell(hide_code=True)
def baseline_arch():
    # ## baseline 架构
    # class BaseThreeTower(nn.Module):
    #     def __init__(self, tcrDim=1280, hlaDim=1280, epitopeDim=1280, hiddemDim=128, outDim=1):
    #         super(BaseThreeTower, self).__init__()
    #         # 这里 tcrDim 等是自定义的, 实际上也可以不用这些参数, 直接在后面构造线形层的时候写死
    #         # super 初始化是为了初始化父类构造, 否则, 即使继承了父类的方法, 缺失 Module 类初始化时的实例属性, 也是没法跑的

    #         # TCR 塔
    #         self.tcrTower = nn.Sequential(
    #             nn.Linear(tcrDim, 256), # 256 个节点的线性层
    #             nn.ReLU(), # 激活函数
    #             nn.Linear(256, hiddemDim),
    #             nn.ReLU()
    #         )

    #         # HLA 塔
    #         self.hlaTower = nn.Sequential(
    #             nn.Linear(hlaDim, 128), 
    #             nn.ReLU(), 
    #             nn.Linear(128, hiddemDim), 
    #             nn.ReLU()
    #         )

    #         # epitope 塔
    #         self.epitopeTower = nn.Sequential(
    #             nn.Linear(epitopeDim, 256), 
    #             nn.ReLU(),
    #             nn.Linear(256, hiddemDim),
    #             nn.ReLU()
    #         )

    #         # 分类器
    #         self.classifier = nn.Sequential(
    #             nn.Linear(hiddemDim*3, 128), 
    #             nn.ReLU(),
    #             nn.Linear(128, 32), 
    #             nn.ReLU(), 
    #             nn.Linear(32, 1),
    #             nn.Sigmoid()
    #         )

    #     # 重写 forwoard 方法, 替换父类的
    #     def forward(self, tcrEmbed, hlaEmbed, epipoteEmbed):
    #         # 各塔的输入输出
    #         tcrFeat = self.tcrTower(tcrEmbed)
    #         hlaFeat = self.hlaTower(hlaEmbed)
    #         epipoteFeat = self.epitopeTower(epipoteEmbed)
    #         # 分类器的输入输出
    #         predScore = self.classifier(torch.cat([tcrFeat, hlaFeat, epipoteFeat], dim=1))
    #         return predScore.squeeze(-1)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### baseline 模型训练
    """)
    return


@app.cell(hide_code=True)
def baseline_dataload():
    # ## 读取训练数据, 实例化数据集
    # trainFile = fileDir / "../data/Train.DownSampling.csv"
    # trainData = pd.read_csv(trainFile, header=0, sep="\t")
    # trainDataset = ThreeTowerDataset(
    #     dataTrain = trainData, 
    #     tcrPtPath = fileDir / "../feature/tcr.GlobalEmbedding.pt", 
    #     hlaPtPath = fileDir / "../feature/hla_pseudo.GlobalEmbedding.pt", 
    #     epitopePtPath = fileDir / "../feature/epitope.GlobalEmbedding.pt"
    # )
    return


@app.cell(hide_code=True)
def baseline_hook_test():
    # from collections import defaultdict

    # def make_hook(layerName, saveDict):
    #     def InnerHook(module, input, output):
    #         saveDict[layerName].append(output.detach().cpu())
    #     return InnerHook
    return


@app.cell(hide_code=True)
def baseline_train():
    # ## 每折训练
    # foldSplit = StratifiedGroupKFold(n_splits=3)
    # # 每折指标记录
    # trainLog = {foldId: {
    #     "testAntigenFrac": 0,
    #     "trainEpochLoss": [], # [1 * epoch_num]
    #     "trainBatchLoss": [], # [epoh_num * batch_num]
    #     "testEpochLoss": [], # [1 * epoch_num]
    #     "testBatchLoss": [], # [epoh_num * batch_num]
    #     "accuracy": [], # [1 * epoch_num]
    #     "recall": [], # [1 * epoch_num]
    #     "ROC-AUC": [], # [1 * epoch_num]
    #     "PR-AUC": [], # [1 * epoch_num]
    #     # "weights": {},
    #     # "outputs": {}, 
    # } for foldId in range(1,4)}


    # for fold, (trainIdx, testIdx) in enumerate(foldSplit.split(
    #     X=trainDataset.data, 
    #     y=trainDataset.labels, 
    #     groups=trainDataset.data['epitope'].values
    # )):   
    #     foldId = fold + 1

    #     # 每个 fold 要重置模型
    #     ## 模型
    #     model = BaseThreeTower().to(device)
    #     ## 损失函数
    #     loss = nn.BCELoss()
    #     ## 优化器
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    #     # 拆分测试和训练fold
    #     trainSub = Subset(trainDataset, trainIdx)
    #     testSub = Subset(trainDataset, testIdx)
    #     trainLoader = DataLoader(trainSub, batch_size=512, shuffle=True)
    #     testLoader = DataLoader(testSub, batch_size=512, shuffle=False)

    #     ## 追踪当前 fold 测试集抗原肽数量比例
    #     trainLog[foldId]["testAntigenFrac"] = testSub.dataset.data.iloc[testIdx]['epitope'].nunique()

    #     for epoch in tqdm(range(10)):
    #         model.train()

    #         ## 追踪 epoch loss
    #         epochTrainLoss = 0.0
    #         epochTestLoss = 0.0
    #         ## 追踪 batch loss
    #         trainLog[foldId]["trainBatchLoss"].append([])
    #         trainLog[foldId]["testBatchLoss"].append([])
    #         ## 追踪权重和输出
    #         # trainLog[foldId]["weights"][epoch] = {}
    #         # trainLog[foldId]["outputs"] = {}
    #         epochWeights = {}
    #         batchIdx = 1

    #         # 训练阶段
    #         for btcr, bhla, bepitope, blabel in trainLoader:
    #             btcr, bhla = btcr.to(device), bhla.to(device)
    #             bepitope, blabel = bepitope.to(device), blabel.to(device)
    #             # 清除历史梯度
    #             optimizer.zero_grad()

    #             ## 追踪 输出
    #             # hookHandles = []
    #             # for layerName, layerModule in model.named_modules():
    #             #     if len(list(layerModule.children())) == 0:
    #             #         handle = layerModule.register_forward_hook(make_hook(layerName, trainLog[foldId]["outputs"][batchIdx]))
    #             #         hook_handles.append(handle)
    #             # 前向传播
    #             bpred = model(btcr,bhla,bepitope)
    #             # 计算损失
    #             bloss = loss(bpred, blabel)
    #             # 反向传播
    #             bloss.backward()
    #             # 梯度更新
    #             optimizer.step()

    #             ## 清除hook
    #             # for h in hookHandles:
    #             #     h.remove()

    #             # 累加损失
    #             epochTrainLoss += bloss.item()

    #             ## 追踪 batch loss
    #             trainLog[foldId]["trainBatchLoss"][-1].append(bloss.item())

    #             ## 追踪权重和输出
    #             # trainLog[foldId]["weights"][epoch][batchIdx] = {}
    #             # trainLog[foldId]["outputs"][batchIdx] = defaultdict(list)
    #             epochWeights[batchIdx] = {}

    #             ### 层均值, 范式, 
    #             # for layerName, layerModule in model.named_modules():
    #             #     if isinstance(layerModule, nn.Linear):
    #             #         layerWeights = layerModule.weight.data.detach()
    #             #         layerWeightsGrad = layerModule.weight.grad.detach()
    #             #         layerBias = layerModule.bias.data.detach()
    #             #         layerBiasGrad = layerModule.bias.grad.detach()
    #             #         epochWeights[batchIdx][layerName] = {
    #             #             "layerWeights": layerWeights.cpu().numpy(),
    #             #             "layerWeightsGrad": layerWeightsGrad.cpu().numpy(), 
    #             #             "layerBias": layerBias.cpu().numpy(), 
    #             #             "layerBiasGrad": layerBiasGrad.cpu().numpy()
    #             #         }


    #             batchIdx += 1


    #         epochAveTrainLoss = epochTrainLoss / len(trainLoader)
    #         ## 追踪 epoch loss
    #         trainLog[foldId]['trainEpochLoss'].append(epochAveTrainLoss)
    #         ## 每个epoch写出权重
    #         # torch.save(epochWeights, fileDir / f"baseline.onlyWeight.Fold{foldId}.epoch{epoch}.pt")

    #         # 验证阶段
    #         model.eval()

    #         epochPred = []
    #         epochLabel = []
    #         with torch.no_grad():
    #             for btcr, bhla, bepitope, blabel in testLoader:
    #                 btcr, bhla = btcr.to(device), bhla.to(device)
    #                 bepitope, blabel = bepitope.to(device), blabel.to(device)
    #                 bpred = model(btcr, bhla, bepitope)
    #                 epochPred.extend(bpred.cpu().numpy())
    #                 epochLabel.extend(blabel.cpu().numpy())
    #                 bloss = loss(bpred, blabel)
    #                 epochTestLoss += bloss.item()

    #                 ## 追踪 batch loss
    #                 trainLog[foldId]["testBatchLoss"][-1].append(bloss.item())

    #         epochAveTestLoss = epochTestLoss / len(testLoader)
    #         ## 追踪 epoch loss
    #         trainLog[foldId]['testEpochLoss'].append(epochAveTestLoss)

    #         ## 追踪 epoch 性能
    #         epochPred, epochLabel = np.array(epochPred), np.array(epochLabel)
    #         epochPredBin = (epochPred > 0.5).astype(int)
    #         trainLog[foldId]['accuracy'].append(accuracy_score(epochLabel, epochPredBin))
    #         trainLog[foldId]['recall'].append(recall_score(epochLabel, epochPredBin, pos_label=1))
    #         trainLog[foldId]['ROC-AUC'].append(roc_auc_score(epochLabel, epochPred))
    #         _precision, _recallPoint, _thresholds = precision_recall_curve(epochLabel, epochPred)
    #         trainLog[foldId]['PR-AUC'].append(auc(_recallPoint, _precision))

    # torch.save(trainLog, fileDir / f"baseline.withoutWeight.Log.epoch10.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## v0.1 logit + weighted BCE
    - 输出层舍弃 sigmoid
    - 损失函数使用加权 BCE
    - 先观察性能指标, 然后才打印看 weights, 因为以后可能会训练更复杂的模型, 不可能总是能打印weights
    """)
    return


@app.cell(hide_code=True)
def v0_1_arch(nn, torch):
    ## v0.1 架构
    ## baseline 架构
    class BaseThreeTowerV0_1(nn.Module):
        def __init__(self, tcrDim=1280, hlaDim=1280, epitopeDim=1280, hiddemDim=128, outDim=1):
            super(BaseThreeTowerV0_1, self).__init__()
            # 这里 tcrDim 等是自定义的, 实际上也可以不用这些参数, 直接在后面构造线形层的时候写死
            # super 初始化是为了初始化父类构造, 否则, 即使继承了父类的方法, 缺失 Module 类初始化时的实例属性, 也是没法跑的

            # TCR 塔
            self.tcrTower = nn.Sequential(
                nn.Linear(tcrDim, 256), # 256 个节点的线性层
                nn.ReLU(), # 激活函数
                nn.Linear(256, hiddemDim),
                nn.ReLU()
            )

            # HLA 塔
            self.hlaTower = nn.Sequential(
                nn.Linear(hlaDim, 128), 
                nn.ReLU(), 
                nn.Linear(128, hiddemDim), 
                nn.ReLU()
            )

            # epitope 塔
            self.epitopeTower = nn.Sequential(
                nn.Linear(epitopeDim, 256), 
                nn.ReLU(),
                nn.Linear(256, hiddemDim),
                nn.ReLU()
            )

            # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(hiddemDim*3, 128), 
                nn.ReLU(),
                nn.Linear(128, 32), 
                nn.ReLU(), 
                nn.Linear(32, 1)
            )

        # 重写 forwoard 方法, 替换父类的
        def forward(self, tcrEmbed, hlaEmbed, epipoteEmbed):
            # 各塔的输入输出
            tcrFeat = self.tcrTower(tcrEmbed)
            hlaFeat = self.hlaTower(hlaEmbed)
            epipoteFeat = self.epitopeTower(epipoteEmbed)
            # 分类器的输入输出
            predScore = self.classifier(torch.cat([tcrFeat, hlaFeat, epipoteFeat], dim=1))
            return predScore.squeeze(-1)

    return (BaseThreeTowerV0_1,)


@app.cell
def v0_1_dataload():
    # ## v0.1 数据集
    # trainDataset = TrainDataLoad()
    return


@app.cell
def _():
    # ## v0.1 训练

    # # 日志
    # trainLogV0_1 = {foldId: {
    #     "testAntigenFrac": 0,
    #     "trainEpochLoss": [], # [1 * epoch_num]
    #     "trainBatchLoss": [], # [epoh_num * batch_num]
    #     "testEpochLoss": [], # [1 * epoch_num]
    #     "testBatchLoss": [], # [epoh_num * batch_num]
    #     "accuracy": [], # [1 * epoch_num]
    #     "recall": [], # [1 * epoch_num]
    #     "ROC-AUC": [], # [1 * epoch_num]
    #     "PR-AUC": [], # [1 * epoch_num]
    # } for foldId in range(1,4)}
    # # 分折遍历
    # foldSplit = StratifiedGroupKFold(n_splits=3)

    # for fold, (trainFoldIdx, testFoldIdx) in enumerate(foldSplit.split(
    #     X = trainDataset.data, 
    #     y = trainDataset.labels, 
    #     groups = trainDataset.data['epitope'].values
    # )):
    #     foldId = fold + 1
    #     foldLog = trainLogV0_1[foldId]
    #     # print(foldLog)

    #     # 加载模型
    #     model = BaseThreeTowerV0_1().to(device)
    #     # 损失函数
    #     loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([49]).to(device))
    #     # 优化器
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    #     # 拿拆分数据
    #     trainSub, testSub, trainLoader, testLoader = FoldSplit(
    #         trainDataset = trainDataset, 
    #         trainFoldIdx = trainFoldIdx, trainShuffle = True,
    #         testFoldIdx  = testFoldIdx,  testShuffle= False
    #     )
    #     ## [日志: 当前 fold 中测试集抗原数量比例]
    #     foldLog["testAntigenFrac"] = testSub.dataset.data.iloc[testFoldIdx]['epitope'].nunique() / 93
    #     # fold 训练
    #     for epoch in tqdm(range(1, 151)):
    #         model.train()
    #         ## [日志: 当前 epoch, batch loss]
    #         epochTrainLoss, epochTestLoss = 0, 0
    #         foldLog['trainBatchLoss'].append([])
    #         foldLog['testBatchLoss'].append([])
    #         # batchId = 0

    #         # 分 batch 训练
    #         for btcr, bhla, bepitope, blabel in trainLoader:
    #             # batchId += 1
    #             btcr, bhla = btcr.to(device), bhla.to(device)
    #             bepitope, blabel = bepitope.to(device), blabel.to(device)
    #             optimizer.zero_grad() ##[注意] 如果加动量是否还要删除梯度

    #             # 前向传播
    #             bpred = model(btcr,bhla,bepitope)
    #             # 计算损失
    #             bloss = loss(bpred, blabel)
    #             # 反向传播
    #             bloss.backward()
    #             # 更新梯度
    #             optimizer.step()

    #             ## [日志: batch 损失]
    #             foldLog['trainBatchLoss'][-1].append(bloss.item())
    #             ## [日志: epoch 损失]
    #             epochTrainLoss += bloss.item()
    #         ## [日志: epoch 损失]
    #         foldLog['trainEpochLoss'].append(epochTrainLoss / len(trainLoader))

    #         # epoch fold 验证
    #         model.eval()
    #         epochPred = []
    #         epochLabel = []
    #         with torch.no_grad():
    #             for btcr, bhla, bepitope, blabel in testLoader:
    #                 btcr, bhla = btcr.to(device), bhla.to(device)
    #                 bepitope, blabel = bepitope.to(device), blabel.to(device)
    #                 bpred = model(btcr, bhla, bepitope)
    #                 epochPred.extend(bpred.cpu().numpy())
    #                 epochLabel.extend(blabel.cpu().numpy())
    #                 bloss = loss(bpred, blabel)
    #                 ## [日志: batch 损失]
    #                 foldLog['testBatchLoss'][-1].append(bloss.item())
    #                 ## [日志: epoch 损失]
    #                 epochTestLoss += bloss.item()
    #         epochPred, epochLabel = np.array(epochPred), np.array(epochLabel)
    #         epochPredBin = (epochPred > 0).astype(int)
    #         sigmoid = nn.Sigmoid()
    #         epochPredSocre = sigmoid(torch.tensor(epochPred)).numpy()
    #         ## [日志: epoch 损失]
    #         foldLog['testEpochLoss'].append(epochTestLoss / len(testLoader))
    #         ## [日志: 性能]
    #         foldLog['accuracy'].append(accuracy_score(epochLabel, epochPredBin))
    #         foldLog['recall'].append(recall_score(epochLabel, epochPredBin, pos_label=1))
    #         foldLog['ROC-AUC'].append(roc_auc_score(epochLabel, epochPredSocre))
    #         _precision, _recallPoint, _thresholds = precision_recall_curve(epochLabel, epochPredSocre)
    #         foldLog['PR-AUC'].append(auc(_recallPoint, _precision))

    # torch.save(trainLogV0_1, fileDir / f"v0_1.withoutWeight.Log.epoch100.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## v0.2 在0.1基础上引入动量
    - momentum=0.9
    - nesterov=True
    - batchsize=1600
    - 架构不变
    - 先练 60 个 epoch
    """)
    return


@app.cell
def _(TrainDataLoad):
    ## v0.2 数据
    trainDataset = TrainDataLoad()
    return (trainDataset,)


@app.cell
def _(
    BaseThreeTowerV0_1,
    FoldSplit,
    StratifiedGroupKFold,
    accuracy_score,
    auc,
    device,
    fileDir,
    nn,
    np,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    torch,
    tqdm,
    trainDataset,
):
    ## v0.2 训练

    # 日志
    trainLogV0_2 = {foldId: {
        "testAntigenFrac": 0,
        "trainEpochLoss": [], # [1 * epoch_num]
        "trainBatchLoss": [], # [epoh_num * batch_num]
        "testEpochLoss": [], # [1 * epoch_num]
        "testBatchLoss": [], # [epoh_num * batch_num]
        "accuracy": [], # [1 * epoch_num]
        "recall": [], # [1 * epoch_num]
        "ROC-AUC-train": [], # [1 * epoch_num]
        "PR-AUC-train": [], # [1 * epoch_num]
        "ROC-AUC-test": [], # [1 * epoch_num]
        "PR-AUC-test": [], # [1 * epoch_num]
    } for foldId in range(1,4)}
    # 分折遍历
    foldSplit = StratifiedGroupKFold(n_splits=3)

    for fold, (trainFoldIdx, testFoldIdx) in enumerate(foldSplit.split(
        X = trainDataset.data, 
        y = trainDataset.labels, 
        groups = trainDataset.data['epitope'].values
    )):
        foldId = fold + 1
        foldLog = trainLogV0_2[foldId]
        # print(foldLog)

        # 加载模型
        model = BaseThreeTowerV0_1().to(device)
        # 损失函数
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([49]).to(device))
        # 优化器
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
        # 拿拆分数据
        trainSub, testSub, trainLoader, testLoader = FoldSplit(
            trainDataset = trainDataset, 
            trainFoldIdx = trainFoldIdx, trainShuffle = True,
            testFoldIdx  = testFoldIdx,  testShuffle= False
        )
        ## [日志: 当前 fold 中测试集抗原数量比例]
        foldLog["testAntigenFrac"] = testSub.dataset.data.iloc[testFoldIdx]['epitope'].nunique() / 93
        # fold 训练
        for epoch in tqdm(range(1, 61)):
            model.train()
            ## [日志: 当前 epoch, batch loss, auc]
            epochTrainLoss, epochTestLoss = 0, 0
            epochPred = []
            epochLabel = []
            foldLog['trainBatchLoss'].append([])
            foldLog['testBatchLoss'].append([])
            # batchId = 0

            # 分 batch 训练
            for btcr, bhla, bepitope, blabel in trainLoader:
                # batchId += 1
                btcr, bhla = btcr.to(device), bhla.to(device)
                bepitope, blabel = bepitope.to(device), blabel.to(device)
                optimizer.zero_grad() ##[注意] 如果加动量是否还要删除梯度

                # 前向传播
                bpred = model(btcr,bhla,bepitope)
                # 计算损失
                bloss = loss(bpred, blabel)
                # 反向传播
                bloss.backward()
                # 更新梯度
                optimizer.step()

                ## [日志: batch 损失]
                foldLog['trainBatchLoss'][-1].append(bloss.item())
                ## [日志: epoch 损失]
                epochTrainLoss += bloss.item()
                ## [日志: epoch auc train]
                epochPred.extend(bpred.detach().cpu().numpy().flatten())
                epochLabel.extend(blabel.detach().cpu().numpy().flatten())
            ## [日志: 训练auc]
            epochPred, epochLabel = np.array(epochPred), np.array(epochLabel)
            epochPredBin = (epochPred > 0).astype(int)
            sigmoid = nn.Sigmoid()
            epochPredSocre = sigmoid(torch.tensor(epochPred)).numpy()
            foldLog['ROC-AUC-train'].append(roc_auc_score(epochLabel, epochPredSocre))
            _precision, _recallPoint, _thresholds = precision_recall_curve(epochLabel, epochPredSocre)
            foldLog['PR-AUC-train'].append(auc(_recallPoint, _precision))
            ## [日志: epoch 损失]
            foldLog['trainEpochLoss'].append(epochTrainLoss / len(trainLoader))

            # epoch fold 验证
            model.eval()
            epochPred = []
            epochLabel = []
            with torch.no_grad():
                for btcr, bhla, bepitope, blabel in testLoader:
                    btcr, bhla = btcr.to(device), bhla.to(device)
                    bepitope, blabel = bepitope.to(device), blabel.to(device)
                    bpred = model(btcr, bhla, bepitope)
                    epochPred.extend(bpred.cpu().numpy().flatten())
                    epochLabel.extend(blabel.cpu().numpy().flatten())
                    bloss = loss(bpred, blabel)
                    ## [日志: batch 损失]
                    foldLog['testBatchLoss'][-1].append(bloss.item())
                    ## [日志: epoch 损失]
                    epochTestLoss += bloss.item()
            epochPred, epochLabel = np.array(epochPred), np.array(epochLabel)
            epochPredBin = (epochPred > 0).astype(int)
            sigmoid = nn.Sigmoid()
            epochPredSocre = sigmoid(torch.tensor(epochPred)).numpy()
            ## [日志: epoch 损失]
            foldLog['testEpochLoss'].append(epochTestLoss / len(testLoader))
            ## [日志: 性能]
            foldLog['accuracy'].append(accuracy_score(epochLabel, epochPredBin))
            foldLog['recall'].append(recall_score(epochLabel, epochPredBin, pos_label=1))
            foldLog['ROC-AUC-test'].append(roc_auc_score(epochLabel, epochPredSocre))
            _precision, _recallPoint, _thresholds = precision_recall_curve(epochLabel, epochPredSocre)
            foldLog['PR-AUC-test'].append(auc(_recallPoint, _precision))

    torch.save(trainLogV0_2, fileDir / f"v0_2.withoutWeight.Log.epoch60.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## v0.3 当前网络下直接最优化
    - 每个线性层都归一化
    - silu 激活函数
    - adamw 优化
    """)
    return


@app.cell
def v0_3_dataload(TrainDataLoad):
    ## v0.3 数据
    trainDataset = TrainDataLoad()
    return (trainDataset,)


@app.cell(hide_code=True)
def v0_3_arch(nn, torch):
    ## v0.3 架构
    ## MLP 先进架构
    class BaseThreeTowerV0_3(nn.Module):
        def __init__(self, tcrDim=1280, hlaDim=1280, epitopeDim=1280, hiddemDim=128, outDim=1):
            super(BaseThreeTowerV0_3, self).__init__()
            # 这里 tcrDim 等是自定义的, 实际上也可以不用这些参数, 直接在后面构造线形层的时候写死
            # super 初始化是为了初始化父类构造, 否则, 即使继承了父类的方法, 缺失 Module 类初始化时的实例属性, 也是没法跑的

            # TCR 塔
            self.tcrTower = nn.Sequential(
                nn.Linear(tcrDim, 256, dtype=torch.bfloat16), # 256 个节点的线性层
                nn.LayerNorm(256, dtype=torch.bfloat16), 
                nn.SiLU(), # 激活函数
                nn.Linear(256, hiddemDim, dtype=torch.bfloat16),
                nn.LayerNorm(128, dtype=torch.bfloat16),
                nn.SiLU()
            )

            # HLA 塔
            self.hlaTower = nn.Sequential(
                nn.Linear(hlaDim, 128, dtype=torch.bfloat16), 
                nn.LayerNorm(128, dtype=torch.bfloat16), 
                nn.SiLU(), 
                nn.Linear(128, hiddemDim, dtype=torch.bfloat16), 
                nn.LayerNorm(128, dtype=torch.bfloat16),
                nn.SiLU()
            )

            # epitope 塔
            self.epitopeTower = nn.Sequential(
                nn.Linear(epitopeDim, 256, dtype=torch.bfloat16), 
                nn.LayerNorm(256, dtype=torch.bfloat16), 
                nn.SiLU(),
                nn.Linear(256, hiddemDim, dtype=torch.bfloat16),
                nn.LayerNorm(128, dtype=torch.bfloat16), 
                nn.SiLU()
            )

            # 分类器
            self.classifier = nn.Sequential(
                nn.Linear(hiddemDim*3, 128, dtype=torch.bfloat16), 
                nn.LayerNorm(128, dtype=torch.bfloat16), 
                nn.SiLU(),
                nn.Linear(128, 32, dtype=torch.bfloat16), 
                nn.LayerNorm(32, dtype=torch.bfloat16), 
                nn.SiLU(), 
                nn.Linear(32, 1, dtype=torch.bfloat16)
            )

        # 重写 forwoard 方法, 替换父类的
        def forward(self, tcrEmbed, hlaEmbed, epipoteEmbed):
            # 各塔的输入输出
            tcrFeat = self.tcrTower(tcrEmbed)
            hlaFeat = self.hlaTower(hlaEmbed)
            epipoteFeat = self.epitopeTower(epipoteEmbed)
            # 分类器的输入输出
            predScore = self.classifier(torch.cat([tcrFeat, hlaFeat, epipoteFeat], dim=1))
            return predScore.squeeze(-1)

    return (BaseThreeTowerV0_3,)


@app.cell
def _(
    BaseThreeTowerV0_3,
    FoldSplit,
    StratifiedGroupKFold,
    accuracy_score,
    auc,
    device,
    fileDir,
    nn,
    np,
    precision_recall_curve,
    recall_score,
    roc_auc_score,
    torch,
    tqdm,
    trainDataset,
):
    # v0.3 训练

    # 日志
    trainLogV0_3 = {foldId: {
        "testAntigenFrac": 0,
        "trainEpochLoss": [], # [1 * epoch_num]
        "trainBatchLoss": [], # [epoh_num * batch_num]
        "testEpochLoss": [], # [1 * epoch_num]
        "testBatchLoss": [], # [epoh_num * batch_num]
        "accuracy": [], # [1 * epoch_num]
        "recall": [], # [1 * epoch_num]
        "ROC-AUC-train": [], # [1 * epoch_num]
        "PR-AUC-train": [], # [1 * epoch_num]
        "ROC-AUC-test": [], # [1 * epoch_num]
        "PR-AUC-test": [], # [1 * epoch_num]
    } for foldId in range(1,4)}
    # 分折遍历
    foldSplit = StratifiedGroupKFold(n_splits=3)

    for fold, (trainFoldIdx, testFoldIdx) in enumerate(foldSplit.split(
        X = trainDataset.data, 
        y = trainDataset.labels, 
        groups = trainDataset.data['epitope'].values
    )):
        foldId = fold + 1
        foldLog = trainLogV0_3[foldId]
        # print(foldLog)

        # 加载模型
        model = BaseThreeTowerV0_3().to(device)
        # 损失函数
        loss = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([49], dtype=torch.bfloat16).to(device))
        # 优化器
        optimizer = torch.optim.AdamW(model.parameters(), lr=0.01)
        # 拿拆分数据
        trainSub, testSub, trainLoader, testLoader = FoldSplit(
            trainDataset = trainDataset, 
            trainFoldIdx = trainFoldIdx, trainShuffle = True,
            testFoldIdx  = testFoldIdx,  testShuffle= False
        )
        ## [日志: 当前 fold 中测试集抗原数量比例]
        foldLog["testAntigenFrac"] = testSub.dataset.data.iloc[testFoldIdx]['epitope'].nunique() / 93
        # fold 训练
        for epoch in tqdm(range(1, 61)):
            model.train()
            ## [日志: 当前 epoch, batch loss, auc]
            epochTrainLoss, epochTestLoss = 0, 0
            epochPred = []
            epochLabel = []
            foldLog['trainBatchLoss'].append([])
            foldLog['testBatchLoss'].append([])
            # batchId = 0

            # 分 batch 训练
            for btcr, bhla, bepitope, blabel in trainLoader:
                # batchId += 1
                btcr, bhla = btcr.to(device), bhla.to(device)
                bepitope, blabel = bepitope.to(device), blabel.to(device)
                model.zero_grad() ##[注意] 如果加动量是否还要删除梯度

                # 前向传播
                bpred = model(btcr,bhla,bepitope)
                # 计算损失
                bloss = loss(bpred, blabel)
                # 反向传播
                bloss.backward()
                # 更新梯度
                optimizer.step()

                ## [日志: batch 损失]
                foldLog['trainBatchLoss'][-1].append(bloss.item())
                ## [日志: epoch 损失]
                epochTrainLoss += bloss.item()
                ## [日志: epoch auc train]
                epochPred.extend(bpred.detach().cpu().float().numpy().flatten())
                epochLabel.extend(blabel.detach().cpu().float().numpy().flatten())
            ## [日志: 训练auc]
            epochPred, epochLabel = np.array(epochPred), np.array(epochLabel)
            epochPredBin = (epochPred > 0).astype(int)
            sigmoid = nn.Sigmoid()
            epochPredSocre = sigmoid(torch.tensor(epochPred)).float().numpy()
            foldLog['ROC-AUC-train'].append(roc_auc_score(epochLabel, epochPredSocre))
            _precision, _recallPoint, _thresholds = precision_recall_curve(epochLabel, epochPredSocre)
            foldLog['PR-AUC-train'].append(auc(_recallPoint, _precision))
            ## [日志: epoch 损失]
            foldLog['trainEpochLoss'].append(epochTrainLoss / len(trainLoader))

            # epoch fold 验证
            model.eval()
            epochPred = []
            epochLabel = []
            with torch.no_grad():
                for btcr, bhla, bepitope, blabel in testLoader:
                    btcr, bhla = btcr.to(device), bhla.to(device)
                    bepitope, blabel = bepitope.to(device), blabel.to(device)
                    bpred = model(btcr, bhla, bepitope)
                    epochPred.extend(bpred.cpu().float().numpy().flatten())
                    epochLabel.extend(blabel.cpu().float().numpy().flatten())
                    bloss = loss(bpred, blabel)
                    ## [日志: batch 损失]
                    foldLog['testBatchLoss'][-1].append(bloss.item())
                    ## [日志: epoch 损失]
                    epochTestLoss += bloss.item()
            epochPred, epochLabel = np.array(epochPred), np.array(epochLabel)
            epochPredBin = (epochPred > 0).astype(int)
            sigmoid = nn.Sigmoid()
            epochPredSocre = sigmoid(torch.tensor(epochPred)).float().numpy()
            ## [日志: epoch 损失]
            foldLog['testEpochLoss'].append(epochTestLoss / len(testLoader))
            ## [日志: 性能]
            foldLog['accuracy'].append(accuracy_score(epochLabel, epochPredBin))
            foldLog['recall'].append(recall_score(epochLabel, epochPredBin, pos_label=1))
            foldLog['ROC-AUC-test'].append(roc_auc_score(epochLabel, epochPredSocre))
            _precision, _recallPoint, _thresholds = precision_recall_curve(epochLabel, epochPredSocre)
            foldLog['PR-AUC-test'].append(auc(_recallPoint, _precision))

    torch.save(trainLogV0_3, fileDir / f"v0_3.withoutWeight.Log.epoch60.pt")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## v1系列
    通过前面的实验观察到, tcr-pMHC 序列的横向拼接, 模型并不会关注真正的空间关联, 转而很容易记住对一某一种 pMHC, 什么样的 tcr 序列是阳性, 什么样的是阴性. 而面对测试集完全没见过的 pMHC, 等同于瞎猜. 这种情况下优化器已经难以发挥作用了. 必须修改 embedding 方式和网络架构了. 此时有两个分支:

    1. transformer
    2. “层次化交叉注意力”
    """)
    return


if __name__ == "__main__":
    app.run()
