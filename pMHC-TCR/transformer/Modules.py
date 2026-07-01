import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 构建可直接用于 TCR-pMHC 的 Transformer 各模块
    1. 输入与padding 模块
    2. 降维模块:
       - ESM2 1280维特征包含进化相似度等等对该任务不是很重要的特征
       - 降维就是让模型能自主学习如何组合原始语义特征, 提取中真正关注的结构特征
    4. 位置编码模块
       - 输入的 embedding 已经包含了序列自身的相对位置, 此时还需要位置编码吗? 或者说, 还需要序列内位置编码吗?
       - TCR-pMHC 实现
       - 预检验: 训练前就看一下编码方法可能好不好, 热图法、表征相似度分析
    5. 多头自注意力模块
    6. 残差连接
    7. 层归一化模块
    8. 前馈网络
    """)
    return


@app.cell
def need_module_import():
    import marimo as mo

    import os
    import random
    import torch
    import math
    import torch.nn as nn
    import numpy as np
    import pandas as pd

    from tqdm import tqdm
    from pathlib import Path
    from typing import List, Literal, Tuple, Dict



    return Path, math, mo, nn, np, os, random, torch


@app.cell
def need_global_variable(Path, np, os, random, torch):
    # project dir
    projDir = Path(os.getcwd()) / "pMHC-TCR/"
    # GPU
    device = torch.accelerator.current_accelerator()

    # set seed
    def FixRandomSeed():
        random.seed(42)
        np.random.seed(42)
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)
        torch.cuda.manual_seed_all(42)
        torch.mps.manual_seed(42)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
    FixRandomSeed()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## padding module

    1. 初始化: tcr、hla、antigen 各自的最大 padding 长度, 以及特征维度
    2. 输入 List of tensor, 每个 tensor 是每条序列的 embedding matrix, tcr、hla、antigen 各自输入一个 list
    3. 输出 tensor [B * L * D], 其中
       - B: batch size, 即 pMHC-TCR 的序列对数, 等于输入中的 list 长度
       - L: length of concat sequence, 拼接矩阵的长度, 即 三种序列各自最大 padding 长度的和
       - D: embedding 特征维度, 与输入列表中维度一致, 用 ESM-2 650M 模型 embedding 即 1280 维
    """)
    return


@app.cell
def padding_module(nn, torch):
    class ThreeSeqPadding(nn.Module):
        # read embedding and padding module
        # input: tcr tensor batch, for each element, it's the embeding matrix of one sequence, the matrix = [SeqLength * EmbedFeatures]
        # output: junc matrix[Batch * Length * 1280], where Length = tcrMaxLen + hlaMaxLen + atgMaxLen, Batch is length of input list, equals to the number of TCR-pMHC pairs
        def __init__(self, tcrMaxLen=25, hlaMaxLen=34, atgMaxLen=13, embedDim = 1280):
            super(ThreeSeqPadding, self).__init__()
            self.tcrMaxLen = tcrMaxLen
            self.hlaMaxLen = hlaMaxLen
            self.atgMaxLen = atgMaxLen
            self.embedDim = embedDim

        def __padding(self, inMat, inMaxLen):
            # assert
            assert inMat.dim() == 2
            assert inMat.shape[0] <= inMaxLen
            # padding
            padLen = inMaxLen-inMat.shape[0]
            outMat = nn.functional.pad(inMat, pad=(0,0,0,padLen), mode="constant", value=0.0)
            return outMat

        def __padList(self, inMatList, inMaxLen):
            device = inMatList[0].device
            batchSize = inMatList.shape[0]
            padTensor = torch.zeros(batchSize, inMaxLen, self.embedDim, device=device, dtype=torch.bfloat16)
            maskTensor = torch.zeros(batchSize, inMaxLen, device=device, dtype=torch.bool)
            for inIdx, inMat in enumerate(inMatList):
                padTensor[inIdx] = self.__padding(inMat=inMat, inMaxLen=inMaxLen)
                maskTensor[inIdx, :inMat.shape[0]] = True # mask 的 idx 个序列, 前 inMat.shape[0] 行改为 True
            return padTensor, maskTensor


        def __concatList(self, tcrBatchTensor, hlaBatchTensor, atgBatchTensor):
            # assert
            assert tcrBatchTensor.dim() == hlaBatchTensor.dim() == atgBatchTensor.dim()
            assert tcrBatchTensor.shape[0] == hlaBatchTensor.shape[0] == atgBatchTensor.shape[0]
            # concat
            concatTensor = torch.cat([tcrBatchTensor, hlaBatchTensor, atgBatchTensor], dim=1)
            return concatTensor

        def forward(self, tcrMatList, hlaMatList, atgMatList):
            tcrBatch, tcrMask = self.__padList(inMatList=tcrMatList, inMaxLen=self.tcrMaxLen)
            hlaBatch, hlaMask = self.__padList(inMatList=hlaMatList, inMaxLen=self.hlaMaxLen)
            atgBatch, atgMask= self.__padList(inMatList=atgMatList, inMaxLen=self.atgMaxLen)
            conBatch = self.__concatList(tcrBatchTensor=tcrBatch, hlaBatchTensor=hlaBatch, atgBatchTensor=atgBatch)
            conMask = self.__concatList(tcrBatchTensor=tcrMask, hlaBatchTensor=hlaMask, atgBatchTensor=atgMask)
            return conBatch, conMask

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## position encoding module, 位置编码模块
    """)
    return


@app.cell
def _(nn, torch):
    class PositionBiasMat(nn.Module):
        # create position bias matrix which will add to the attention matrix (Q@K.T + bias)
        # input: tensor [batch, matrix], martix = Q@K.T
        # output: position bias matrix [batch, matrix], where length of matrix = padding tcr length + padding hla length + padding antigen length
        def __init__(self, tcrMaxLen=25, hlaMaxLen=34, atgMaxLen=13, maxRelDist=16):
            super(PositionBiasMat, self).__init__()
            self.tcrMaxLen = tcrMaxLen
            self.hlaMaxLen = hlaMaxLen
            self.atgMaxLen = atgMaxLen
            self.concatLen = tcrMaxLen + hlaMaxLen + atgMaxLen
            self.maxRelDist = maxRelDist
            # 链内关系
            self.inSeqPosi = 2 * maxRelDist + 1 # (-16,-15,-14,...0,...14,15,16)
            # 链间关系
            self.interSeqPosi = [2*maxRelDist+i for i in range(1,7)] # tcr-hla, tcr-atg, hla-tcr, hla-atg, atg-tcr, atg-hla
            # 总关系
            self.allPosi = self.inSeqPosi + len(self.interSeqPosi)
            self.posTable = nn.Embedding(self.allPosi, 1)
            # 固定一次位置索引矩阵
            idxMat = self.__biasPositionIdx()
            self.register_buffer("Fixed_Related_Position_Idx_Matrix", idxMat)

        def __biasPositionIdx(self):
            maxD = self.maxRelDist
            # tcr 部分矩阵
            tcrBias = torch.tensor([range(i-1, i-(self.tcrMaxLen+1), -1) for i in range(1,1+self.tcrMaxLen)], dtype=torch.long)
            tcrBias.clamp_(max=maxD, min=-maxD).add_(maxD)
            tcrBias = nn.functional.pad(tcrBias, pad=[0,34,0,0], mode="constant", value=self.interSeqPosi[0]) # pad hla 
            tcrBias = nn.functional.pad(tcrBias, pad=[0,13,0,0], mode="constant", value=self.interSeqPosi[1]) # pad atg
            # hla 部分矩阵
            hlaBias = torch.tensor([range(i-1, i-(self.hlaMaxLen+1), -1) for i in range(1,1+self.hlaMaxLen)], dtype=torch.long)
            hlaBias.clamp_(max=maxD, min=-maxD).add_(maxD)
            hlaBias = nn.functional.pad(hlaBias, pad=[25,0,0,0], mode="constant", value=self.interSeqPosi[2])
            hlaBias = nn.functional.pad(hlaBias, pad=[0,13,0,0], mode="constant", value=self.interSeqPosi[3])
            # atg 部分矩阵
            atgBias = torch.tensor([range(i-1, i-(self.atgMaxLen+1), -1) for i in range(1, 1+self.atgMaxLen)], dtype=torch.long)
            atgBias.clamp_(max=maxD, min=-maxD).add_(maxD)
            atgBias = nn.functional.pad(atgBias, pad=[34,0,0,0], mode="constant", value=self.interSeqPosi[5]) # hla pad, 从右往左 pad, 所以先 hla
            atgBias = nn.functional.pad(atgBias, pad=[25,0,0,0], mode="constant", value=self.interSeqPosi[4]) # tcr pad
            # 合并
            posBias = torch.cat([tcrBias, hlaBias, atgBias], dim=0) # [72*72]
            # 映射, 这里只索引, 不映射, 因为 embedding 是可学习的, 每次映射都会变
            # posBias = self.posTable(posBias) # [72*72*1]
            return posBias

        def __maskBias(self, batchPosBias, batchMask):
            # input: 
                # batchPosBias: [1*72*72]
                # batchMask: [B*72]
            # output:
                # batchPosBias: [B*72*72]
            maskRaw = batchMask.unsqueeze(1)
            maskCol = batchMask.unsqueeze(2)
            maskMat = ~(maskRaw & maskCol) # 取非, 即没有氨基酸的位置要修改, 此时 [B*72*72]
            batchMaskBias = batchPosBias.masked_fill(maskMat, -1e9)
            return batchMaskBias

        def forward(self, batchRawAttention, batchMask):
            # input:
                # batchRawAttention: [B*72*72]
                # batchMask: [B*72]
            # output:
                # batchRawAttention: [B*72*72]
            device = batchMask.device
            posBias = self.posTable(self.Fixed_Related_Position_Idx_Matrix).squeeze(-1).unsqueeze(0) # 72*72*1 -> 72*72 -> 1*72*72, 1为了广播Batch
            batchMaskPosBias = self.__maskBias(batchPosBias=posBias, batchMask=batchMask)
            batchRawAttention.add_(batchMaskPosBias)
            return batchRawAttention
        

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 多头注意力模块
    """)
    return


@app.cell
def _(math, nn, torch):
    class MultiHeadAttention(nn.Module):
        # 多头注意力机制
        # input: [B, length(72),hiddenDim]
        # output: [B, length, hiddenDim]
        def __init__(self, hiddenDim = 256, numberHead = 8):
            super(MultiHeadAttention, self).__init__()
            self.hiddenDim = hiddenDim
            self.numberHead = numberHead

            # 多头权重矩阵
            self.Wq = nn.Linear(hiddenDim, hiddenDim)
            self.Wk = nn.Linear(hiddenDim, hiddenDim)
            self.Wv = nn.Linear(hiddenDim, hiddenDim)
            self.Dk = hiddenDim // numberHead
            self.scale = math.sqrt(self.Dk)
            # 上下文线性融合
            self.ConLinear = nn.Linear(hiddenDim, hiddenDim)
        
        def forward(self, 
            X, # [batch * length(72) * hiddenDim]
            posiBias, # [batch * length * length]
        ):
            device = X.device
            # 身份区分
            multiQ = self.Wq(X) # [batch*72*hiddenDim]
            multiK = self.Wk(X)
            multiV = self.Wv(X)
            # 多头拆分
            batchSize, seqLeng = X.shape[0], X.shape[1]
            multiQ = multiQ.view(batchSize, seqLeng, self.numberHead, self.Dk).permute(0,2,1,3)
                # view 方法用新的形状[batch * length * numberHead * dk] 来解读原来的 [batch * length * hiddenDim], 返回各维度索引, 
                # permute方法重排维度, 即[batch * numberHead * length * dk]
            multiK = multiK.view(batchSize, seqLeng, self.numberHead, self.Dk).permute(0,2,1,3)
            multiV = multiV.view(batchSize, seqLeng, self.numberHead, self.Dk).permute(0,2,1,3)
            # QK 原始注意力
            QK = multiQ @ multiK.transpose(-2, -1) / self.scale # [batch * head * length * length]
                # transpose 方法, 交换 -2 和 -1 两个维度, 变成[batch * numberHead * dk * length]
            # 加上位置偏置
            QK = QK + posiBias.unsqueeze(1) # 先把位置偏置变成 [batch * 1 * length * length]
            # softmax
            A = torch.softmax(QK, dim=-1) # [batch * head * length * length]
            # QKV
            AV = torch.matmul(A, multiV) # [batch * head * length * dk]
            # 合并多头
            context = AV.permute(0, 2, 1, 3).contiguous().view(batchSize, seqLeng, self.hiddenDim)
            context = self.ConLinear(context)
            return context # [batch * length(72) * hiddenDim]
        
        

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 前馈神经网络模块
    """)
    return


@app.cell
def _(nn):
    class FNN(nn.Module):
        # input: [batch, length, hiddenDim], 多头注意力模块输出的维度
        # output: [batch, length, hiddenDim], 与输入完全相同的形状
        def __init__(self, hiddenDim=256, hiddenFold=4, dropoutP=0.1):
            super(FNN, self).__init__()
            # 参数属性
            self.hiddenDim = hiddenDim
            self.hiddenFold = hiddenFold
            self.dropoutP = dropoutP
            self.upDim = self.hiddenDim * self.hiddenFold
            # 两层网络
            self.up = nn.Linear(self.hiddenDim, self.upDim)
            self.down = nn.Linear(self.upDim, self.hiddenDim)
            # 激活函数
            self.activation = nn.GELU()
            # dropout
            self.drop = nn.Dropout(self.dropoutP)

        def forward(self, linearContext, ):
            out = self.up(linearContext) # [batch, length, upDim]
            out = self.activation(out)
            out = self.drop(out)
            out = self.down(out) # [batch, length, hiddenDim]
            return out
        

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## encoder 框架
    """)
    return


app._unparsable_cell(
    r"""
    class pTcrEncoder(nn.Module):
        # in
        # out
        def __init__(self, rowDim=1280, featureDim=256, attentionDim=256, upDimFold=4, dropoutP=0.1, numberHead=8):
            # 属性
            self.rowDim = rowDim
            self.featureDim = featureDim # 对 ESM2 提取的 1280 维特征降维提取
            self.attentionDim = attentionDim # 注意力机制中的隐藏层维度
            self.numberHead = numberHead # 注意力头数
            self.upDimFold = upDimFold # 前馈神经网络中对 注意力context 升维的倍数
            self.dropoutP = dropoutP # dropout 概率
            # 架构
            self.featExtract = nn.Linear(self.rowDim, self.featureDim) # [batch, length, rowDim] -> [batch, length, featureDim]
            self.layerNorm = nn.LayerNorm(self.featureDim) # 用于多头注意力机制的输入
            self.multiAttention = MultiHeadAttention(hiddenDim=self.attentionDim, numberHead=self.numberHead)
            self.dropAttention = nn.Dropout(self.dropoutP) # 多头注意力输出的dropout
            self.
    """,
    name="_"
)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 测试
    """)
    return


@app.cell
def _(torch):
    Q = torch.rand(2,2,3,4)
    K = torch.rand(2,2,3,4)

    return K, Q


@app.cell
def _(Q):
    Q
    return


@app.cell
def _(K):
    K
    return


@app.cell
def _(K, Q):
    Q@K.transpose(-2,-1)
    return


@app.cell
def _(tmask):
    tmask.unsqueeze(2).unsqueeze(3).shape
    return


@app.cell
def _(tmask):
    tmask.unsqueeze(1).unsqueeze(3).shape
    return


@app.cell
def _(nn):
    tsoft = nn.Softmax()
    return (tsoft,)


@app.cell
def _(torch, tsoft):
    tsoft(torch.tensor([1e-4, 0.1, 0.2, 1.1, 0.8]))
    return


@app.cell
def _(torch):
    tmask = torch.tensor([[True, False, True, True], [False, False, True, True]])
    return (tmask,)


@app.cell
def _(tmask):
    tmaskRaw = tmask.unsqueeze(1)
    tmaskCOl = tmask.unsqueeze(2)
    return tmaskCOl, tmaskRaw


@app.cell
def _(tmaskCOl, tmaskRaw):
    tmaskCOl & tmaskRaw
    return


@app.cell
def _(tmaskRaw):
    tmaskRaw
    return


@app.cell
def _(tmaskCOl):
    tmaskCOl.device
    return


@app.cell
def _(torch):
    torch.rand(2,3,4).unsqueeze(1).shape
    return


@app.cell
def _(nn, torch):
    testEmbd = nn.Embedding(100, 1)
    testEmbd(torch.arange(48).reshape(6,8)).squeeze().unsqueeze(0)
    return


@app.cell
def _(torch):
    atgCore = torch.rand(3,3)
    atgCore
    return (atgCore,)


@app.cell
def _(atgCore, nn):
    atgC = nn.functional.pad(atgCore, pad=[2,0,0,0],mode="constant", value=1)
    atgC
    return (atgC,)


@app.cell
def _(atgC, nn):
    nn.functional.pad(atgC, pad=[2,0,0,0],mode="constant", value=1.2)
    return


if __name__ == "__main__":
    app.run()
