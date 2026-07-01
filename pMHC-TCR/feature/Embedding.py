import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    from pathlib import Path
    from transformers import AutoTokenizer, EsmModel
    import numpy as np
    from tqdm import tqdm
    import pandas as pd 
    import torch

    fileDir = Path(__file__).resolve().parent
    device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else 'cpu'
    batchSize = 64
    return (
        AutoTokenizer,
        EsmModel,
        batchSize,
        device,
        fileDir,
        mo,
        np,
        pd,
        torch,
        tqdm,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 使用 ESM-2 对序列 embedding
    """)
    return


@app.cell
def _(AutoTokenizer):
    # 加载分词器, 分词器是将氨基酸转换成 token, 序列则成为编码器能理解的 token 向量
    modelName = "facebook/esm2_t33_650M_UR50D"
    modelVersion = '08e4846e537177426273712802403f7ba8261b6c'
    tokenizer = AutoTokenizer.from_pretrained(modelName, revision=modelVersion)
        # 从 hugging face 的该模型中下载分词器, 如果模型中有完整分词器相关文件, 如 tokenizer_config.json 等, 就可以直接使用
        # 参数: 
            # cache_dir 可以指定读取和下载位置, 默认在 ~/.cache/huggingface
            # force_download 缓存损坏时强制重新下载
            # resume_download 开启断点续传
            # local_files_only 不下载, 只找本地缓存
            # revision 指定模型版本, 一个模型就是一个git仓库, 实际上就是知道分支或者commit hash
            # proxies 设置代理

    return modelName, modelVersion, tokenizer


@app.cell
def _(EsmModel, device, modelName, modelVersion):
    ## 加载模型
    model = EsmModel.from_pretrained(modelName, revision=modelVersion).to(device)
        # 模型类中 from_pretrained 方法的特殊参数:
            # torch_dtype 指定权重加载精度，常用 torch.float16 或 torch.bfloat16(更快占用更小损失精度)
            # device_map 自动分配模型到计算设备, 开启后无需再写 .to(DEVICE)
    model.eval
    return (model,)


@app.cell
def _(batchSize, device, fileDir, model, np, pd, tokenizer, torch, tqdm):
    ## embedding 并 平均池化
    def BatchEmbedding(
        allSeqList :list, 
        tokenizer, # 分词器
        embedModel, # embed 模型
        device,
        batchSize=64
    ):
        # 结果缓存
        embeddingRes = []

        # 批遍历 embedding
        for batchIdx in tqdm(range(0, len(allSeqList), batchSize)):
            batchSeqs = allSeqList[batchIdx: batchIdx + 64]

            # tokenize
            batchToken = tokenizer(batchSeqs, padding=True, return_tensors="pt").to(device)

            # embed, 推理模式
            with torch.no_grad():
                batchOutput = model(**batchToken)

            # 去除feature
            batchFeature  = batchOutput.last_hidden_state # 此时是 batch(序列数)*row(最大氨基酸长度)*col(feature数,该模型是1280)
            # 注意力掩码和长度
            batchMask = batchToken.attention_mask.unsqueeze(-1) # batch*row*1
            batchLength = batchMask.sum(dim=1) # batch*col(每个batch的氨基酸长度)
            # 平均池化
            batchFeatureAve = (batchFeature * batchMask).sum(dim=1) / batchLength # batch * col(特征数1280)
            # 合并
            embeddingRes.append(batchFeatureAve.cpu().numpy())

        # 展开外层嵌套
        return np.vstack(embeddingRes)

    ## 对 hla, tcr, antigen 序列, 各自取 unique 序列, 然后 embedding, 存储为 pth
    trainData = pd.read_csv(fileDir / "../data/Train.DownSampling.csv", sep="\t", header=0)
    testData = pd.read_csv(fileDir / "../data/Test.csv", sep="\t", header=0)
    globalData = pd.concat([trainData, testData])
    for col in ['hla_pseudo','epitope','tcr']:
        # 取序列
        uniqueSeqs = globalData[col].unique().tolist()
        # embedding
        uniqueEmbedding = BatchEmbedding(
            allSeqList = uniqueSeqs,
            tokenizer = tokenizer, 
            embedModel = model, 
            device = device, 
            batchSize = batchSize
        )
        # 字典
        uniqueDict = {seq: vec for seq,vec in zip(uniqueSeqs,uniqueEmbedding)}
        # 储存
        print(f"嵌入向量长度: {len(uniqueDict)}; 已知序列长度: {globalData[col].nunique()}")
        saveFile = fileDir / f"{col}.GlobalEmbedding.pt"
        torch.save(uniqueDict, saveFile)
    
    
    return


if __name__ == "__main__":
    app.run()
