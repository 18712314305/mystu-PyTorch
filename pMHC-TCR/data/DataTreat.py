import marimo

__generated_with = "0.23.8"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd 
    from pathlib import Path
    import numpy as np
    import random
    from tqdm import tqdm

    return Path, mo, np, pd, random, tqdm


@app.cell
def _(Path, pd):
    def ReadPseudoHLA():
        # 同本脚本目录下读取文件, 去重后转换成字典
        hlaData = pd.read_table(
            Path(__file__).resolve().parent / 'MHC_pseudo.dat',
            header=None, names=['hla','pseudo.seq'], sep=r'\s+'
        ).drop_duplicates()

        # hlaData['hla'] = hlaData['hla'].apply(lambda x: x.replace("*",""))
        hlaData = hlaData.set_index('hla').to_dict()['pseudo.seq']

        return hlaData

    def FormatNameHLA(hlaName):
        # 规范 HLA 命名到蛋白型

        if hlaName.count(":") > 1:
            colonIdx = [idx for idx,char in enumerate(hlaName) if char == ':']
            hlaName = hlaName[:colonIdx[1]]
        return hlaName

    def ReadTcrFile(filePath):
        # 读取一个文件中的所有 TCR aa 序列
        tcrData = pd.read_csv(filePath, index_col=0)
        tcrSet = set(tcrData['AASeq'].dropna())
        return tcrSet

    return FormatNameHLA, ReadPseudoHLA, ReadTcrFile


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 先观察数据
    - 包含 I 型和 II 型: 14831 + 75
    - 包含人和小鼠: 14845 + 61
    - tcr beta 包含修正前后的, 统一采用修正后的吧, 修正后的没有缺失值
    """)
    return


@app.cell
def _(pd):
    rawData = pd.read_csv(
        "hf://datasets/isalgo/vdjdb_structure_models/vdjdb_structures_metadata.tsv.gz", 
        sep="\t"
    )
    return (rawData,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 阳性数据清洗
    """)
    return


@app.cell
def _(FormatNameHLA, Path, ReadPseudoHLA, np, rawData):
    ## 筛选物种
    rawTreat = rawData[rawData['species'] == 'HomoSapiens']

    ## 筛选MHC
    rawTreat = rawTreat[rawTreat['mhc.class'] == 'MHCI']

    ## 提取标准 cdr3 beta 序列
    rawTreat['cdr3.beta.norm'] = rawTreat['cdr3fix.beta'].apply(lambda x: eval(x)['cdr3'])

    ## HLA序列转换
    hlaPseudo = ReadPseudoHLA()
    rawTreat['hla'] = rawTreat['mhc.a'].apply(lambda x: FormatNameHLA(x).replace('*',''))
    rawTreat['hla.pseudo'] = rawTreat['hla'].apply(lambda x: hlaPseudo[x] if x in hlaPseudo else np.nan)

    ## 去除未知假序列的分型
    rawTreat = rawTreat[rawTreat['hla.pseudo'].notna()]

    ## 更新表头
    rawTreat = rawTreat[['hla','cdr3.beta.norm','hla.pseudo','antigen.epitope']]
    rawTreat.columns = ['hla','tcr','hla_pseudo','epitope']

    ## 写出
    rawTreat.to_csv(
        Path(__file__).resolve().parent / 'Raw.Posi.Clean.csv',
        index= False, sep='\t'
    )
    return (rawTreat,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 健康人群 TCR 库提取做阴性
    从 TCRdb2.0 下载了三个数据集
    """)
    return


@app.cell
def _(Path, ReadTcrFile, rawTreat):
    ## 提取总库
    rawDir = Path(__file__).resolve().parent / 'HealthyTcrRepoRaw'
    healthyTCR = set().union(*[ReadTcrFile(f) for f in rawDir.iterdir()])

    ## 去除阳性中有的 tcr
    posiTCR = set(rawTreat['tcr'])
    healthyTCR = healthyTCR.difference(posiTCR)
    return (healthyTCR,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 同步构建阴性和区分训练测试集
    - 按抗原肽序列分组, 遍历抗原肽组, 以 0.1 概率决定这条肽是否进入测试集
    - 在该组中遍历每个 tcr-pMHC, 随机抽取 49 个背景 tcr 作为阴性, 这 50 条同时进入训练集或测试集
    """)
    return


@app.cell
def _(Path, healthyTCR, pd, random, rawTreat, tqdm):
    # 训练集和测试集
    healthyTcrList = list(healthyTCR)
    trainData = []
    testData = []
    # pd.DataFrame(columns=['hla.name','hla.pseudo','antigen','tcr','label'])

    # 构建
    for antigen, antiData in tqdm(rawTreat.groupby("epitope")):
        # 决定是否加入阳性, 并且缓存要扩充的阴性
        toTest = 1 if random.random() < 0.20 else 0
        antiBatch = []
        for row in antiData.itertuples(index=False):
            hla, pseudo, epitope, tcr = row.hla, row.hla_pseudo, row.epitope, row.tcr
            # 当前肽段是阳性
            antiBatch.append([hla, pseudo, epitope, tcr, 1])
            # 添加阴性
            antiNega = random.sample(healthyTcrList, 49)
            for nega in antiNega:
                antiBatch.append([hla, pseudo, epitope, nega, 0])
        # 加入训练或测试集
        if toTest:
            testData.extend(antiBatch)
        else:
            trainData.extend(antiBatch)

        
    header = ['hla','hla_pseudo','epitope','tcr','label']
    trainData = pd.DataFrame(trainData, columns=header)
    testData = pd.DataFrame(testData, columns=header)

    trainData.to_csv(Path(__file__).resolve().parent / "Train.Raw.csv", index=False, sep='\t')
    testData.to_csv(Path(__file__).resolve().parent / "Test.csv", index=False, sep='\t')
    return (trainData,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 数据下采样
    因为要用训练集进行交叉验证, 此时训练集中的最大两个抗原家族, 分别就占据了 220k 和 180k 的数据, 超过了 总数据量 600k 的 1/5(对于5折交叉验证的话), 这样就得对两个最大家族进行下采样, 采集到大概 5 折的一半好了, 即 60k 左右
    """)
    return


@app.cell
def _(pd, trainData):
    trainSample = pd.DataFrame()

    # 遍历抗原家族, 如果数据量超过 60k, 就下采样
    for antigen_, antiData_ in trainData.groupby('epitope'):
        if antiData_.shape[0] > 60000:
            # 阳性阴性分开
            antiPosi_ = antiData_[antiData_['label'] == 1]
            antiNega_ = antiData_[antiData_['label'] == 0]
            # 采样比例
            sampleRate_ = 60000 / antiData_.shape[0]
            # 采样
            antiPosiDown_ = antiPosi_.sample(frac=sampleRate_, random_state=42)
            antiNegaDown_ = antiNega_.sample(frac=sampleRate_, random_state=42)
            # 拼接
            trainSample = pd.concat([trainSample, antiPosiDown_, antiNegaDown_])
        else:
            # 直接拼接
            trainSample = pd.concat([trainSample, antiData_])
        
    return (trainSample,)


@app.cell
def _(Path, trainSample):
    trainSample.to_csv(Path(__file__).resolve().parent / "Train.DownSampling.csv", index=False, sep='\t')
    return


if __name__ == "__main__":
    app.run()
