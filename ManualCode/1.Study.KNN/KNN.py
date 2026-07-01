import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium")


@app.cell
def module():
    import marimo as mo
    import importlib
    import numpy as np
    import pandas as pd 
    import os 
    from pathlib import Path
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.preprocessing import MinMaxScaler

    import Utils as ut
    importlib.reload(ut)
    return (
        KNeighborsClassifier,
        MinMaxScaler,
        Path,
        accuracy_score,
        mo,
        np,
        os,
        pd,
        ut,
    )


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 案例1
    """)
    return


@app.cell
def _(ut):
    # 读取数据集
    dataset, labels, data = ut.ExtractTrainDataset(file_path="./data/Ch02-KNN/datingTestSet2.txt")

    # 拆分出训练集和测试集, 用于 sklearn 封装的 KNN
    trn_datas, trn_label, tst_datas, tst_label = ut.SplitDataset(dataset=dataset, labels=labels, ratio=0.1)
    return dataset, labels, trn_datas, trn_label, tst_datas, tst_label


@app.cell
def _(
    KNeighborsClassifier,
    MinMaxScaler,
    accuracy_score,
    dataset,
    labels,
    pd,
    trn_datas,
    trn_label,
    tst_datas,
    tst_label,
    ut,
):
    # 对比我的 KNN 和封装的 KNN
    my_pred, my_acc = ut.AccurrcyModel(
        dataset=dataset, 
        labels=labels,
        ratio=0.1,
        K=7, 
        given_params=(trn_datas, trn_label, tst_datas, tst_label)
    )

    # 封装 KNN
    KNN = KNeighborsClassifier(
        n_neighbors=7,
        weights="uniform",
        algorithm="brute",
        metric="euclidean"
    )
    Scaler = MinMaxScaler()
    KNN.fit(Scaler.fit_transform(trn_datas), trn_label)
    KNN_pred = KNN.predict(Scaler.transform(tst_datas))
    KNN_acc = accuracy_score(tst_label, KNN_pred)

    print(f"你的模型准确率为 {my_acc}, KNN 模型准确率为 {KNN_acc}", )

    pred_data = pd.DataFrame({
        "Ture_Label": tst_label, 
        "My_Pred": my_pred, 
        "KNN_Pred": KNN_pred
    })
    

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 2. 案例2: 数字识别
    """)
    return


@app.cell(hide_code=True)
def _(Path, np, os, pd):
    def ParseFig(
        fig_path: Path, # 解析的图片路径
    ):
        """解析突变, 返回其 label 和转换的 vector"""
        label = fig_path.name.split("_")[0]
        vector = []
        # 逐行添加
        data = pd.read_table(fig_path, header=None, names=['rows'])
        for row in data['rows'].tolist():
            vector += [int(i) for i in list(row)]

        return label, np.array(vector)

    def ParseTotalFigs(
        fig_dir: str, # 解析图片的存放目录
    ):
        """从目录中解析所有图片"""
        fig_paths = os.listdir(fig_dir)
        parse_res = []
        for fpath in fig_paths:
            fpath = Path(fig_dir).joinpath(fpath)
            parse_res.append(ParseFig(fpath))

        labels, vectors = zip(*parse_res)
        return np.array(labels), np.array(vectors)
    

    trn_label_case2, trn_datas_case2 = ParseTotalFigs("./data/Ch02-KNN/trainingDigits/")
    tst_label_case2, tst_datas_case2 = ParseTotalFigs("./data/Ch02-KNN/testDigits/")
    return trn_datas_case2, trn_label_case2, tst_datas_case2, tst_label_case2


@app.cell
def _(
    KNeighborsClassifier,
    accuracy_score,
    np,
    trn_datas_case2,
    trn_label_case2,
    tst_datas_case2,
    tst_label_case2,
    ut,
):
    my_pred_case2, my_acc_case2 = ut.AccurrcyModel(
        dataset = np.zeros(10),
        labels = np.zeros(10), 
        ratio = 0,
        K = 7, 
        given_params = [trn_datas_case2, trn_label_case2, tst_datas_case2, tst_label_case2],
        normaize = False
    )

    KNN_case2 = KNeighborsClassifier(
        n_neighbors=7,
        weights="uniform",
        algorithm="brute",
        metric="euclidean"
    )
    KNN_case2.fit(trn_datas_case2, trn_label_case2)
    KNN_pred_case2 = KNN_case2.predict(tst_datas_case2)
    KNN_acc_case2 = accuracy_score(tst_label_case2, KNN_pred_case2)
    return KNN_acc_case2, my_acc_case2


@app.cell
def _(KNN_acc_case2, my_acc_case2):
    print(my_acc_case2, KNN_acc_case2)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 9. 测试单元
    """)
    return


if __name__ == "__main__":
    app.run()
