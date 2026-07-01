"""函数定义脚本"""

import numpy as np
import pandas as pd 

# 预测器
class KNNpredictor():

    def __init__(self):
        pass


def KnnClassfier(
    ref_dataset,   # 参考/训练数据集
    query_data,  # 待测数据
    ref_label,  # 参考/训练数据集的label
    K,  # 分类选择前k项      
):
    """KNN 分类器"""
    assert len(ref_label) == len(ref_dataset), "参考数据与其label长度不一致"

    # 计算 query 到每个 ref 到距离
        # 构建同 shape 矩阵, 行扩充到跟测试数据集一样, 列不用扩充, 因为数据 feature 结构一样
    query_data = np.tile(query_data, (ref_dataset.shape[0], 1))
        # 做差, 直接做差结果一样啊, 不需要前面的 tile
    try:
        diff_data = query_data - ref_dataset
    except Exception as err:
        print(query_data, ref_dataset)
        assert 1==0, "停止"
        
        # 平方
    sq_diff_data = diff_data ** 2
        # 平方和, 对每一行求和, 即 query 到 dataset 中每个
    sum_sq_diff_data = sq_diff_data.sum(axis=1)
        # 开方即距离
    dis_data = sum_sq_diff_data ** 0.5

    # 对距离排序
    sort_dis_idx = dis_data.argsort()

    # 取排序的前 K 项的 label
    vote_label = {}
    for idx in range(K):
        curr_label = ref_label[sort_dis_idx[idx]]
        vote_label[curr_label] = vote_label.get(curr_label, 0) + 1
    
    # 取频数最高的 label
    sort_label = sorted(vote_label.items(), key=lambda x: x[1], reverse=True)
    return sort_label[0][0]


def ExtractTrainDataset(file_path):
    data = pd.read_table(
        file_path, header=None, names=['FlightMile','GameTime','IceKilo','Label']
    )

    dataset = data[data.columns[:-1]].values
    labels = data[data.columns[-1]].values
    return dataset, labels, data


def NormDataset(
    dataset,    # 需要归一化的数据集
    given_param = None, # 测试集的归一化参数
):
    """对每一列采用 max-min 归一化"""
    if not given_param:
        cmax, cmin = dataset.max(axis=0), dataset.min(axis=0)
        range = cmax - cmin 
    else:
        cmin, range = given_param
    norm_dataset = (dataset - cmin) / range 
    return norm_dataset, cmin, range


def AccurrcyModel(
    dataset: np.array, # 完整数据集, 需要拆分成测试集和训练集,
    labels: np.array, # 完整标签, 需要同样拆分
    ratio: float, # 测试集比例
    K: int, # KNN 算法投票取前 K 项,
    given_params = None, # 给的数据集和参数
    normaize: bool = True, # 是否进行归一化
):
    """拆分数据集进行训练并计算准确率"""
    if not given_params:
        # 随机拆分, 
        train_data, train_label, test_data, test_label = SplitDataset(
            dataset=dataset, labels=labels, ratio=ratio
        )
    else:
        train_data, train_label, test_data, test_label = given_params

    if normaize:
        # 训练集归一化和归一化参数
        norm_train, train_min, train_range = NormDataset(train_data)
        # 测试集归一化
        norm_test, _, _ = NormDataset(test_data, given_param=(train_min, train_range))
    else:
        norm_train =  train_data
        norm_test =  test_data

    # 性能计算
    acc = np.zeros(shape=norm_test.shape[0])
    preds = []
    for idx, vector in enumerate(norm_test):
        true_label = test_label[idx]
        pred_label = KnnClassfier(
            ref_dataset=norm_train,
            query_data=vector,
            ref_label=train_label,
            K=K
        )
        preds.append(pred_label)
        if true_label == pred_label:
            acc[idx] = 1
    return preds, acc.sum() / acc.shape[0]


def SplitDataset(dataset, labels, ratio):
    # 随机拆分, 
    train_shape = int(dataset.shape[0] * (1 - ratio))
    rand_idxs = np.random.permutation(dataset.shape[0])
    train_idxs = rand_idxs[:train_shape]
    test_idxs = rand_idxs[train_shape:]

    train_data = dataset[train_idxs]
    train_label = labels[train_idxs]
    test_data = dataset[test_idxs]
    test_label = labels[test_idxs]

    return train_data, train_label, test_data, test_label

