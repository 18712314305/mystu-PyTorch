"""
函数定义脚本 for DecisionTree.py
"""

import numpy as np
import pandas as pd 
from typing import Dict 
from typing_extensions import TypeGuard
from matplotlib import pyplot as plt


np.random.seed(42)

class Dataset(object):
    """数据集对象, 集成了提取标签, 拆分训练和测试, 对指定列归一化的功能"""
    def __init__(self, data, label, need_split, range = None):
        # 初始化属性
        assert isinstance(data, pd.DataFrame), "请传入含有特征名/列名的 DataFrame 格式作为 data"
        self.data :pd.DataFrame = data # 接受一个 DataFrame 作为输入
        self.size :int = data.shape[0]
        self.SplitLabel :bool = False # 数据集中是否拆分出了 label
        self.label :np.ndarray = self.GetLabel(label) # 接受一个列名或直接传入一个 label 数组
        self.NeedSplit :bool = need_split # 是否需要切分数据集
        self.range :str = self.ParseRange(range) # 数据集中数据范围, train/test/train+test
        
        ## 初始化(SplitLabel)关联属性, 这里不再显示定义, 不然会被重新初始化
        ## self.SplitLabel :bool = False # 数据集中是否拆分出了 label
        ## self.format_data :np.ndarray = None 
        ## self.features :str = None
        
        # 方法维护属性, 这里可以显示定义, 提示这些属性可直接调用, 方法调用时会更新
        self.IsSplited :bool = False 
        self.train_data :pd.DataFrame = None
        self.train_format :np.ndarray = None
        self.train_label :np.ndarray = None
        self.train_idx :np.ndarray = None 
        self.test_data :pd.DataFrame = None 
        self.test_format :np.ndarray = None 
        self.test_label :np.ndarray = None
        self.test_idx : np.ndarray = None 
        self.norm_data :pd.DataFrame = None 
        self.norm_format :np.ndarray = None
        self.norm_train :np.ndarray = None
        self.norm_test :np.ndarray = None 
        self.norm_mins :dict = dict() # 用 update 更新, 可以对不同 feature 不同时归一化
        self.norm_ranges :dict = {}
        self.SplitRatio :float = 0
        
    def Infos(self):
        """返回属性总览"""
        info = {
            key: {"type":type(value), "shape":value.shape} if type(value) in [pd.DataFrame, np.ndarray] else value 
            for key,value in vars(self).items()
        }
        return info
        
    def GetLabel(self, label):
        """根据实例化的 label 参数, 记录或解析 label"""
        # 如果传入的是一个数组, 检查长度直接作为 label
        if isinstance(label, np.ndarray):
            assert len(label) == self.data.shape[0], f"传入的数组长度与数据集长度不一致 !"
            label_ = label
            self.SplitLabel = True # 标记 data 中没有 label 列
            self.format_data = self.data.values
            self.features = self.data.columns
        elif isinstance(label, str):
            assert label in self.data.columns, f"数据集中不存在传入的列名: {label} !"
            label_ = self.data[label].values
            self.data = self.data.drop(label, axis=1) # 取出 label 列
            self.SplitLabel = True # 标记 data 中没有 label 列
            self.format_data = self.data.values
            self.features = self.data.columns
        else:
            assert 1 == 0, f"请传入标签数组或数据集中 label 列名作为 label 参数 !"
        return label_

    def ParseRange(self, range):
        """根据 range 参数和是否需要拆分判断数据集数据范围"""
        if range:
            legal_range = ['Train','Test','TrainTest']
            assert range in legal_range, f"不支持的 range 参数: {range}, 请在 {legal_range} 中选择一项"
            return range
        # 如果没有传递参数, 根据是否需要拆分, 需要则both,不需要则默认训练集
        else:
            range = 'TrainTest' if self.NeedSplit else 'Test'
            return range
    
    def SplitData(self, ratio, train_idx = None, test_idx = None):
        """按给定比例切分数据集, ratio 是测试集的比例"""
        assert self.SplitLabel, "请分离 label 后再拆分"
        # 如果没切分过就重新取 idx
        if (train_idx is None) or (test_idx is None):
            train_idx, test_idx = self._RandomIdx(self.data.index.tolist(), ratio)
        self.train_data = self.data.loc[train_idx,:] if self.range != 'Test' else None
        self.train_format = self.train_data.values if self.range != 'Test' else None
        self.train_label = self.label[train_idx] if self.range != 'Test' else None 
        self.train_idx = train_idx if self.range != 'Test' else None 
        self.test_data = self.data.loc[test_idx,:] if self.range != 'Train' else None 
        self.test_format = self.test_data.values if self.range != 'Train' else None 
        self.test_label = self.label[test_idx] if self.range != 'Train' else None 
        self.test_idx = test_idx if self.range != 'Train' else None 
        self.IsSplited = True
        self.SplitRatio = ratio # 记录下切分比例, 
        return        

    def MapFeatureClass(self, feature_maps):
        # 目前的转换方式, 似乎不支持 self.data 中有缺失值
        """
        根据给定的 map 对其中的类别型 feature 转换为数字.
        feature_map 的格式为 {feature_1: {class_1: value_1},...}
        """
        self._MapFormatCheck(feature_maps)
        # 检查列名是否正确
        for feature, fmap in feature_maps.items():
            assert feature in self.features, f"特征转换字典中 {feature} 不在列名中"
            diff_clas = set(self.data[feature].unique()) -  set(fmap.keys())
            assert len(diff_clas) == 0, f"特征转换字典中, {feature} 子字典中不包含 {diff_clas} 类别, 该类别在 data 中"
        # 遍历每个特征修改
        for feature, fmap in feature_maps.items():
            self.data[feature] = self.data[feature].apply(lambda x: fmap[x])

        # 更新相关数据集, 如果切分过, 重新切分
        self.format_data = self.data.values
        if self.IsSplited:
            self.SplitData(self.SplitRatio, train_idx=self.train_idx, test_idx=self.test_idx)
        return            

    # 还是应该尝试将数据集继承出来
    def NormData(self, features: list = None, method = "Max-Min", norm_mins = None, norm_ranges = None, ratio = None):
        """
        归一化数据集的 features 列, 先只支持 Max-Min 归一化
        如果是 TrainTest 且没拆分, 会先进行拆分操作, 否则测试集的数据会提前学习
        """
        # 如果没指定要归一化的 features, 用归一化所有feature
        if not features:
            features = self.features
        # 检查有没有非数值内容
        for feature in features:
            assert self.data[feature].dtype != 'O', f"{feature} 列包含非数值类型, 请转换后再进行归一化"

        # 如果只有训练集, 完全归一化, 并返回
        if self.range == 'Train':
            norm_data = self.data.copy()
            norm_mins = {}
            norm_ranges = {}
            for feature in features:
                norm_mins[feature] = norm_data[feature].min()
                norm_ranges[feature] = norm_data[feature].max() - norm_data[feature].min()
                norm_data[feature] = (norm_data[feature] - norm_mins[feature]) / norm_ranges[feature]
            # 更新属性
            self.norm_data = norm_data
            self.norm_format = self.norm_data.values
            self.norm_train = self.norm_format
            self.norm_mins = self.norm_mins.update(norm_mins)
            self.norm_ranges = self.norm_ranges.update(norm_ranges)
            return 

        # 如果只有测试集, 给定参数归一化
        elif self.range == "Test":
            # 检查是否提供了参数
            assert norm_mins, "对于测试集, 需要传入归一化参数 norm_min 和 norm_range"
            assert norm_ranges, "对于测试集, 需要传入归一化参数 norm_min 和 norm_range"
            # 检查要归一化的列是否都提供了参数
            assert set(features) <= set(norm_mins.keys()), f"传入的归一化参数不全, norm_mins: {set(features)-set(norm_mins.keys())}"
            assert set(features) <= set(norm_ranges.keys()), f"传入的归一化参数不全, norm_ranges: {set(features)-set(norm_ranges.keys())}"
            
            norm_data = self.data.copy()
            for feature in features:
                norm_data[feature] = (norm_data[feature] - norm_mins[feature]) / norm_ranges[feature]
            # 更新属性
            self.norm_data = norm_data
            self.norm_format = self.norm_data.values
            self.norm_test = self.norm_format
            self.norm_mins = self.norm_mins.update(norm_mins)
            self.norm_ranges = self.norm_ranges.update(norm_ranges)
            return
        # 如果都有, 根据需要先拆分
        else:
            if not self.IsSplited:
                assert ratio, "当前数据集未拆分, 请传入 ratio 参数, 归一化方法将自动拆分后归一化"
                _ = self.SplitData(ratio)
            # 如果拆分过, 取训练集计算归一化参数
            norm_train_data = self.train_data.copy()
            norm_mins = {}
            norm_ranges = {}
            for feature in features:
                norm_mins[feature] = norm_train_data[feature].min()
                norm_ranges[feature] = norm_train_data[feature].max() - norm_mins[feature]
                norm_train_data[feature] = (norm_train_data[feature] - norm_mins[feature]) / norm_ranges[feature]
            self.norm_train = norm_train_data.values
            self.norm_mins = self.norm_mins.update(norm_mins)
            self.norm_ranges = self.norm_ranges.update(norm_ranges)
            # 再归一化测试集
            norm_test_data = self.test_data.copy()
            for feature in features:
                norm_test_data[feature] = (norm_test_data[feature] - norm_mins[feature]) / norm_ranges[feature]
            self.norm_test = norm_test_data.values
            return       
    
    @staticmethod
    def _RandomIdx(indexs, ratio):
        """对给定长度的数组, 返回 ratio 组和剩下的组分"""
        srt_idx = np.random.permutation(indexs)
        part_ratio = srt_idx[int(len(indexs) * (1-ratio)):]
        part_other = srt_idx[:int(len(indexs) * (1-ratio))]
        return part_other, part_ratio

    @staticmethod
    def _MapFormatCheck(feature_maps) -> TypeGuard[Dict[str, Dict[str, float]]]:
        assert isinstance(feature_maps, dict), "特征转换字典格式不正确, 正确格式为 [Dict[str, Dict[str, float/int]], 形如 {feature_1: {class_1: value_1},...} !"
        for feature, fmap in feature_maps.items():
            assert isinstance(fmap, dict), "特征转换字典格式不正确, 正确格式为 [Dict[str, Dict[str, float/int]], 形如 {feature_1: {class_1: value_1},...} !"
            for clas, value in fmap.items():
                assert type(value) in [int, float], "特征转换字典格式不正确, 正确格式为 [Dict[str, Dict[str, float/int]], 形如 {feature_1: {class_1: value_1},...} !"
        return True


# 信息增益计算相关函数
def CalEntropy(array: np.ndarray[any]):
    """计算一组 label 的熵"""
    _, counts = np.unique(array, return_counts=True)
    probs = counts / len(array)
    erp_ = -np.sum(probs * np.log2(probs))
    return erp_


def WeightSumErps(arrays: list[np.ndarray[any]]):
    """计算多组数据的加权信息熵, 加权默认数量加权"""
    len_arr = np.array([len(arr) for arr in arrays])
    size = len_arr.sum()
    weight_arr = len_arr / size
    erp_arr = np.array([CalEntropy(arr) for arr in arrays])
    weight_sum_ = erp_arr.dot(weight_arr)
    return weight_sum_


def CreateTree(
    data :pd.DataFrame, # 包含 LABEL 标签的数据集, 列名是 Label, 且是最后一列
):
    # 边界条件1: 如果当前数据集中只有一种类
    if data['Label'].unique().size == 1:
        return data['Label'].values[0]
    # 边界条件2: 没有特征可以继续分
    if data.columns.size == 1:  # 只剩下 Label 列
        return VoteClass(data['Lable'].values)
    # 递归长枝
    bfeature = SelectBestFeature(data)
    btree = {bfeature: {}}
    for fclas, fc_data in data.groupby(by=bfeature):
        btree[bfeature][fclas] = CreateTree(fc_data.drop(columns=bfeature))
    return btree


def SelectBestFeature(
    data :pd.DataFrame, # 包含 LABEL 标签的数据集, 列名是 Label, 且是最后一列
):
    """选择最优的分组标签"""
    avail_features = data.columns[:-1]      # 除去 Label 列都是标签
    root_erp = CalEntropy(data['Label'])
    bfeature, bf_erpGain = "", 0
    for feature in avail_features:
        # 计算分枝加权熵和信息增益
        child_labels = []
        for _, fc_data in data.groupby(by=feature):
            child_labels.append(fc_data['Label'].values)
        child_erp = WeightSumErps(child_labels)
        erp_Gain = root_erp - child_erp
        # 更新最优标签和最优信息增益
        if erp_Gain >= bf_erpGain:
            bfeature = feature
            bf_erpGain = erp_Gain
    return bfeature


def VoteClass(
    labels: np.ndarray, # 标签数组
):
    """投票选择当前数组中最多的类"""
    clas, count = np.unique(labels, return_counts=True)
    votes = sorted(zip(clas, count), key=lambda x: x[1], reverse=True)
    return votes[0][0]


def plotTree(
    tree :dict, # 决策树
):
    """绘图决策树"""
    pass