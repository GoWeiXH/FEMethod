import math

import pandas as pd
import numpy as np


def simple_match(data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
    """
    简单匹配系数
    """

    result = data_1 + data_2

    counts = result.value_counts()
    f_00 = counts[0]
    f_11 = counts[2]
    f_10_01 = counts[1]

    smc = f_11 / (f_00 + f_10_01)
    return round(smc, digits)


def jaccard(data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
    """
    Jaccard 系数，忽略二者相等且为 0 的数据，可以应用于稀疏数据。

    jc = f_11 / (f_10 + f_01 + f_11)

    f_11: 取值全为 1
    f_01、f_10: 取值不同

    完全相同则 Jaccard 系数为 1，完全不同则为 0；
    """

    result = data_1 + data_2

    counts = result.value_counts()
    f_11 = counts[2]
    f_10_01 = counts[1]

    smc = f_11 / f_10_01
    return round(smc, digits)


def dis_l1(data_1: pd.Series, data_2: pd.Series) -> float:
    """
    曼哈顿距离

    距离易受取值范围较大特征的影响，应注意数据规范化。
    """

    return float(np.sum(np.abs(data_1 - data_2)))


def dis_l2(data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
    """
    欧氏距离

    距离易受取值范围较大特征的影响，应注意数据规范化。

    dis_l2 = sqrt(sum((data_1 - data_2)**2))
    """

    return round(math.sqrt(np.sum((data_1 - data_2)**2)), digits)


def pearson(data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
    """
    Pearson 系数

    正相关等于 1
    负相关等于 -1
    无线性关系 0
    """

    mean_1 = np.mean(data_1)
    mean_2 = np.mean(data_2)

    data_1_diff = data_1 - mean_1
    data_2_diff = data_2 - mean_2

    cov = np.dot(data_1_diff, data_2_diff)
    std_1 = np.sum(np.square(data_1_diff))
    std_2 = np.sum(np.square(data_2_diff))
    pc = cov / np.sqrt(std_1 * std_2)

    return round(pc, digits)


def spearman(data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
    """
    Spearman 相关系数

    如果两个变量存在秩次，则

        sc = 1 - 6*sum((x-y)^2) / n(n^2-1)

    如果不存在秩次关系，仍然使用 pearson 系数计算
    """

    return round(data_1.corr(data_2, method='spearman'), digits)


def cosine(data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
    """
    余弦相似度

    cos = A·B / ||A||·||B||
    """

    dot = np.dot(data_1, data_2)
    data_1_l2 = np.sqrt(np.sum(np.square(data_1)))
    data_2_l2 = np.sqrt(np.sum(np.square(data_2)))
    cos = dot / (data_1_l2 * data_2_l2)

    return round(cos, digits)