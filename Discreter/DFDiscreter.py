import pandas as pd


def points():
    """
    规则硬分，根据给定分割点进行划分

    :return:
    """


def freq(col_data: pd.DataFrame, n_bins: int, labels=None) -> pd.DataFrame:
    """
    等频分箱，各离散值拥有相等数量的样本

    :param col_data: 需要离散化的列数据
    :param n_bins: 要求划分成的类别个数
    :param labels: 是否将离散化后的类别映射成 序号索引，默认 True
    :return: 离散化后 DataFrame 数据
    """

    return pd.qcut(col_data, n_bins, labels=labels)


def width(col_data: pd.DataFrame, n_bins: int, labels=None) -> pd.DataFrame:
    """
    等宽分箱，将取值区间均分为若干个子区间

    :param col_data: 需要离散化的列数据
    :param n_bins: 要求划分成的类别个数
    :param labels: 是否将离散化后的类别映射成 序号索引，默认 True
    :return: 离散化后 DataFrame 数据
    """

    return pd.cut(col_data, n_bins, labels=labels)


def hashtable():
    """
    Hash分桶，利用 Hash table 方法离散数据

    :return:
    """


def cluster():
    """
    使用一维聚类算法对数据进行离散化

    :return:
    """
