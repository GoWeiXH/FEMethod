from __Helper import __new_col_data
import pandas as pd


def points(col_data: pd.Series, ps: list, **kwargs) -> pd.DataFrame:
    """
    规则硬分，根据给定分割点进行划分

    :param col_data: 需要离散化的列数据
    :param ps: 包含切分点的 list
    :return: 离散化后 DataFrame 数据
    """
    ps = [min(col_data)-0.001] + ps + [max(col_data)]

    def to_range(x):
        for i in range(len(ps)-1):
            if ps[i] < x <= ps[i+1]:
                if labels or labels is None:
                    return f'[{ps[i]}, {ps[i+1]})'
                else:
                    return i

    labels = kwargs.get('labels')
    points_data = col_data.map(to_range)

    return __new_col_data(points_data, 'cut')


def freq(col_data: pd.Series, n_bins: int, **kwargs) -> pd.DataFrame:
    """
    等频分箱，各离散值拥有相等数量的样本

    :param col_data: 需要离散化的列数据
    :param n_bins: 要求划分成的类别个数
    :return: 离散化后 DataFrame 数据
    """

    freq_data = pd.qcut(col_data, n_bins, **kwargs)
    return __new_col_data(freq_data, 'cut')


def width(col_data: pd.Series, n_bins: int, **kwargs) -> pd.DataFrame:
    """
    等宽分箱，将取值区间均分为若干个子区间

    :param col_data: 需要离散化的列数据
    :param n_bins: 要求划分成的类别个数
    :return: 离散化后 DataFrame 数据
    """

    width_data = pd.cut(col_data, n_bins, **kwargs)
    return __new_col_data(width_data, 'cut')


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
