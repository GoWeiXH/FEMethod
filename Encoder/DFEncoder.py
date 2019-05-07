import pandas as pd
import numpy as np

from collections import Counter
import math


def mapping(data: pd.DataFrame, col: str, map_dict: dict, layer=False, drop=True) -> pd.DataFrame:
    """
    将数据中某一列数据根据给定的映射字典进行替换
    具有两种映射形式：普通映射、分层映射

    :param data: ->int 原始完整 DataFrame 数据
    :param col: 需要使用序号编码表示的列名称
    :param map_dict: 指定映射字典
    :param layer: 是否分层映射，如果 True，则 map_dict 中的 values 应为可迭代对象
    :param drop: 是否删除原列数据
    :return: 替换后的 DataFrame 数据

    例：
        1)普通映射 传入的字典: {'jack': 'a',
                          'jerry': 'b',
                          'tom': 'c'}
           以上键值一一对应，一对一进行映射

        2)分层映射 传入的字典：{'a': ['jack', 'jerry'],
                          'b': ['tom'], }
          以上键值一对多，意味着将 value 中包含的元素映射为 key

    """

    if layer:
        new_dict = dict()
        for k, v in map_dict.items():
            if isinstance(v, list):
                for i in v:
                    new_dict[i] = k
            else:
                raise TypeError('The value of map_dict must be a list')
        map_dict = new_dict

    col_data = pd.DataFrame(data[col].map(map_dict))
    col_data.columns = [f'{col}_mapped']

    return concat_data(drop, data, col_data, col)


def ordinary(data: pd.DataFrame, col: str, step=1, reverse=False, drop=True) -> pd.DataFrame:
    """
    将数据中的某一列使用序号编码进行替换，
    序号取值从 0 开始，其取值个数与 col 列中的取值个数一致

    :param data: 原始完整 DataFrame 数据
    :param col: 需要使用序号编码表示的列名称
    :param step: 默认以 1 步长增长，可以指定
    :param reverse: 默认 False，取值出现次数越多值越大；True 则相反
    :param drop: 是否删除原列数据
    :return: 替换后的 DataFrame 数据
    """

    # 以字典形式统计当前所选列数据共有多少种取值
    unique = Counter(data[col])

    # 根据每个取值出现的次数进行排序，出现次数多的取值大
    unique = sorted(unique, key=lambda k: unique[k], reverse=reverse)

    # 生成原始数据与序号的映射
    k_dict = dict(zip(unique, range(0, step*len(unique), step)))

    # 对数据进行替换
    col_data = pd.DataFrame(data[col].map(k_dict))
    col_data.columns = [f'{col}_ord']

    return concat_data(drop, data, col_data, col)


def binary(data: pd.DataFrame, col: str, reverse=False, drop=True) -> pd.DataFrame:
    """
    将数据中的某一列使用二进制编码进行替换

    :param data: 原始完整 DataFrame 数据
    :param col: 需要使用二进制编码表示的列名称
    :param reverse: 默认出现次数多的取值所对应的二进制数大，reverse=True 则相反
    :param drop: 是否删除原列数据
    :return: 替换后的 DataFrame 数据
    """

    # 以字典形式统计当前所选列数据共有多少种取值
    k_dict = dict(Counter(data[col]))
    unique_list = sorted(k_dict, key=lambda x: k_dict[x], reverse=reverse)

    # 计算根据当前取值个数所需要的二进制位数，向上取整数位
    max_len = int(math.ceil(math.log2(int(len(unique_list)))))

    # 对每个取值进行二进制映射
    for n, k in enumerate(unique_list):

        # 计算当前取值的二进制形式
        r = list(bin(n))[2:]

        # 以 0 补足缺少的位数
        r = ['0'] * (max_len - len(r)) + r
        k_dict[k] = "".join(r)

    # 使用二进制映射，对所选列数据中的元素进行替换
    # 例如：以 '001' 进行替换
    col_data = data[col].map(k_dict)

    # 将一列数据以 '位' 进行拆分
    bin_data = pd.DataFrame()

    for i in range(max_len):
        bin_data[f'{col}_index{i}'] = col_data.map(lambda x: x[i])

    # 返回数据
    return concat_data(drop, data, bin_data, col)


def one_hot(data: pd.DataFrame, col: str, engine='pd', drop=True) -> pd.DataFrame:
    """
    将数据中的某一列使用 one-hot 编码进行替换

    :param data: 原始完整 DataFrame 数据
    :param col: 需要使用 one-hot 编码表示的列名称
    :param drop: 是否删除原列数据
    :param engine: 默认使用 pandas.get_dummies()，否则使用自己实现方式
    :return: 替换后的 DataFrame 数据
    """

    if engine == 'pd':
        return pd.concat((pd.get_dummies(data, columns=[col]), data[col]), axis=1)

    # 获取数据
    col_data = data[col]

    # 计算该列取值个数
    unique = set(col_data)
    unique_length = len(unique)

    # 得到取值与列索引的对应关系
    k_dict = dict(zip(unique, range(unique_length)))

    # 将数据转换成向量（所在索引取值为 1 ）
    # 全 0 初始化矩阵
    init_matrix = np.zeros((len(col_data), unique_length), dtype=np.int)

    # 将第 r 行的数据根据列索引修改为 1
    col_data = col_data.map(k_dict)
    init_matrix[range(len(col_data)), col_data] = 1

    # 添加 one-hot 列名，生成 DataFrame 数据
    k_dict_inverse = dict(zip(k_dict.values(), k_dict.keys()))
    columns = [f'{col}_{k_dict_inverse[i]}' for i in range(len(unique))]
    oh_data = pd.DataFrame(init_matrix, columns=columns)

    # 返回数据
    return concat_data(drop, data, oh_data, col)


def ratio(data: pd.DataFrame, col: str, n_digits=3, drop=True) -> pd.DataFrame:
    """
    将数据中的某一列使用其出现频率进行编码

    :param data: 原始完整 DataFrame 数据
    :param col: 需要使用频率编码表示的列名称
    :param n_digits: 计算频率保留小数位数
    :param drop: 是否删除原列数据
    :return: 替换后的 DataFrame 数据
    """

    # 获取所选列数据
    col_data = data[col]

    # 计算各取值的个数
    k_dict = Counter(col_data)

    # 计算各取值的频率，生成取值与频率的映射字典
    total = len(col_data)
    for k, v in k_dict.items():
        k_dict[k] = round(v / total, n_digits)

    # 利用取值频率映射字典修改数据
    col_data = pd.DataFrame(col_data.map(k_dict))
    col_data.columns = [f'{col}_ratio']

    # 返回数据
    return concat_data(drop, data, col_data, col)


def concat_data(drop: bool, data: pd.DataFrame, new_data: pd.DataFrame, col: str) -> pd.DataFrame:
    result = pd.concat([data, new_data], axis=1)
    if drop:
        return result.drop(columns=[col])
    else:
        return result
