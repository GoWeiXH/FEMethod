from __Helper import __new_col_data

import pandas as pd
import numpy as np

from collections import Counter
import math


def ordinary(col_data: pd.Series, step=1, reverse=False) -> pd.DataFrame:
    """
    将数据中的某一列使用序号编码进行替换，
    序号取值从 0 开始，其取值个数与 col 列中的取值个数一致

    :param col_data: 需要编码表示的列数据
    :param step: 默认以 1 步长增长，可以指定
    :param reverse: 默认 False，取值出现次数越多值越大；True 则相反
    :return: 编码后的 DataFrame 数据

    例:
        >> data = [['jack', 'old', 180], ['jack', 'young', 175],
                   ['jerry', 'middle', 170], ['tom', 'young', 175]]
        >> pd_data = pd.DataFrame(data, columns=['name', 'age', 'height'])
        >> ordinary(pd_data['name'])
               name_ord
        0         2
        1         2
        2         0
        3         1

    """

    # 以字典形式统计当前所选列数据共有多少种取值
    unique = Counter(col_data)

    # 根据每个取值出现的次数进行排序，出现次数多的取值大
    unique = sorted(unique, key=lambda k: unique[k], reverse=reverse)

    # 生成原始数据与序号的映射
    k_dict = dict(zip(unique, range(0, step*len(unique), step)))

    # 对数据进行替换
    col_data = col_data.map(k_dict)

    return __new_col_data(col_data, 'ord')


def binary(col_data: pd.Series, reverse=False) -> pd.DataFrame:
    """
    将数据中的某一列使用二进制编码进行替换

    :param col_data: 需要编码表示的列数据
    :param reverse: 默认出现次数多的取值所对应的二进制数大，reverse=True 则相反
    :return: 编码后的 DataFrame 数据

    例:
        >> data = [['jack', 'old', 180], ['jack', 'young', 175],
                   ['jerry', 'middle', 170], ['tom', 'young', 175]]
        >> pd_data = pd.DataFrame(data, columns=['name', 'age', 'height'])
        >> binary(pd_data['name'])
            name_bin_0 name_bin_1
        0          1          0
        1          1          0
        2          0          0
        3          0          1

    """

    # 以字典形式统计当前所选列数据共有多少种取值
    k_dict = dict(Counter(col_data))
    unique_list = sorted(k_dict, key=lambda x: k_dict[x], reverse=reverse)

    # 计算根据当前取值个数所需要的二进制位数，向上取整数位
    max_len = int(math.ceil(math.log2(int(len(unique_list)))))

    # 对每个取值进行二进制映射
    for n, k in enumerate(unique_list):

        # 计算当前取值的二进制形式
        r = list(bin(n))[2:]

        # 以 0 补足缺少的位数
        r = ['0'] * (max_len - len(r)) + r
        # k_dict[k] = "".join(r)
        k_dict[k] = r

    # 使用二进制映射，对所选列数据中的元素进行替换
    # 例如：以 '001' 进行替换
    cols_data = np.array(list(col_data.map(k_dict)))

    # 将一列数据以 '位' 进行拆分
    bin_data = pd.DataFrame()

    col_name = getattr(col_data, 'name')
    for i in range(cols_data.shape[1]):
        bin_data[f'{col_name}_bin_{i}'] = cols_data[:, i]

    # 返回数据
    return bin_data


def one_hot(col_data: pd.Series, engine='pd') -> pd.DataFrame:
    """
    将数据中的某一列使用 one-hot 编码进行替换

    :param col_data: 需要编码表示的列数据
    :param engine: 默认使用 pandas.get_dummies()，否则使用自己实现方式
    :return: 编码后的 DataFrame 数据

    例：
        >> data = [['jack', 'old', 180], ['jack', 'young', 175],
                   ['jerry', 'middle', 170], ['tom', 'young', 175]]
        >> pd_data = pd.DataFrame(data, columns=['name', 'age', 'height'])

    1)
        >> one_hot(pd_data['name'])
                   name_jack  name_jerry  name_tom
        0          1           0         0
        1          1           0         0
        2          0           1         0
        3          0           0         1

    2)
        >> one_hot(pd_data['name'], engine=None)
                   name_jack  name_tom  name_jerry
        0          1         0           0
        1          1         0           0
        2          0         0           1
        3          0         1           0
    """

    col_name = getattr(col_data, 'name')
    if engine == 'pd':
        oh_data = pd.get_dummies(col_data, columns=[col_name])
        oh_data.columns = [f'{col_name}_{name}' for name in oh_data.columns]
        return oh_data

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
    columns = [f'{col_name}_{k_dict_inverse[i]}' for i in range(len(unique))]
    oh_data = pd.DataFrame(init_matrix, columns=columns)

    # 返回数据
    return oh_data


def ratio(col_data: pd.Series, n_digits=3) -> pd.DataFrame:
    """
    将数据中的某一列使用其出现频率进行编码

    :param col_data: 需要编码表示的列数据
    :param n_digits: 计算频率保留小数位数
    :return: 编码后的 DataFrame 数据

    例：
        >> data = [['jack', 'old', 180], ['jack', 'young', 175],
                   ['jerry', 'middle', 170], ['tom', 'young', 175]]
        >> pd_data = pd.DataFrame(data, columns=['name', 'age', 'height'])
        >> ratio(pd_data['name'])
            name_ratio
        0        0.50
        1        0.50
        2        0.25
        3        0.25

    """

    # 计算各取值的个数
    k_dict = Counter(col_data)

    # 计算各取值的频率，生成取值与频率的映射字典
    total = len(col_data)
    for k, v in k_dict.items():
        k_dict[k] = round(v / total, n_digits)

    # 利用取值频率映射字典修改数据
    col_data = col_data.map(k_dict)

    # 返回数据
    return __new_col_data(col_data, 'ratio')
