from __Helper import __new_col_data
import pandas as pd


def key_value(col_data: pd.DataFrame, map_dict: dict, layer=False) -> pd.DataFrame:
    """
    将数据中某一列数据根据给定的映射字典进行替换
    具有两种映射形式：普通映射、分层映射

    :param col_data: 需要映射表示的列数据
    :param map_dict: 指定映射字典
    :param layer: 是否分层映射，如果 True，则 map_dict 中的 values 应为可迭代对象
    :return: 映射后的 DataFrame 数据

    例：
        >> data = [['jack', 'old', 180],
                ['jack', 'young', 175],
                ['jerry', 'middle', 170],
                ['tom', 'young', 175]]

        >> pd_data = pd.DataFrame(data, columns=['name', 'age', 'height'])

        1) 普通映射 传入的字典: {'jack': 'a',
                             'jerry': 'b',
                             'tom': 'c'}
           以上键值一一对应，一对一进行映射

            >> map_dict = {'jack': 'a', 'jerry': 'b', 'tom': 'c'}
            >> key_value(pd_data['name'], map_dict)
              name_mapped
            0           a
            1           a
            2           b
            3           c


        2) 分层映射 传入的字典：{'a': ['jack', 'jerry'],
                             'b': ['tom'], }
          以上键值一对多，意味着将 value 中包含的元素映射为 key

            >> map_dict = {'a': ['jack', 'jerry'], 'b': ['tom']}
            >> key_value(pd_data['name'], map_dict, layer=True)
              name_mapped
            0           a
            1           a
            2           a
            3           b

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

    col_data = col_data.map(map_dict)

    return __new_col_data(col_data, 'mapped')


def max_min(col_data: pd.DataFrame)-> pd.DataFrame:
    """
    最大-最小归一化

    """

    try:
        max_v = col_data.max()
        min_v = col_data.min()
        d = max_v - min_v
        col_data = (col_data - min_v) / d
    except TypeError:
        raise TypeError('Data type must be consistent (int or float)')

    return __new_col_data(col_data, 'scale')


def mean_std(col_data: pd.DataFrame)-> pd.DataFrame:
    """
    使用均值、标准差对数据进行变换

    """

    try:
        mean = col_data.mean()
        std = col_data.std()
        col_data = (col_data - mean) / std
    except TypeError:
        raise TypeError('Data type must be consistent (int or float)')

    return __new_col_data(col_data, 'scale')


def median_abs(col_data: pd.DataFrame)-> pd.DataFrame:
    """
    利用中位数、绝对差对数据进行变换
    相比 mean_std() 更不易受离群点影响

    """

    try:
        med = col_data.median()
        abs_d = sum(abs(col_data - med)) / len(col_data)
        col_data = (col_data - med) / abs_d
    except TypeError:
        raise TypeError('Data type must be consistent (int or float)')

    return __new_col_data(col_data, 'scale')


def non_linear(col_data: pd.DataFrame)-> pd.DataFrame:
    """
    非线性变换: x / (1+x)
    适合对值域较大的数据

    """
    try:
        col_data = col_data.map(lambda x: x * 10 / (1 + x) - 9)
    except TypeError:
        raise TypeError('Data type must be consistent (int or float)')

    return __new_col_data(col_data, 'scale')


def one_step():
    """
    一步计算，利用函数：

        sqrt(x)、log(x)、1/x、x^k、exp(x)、sin(x)、|x|

    对数据进行映射

    :return:
    """
