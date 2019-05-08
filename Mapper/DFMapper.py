import pandas as pd


def key_value(data: pd.DataFrame, col: str, map_dict: dict, layer=False, drop=True) -> pd.DataFrame:
    """
    将数据中某一列数据根据给定的映射字典进行替换
    具有两种映射形式：普通映射、分层映射

    :param data: 原始完整 DataFrame 数据
    :param col: 需要使用序号编码表示的列名称
    :param map_dict: 指定映射字典
    :param layer: 是否分层映射，如果 True，则 map_dict 中的 values 应为可迭代对象
    :param drop: 是否删除原列数据
    :return: 替换后的 DataFrame 数据

    例：
        1) 普通映射 传入的字典: {'jack': 'a',
                             'jerry': 'b',
                             'tom': 'c'}
           以上键值一一对应，一对一进行映射

        2) 分层映射 传入的字典：{'a': ['jack', 'jerry'],
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

    return __concat_data(drop, data, col_data, col)


def max_min():
    """
    最大-最小归一化

    :return:
    """


def mean_std():
    """
    使用均值、标准差对数据进行变换

    :return:
    """


def median_abs():
    """
    利用中位数、绝对标准差对数据进行变换

    :return:
    """


def non_linear():
    """
    非线性变换

    :return:
    """


def one_step():
    """
    一步计算，利用函数：

        sqrt(x)、log(x)、1/x、x^k、exp(x)、sin(x)、|x|

    对数据进行映射

    :return:
    """


def __concat_data(drop: bool, data: pd.DataFrame, new_data: pd.DataFrame, col: str) -> pd.DataFrame:
    result = pd.concat([data, new_data], axis=1)
    if drop:
        return result.drop(columns=[col])
    else:
        return result
