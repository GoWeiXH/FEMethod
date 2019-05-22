from collections import Counter
import math
import re

from sklearn.feature_extraction import text
import pandas as pd
import numpy as np
import jieba


class MapProcess:

    def key_value(self, col_data: pd.Series, map_dict: dict, layer=False) -> pd.DataFrame:
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
                    raise ValueError('The value of map_dict must be a list')

            map_dict = new_dict

        default_value = 'no_mapped'
        for k in set(col_data):
            map_dict.setdefault(k, default_value)

        col_data = col_data.map(map_dict)

        return _new_col_data(col_data, 'mapped')

    def max_min(self, col_data: pd.Series) -> pd.DataFrame:
        """
        最大-最小归一化

        """

        max_v = col_data.max()
        min_v = col_data.min()
        d = max_v - min_v
        col_data = (col_data - min_v) / d

        return _new_col_data(col_data, 'scale')

    def mean_std(self, col_data: pd.Series) -> pd.DataFrame:
        """
        使用均值、标准差对数据进行变换

        """

        mean = col_data.mean()
        std = col_data.std()
        col_data = (col_data - mean) / std

        return _new_col_data(col_data, 'scale')

    def median_abs(self, col_data: pd.Series) -> pd.DataFrame:
        """
        利用中位数、绝对差对数据进行变换
        相比 mean_std() 更不易受离群点影响

        """

        med = col_data.median()
        abs_d = sum(abs(col_data - med)) / len(col_data)
        col_data = (col_data - med) / abs_d

        return _new_col_data(col_data, 'scale')

    def non_linear(self, col_data: pd.Series) -> pd.DataFrame:
        """
        非线性变换: x / (1+x)
        适合对值域较大的数据

        """

        col_data = col_data.map(lambda x: x * 10 / (1 + x) - 9)

        return _new_col_data(col_data, 'scale')

    # 对所有类方法进行异常捕获
    # def __getattribute__(self, func):
    #     def wrapper(*args, **kwargs):
    #         try:
    #             r = eval('Map.' + func)(self, *args, **kwargs)
    #             return r
    #         except TypeError as e:
    #             raise TypeError(f'Data type must be consistent, {e}')
    #         except Exception as e:
    #             raise e
    #
    #     return wrapper


class EncodeProcess:

    def ordinary(self, col_data: pd.Series, step=1, reverse=False) -> pd.DataFrame:
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
        k_dict = dict(zip(unique, range(0, step * len(unique), step)))

        # 对数据进行替换
        col_data = col_data.map(k_dict)

        return _new_col_data(col_data, 'ord')

    def binary(self, col_data: pd.Series, reverse=False) -> pd.DataFrame:
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

    def one_hot(self, col_data: pd.Series, engine='pd') -> pd.DataFrame:
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

    def ratio(self, col_data: pd.Series, n_digits=3) -> pd.DataFrame:
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
        return _new_col_data(col_data, 'ratio')


class DiscreteProcess:

    def points(self, col_data: pd.Series, ps: list, **kwargs) -> pd.DataFrame:
        """
        规则硬分，根据给定分割点进行划分

        :param col_data: 需要离散化的列数据
        :param ps: 包含切分点的 list
        :return: 离散化后 DataFrame 数据
        """
        ps = [min(col_data) - 0.001] + ps + [max(col_data)]

        def to_range(x):
            for i in range(len(ps) - 1):
                if ps[i] < x <= ps[i + 1]:
                    if labels or labels is None:
                        return f'[{ps[i]}, {ps[i + 1]})'
                    else:
                        return i

        labels = kwargs.get('labels')
        points_data = col_data.map(to_range)

        return _new_col_data(points_data, 'cut')

    def freq(self, col_data: pd.Series, n_bins: int, **kwargs) -> pd.DataFrame:
        """
        等频分箱，各离散值拥有相等数量的样本

        :param col_data: 需要离散化的列数据
        :param n_bins: 要求划分成的类别个数
        :return: 离散化后 DataFrame 数据
        """

        freq_data = pd.qcut(col_data, n_bins, **kwargs)
        return _new_col_data(freq_data, 'cut')

    def width(self, col_data: pd.Series, n_bins: int, **kwargs) -> pd.DataFrame:
        """
        等宽分箱，将取值区间均分为若干个子区间

        :param col_data: 需要离散化的列数据
        :param n_bins: 要求划分成的类别个数
        :return: 离散化后 DataFrame 数据
        """

        width_data = pd.cut(col_data, n_bins, **kwargs)
        return _new_col_data(width_data, 'cut')

    def hashtable(self, ):
        """
        Hash分桶，利用 Hash table 方法离散数据

        :return:
        """

    def cluster(self, ):
        """
        使用一维聚类算法对数据进行离散化

        :return:
        """


class TextProcess:

    punctuation = ",.;:/!?'`\"<>{}[]()" + "，。、；？！《》【】’‘“”：·…—"
    blank = [' ', '\n', '\r', '\r\n', '\t']

    def n_gram(self, corpus: list, n=(1, 1), encode='count', lang='CN', *args, **kwargs) -> pd.DataFrame:
        """
        n-gram 转换器

        :param corpus: 需要转换的数据，['sent_0', 'sent_1', ...]
        :param n: 指定 n-gram 范围，（1，2）表示既有 1-gram 也有 2-gram
                                   （1，1）表示只有 1-gram
        :param encode: 指定使用的编码方式，默认 'count'
                            'count' 以 出现次数 表示每个词，并句子向量；
                            'tf-idf' 以各句子各词的 tf-idf 值表示，并在组成句子向量
        :param lang: 所面向的语言，默认 ‘CN' 中文
        :return: 拆分表示后的 DataFrame 数据
        """

        # 初始化
        count = text.CountVectorizer(ngram_range=n, *args, **kwargs)
        tf_idf = text.TfidfVectorizer(ngram_range=n, *args, **kwargs)
        encode_dict = {'count': count, 'tf-idf': tf_idf}
        n_gram_worker = encode_dict.get(encode, count)

        if lang == 'CN':
            # sklearn 中只接受 str，并按 ‘空格’ 切分句子
            # 所以对于中文来讲，先使用 jieba 分词，再用 ‘空格’ 拼接
            corpus = [' '.join(jieba.cut(sent)) for sent in corpus]

        # 转换数据，获取字段名称
        new_data = n_gram_worker.fit_transform(corpus)
        columns = n_gram_worker.get_feature_names()

        # 构建新数据
        new_data = pd.DataFrame(new_data.toarray())
        new_data.columns = columns

        return new_data

    def del_stop_word(self, ori_sent: list, stop_words: list) -> list:
        """
        剔除停用词
        """

        clear_sent = [w for w in ori_sent if w not in stop_words]

        return clear_sent

    def del_punctuation(self, ori_sent: list, punctuation='') -> list:
        """
        剔除标点
        """
        if punctuation != '':
            lib = punctuation
        else:
            lib = self.punctuation

        lib = lib.split()
        print('Delete items:', self.punctuation)
        clear_sent = [w for w in ori_sent if w not in lib]

        return clear_sent

    def del_blank(self, ori_sent: list) -> list:
        """
        去除空格、回车 \r\n
        """
        print('Delete items:', self.blank)
        lib = self.blank
        clear_sent = [w for w in ori_sent if w not in lib]

        return clear_sent

    def word2v(self, ):
        ...

    def wv2sv(self, ):
        """
        词向量转换成句子向量

        :return:
        """


class DateProcess:

    def extract(self, subject):
        """
        subject:year/month/day/hour/min

        :return:
        """

    def season(self, ):
        """

        :return:
        """

    def weekend(self, ):
        """

        :return:
        """

    def segment(self, ):
        """
        rule: 提供时段规则

        :return:
        """

    def tradition_hour(self, ):
        """
        十二时辰

        :return:
        """


class FeatureSimilar:

    def smc_sim(self, data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
        """
        SMC (Simple Match Coefficient) 简单匹配系数

        针对二值数据 (1,0) 计算相似度

        jc = f_11 + f_00 / (f_10 + f_01 + f_11 + f_00)

        f_11、f_00: 取值相同
        f_01、f_10: 取值不同
        """

        for data in (data_1, data_2):
            if set(Counter(data)) - {0, 1}:
                raise ValueError(f"Make sure the value of data is 0 or 1: column['{data.name}']")

        result = data_1 + data_2

        counts = result.value_counts()
        f_00 = counts[0]
        f_11 = counts[2]
        f_10_01 = counts[1]

        smc = f_11 / (f_00 + f_10_01)
        return round(smc, digits)

    def jcd_sim(self, data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
        """
        Jaccard 系数，忽略二者相等且为 0 的数据，可以应用于稀疏数据。

        针对二值数据 (1,0) 计算相似度。

        jc = f_11 / (f_10 + f_01 + f_11)

        f_11: 取值全为 1
        f_01、f_10: 取值不同

        完全相同则 Jaccard 系数为 1，完全不同则为 0；
        """

        for data in (data_1, data_2):
            if set(Counter(data)) - {0, 1}:
                raise ValueError(f"Make sure the value of data is 0 or 1: column['{data.name}']")

        result = data_1 + data_2

        counts = result.value_counts()
        f_11 = counts[2]
        f_10_01 = counts[1]

        smc = f_11 / (f_10_01 + f_11)
        return round(smc, digits)

    def jcd_dis(self, data_1: pd.Series, data_2: pd.Series, digits=4) -> float:

        return 1 - self.jcd_sim(data_1, data_2, digits)

    def l1_dis(self, data_1: pd.Series, data_2: pd.Series) -> float:
        """
        曼哈顿距离

        距离易受取值范围较大特征的影响，应注意数据规范化。
        """

        return float(np.sum(np.abs(data_1 - data_2)))

    def l2_dis(self, data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
        """
        欧氏距离

        距离易受取值范围较大特征的影响，应注意数据规范化。

        dis_l2 = sqrt(sum((data_1 - data_2)**2))
        """

        return round(math.sqrt(np.sum((data_1 - data_2)**2)), digits)

    def pearson(self, data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
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

    def spearman(self, data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
        """
        Spearman 相关系数

        如果两个变量存在秩次，则

            sc = 1 - 6*sum((x-y)^2) / n(n^2-1)

        如果不存在秩次关系，仍然使用 pearson 系数计算
        """

        return round(data_1.corr(data_2, method='spearman'), digits)

    def cosine(self, data_1: pd.Series, data_2: pd.Series, digits=4) -> float:
        """
                余弦相似度

                cos = A·B / ||A||·||B||
                """

        dot = np.dot(data_1, data_2)
        data_1_l2 = np.sqrt(np.sum(np.square(data_1)))
        data_2_l2 = np.sqrt(np.sum(np.square(data_2)))
        cos = dot / (data_1_l2 * data_2_l2)

        return round(cos, digits)


def _new_col_data(col_data: pd.Series, func_name: str) -> pd.DataFrame:
    """
    对新数据修改列名称

    """
    col_name = getattr(col_data, 'name')
    new_data = pd.DataFrame(col_data)
    new_data.columns = [f'{col_name}_{func_name}']
    return new_data
