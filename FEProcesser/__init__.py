import logging

import jieba

from .FEProcess import DiscreteProcess
from .FEProcess import EncodeProcess
from .FEProcess import TextProcess
from .FEProcess import DateProcess
from .FEProcess import MapProcess

from .FEProcess import FeatureSimilar


jieba.setLogLevel(logging.INFO)


__all__ = ['DiscreteProcess', 'EncodeProcess', 'TextProcess', 'DateProcess', 'MapProcess',
           'FeatureSimilar']

