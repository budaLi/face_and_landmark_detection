# @Time    : 2020/5/11 17:51
# @Author  : Libuda
# @FileName: word_embedding_live.py
# @Software: PyCharm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as tud
from collections import Container
import numpy as np
import random
import math
import pandas as pd
import scipy
import sklearn
from sklearn.metrics.pairwise import cosine_distances

# 固定随机数 使模型训练的结果尽可能一直
random.seed(1)
np.random.seed(1)
torch.manual_seed(1)

# 周围三个单词
C = 3
# 每出现一个正常的词就有100个错误的词
K =100
# 最终这个词汇表有多大
MAX_VOCAB_SIZE = 30000
BATCH_SIZE = 128
LEARNING_RATE = 0.2
EMBEDDING_SIZE = 100

def work_tokenize(text):
    """
    把文件变成一个一个单词
    :param text:
    :return:
    """
    return text.split()


