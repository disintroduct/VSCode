from numpy import *
import time
from numpy import linalg as la
import numpy as np
from scipy import sparse
from numpy.linalg import cholesky
import matplotlib.pyplot as plt
from scipy.stats import binom
import math
import random
import numpy.linalg as nlg


path = u'F:\Code\VSCode\python\MM\svm1.txt'  # 数据文件路径
data = np.loadtxt(path, dtype=float, delimiter='\t')
y, x = np.split(data, (1,), axis=1)
print(x, y)
x1 = x[:].T
x = x[:, 1:]