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

'''
eta = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
GD = [1.23, 0.89, 0.32, 0.11, 0.07, 0.004]
zh = [1.25, 0.91, 0.33, 0.115, 0.078, 0.006]
'''

xiao_vmax = [0.432497, 0.454638, 0.460792, 0.459258, 0.455029, 0.456823]
xiao_vmax = [10.788110, 10.773104, 10.785208, 10.781518, 10.784362, 10.782547]
zh_vmax = [10.355737, 10.355737, 10.432995, 10.634897, 10.698646, 10.742953]
hau_vmax = [810.305505, 429.136783, 213.522255, 110.958499, 52.714707, 25.285056]
nj_vmax = [508.179122, 234.613983, 126.588237, 63.876417, 31.134939, 16.411785]


xiao_rmse = [3.395099, 3.395170, 3.395200, 3.395222, 3.395212, 3.395209]
zh_rmse = [3.388192, 3.391347, 3.393436, 3.394189, 3.394704, 3.394929]
hau_rmse = [57.371773, 30.988090, 16.319872, 8.664005, 5.227499, 3.902674]
nj_rmse = [38.593371, 20.194134, 10.673008, 6.096408, 4.217834, 3.617418]

eta = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]

'''
eta = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
GD = [1.6132, 1.5030, 1.4872, 1.4735, 1.4713, 1.4711]
zh = [1.6353, 1.5324, 1.5019, 1.4867, 1.4798, 1.4723]
'''
# eta = [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]
NP = [0.1552262, 0.2645495, 0.38532871, 0.48720538, 0.593800385, 0.6697157645, 0.7358, 0.79953003, 0.84829055, 0.88936705, 0.88936705, 0.88936705]
GD = [0.13062, 0.2571122, 0.37589175, 0.4842243, 0.5809523, 0.66518835, 0.73719555, 0.7972491, 0.84640155, 0.88572215, 0.88572215, 0.88572215]
zh = [0.128946, 0.2539647, 0.3720439, 0.4807164, 0.57781795, 0.66203095, 0.73372285, 0.79365545, 0.84299775, 0.88263945, 0.88263945, 0.88263945]
plt.figure(dpi=128, figsize=(10, 6))
# plt.plot(Vmax_random0, label='$Epsilon = 0.1$', linestyle='-')

# plt.legend('Epsilon = 1')
# plt.plot(p4, 'b--', label='$Epsilon = 1/(i*(i+1)) with correction factor i*(i+1)$')
# plt.plot(p5, label='$Epsilon = 1/(i*(i+1)) with correction factor i^2 *(i+1)$')
# plt.plot(Vmax_random2, label='$Epsilon = 1$')

plt.plot(eta, xiao_vmax, 'b--', label='$xiao$')
plt.plot(eta, zh_vmax, ':', label='$zh$')
plt.plot(eta, hau_vmax, '-.', label='$hau$')
plt.plot(eta, nj_vmax, '-', label='$NJ$')

#plt.plot(x, zh_prediction_2, label='$Epsilon = 0.6$')
#plt.plot(x, zh_prediction_3, label='$Epsilon = 1$')

# plt.plot(x, eta_10, label='$Epsilon = 10$')
# plt.plot(x, frency_NP, label='$NP$')
# plt.plot(x, eta_0_1, label='$Epsilon = 0.1$')
# plt.plot(x, eta_1, label='$Epsilon = 1$')
# plt.plot(x, random_select, label='$Random Select$')

# plt.plot(pe4_new, label='$CF 1/(i*(i+1)) with new_U$')
# plt.legend('Epsilon = 1')
# plt.xlim(0, 10000)
# plt.ylim(0.6, 1)
plt.xlabel('Vmax')
plt.ylabel('Epsilon')
plt.legend()

plt.draw()
plt.pause(10)

plt.savefig('F:\\50.png')

plt.close()
