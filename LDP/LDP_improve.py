import common_function
from numpy import *
import numpy as np
import scipy.sparse
from scipy import sparse


def machine_learning(sum11, U, V, r, c, ratings, m, n, d, lu, lv, k, sum, D, D3):
    r = np.load('D:\Labor\\n.npy')
    c = np.load('D:\Labor\m.npy')
    ratings = np.load('D:\Labor\\r.npy')
    D = np.load('D:\Labor\D.npy')
    D3 = np.load('D:\Labor\D3.npy')
    m = 131262
    # n = 138493
    n = 138493
    d = 15
    lu = 10 ** -8
    lv = 10 ** -8
    k = 10
    sum = 20000263
    n_number = 1000
    # U = load("D:\\U8.npy")
    # print(U[:3], V[:3])
    # V = load('D:\V8.npy')
    for it in range(k):
        rt = 1 / (it + 1)
        print(str(it) + ' iterations！')
        common_function.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))

        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum1 += 1
            if sum1 < sum:
                try:
                    test = r[sum1]
                except Exception as er:
                    print(er)
                    break
                if r[sum1] != i:
                    i += 1
                    dB = dB + dV
                    # laplace_n = np.random.laplace(0, 10 * 15 / sum11, (m, d))
                    dV = np.zeros((m, d), float16)
            else:
                dB = dB + dV
        laplace_n = np.random.laplace(0, 10 * 10 / (20 - sum11 * 2), (m, d))
        dV = dB / n + laplace_n
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        common_function.show_time()
    print('\n' + f'I have done it! {sum11}')
    save(f'D:\\U7_{sum11}', U)
    save(f'D:\V7_{sum11}', V)

if __name__ == '__main__':
    print('This is the improvement of least progress!')
    
    r = np.load('D:\Labor\\n.npy')
    c = np.load('D:\Labor\m.npy')
    ratings = np.load('D:\Labor\\r.npy')
    # D = np.load('D:\Labor\D.npy')
    # D3 = np.load('D:\Labor\D3.npy')
    m = 131262
    # n = 138493
    n = 138493
    d = 15
    lu = 10 ** -8
    lv = 10 ** -8
    k = 10
    sum = 20000263
    n_number = 1000
    # U = load("D:\\U8.npy")
    # print(U[:3], V[:3])
    # V = load('D:\V8.npy')

