import iterationRelated as iR
import numpy as np 


def LibimSeTi():
    eta_list = [0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    # n, m = find_n_m()
    # d_limbi(m, n)
    # n, m = find_n_m()
    index = 'E:\\limbi\\'
    n = 135359
    m = 26509
    d = 20
    lu = 10 ** -8
    lv = 10 ** -8
    q = 2700
    k = 10
    l_i = 50
    l_q = 5
    # second step
    # u = np.random.rand(n, d)
    # v = np.random.rand(m, d)
    r = np.load(index + 'n.npy')
    c = np.load(index + 'm.npy')
    ratings = np.load(index + 'r.npy')
    u = np.load(index + 'u.npy')
    v = np.load(index + 'v.npy')
    for i in range(6):
        if i > 0:
            iR.get_items(r, c, ratings, m, n, i, k, index, eta_list[i])
        iR.get_granted(r, c, ratings, m, n, d, i, u, v, k, index, eta_list[i], 1, 0)
        iR.get_granted(r, c, ratings, m, n, d, i, u, v, k, index, eta_list[i], k/eta_list[i], 2)


if __name__ == "__main__":
    LibimSeTi()
    # 思路：1.上传物品梯度最大为k个，故可以先选l个备选远大于k，每次挑一个，但是噪声都为k的时候的噪声
    # 2.每次不放会抽样，抽k次，每次不同的物品梯度
