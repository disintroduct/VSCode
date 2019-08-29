import common_function as cf
import numpy as np
import math
import libim


def readfile(filename):
    fr = open(filename)
    readLines = fr.readlines()
    s = np.alen(readLines)
    n = []
    m = []
    r = []
    # print(readLines)
    k = 1
    for line in readLines:
        # print(line)
        if k == 1:
            k += 1
            continue
        ListFormLine = line.strip()
        ListFormLine = ListFormLine.split(',')
        n.append(int(ListFormLine[0]) - 1)
        m.append(int(ListFormLine[1]) - 1)
        r.append(float(ListFormLine[2]))
    fr.close()
    np.save('E:\\movielens\\n', n)
    np.save('E:\\movielens\\m', m)
    np.save('E:\\movielens\\r', r)
    print('第一步：已获得m，n的值。', s)
    return True


def adjustTheDataset():
    n = np.load('E:\\movielens\\n.npy')
    m = np.load('E:\\movielens\\m.npy')
    r = np.load('E:\\movielens\\r.npy')
    ratings_number = len(n)
    r_arr = np.zeros((ratings_number, 3))
    b = []
    for i in range(ratings_number):
        if n[i] not in b:
            b.append(n[i])
    sum_number = 0
    l_b = len(b)
    n_nb = l_b
    for i in range(ratings_number):
        for j in range(l_b):
            if n[i] == b[j]:
                n[i] = j
                sum_number += 1
                break

    b = []
    for i in range(ratings_number):
        if m[i] not in b:
            b.append(m[i])
    sum_number = 0
    l_b = len(b)
    m_mb = l_b
    for i in range(ratings_number):
        for j in range(l_b):
            if m[i] == b[j]:
                m[i] = j
                sum_number += 1
                break
    print(sum_number)
    np.save('E:\\movielens\\n', n)
    np.save('E:\\movielens\\m', m)
    print(n[:10], m[:10])
    return True


def get_rmse(U, V):
    # r = np.load('E:\\limbi\\n.npy')
    # c = np.load('E:\\limbi\\m.npy')
    # ratings = np.load('E:\\limbi\\r.npy')
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    n = len(r)
    rmse = 0
    i = 0
    for m in range(n):
        i = r[m]
        j = c[m]
        rmse += ((ratings[m] - np.dot(U[i], V[j].T)) ** 2)
    rmse = (rmse / n) ** 0.5
    print(rmse)
    return rmse


def find_n_m():
    n = np.load('E:\\movielens\\n.npy')
    m = np.load('E:\\movielens\\m.npy')
    n0 = 0
    m0 = 0

    for i in range(len(n)):
        if n[i] > n0:
            n0 = n[i]
    for i in range(len(m)):
        if m[i] > m0:
            m0 = m[i]
    print(n0, m0, len(n))
    return n0, m0


def produce_D_D3(m, q):
    D4 = np.random.randn(q, m) / (q ** 0.5)
    D = np.dot(D4, D4.T)
    D1 = np.linalg.inv(D)
    D5 = np.dot(D4.T, D1)
    np.save('E:\\movielens\\Dl1', D4)
    np.save('E:\\movielens\\Dl3', D5)


def machine_learning(m, n, d, sum11, U, V, k):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    # c = np.load('E:\\movielens\column_list.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    sum_ratings = len(ratings)
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    # k = 1
    for it in range(k):
        rt = 1 / (it + 1)
        # 验证实验
        # rt = 1 / (it+1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    dB = dB + dV
                    dV = np.zeros((m, d))
                    i += 1
            else:
                dB = dB + dV
            sum1 += 1
        dV = dB / n
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_0{sum11}', U)
    np.save(f'E:\\movielens\\V_0{sum11}', V)


# 满足差分隐私（改进安全性）
def ML_LDP_prove(m, n, sum_ratings, d, sum11, U, V, eta, k):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    D = np.load('E:\\movielens\\D.npy')
    D3 = np.load('E:\\movielens\\D3.npy')
    # m = 131262
    # n = 138493
    # d = 15
    q = 2700
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    k = 10
    # sum = 20000263
    # eta = 0.1
    eta /= d
    t_sum = q * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_s = math.exp(eta/k)
    for it in range(k):
        rt = 1 / (it + 1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((q, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            # if j >= 0:
            dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum1 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum1] != i:
                    x = np.dot(D, dV)
                    for ls in range(d):
                        s = np.random.randint(0, q)
                        t = x[s][ls]
                        if t > 1:
                            t = 1
                        elif t < -1:
                            t = -1
                        T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                        random_t = np.random.random()
                        if random_t <= T:
                            x[s][ls] = t_sum
                        else:
                            x[s][ls] = -t_sum
                    i += 1
                    dB = dB + x
                    dV = np.zeros((m, d))
            else:
                dB = dB + x
            sum1 += 1
        dB = dB / n
        dU /= n
        dV = np.dot(D3, dB)
        # dV_t = np.dot(D, dV)
        # dV = np.dot(D3, dV_t)
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_1{sum11}', U)
    np.save(f'E:\\movielens\\V_1{sum11}', V)


def ml_LDP(m, n, d, sum11, U, V, eta, k):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    D = np.load('E:\\movielens\\D.npy')
    D3 = np.load('E:\\movielens\\D3.npy')
    sum_ratings = len(ratings)
    # m = 131262
    # n = 138493
    # d = 15
    q = 2700
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    # k = 10
    # sum = 20000263
    # eta = 0.1
    t_sum = q * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_sum_1 = math.exp(eta/k)
    for it in range(k):
        rt = 1 / (it + 1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((q, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            # if j >= 0:
            dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum1 += 1
            if sum1 < sum_ratings:
                try:
                    test = r[sum1]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum1] != i:
                    i += 1
                    x = np.dot(D, dV)
                    s = np.random.randint(0, q)
                    ls = np.random.randint(0, d)
                    t = x[s][ls]
                    if t > 1:
                        t = 1
                    elif t < -1:
                        t = -1
                    T = (t * (t_sum_1 - 1) + t_sum_1 + 1) / (2 * (t_sum_1 + 1))
                    random_t = np.random.random()
                    if random_t <= T:
                        x[s][ls] = t_sum
                    else:
                        x[s][ls] = -t_sum
                    dB = dB + x
                    dV = np.zeros((m, d))
            else:
                dB = dB + x
        dB = dB / n
        dU /= n
        dV = np.dot(D3, dB)
        # dV_t = np.dot(D, dV)
        # dV = np.dot(D3, dV_t)
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_2{sum11}', U)
    np.save(f'E:\\movielens\\V_2{sum11}', V)


def ml_LDP_hau(sum11, U, V, eta, m, n, d, k):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    sum_ratings = len(r)
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    # k = 10
    # eta = 0.1
    t_sum = m * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_sum_1 = math.exp(eta/k)
    for it in range(k):
        rt = 1 / (it + 1)
        # rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            # if j >= 0:
            dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    s = np.random.randint(0, m)
                    ls = np.random.randint(0, d)
                    t = dV[s][ls]
                    if t > 1:
                        t = 1
                    elif t < -1:
                        t = -1
                    T = (t * (t_sum_1 - 1) + t_sum_1 + 1) / (2 * (t_sum_1 + 1))
                    random_t = np.random.random()
                    if random_t <= T:
                        dB[s][ls] += t_sum
                    else:
                        dB[s][ls] += -t_sum
                    dV = np.zeros((m, d))
                    i += 1
            else:
                s = np.random.randint(0, m)
                ls = np.random.randint(0, d)
                t = dV[s][ls]
                if t > 1:
                    t = 1
                elif t < -1:
                    t = -1
                T = (t * (t_sum_1 - 1) + t_sum_1 + 1) / (2 * (t_sum_1 + 1))
                random_t = np.random.random()
                if random_t <= T:
                    dB[s][ls] += t_sum
                else:
                    dB[s][ls] += -t_sum
                break
            sum1 += 1
        dV = dB / n
        dU /= n
        # dV_t = np.dot(D, dV)
        # dV = np.dot(D3, dV_t)
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_3{sum11}', U)
    np.save(f'E:\\movielens\\V_3{sum11}', V)


def ml_LDP_NJ(sum11, U, V, eta, m, n, sum, d):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    # c = np.load('E:\\movielens\column_list.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    lu = 10 ** -8
    lv = 10 ** -8
    k = 10
    # k = 1
    for it in range(k):
        # rt = 1 / (it + 1)
        rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
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
                    print(er, test)
                    break
                if r[sum1] != i:
                    i += 1
                    dB = dB + dV
                    dV = np.zeros((m, d))
            else:
                dB = dB + dV
        dV = dB / n
        dV = dV + np.random.laplace(0, 10 * (d ** 0.5) / eta, (m, d))
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_4{sum11}', U)
    np.save(f'E:\\movielens\\V_4{sum11}', V)


def s_items(m, n, d, sum11, U, V, eta, k, l_i):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    t_sum = math.exp(eta/(2*l_i))
    sum_ratings = len(r)
    cf.show_time()
    sum1 = 0
    user_items = []
    sum_items = []
    for ii in range(m):
        sum_items.append(ii)
    other_items = sum_items[:]
    s_i = []
    l_array = np.zeros((n, l_i))
    i = 0
    while sum1 < sum_ratings and i < n:
        j = c[sum1]
        user_items.append(j)
        other_items.remove(j)
        sum2 = sum1 + 1
        if sum2 < sum_ratings and i < n:
            try:
                test = r[sum1]
            except Exception as er:
                print(er, test)
                break
            if r[sum2] != i:
                s = len(user_items)
                if s >= l_i:
                    if s > l_i:
                        for o in range(s - l_i):
                            l_o = len(user_items)
                            rand_number = np.random.randint(0, l_o-1)
                            del user_items[rand_number]
                else:
                    for o in range(l_i-s):
                        l_o = len(other_items)
                        tt = np.random.randint(0, l_o - 1)
                        user_items.append(other_items[tt])
                        del other_items[tt]
                for o in range(l_i):
                    rand_number = np.random.random()
                    if rand_number <= t_sum / (t_sum + 1):
                        s_i.append(user_items[o])
                # l_other = s_r(0, m-l_i-1, l_i-len(s_i))
                for o in range(l_i-len(s_i)):
                    l_o = len(other_items)
                    tt = np.random.randint(0, l_o - 1)
                    s_i.append(other_items[tt])
                    del other_items[tt]
                s_i.sort()
                l_array[i] = s_i
                i += 1
                other_items = sum_items[:]
                s_i = []
                user_items = []
        sum1 += 1
    cf.show_time()
    print(l_array[0])
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\l_array{sum11}', l_array)


def get_flag(m, n, sum11, l_array):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    sum_ratings = len(r)
    c_flag = []
    for i in range(len(r)):
        c_flag.append(1)
    cf.show_time()
    sum1 = 0
    i = 0
    for sum1 in range(sum_ratings):
        if r[sum1] == i:
            if c[sum1] in l_array[i]:
                c_flag[sum1] = 1
            else:
                c_flag[sum1] = 0
        else:
            i += 1
            if c[sum1] in l_array[i]:
                c_flag[sum1] = 1
            else:
                c_flag[sum1] = 0
    cf.show_time()
    print(len(c_flag))
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\c_flag{sum11}', c_flag)


# 满足差分隐私（改进实用性）削减参加计算的个人用户物品梯度
def ML_LDP_part(m, n, d, sum11, U, V, eta, k, c_flag, l_array, r_q):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    D = np.load('E:\\movielens\\Dl1.npy')
    D3 = np.load('E:\\movielens\\Dl3.npy')
    n, l_items = np.shape(l_array)
    q = r_q
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    sum_ratings = len(r)
    eta /= d
    t_sum = q * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_s = math.exp(eta/k)
    print(l_array[:10][:10], c_flag[:10])
    for it in range(k):
        # rt = 1 / (it + 1) / (k ** 2)
        # rt = 1 / (it + 1)
        rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        dl = np.zeros((l_items, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if c_flag[sum1] == 1:
                dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    for o in range(l_items):
                        tts = int(l_array[i][o])
                        dl[o] = dV[tts]
                    x = np.dot(D, dl)
                    for ls in range(d):
                        s = np.random.randint(0, q)
                        t = x[s][ls]
                        if t > 1:
                            t = 1
                        elif t < -1:
                            t = -1
                        T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                        random_t = np.random.random()
                        if random_t <= T:
                            x[s][ls] = t_sum
                        else:
                            x[s][ls] = -t_sum
                    x_z = np.dot(D3, x)
                    for o in range(l_items):
                        tts = int(l_array[i][o])
                        dB[tts] += x_z[o]
                    dV = np.zeros((m, d))
                    i += 1
            else:
                for o in range(l_items):
                    tts = int(l_array[i][o])
                    dl[o] = dV[tts]
                x = np.dot(D, dl)
                for ls in range(d):
                    s = np.random.randint(0, q)
                    t = x[s][ls]
                    if t > 1:
                        t = 1
                    elif t < -1:
                        t = -1
                    T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                    random_t = np.random.random()
                    if random_t <= T:
                        x[s][ls] = t_sum
                    else:
                        x[s][ls] = -t_sum
                x_z = np.dot(D3, x)
                for o in range(l_items):
                    tts = int(l_array[i][o])
                    dB[tts] += x_z[o]
                dV = np.zeros((m, d))
                i += 1
            sum1 += 1
        dV = dB / n
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_5{sum11}', U)
    np.save(f'E:\\movielens\\V_5{sum11}', V)


# 满足差分隐私（改进实用性）削减参加计算的个人用户物品梯度
def ML_LDP_part_GD(m, n, d, sum11, U, V, eta, k, c_flag, l_array, r_q):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    D = np.load('E:\\movielens\\Dl1.npy')
    D3 = np.load('E:\\movielens\\Dl3.npy')
    n, l_items = np.shape(l_array)
    q = r_q
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    sum_ratings = len(r)
    eta /= d
    t_sum = q * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_s = math.exp(eta/k)
    # print(l_array[:10][:10], c_flag[:10])
    for it in range(k):
        # rt = 1 / (it + 1) / (k ** 2)
        # rt = 1 / (it + 1)
        rt = 1 / (it + 1) / k
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        dl = np.zeros((l_items, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if c_flag[sum1] == 1:
                dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    for o in range(l_items):
                        tts = int(l_array[i][o])
                        dl[o] = dV[tts]
                    x = np.dot(D, dl)
                    for ls in range(d):
                        s = np.random.randint(0, q)
                        t = x[s][ls]
                        if t > 1:
                            t = 1
                        elif t < -1:
                            t = -1
                        T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                        random_t = np.random.random()
                        if random_t <= T:
                            x[s][ls] = t_sum
                        else:
                            x[s][ls] = -t_sum
                    x_z = np.dot(D3, x)
                    for o in range(l_items):
                        tts = int(l_array[i][o])
                        dB[tts] += x_z[o]
                    dV = np.zeros((m, d))
                    i += 1
            else:
                for o in range(l_items):
                    tts = int(l_array[i][o])
                    dl[o] = dV[tts]
                x = np.dot(D, dl)
                for ls in range(d):
                    s = np.random.randint(0, q)
                    t = x[s][ls]
                    if t > 1:
                        t = 1
                    elif t < -1:
                        t = -1
                    T = (t * (t_s - 1) + t_s + 1) / (2 * (t_s + 1))
                    random_t = np.random.random()
                    if random_t <= T:
                        x[s][ls] = t_sum
                    else:
                        x[s][ls] = -t_sum
                x_z = np.dot(D3, x)
                for o in range(l_items):
                    tts = int(l_array[i][o])
                    dB[tts] += x_z[o]
                dV = np.zeros((m, d))
                i += 1
            sum1 += 1
        dV = dB / n
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_6{sum11}', U)
    np.save(f'E:\\movielens\\V_6{sum11}', V)


# 获得预测百分比
def get_percent(U, V, sum_ratings):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    # c = np.load('E:\movielens\column_list.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    error_list = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    sum1 = 0
    i = 0
    while sum1 < sum_ratings:
        j = c[sum1]
        T = np.dot(U[i], V[j].T)
        sum1 += 1
        if abs(ratings[sum1]-T) < 0.5:
            error_list[0] += 1
        elif abs(ratings[sum1]-T) < 1:
            error_list[1] += 1
        elif abs(ratings[sum1]-T) < 1.5:
            error_list[2] += 1
        elif abs(ratings[sum1]-T) < 2:
            error_list[3] += 1
        elif abs(ratings[sum1]-T) < 2.5:
            error_list[4] += 1
        elif abs(ratings[sum1]-T) < 3:
            error_list[5] += 1
        elif abs(ratings[sum1]-T) < 3.5:
            error_list[6] += 1
        elif abs(ratings[sum1]-T) < 4:
            error_list[7] += 1
        elif abs(ratings[sum1]-T) < 4.5:
            error_list[8] += 1
        elif abs(ratings[sum1]-T) < 5:
            error_list[9] += 1
        if sum1 < sum_ratings:
            try:
                test = r[sum1]
            except Exception as er:
                print(er, test)
                break
            if r[sum1] != i:
                i += 1
        else:
            for i in range(11):
                error_list[i+1] += error_list[i]
            for i in range(12):
                error_list[i] = error_list[i] / sum_ratings
    print(error_list)


def LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q):
    # use ml_LDP_part algrithm
    # produce_D_D3(l_i, l_q)
    for j in range(2):
        i = j + 5
        eta = eta_list[i] / 2
        s_items(m, n, d, i, u, v, eta, k, l_i)
        print('get s_items')
        l_array = np.load(f'E:\\movielens\\l_array{i}.npy')
        get_flag(m, n, i, l_array)
        print('get_flag')
        c_flag = np.load(f'E:\\movielens\\c_flag{i}.npy')
        # ML_LDP_part(m, n, d, i, u, v, eta, k, c_flag, l_array, l_q)
        ML_LDP_part_GD(m, n, d, i, u, v, eta, k, c_flag, l_array, l_q)
        print('get the granted')


def ldp_iteration(U, V, eta_list, m, n, d, k):
    for j in range(1):
        i = j + 5
        ml_LDP_hau(i, U, V, eta_list[i], m, n, d, k)


def libim_funcation():
    eta_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    # n, m = find_n_m()
    # d_limbi(m, n)
    # n, m = find_n_m()
    n = 135359
    m = 26509
    d = 20
    # q = 2700
    k = 10
    l_i = 50
    l_q = 5
    # second step
    # u = np.random.rand(n, d)
    # v = np.random.rand(m, d)
    u = np.load('E:\\limbi\\u.npy')
    v = np.load('E:\\limbi\\v.npy')
    # get_vmax()
    # np.save('E:\\limbi\\u', u)
    # np.save('E:\\limbi\\v', v)
    # produce_D_D3(m, q)
    # machine_learning(m, n, d, 0, u, v, k)
    libim.LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q)
    libim.ldp_iteration(u, v, eta_list, m, n, d)


def number_of_iterations(m, n, d, u, v, l_q):
    k_list = [1, 2, 3, 4, 5, 20, 50]
    eta = 0.1
    l_array = np.load('E:\\movielens\\l_array1.npy')
    c_flag = np.load('E:\\movielens\\c_flag1.npy')
    for i in range(7):
        j = i + 6
        machine_learning(m, n, d, j, u, v, k_list[i])
        # ml_LDP(m, n, d, j, u, v, eta, k_list[i])
        ml_LDP_hau(j, u, v, eta, m, n, d, k_list[i])
        ML_LDP_part_GD(m, n, d, j, u, v, 0.05, k_list[i], c_flag, l_array, l_q)
        ML_LDP_part(m, n, d, j, u, v, eta/2, k_list[i], c_flag, l_array, l_q)


def get_vmax(ss):
    v = np.load('E:\\limbi\\v_00.npy')
    vmax = []
    rmse = []
    for i in range(6):
        j = i
        # ML_LDP_prove(m, n, sum_ratings, d, i, u, v, eta_list[i], k)
        # ml_LDP(m, n, d, j, u, v, eta_list[j])
        # ml_LDP_hau(i, u, v, eta_list[i], m, n, sum_ratings, d)
        # ml_LDP_NJ(i, u, v, eta_list[i], m, n, sum_ratings, d)
        v1 = np.load(f'E:\\limbi\\V_{ss}{j}.npy')
        u1 = np.load(f'E:\\limbi\\U_{ss}{j}.npy')
        rmse_u = get_rmse(u1, v1)
        rmse.append(rmse_u)
        temp = np.linalg.norm(v-v1, np.inf)
        vmax.append(temp)
        print('I have finish the epslion of ', ss)
    print(vmax)
    print(rmse)


if __name__ == '__main__':
    cf.show_title()

    q = 2700
    sum_ratings = 20000264
    m = 26744
    n = 138493
    d = 15
    l_i = 50
    l_q = 5
    k = 10
    lambda_u = 10 ** -8
    lambda_v = 10 ** -8
    eta_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4]
    u = np.load('E:\\movielens\\u.npy')
    v = np.load('E:\\movielens\\v.npy')

    number_of_iterations(m, n, d, u, v, l_q)

    # first step
    # readfile('E:\\movielens\\ratings.txt')
    # adjustTheDataset()
    # find_n_m()

    # second step
    # u = np.random.rand(n, d)
    # v = np.random.rand(m, d)
    # np.save('E:\\movielens\\u', u)
    # np.save('E:\\movielens\\v', v)
    # produce_D_D3(l_i, l_q)
    # machine_learning(m, n, sum_ratings, d, 0, u, v, k)

    # LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q)
    # get_rmse(u, v)
    # s_items(m, n, d, 0, u, v, 0.025, k, l_i)

    # ldp_iteration(u, v, eta_list, m, n, d)
    # libim_funcation()

    # use ml_LDP_part algrithm
    # produce_D_D3(l_i, l_q)

    # m_list = np.load('E:\\movielens\\m.npy')
    # print(m_list[:10])

    # ML_LDP_part(m, n, d, 1, u, v, 0.05, k, c_flag, l_array, l_q)

    # 检验实验精准性原因
    # machine_learning(m, n, d, 1, u, v, k)
