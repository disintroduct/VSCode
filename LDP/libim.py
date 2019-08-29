import common_function as cf
import numpy as np
import improve
import math


def readfile(filename):
    fr = open(filename)
    readLines = fr.readlines()
    s = np.alen(readLines)
    n = []
    m = []
    r = []
    # print(readLines)
    for line in readLines:
        # print(line)
        ListFormLine = line.strip()
        ListFormLine = ListFormLine.split(',')
        n.append(int(ListFormLine[0]) - 1)
        m.append(int(ListFormLine[1]) - 1)
        r.append(int(ListFormLine[2]))
    fr.close()
    np.save('E:\\limbi\\n', n)
    np.save('E:\\limbi\\m', m)
    np.save('E:\\limbi\\r', r)
    print('第一步：已获得m，n的值。', s)
    return True


def adjustTheDataset():
    n = np.load('E:\\limbi\\n.npy')
    m = np.load('E:\\limbi\\m.npy')
    print(n[:10], m[:10])
    a0 = n[0]
    lnn = len(n)
    for i in range(lnn):
        n[i] -= a0
    b = []
    lnm = len(m)
    for i in range(lnm):
        if m[i] in b:
            continue
        else:
            b.append(m[i])
    sum_number = 0
    for i in range(lnm):
        for j in range(len(b)):
            if m[i] == b[j]:
                m[i] = j
                sum_number += 1
                break
    print(sum_number)
    np.save('E:\\limbi\\n', n)
    np.save('E:\\limbi\\m', m)
    print(n[:10], m[:10])
    return True


def find_n_m():
    n = np.load('E:\\limbi\\n.npy')
    m = np.load('E:\\limbi\\m.npy')
    n_max = 0
    m_max = 0
    for i in range(len(n)):
        if n[i] > n_max:
            n_max = n[i]
    for i in range(len(m)):
        if m[i] > m_max:
            m_max = m[i]
    n_max += 1
    m_max += 1
    print(n_max, m_max, len(n))
    return n_max, m_max


def produce_D4_D5(m, q):
    D4 = np.random.randn(q, m) / (q ** 0.5)
    D = np.dot(D4, D4.T)
    D1 = np.linalg.inv(D)
    D5 = np.dot(D4.T, D1)
    np.save('E:\\limbi\\Dl1', D4)
    np.save('E:\\limbi\\Dl3', D5)


def produce_D_D3(m, q):
    D4 = np.random.randn(q, m) / (q ** 0.5)
    D = np.dot(D4, D4.T)
    D1 = np.linalg.inv(D)
    D5 = np.dot(D4.T, D1)
    np.save('E:\\limbi\\D', D4)
    np.save('E:\\limbi\\D3', D5)


def getTop_k(m, sum_ratings, eta1, number_top_k):
    number_top_k = 5000
    all_list = improve.select_the_frency(m, sum_ratings, eta1)
    list_in_order4 = improve.top_k_order(all_list, number_top_k)
    np.save('E:\\limbi\\list_in_order4.npy', list_in_order4)


def get_rmse(U, V):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
    ratings = np.load('E:\\limbi\\r.npy')
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


def d_limbi(m_number, n_number):
    '''
    m = np.load('E:\\limbi\\m.npy')
    n = np.load('E:\\limbi\\n.npy')
    r = np.load('E:\\limbi\\r.npy')
    m_array = np.zeros((3, len(m)))
    m_array[0] = n[:]
    m_array[1] = m[:]
    m_array[2] = r[:]
    m_list = np.zeros((1, m_number))
    for i in range(len(m)):
        j = m[i]
        m_list[0][j] += 1
    long_list = []
    for i in range(m_number):
        if m_list[0][i] > 140:
            long_list.append(i)
    print('get score is over 140')
    t = len(m)
    m_a = []
    n_a = []
    r_a = []
    for i in range(t):
        if m[i] in long_list:
            m_a.append(m[i])
            n_a.append(n[i])
            r_a.append(r[i])
    print('get the new set')
    np.save('E:\\limbi\\n', n_a)
    np.save('E:\\limbi\\m', m_a)
    np.save('E:\\limbi\\r', r_a)
    '''
    # test = np.zeros((20, 30))
    # print(test)
    '''
    data = np.zeros((n_number, m_number), np.float16)
    n_a = np.load('E:\\limbi\\n.npy')
    m_a = np.load('E:\\limbi\\m.npy')
    r_a = np.load('E:\\limbi\\r.npy')
    t = len(n_a)
    for i in range(t):
        s_t = int(n_a[i])
        l_t = int(m_a[i])
        data[s_t][l_t] = r_a[i]
    print('set the old dateset')
    cf.show_time()
    n_a = []
    m_a = []
    r_a = []
    m_click = 0
    for i in range(m_number):
        s_t = 0
        for j in range(n_number):
            ratings_s = data[j][i]
            if ratings_s != 0:
                s_t += 1
                n_a.append(j)
                m_a.append(m_click)
                r_a.append(ratings_s)
        if s_t != 0:
            m_click += 1
            print(m_click)
    np.save('E:\\limbi\\n', n_a)
    np.save('E:\\limbi\\m', m_a)
    np.save('E:\\limbi\\r', r_a)
    print('order the dataset')
    '''
    n_a = np.load('E:\\limbi\\n.npy')
    m_a = np.load('E:\\limbi\\m.npy')
    r_a = np.load('E:\\limbi\\r.npy')
    t = len(n_a)
    m_number = 26509
    cf.show_time()
    data = np.zeros((n_number, m_number))
    for i in range(t):
        s_t = int(n_a[i])
        l_t = int(m_a[i])
        data[s_t][l_t] = r_a[i]
    print('get new dataset')
    cf.show_time()
    n_a = []
    m_a = []
    r_a = []
    for i in range(n_number):
        for j in range(m_number):
            ratings_s = data[i][j]
            if ratings_s != 0:
                n_a.append(i)
                m_a.append(j)
                r_a.append(ratings_s)
        print(i)
    print('get the order list')
    cf.show_time()
    '''
    m_adjust = []
    m_adjust.append(m_a[0])
    for i in range(len(m_a)):
        if m_a[i] not in m_adjust:
            m_adjust.append(m_a[i])
    np.save('E:\\limbi\\m_adjust', m_adjust)
    print('Get new m order')
    cf.show_time()
    for i in range(len(m_a)):
        for j in range(len(m_adjust)):
            if m_a[i] == m_adjust[j]:
                m_a[i] = j
                print(j)
                break
    cf.show_time()
    print('order the m')
    n_adjust = [0]
    ss = 0
    for i in range(len(n_a)-1):
        j = i + 1
        if n_a[j] == n_a[i]:
            n_adjust.append(ss)
        else:
            ss += 1
            n_adjust.append(ss)
    '''
    print('order the n')
    np.save('E:\\limbi\\n', n_a)
    np.save('E:\\limbi\\m', m_a)
    np.save('E:\\limbi\\r', r_a)


def machine_learning(m, n, d, sum11, U, V, k):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
    # c = np.load('E:\\limbi\column_list.npy')
    ratings = np.load('E:\\limbi\\r.npy')
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
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            dV[j] += -2 * U[i] * (ratings[sum1] - T)
            dU[i] += (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    i += 1
            sum1 += 1
        dV = dV / n
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\limbi\\U_0{sum11}', U)
    np.save(f'E:\\limbi\\V_0{sum11}', V)


def ml_LDP(m, n, d, sum11, U, V, eta, k):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
    ratings = np.load('E:\\limbi\\r.npy')
    D = np.load('E:\\limbi\\D.npy')
    D3 = np.load('E:\\limbi\\D3.npy')
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
    np.save(f'E:\\limbi\\U_2{sum11}', U)
    np.save(f'E:\\limbi\\V_2{sum11}', V)


def s_items(m, n, d, sum11, U, V, eta, k, l_i):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
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
    np.save(f'E:\\limbi\\l_array{sum11}', l_array)


def get_flag(m, n, sum11, l_array):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
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
    np.save(f'E:\\limbi\\c_flag{sum11}', c_flag)


# 满足差分隐私（改进实用性）削减参加计算的个人用户物品梯度
def ML_LDP_part_GD(m, n, d, sum11, U, V, eta, k, c_flag, l_array, r_q):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
    ratings = np.load('E:\\limbi\\r.npy')
    D = np.load('E:\\limbi\\Dl1.npy')
    D3 = np.load('E:\\limbi\\Dl3.npy')
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
    np.save(f'E:\\limbi\\U_6{sum11}', U)
    np.save(f'E:\\limbi\\V_6{sum11}', V)


def LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q):
    # use ml_LDP_part algrithm
    # produce_D4_D5(l_i, l_q)
    for j in range(6):
        i = j
        eta = eta_list[i] / 2
        # s_items(m, n, d, i, u, v, eta, k, l_i)
        print('get s_items')
        l_array = np.load(f'E:\\limbi\\l_array{i}.npy')
        # get_flag(m, n, i, l_array)
        print('get_flag')
        c_flag = np.load(f'E:\\limbi\\c_flag{i}.npy')
        # ML_LDP_part(m, n, d, i, u, v, eta, k, c_flag, l_array, l_q)
        ML_LDP_part(m, n, d, i, u, v, eta, k, c_flag, l_array, l_q)
        print('get the granted')


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
        print('I have finish the epslion of ', eta_list[i], ss)
    print(vmax)
    print(rmse)


def ml_LDP_hau(sum11, U, V, eta, m, n, d, k):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
    ratings = np.load('E:\\limbi\\r.npy')
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
    np.save(f'E:\\limbi\\U_3{sum11}', U)
    np.save(f'E:\\limbi\\V_3{sum11}', V)


def ldp_iteration(U, V, eta_list, m, n, d):
    for i in range(6):
        ml_LDP_hau(i, U, V, eta_list[i], m, n, d)


# 满足差分隐私（改进实用性）削减参加计算的个人用户物品梯度
def ML_LDP_part(m, n, d, sum11, U, V, eta, k, c_flag, l_array, r_q):
    r = np.load('E:\\limbi\\n.npy')
    c = np.load('E:\\limbi\\m.npy')
    ratings = np.load('E:\\limbi\\r.npy')
    D = np.load('E:\\limbi\\Dl1.npy')
    D3 = np.load('E:\\limbi\\Dl3.npy')
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
    np.save(f'E:\\limbi\\U_5{sum11}', U)
    np.save(f'E:\\limbi\\V_5{sum11}', V)


def number_of_iterations(m, n, d, u, v, l_q):
    k_list = [1, 2, 3, 4, 5, 20, 50]
    eta = 0.1
    l_array = np.load('E:\\limbi\\l_array1.npy')
    c_flag = np.load('E:\\limbi\\c_flag1.npy')
    for i in range(7):
        j = i + 6
        machine_learning(m, n, d, j, u, v, k_list[i])
        # ml_LDP(m, n, d, j, u, v, eta, k)
        ml_LDP_hau(j, u, v, eta, m, n, d, k_list[i])
        ML_LDP_part_GD(m, n, d, j, u, v, 0.05, k_list[i], c_flag, l_array, l_q)
        ML_LDP_part(m, n, d, j, u, v, eta/2, k_list[i], c_flag, l_array, l_q)


if __name__ == '__main__':
    '''
    filename = 'D:\\ratings.dat'
    readfile(filename)
    adjustTheDataset()
    '''
    cf.show_title()

    # u = np.load('E:\\limbi\\u.npy')
    # v = np.load('E:\\limbi\\v.npy')

    eta_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    # n, m = find_n_m()
    # d_limbi(m, n)
    # n, m = find_n_m()
    n = 135359
    m = 26509
    d = 20
    q = 2700
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

    # LDP_PD(m, n, d, u, v, k, l_i, eta_list, l_q)
    # ldp_iteration(u, v, eta_list, m, n, d)
    # machine_learning(m, n, d, 0, u, v, k)
    # ss_list = [2, 3, 5, 6]
    # for i in range(4):
    # get_vmax(ss_list[i])
    number_of_iterations(m, n, d, u, v, l_q)

    '''
    n, m = limbi.find_n_m_1()
    limbi.d_limbi(m, n)
    n, m = limbi.find_n_m()
    limbi.d_limbi0(n, m)
    limbi.d_limbi1(n, m)
    '''
    # v_ml = np.load('E:\\limbi\\V1000_10.npy')
    # v_ml = np.load('E:\\limbi\\V1000_10.npy')
    # print(np.shape(v_ml), v_ml[1])
