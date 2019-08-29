import common_function as cf
import part1000 as p1
import numpy as np
import improve
import libim
import math


def readfile(filename):
    fr = open(filename, 'r', encoding='UTF-8')
    readLines = fr.readlines()
    s = np.alen(readLines)
    n = []
    m = []
    r = []
    # print(readLines)
    # click = 0
    for line in readLines:
        print(line)
        ListFormLine = line.strip()
        # print(ListFormLine)
        # print(['232323', "12345"])
        ListFormLine = str(ListFormLine)
        ListFormLine = ListFormLine.split(';')
        # print(ListFormLine)
        # ListFormLine[0] = ListFormLine.strip('"')
        print(ListFormLine[0])
        n.append(str(ListFormLine[0]))
        m.append(str(ListFormLine[1]))
        r.append(int(ListFormLine[2][:2]))
    fr.close()
    np.save('E:\\book\\n', n)
    np.save('E:\\book\\m', m)
    np.save('E:\\book\\r', r)
    print('第一步：已获得m，n的值。', s)
    return True


def adjustTheDataset():
    n = np.load('E:\\book\\n.npy')
    m = np.load('E:\\book\\m.npy')
    sst = np.load('E:\\book\\r.npy')
    print(n[:10], m[:10], sst[:10])
    lnn = len(n)
    click_n = 0
    tem_n = n[0]
    adjust_n = []
    for i in range(lnn):
        if n[i] == tem_n:
            adjust_n.append(click_n)
        else:
            tem_n = n[i]
            click_n += 1
            adjust_n.append(click_n)

    b = []
    adjust_m = []
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
                # m[i] = j
                adjust_m.append(j)
                sum_number += 1
                break
    print(sum_number)
    np.save('E:\\book\\n', adjust_n)
    np.save('E:\\book\\m', adjust_m)
    print(n[:10], m[:10])
    return True


def find_n_m():
    n = np.load('E:\\book\\n.npy')
    m = np.load('E:\\book\\m.npy')
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


def produce_D4_D5(m, q, d):
    D4 = np.random.randn(q, m) / (q ** 0.5)
    D = np.dot(D4, D4.T)
    D1 = np.linalg.inv(D)
    D5 = np.dot(D4.T, D1)
    np.save('E:\\book\\D4', D4)
    np.save('E:\\book\\D5', D5)


def produce_D_D3(m, q, d):
    D4 = np.random.randn(q, m) / (q ** 0.5)
    D = np.dot(D4, D4.T)
    D1 = np.linalg.inv(D)
    D5 = np.dot(D4.T, D1)
    np.save('E:\\book\\D', D4)
    np.save('E:\\book\\D3', D5)


def getTop_k(m, sum_ratings, eta1, number_top_k):
    number_top_k = 5000
    all_list = improve.select_the_frency(m, sum_ratings, eta1)
    list_in_order4, list_in_order3 = improve.top_k_order(all_list, number_top_k)
    np.save('E:\\book\\list_in_order4.npy', list_in_order4)
    np.save('E:\\book\\list_in_order3.npy', list_in_order3)
    return True


def classifyMF(m, n, sum_ratings, d, j, u, v, eta_list, k):
    for i in range(4):
        # rmse_ldp = get_rmse(filename+filename_u+f'{s}'+f'{j}'+'.npy', filename+filename_v+f'{s}'+f'{j}'+'.npy')
        # v = np.load(filename+filename_v+f'{s}'+f'{j}'+'.npy')
        # rmse_ldp = np.linalg.norm(v_ml-v, np.inf)
        # rmse.append(rmse_ldp)
        j = i
        p1.machine_learning_LDP(m, n, sum_ratings, d, j, u, v, 1000, eta_list[j])
        p1.machine_learning_LDP_part(m, n, sum_ratings, d, j, u, v, n, eta_list[j], k)
        p1.machine_learning_LDP_hau(j, u, v, eta_list[j], m, n, sum_ratings, d)
        p1.machine_learning_LDP_NJ(j, u, v, eta_list[j], m, n, sum_ratings, d)
    print('I have do it well!', j)


def get_rmse():
    # number_top_k = 5000
    # u = np.load('E:\\book\\u.npy')
    # v = np.load('E:\\book\\v.npy')
    # filename = 'E:\\book\\'
    filename = 'E:\\book\\'
    filename_u = 'U1000_'
    filename_v = 'V1000_'

    # v_ml = np.load('E:\\book\\V1000_10.npy')
    # v_ml = np.load('E:\\book\\V1000_10.npy')
    # print(np.shape(v_ml), v_ml[1])
    for i in range(3):
        s = i + 3
        rmse = []
        for j in range(6):
            rmse_ldp = libim.get_rmse(filename+filename_u+f'{s}'+f'{j}'+'.npy', filename+filename_v+f'{s}'+f'{j}'+'.npy')
            # v = np.load(filename+filename_v+f'{s}'+f'{j}'+'.npy')
            # rmse_ldp = np.linalg.norm(v_ml-v, np.inf)
            rmse.append(rmse_ldp)
        print(rmse)
    return rmse


def percent_ratings(n, m, r, u, v):
    list_len = len(n)
    list_error = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    for i in range(list_len):
        # print(m[:2], n[:2], r[:2], u[:2], v[:2])
        temp = r[i] - np.dot(u[int(n[i])], v[int(m[i])].T)
        for j in range(20):
            if temp < (j * 0.5 + 0.5):
                list_error[j] += 1
                break
    for i in range(19):
        list_error[i+1] = list_error[i] + list_error[i+1]
    list_per = np.array(list_error) / list_len
    print(list_per)


def f_score(user_len, user_top_items, u1, v1, list_top_k):
    cf.show_time()
    f_score_sum = 0
    # user_top_items = np.zeros((user_len, 10))
    for i in range(user_len):
        # user_score_0 = np.dot(u[i], v.T)
        user_score_1 = np.dot(u1[i], v1.T)
        # list_0 = np.argsort(-user_score_0)
        list_1 = np.argsort(-user_score_1)
        # list_0_top10 = list_0[:10]
        # user_top_items[i] = list_0[:10]
        temp_count = 0
        for j in range(10):
            if list_1[j] in user_top_items[i]:
                # if list_1[j] in list_top_k:
                temp_count += 1
        f_score_sum += temp_count / 10
    f_score_final = f_score_sum / user_len
    # np.save('E:\\book\\user_top_items', user_top_items)
    print(f_score_final)
    cf.show_time()
    return f_score_final


def adjust_m(m, top_k_list):
    n = len(m)
    for i in range(n):
        if m[i] not in top_k_list:
            m[i] = -1
    np.save('E:\\book\\m_adjust', m)


def machine_learning_LDP_part(m, n, sum, d, sum11, U, V, top_k_user, eta, k):
    r = np.load('E:\\book\\n.npy')
    # c = np.load('E:\\book\m.npy')
    c = np.load('E:\\book\\m_adjust.npy')
    ratings = np.load('E:\\book\\r.npy')
    list_in_order4 = np.load('E:\\book\\list_in_order4.npy')
    D4 = np.load('E:\\book\\D4.npy')
    D5 = np.load('E:\\book\\D5.npy')
    m_d = 5000
    # q = 2700
    q = 500
    lu = 10 ** -8
    lv = 10 ** -8

    t_sum = q * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    for it in range(k):
        rt = 1 / (it + 1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((q, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum and i < top_k_user:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if j >= 0:
                dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum1 += 1
            if sum1 < sum and i < top_k_user:
                try:
                    test = r[sum1]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum1] != i:
                    i += 1
                    dv_x = np.zeros((m_d, d))
                    for o in range(m_d):
                        dv_x[o] = dV[int(list_in_order4[o])]
                    x = np.dot(D4, dv_x)
                    s = np.random.randint(0, q)
                    ll = np.random.randint(0, d)
                    if x[s][ll] > 0:
                        x[s][ll] = t_sum
                    else:
                        x[s][ll] = -t_sum
                    dB = dB + x
                    dV = np.zeros((m, d))
            else:
                dB = dB + x
        dV = np.zeros((m, d))
        dv_x = dB / n
        dv_xt = np.dot(D5, dv_x)
        for o in range(m_d):
            dV[int(list_in_order4[o])] = dv_xt[o]
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\book\\U1000_31{sum11}', U)
    np.save(f'E:\\book\\V1000_31{sum11}', V)


def machine_learning(m, n, sum, d, sum11, U, V, k, top_k):
    r = np.load('E:\\book\\n.npy')
    c = np.load('E:\\book\\m.npy')
    # c = np.load('E:\\book\column_list.npy')
    ratings = np.load('E:\\book\\r.npy')
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    # k = 1
    for it in range(k):
        rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
        while sum1 < sum and i < top_k:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)

            dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum1 += 1
            if sum1 < sum and i < top_k:
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
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\book\\U1000_11{sum11}', U)
    np.save(f'E:\\book\\V1000_11{sum11}', V)


def ML_LDP_part_prove(m, n, sum, d, sum11, U, V, top_k_user, eta, k):
    r = np.load('E:\\book\\n.npy')
    c = np.load('E:\\book\m.npy')
    c_flag = np.load('E:\\book\\m_adjust.npy')
    ratings = np.load('E:\\book\\r.npy')
    list_in_order4 = np.load('E:\\book\\list_in_order4.npy')
    D4 = np.load('E:\\book\\D4.npy')
    D5 = np.load('E:\\book\\D5.npy')
    m_d = 5000
    # q = 2700
    q = 500
    lu = 10 ** -8
    lv = 10 ** -8

    for it in range(k):
        # rt = 1 / (it + 1) / (k ** 2)
        rt = 1 / (it + 1) / k
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((q, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum and i < top_k_user:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if c_flag[sum1] >= 0:
                dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum1 += 1
            if sum1 < sum and i < top_k_user:
                try:
                    test = r[sum1]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum1] != i:
                    i += 1
                    dv_x = np.zeros((m_d, d))
                    for o in range(m_d):
                        dv_x[o] = dV[int(list_in_order4[o])]
                    x = np.dot(D4, dv_x)
                    s = np.random.randint(0, q)
                    x[s] += np.random.laplace(0, 2*d*k/eta, d)
                    dB = dB + x
                    dV = np.zeros((m, d))
            else:
                dB = dB + x
        dV = np.zeros((m, d))
        dv_x = dB / n
        dv_xt = np.dot(D5, dv_x)
        for o in range(m_d):
            dV[int(list_in_order4[o])] = dv_xt[o]
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\book\\U1000_7{sum11}', U)
    np.save(f'E:\\book\\V1000_7{sum11}', V)


def find_n_m_1():
    n = np.load('E:\\book\\n.npy')
    m = np.load('E:\\book\\m.npy')
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


def d_book(m_number, n_number):

    m = np.load('E:\\book\\m.npy')
    n = np.load('E:\\book\\n.npy')
    r = np.load('E:\\book\\r.npy')
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
    np.save('E:\\book\\n', n_a)
    np.save('E:\\book\\m', m_a)
    np.save('E:\\book\\r', r_a)


# test = np.zeros((20, 30))
# # print(test)
def d_book0(n_number, m_number):
    data = np.zeros((n_number, m_number), np.float16)
    n_a = np.load('E:\\book\\n.npy')
    m_a = np.load('E:\\book\\m.npy')
    r_a = np.load('E:\\book\\r.npy')
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
    np.save('E:\\book\\n', n_a)
    np.save('E:\\book\\m', m_a)
    np.save('E:\\book\\r', r_a)
    print('order the dataset')


def d_book1(n_number, m_number):
    n_a = np.load('E:\\book\\n.npy')
    m_a = np.load('E:\\book\\m.npy')
    r_a = np.load('E:\\book\\r.npy')
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
    m_adjust = []
    m_adjust.append(m_a[0])
    for i in range(len(m_a)):
        if m_a[i] not in m_adjust:
            m_adjust.append(m_a[i])
    np.save('E:\\book\\m_adjust', m_adjust)
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

    print('order the n')
    np.save('E:\\book\\n', n_a)
    np.save('E:\\book\\m', m_a)
    np.save('E:\\book\\r', r_a)


if __name__ == '__main__':
    # adjustTheDataset()
    # filename = 'E:\\book\\BX-book-Ratings.txt'
    # readfile(filename)
    # find_n_m()
    n = 105283
    m = 340556
    sum_ratings = 1149780
    d = 20
    k = 10
    short_k = 1
    q_short = 500
    q = 2700
    number_top_k = 5000

    # produce_D4_D5(number_top_k, q_short, d)
    # D6 = np.random.randn(number_top_k, number_top_k) / (number_top_k ** 0.5)
    # D7 = np.linalg.inv(D6)
    # np.save('E:\\book\\D6', D6)
    # np.save('E:\\book\\D7', D7)
    # produce_D_D3(m, q, d)
    u = np.load('E:\\book\\u.npy')
    v = np.load('E:\\book\\v.npy')
    # v1 = np.load('E:\\book\\V1000_10.npy')
    # v = np.load('E:\\book\\V1000_10.npy')
    # u = np.load('E:\\book\\U1000_10.npy')
    # np.save('E:\\book\\u', u)
    # np.save('E:\\book\\v', v)
    eta_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    # getTop_k(m, sum_ratings, 1, 5000)
    list_top_k = np.load('E:\\book\\list_in_order3.npy')

    k_list = [1, 2, 3, 4, 5]

    # filename = 'E:\\book\\'
    filename = 'E:\\book\\'
    filename_u = 'U1000_'
    filename_v = 'V1000_'
    print('I am fine.')
    # list_top_k = np.load()
    # adjust_m = np.load('E:\\book\\m_adjust.npy')
    # top_k_list = np.load('E:\\book\\list_in_order4.npy')
    # adjust_m(m, top_k_list)

    # n = np.load('E:\\book\\n.npy')
    # m = np.load('E:\\book\\m.npy')
    # r = np.load('E:\\book\\r.npy')
    '''
    
    for i in range(6):
        # machine_learning_LDP_part(m, n, sum_ratings, d, i, u, v, n, 0.1, k_list[i])
        # machine_learning(m, n, sum_ratings, d, i, u, v, k_list[i], m)
        ML_LDP_part_prove(m, n, sum_ratings, d, i, u, v, n, eta_list[i], k)
    '''
    user_top_items = np.load('E:\\book\\user_top_items.npy')
    print(np.shape(user_top_items))
    for i in range(1):
        Vmax = []
        s = i + 6
        f_score_count = []

        for j in range(6):
            v1 = np.load(filename + filename_v + f'{s}' + f'{j}.npy')
            u1 = np.load(filename + filename_u + f'{s}' + f'{j}.npy')
            vt = np.linalg.norm(v1-v, np.inf)
            Vmax.append(vt)
            # percent_ratings(n, m, r, u, v)
            # print(list_top_k)
            temp = f_score(n, user_top_items, u1, v1, list_top_k)
            f_score_count.append(temp)
        print(f_score_count)
        # print(Vmax)


    # p1.machine_learning_LDP_part(m, n, sum_ratings, d, j, u, v, n, eta_list[j], k)
    # print(Vmax)

    # m_list = np.load('E:\\book\\m.npy')
    # print(m_list[:250])

    # p1.machine_learning_LDP_part(m, n, sum_ratings, d, 6, u, v, n, 0.05, k)
    # print('machine_learning 1! done')

    # p1.machine_learning(m, n, sum_ratings, d, 1, u, v, k, 1000)
    # print('machine_learning 2! done')

    # getTop_k(m, sum_ratings, 1, number_top_k)

    # rmse = get_rmse()
