import common_function as cf
import numpy as np
import math


def machine_learning_LDP(m, n, sum, d, sum11, U, V, top_k, eta):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    D = np.load('E:\\movielens\D.npy')
    D3 = np.load('E:\\movielens\D3.npy')
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
        while sum1 < sum and i < top_k:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            # if j >= 0:
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
        dV = dB / n
        dU /= n
        # dV_t = np.dot(D, dV)
        # dV = np.dot(D3, dV_t)
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (np.dot(D3, dV) + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_2{sum11}', U)
    np.save(f'E:\\movielens\\V_2{sum11}', V)


def machine_learning(m, n, sum, d, sum11, U, V, k, top_k):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    # c = np.load('E:\\movielens\column_list.npy')
    ratings = np.load('E:\\movielens\\r.npy')
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
    np.save(f'E:\\movielens\\U_1{sum11}', U)
    np.save(f'E:\\movielens\\V_1{sum11}', V)


def machine_learning_LDP_part(m, n, sum, d, sum11, U, V, top_k_user, eta, k):
    r = np.load('E:\\movielens\\n.npy')
    # c = np.load('E:\\movielens\m.npy')
    c = np.load('E:\\movielens\\m_adjust.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    list_in_order4 = np.load('E:\\movielens\\list_in_order4.npy')
    D4 = np.load('E:\\movielens\\D4.npy')
    D5 = np.load('E:\\movielens\\D5.npy')
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
                    l = np.random.randint(0, d)
                    if x[s][l] > 0:
                        x[s][l] = t_sum
                    else:
                        x[s][l] = -t_sum
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
    np.save(f'E:\\movielens\\U_3{sum11}', U)
    np.save(f'E:\\movielens\\V_3{sum11}', V)


def machine_learning_LDP_hau(sum11, U, V, eta, m, n, sum, d):
    r = np.load('E:\\movielens\\n.npy')
    c = np.load('E:\\movielens\\m.npy')
    ratings = np.load('E:\\movielens\\r.npy')
    lu = 10 ** -8
    lv = 10 ** -8
    # k = 10
    k = 10
    # eta = 0.1
    t_sum = m * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_sum_1 = math.exp(eta/k)
    for it in range(k):
        rt = 1 / (it + 1) / (k ** 2)
        # rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))
        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            # if j >= 0:
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
                        dV[s][ls] = t_sum
                    else:
                        dV[s][ls] = -t_sum
                    dB = dB + dV
                    dV = np.zeros((m, d))
            else:
                dB = dB + dV
        dV = dB / n
        dU /= n
        # dV_t = np.dot(D, dV)
        # dV = np.dot(D3, dV_t)
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\movielens\\U_4{sum11}', U)
    np.save(f'E:\\movielens\\V_4{sum11}', V)


def machine_learning_LDP_NJ(sum11, U, V, eta, m, n, sum, d):
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
        rt = 1 / (it + 1) / (k ** 2)
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
    np.save(f'E:\\movielens\\U_5{sum11}', U)
    np.save(f'E:\\movielens\\V_5{sum11}', V)


if __name__ == '__main__':
    cf.show_title()
    cf.show_time()
    top_k = 138493
    k = 10
    # eta = 1
    experiment = 1
    eta_list = [0.05, 0.1, 0.2, 0.4, 0.8, 1.6]
    eta = eta_list[3]

    U = np.load('E:\\movielens\\U.npy')
    V = np.load('E:\\movielens\V.npy')

    # machine_learning(0, U, V, top_k)
    # machine_learning_LDP_part(0, U, V, top_k, eta, k)
    # machine_learning_LDP(experiment, U, V, top_k, eta)
    # machine_learning_LDP_hau(0, U, V, top_k, eta)
    # machine_learning_LDP_NJ(1, U, V, top_k, eta)

    # machine_learning_LDP_hau(, U, V, top_k, eta)
    # machine_learning_LDP_NJ(3, U, V, top_k, eta)
    '''
    for i in range(1):
        j = i + 5
        eta = eta_list[j]
        print(eta)
        # machine_learning_LDP_part(j, U, V, top_k, eta, k)
        machine_learning_LDP_hau(j, U, V, top_k, eta)
        machine_learning_LDP_NJ(j, U, V, top_k, eta)
    '''

    '''
    machine_learning(0, U, V, top_k, 0.1)
    machine_learning_LDP_part(0, U, V, top_k, 0.1)

    machine_learning_LDP(1, U, V, top_k, 1)
    machine_learning(1, U, V, top_k, 1)
    machine_learning_LDP_part(1, U, V, top_k, 1)

    machine_learning_LDP(2, U, V, top_k, 10)
    machine_learning(2, U, V, top_k, 10)
    machine_learning_LDP_part(2, U, V, top_k, 10)
    # 0 eta = 0.1
    '''
