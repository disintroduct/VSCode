import common_function as cf
import numpy as np
import math
'''
* get top-k
* compute the gradient
* randomize the gradient
* recovery the gradient
* compress the data set
'''


def select_the_frency(m, sum_ratings, eta):
    r = np.load('E:\\book\\n.npy')
    c = np.load('E:\\book\\m.npy')
    sum_set_number = sum_ratings
    sum1 = 0
    i = 0
    all_list = []
    user_list = []
    while sum1 < sum_set_number:
        j = c[sum1]

        if sum1 < sum_set_number:
            try:
                test = r[sum1]
            except Exception as er:
                print(er, test)
                break
            if r[sum1] != i:
                ssr = np.random.randint(0, len(user_list))
                i += 1
                pr = np.random.random()
                if pr > (math.exp(eta/(2 * len(user_list))) / (1 + math.exp(eta/(2 * len(user_list))))):
                    ttr = np.random.randint(0, m)
                    all_list.append(ttr)
                else:
                    all_list.append(user_list[ssr])
                user_list = []
        user_list.append(j)
        sum1 += 1
    return all_list


def top_k_order(all_list, number_top_k):
    top_k = all_list
    # print(sorted(top_k))
    # print(top_k)
    list_in_order = (sorted(top_k))
    list_in_order1 = []
    list_in_order2 = []
    list_in_order1.append(list_in_order[0])
    # list_in_order2.append(0)
    t = list_in_order[0]
    sum_frency = 0
    for i in range(len(top_k)):
        sum_frency += 1
        if t != list_in_order[i]:
            t = list_in_order[i]
            list_in_order1.append(t)
            list_in_order2.append(sum_frency)
            sum_frency = 0
    list_in_order2.append(sum_frency)
    n = len(list_in_order1)
    RR_top_array = np.zeros((2, n))
    RR_top_array[0] = list_in_order1
    RR_top_array[1] = list_in_order2
    save_order_array = RR_top_array.T[np.lexsort(-RR_top_array)].T
    sum_top_1000 = 0
    for i in range(number_top_k):
        sum_top_1000 += save_order_array[1][i]
    print(sum_top_1000 / len(top_k))
    list_in_order3 = save_order_array[0][:number_top_k]
    list_in_order4 = sorted(list_in_order3)
    return list_in_order4, list_in_order3[:10]


def machine_learning(list_in_order, sum11, U, V):
    r = np.load('E:\\book\\n.npy')
    c = np.load('E:\\book\m.npy')
    ratings = np.load('E:\\book\\r.npy')
    # D = np.load('E:\book\D.npy')
    # D3 = np.load('E:\book\D3.npy')
    length_list = len(list_in_order)
    print('length_list', length_list)
    m = 131262
    # n = 138493
    n = 138493
    d = 15
    lu = 10 ** -8
    lv = 10 ** -8
    k = 10
    sum_set_number = 20000263
    # U = load("E:\\U8.npy")
    # print(U[:3], V[:3])
    # V = load('E:\V8.npy')
    for it in range(k):
        rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dB = np.zeros((m, d))

        sum1 = 0
        i = 0
        dV = np.zeros((m, d))
        while sum1 < sum_set_number:
            j = c[sum1]
            if c[sum1] in list_in_order[:]:
                T = np.dot(U[i], V[j].T)
                dV[j] = -2 * U[i] * (ratings[sum1] - T)
            dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
            sum1 += 1
            if sum1 < sum_set_number:
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
    np.save(f'E:\\U8_{sum11}', U)
    np.save(f'E:\V8_{sum11}', V)


def machine_learning_LDP(sum11, U, V):
    r = np.load('E:\book\\n.npy')
    c = np.load('E:\book\m.npy')
    ratings = np.load('E:\book\\r.npy')
    # D = np.load('E:\book\D.npy')
    # D3 = np.load('E:\book\D3.npy')
    m = 131262
    n = 138493
    d = 15
    lu = 10 ** -8
    lv = 10 ** -8
    k = 10
    sum = 20000263
    n_number = 1000
    print('n_number', n_number)
    # U = load("E:\\U8.npy")
    # print(U[:3], V[:3])
    # V = load('E:\V8.npy')
    for it in range(k):
        rt = 1 / (it + 1)
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
                    # laplace_n = np.random.laplace(0, 10 * 15 / sum11, (m, d))
                    dV = np.zeros((m, d))
            else:
                dB = dB + dV
        dV = dB / n
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'E:\\U9_{sum11}', U)
    np.save(f'E:\V9_{sum11}', V)


def get_top_k(m, sum_ratings, eta1, number_top_k):
    # get top-k
    eta1 = 0.1
    number_top_k = 5000
    all_list = select_the_frency(m, sum_ratings, eta1)
    list_in_order4 = top_k_order(all_list, number_top_k)
    np.save('E:\\book\\list_in_order4', list_in_order4)
    print('over!')
    return list_in_order4


if __name__ == '__main__':

    print('I am fine, thanks!')