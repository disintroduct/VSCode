from scipy import sparse
import common_function
import numpy as np
import scipy.sparse
import math


def getNM(filename):
    fr = open(filename)
    readLines = fr.readlines()
    s = np.alen(readLines)
    n = []
    m = []
    r = []
    for line in readLines:
        ListFormLine = line.strip()
        ListFormLine = ListFormLine.split(',')
        n.append(int(ListFormLine[0]) - 1)
        m.append(int(ListFormLine[1]) - 1)
        r.append(float(ListFormLine[2]))
    fr.close()
    np.save('D:\\n', n)
    np.save('D:\m', m)
    np.save('D:\\r', r)
    print('第一步：已获得m，n的值。', s)
    return n


def readfile(n, m, filename):

    # 读取数据集，并生成矩阵
    fr = open(filename)
    dataset = np.zeros((n, m), np.float16)
    dataline = fr.readlines()
    i = 0
    for line in dataline:
        line = line.strip()  # strip() 方法用于移除字符串头尾指定的字符（默认为空格或换行符）或字符序列。
        listfromline = line.split(',')  # Python split() 通过指定分隔符对字符串进行切片，如果参数 num 有指定值，则仅分隔 num 个子字符串，变成列表。
        length = len(listfromline) - 1
        listfromline[0] = int(listfromline[0]) - 1
        # 把字符型化为整形
        listfromline[1] = int(listfromline[1]) - 1
        listfromline[2] = float(listfromline[2])
        dataset[listfromline[0]][listfromline[1]] = listfromline[2]
        # 在集合dataset中，末尾加上整形的listformline
        i = i + 1
    print('有效评论数为：' + str(i))
    print('创建原始数据集成功~得到dataSet')
    fr.close()
    return dataset  # 返回整理过的数据集


def local_differencial_privacy(eta, p):
    r = np.load('D:\Labor\\n.npy')
    c = np.load('D:\Labor\m.npy')
    ratings = np.load('D:\Labor\\r.npy')
    D = np.load('D:\Labor\D.npy')
    D3 = np.load('D:\Labor\D3.npy')
    U = np.load('D:\\U.npy')
    V = np.load('D:\V.npy')
    m = 131262
    n = 138492
    sum = 150629
    print(sum)
    q = 2700
    k = 10
    d = 15
    # 规则化参数
    lu = 10 ** -8
    lv = 10 ** -8
    for s1 in range(1):
        print('This is the ' + str(s1 + 1) + 'iteration!')
        filename = f'D:\V{p}'
        filename1 = f'D:\\U{p}'
        # save(filename1, U)
        for it in range(k):
            # 初始化U，V的偏导矩阵
            print('The ' + str(it) + ' iteration!')
            common_function.show_time()
            # dV = np.zeros((m, d), float16)
            rt = 1 / (it + 1) / (k ** 2)
            dB = np.zeros((q, d))
            dU = np.zeros((n, d))
            sum_N = np.zeros((q, d))
            sum1 = 0
            i = 0
            while i < n:
                dV = np.zeros((m, d))
                N = np.random.laplace(0, 2 * 10 * (15 ** 0.5) / eta, (m, d))
                while i < n:
                    if r[sum1] != i:
                        i = i + 1
                        break
                    j = c[sum1]
                    T = np.dot(U[i], V[j].T)
                    dV[j] = -2 * U[i] * (ratings[sum1] - T)
                    dU[i] = dU[i] + (-2) * V[j] * (ratings[sum1] - T)
                    sum1 += 1
                x = np.dot(D, dV + N)
                y = np.dot(D, N)
                sum_N += y
                dB += x
            dB = (dB - sum_N) / sum
            dU /= sum
            U = U - rt * (dU + 2 * lu * U)
            V = V - rt * (np.dot(D3, dB) + 2 * lv * V)
            common_function.show_time()
        print(V)
        print('Get V！')
        np.save(filename, V)
        np.save(filename1, U)


def machine_learning(sum11, U, V):
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
    for it in range(k):
        rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
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
                    print(er, test)
                    break
                if r[sum1] != i:
                    i += 1
                    dB = dB + dV
                    # laplace_n = np.random.laplace(0, 10 * 15 / sum11, (m, d))
                    dV = np.zeros((m, d))
            else:
                dB = dB + dV
        laplace_n = np.random.laplace(0, 10 * 10 / (20 - sum11 * 2), (m, d))
        dV = dB / n + laplace_n
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        common_function.show_time()
    print('\n' + f'I have done it! {sum11}')
    np.save(f'D:\\U7_{sum11}', U)
    np.save(f'D:\V7_{sum11}', V)


def get_frency_mf(sum11, U, V):
    r = np.load('D:\Labor\\n.npy')
    c = np.load('D:\Labor\m.npy')
    ratings = np.load('D:\Labor\\r.npy')
    m = 131262
    order_list = np.load('D:\RR_order_list.npy')
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
        print(str(it) + '次迭代！')
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
        common_function.show_time()
    print('\n' + 'I have done it!')
    np.save(f'D:\\U{sum11}', U)
    np.save(f'D:\V{sum11}', V)


def get_rmse(filename_U, filename_V):
    common_function.show_time()
    U = np.load(filename_U)
    V = np.load(filename_V)
    r = np.load('E:\Labor\\n.npy')
    c = np.load('E:\Labor\m.npy')
    ratings = np.load('E:\Labor\\r.npy')
    n = len(r)
    rmse = 0
    i = 0
    for m in range(n):
        if r[m] >= 138492:
            break
        i = r[m]
        j = c[m]
        rmse += ((ratings[m] - np.dot(U[i], V[j].T)) ** 2)
    rmse = (rmse / n) ** 0.5
    print(rmse)
    common_function.show_time()
    return rmse


def get_number():
    Y = np.load('D:\Labor\Y.npy')
    n = np.alen(Y)
    m = np.alen(Y[0])
    sum_number = 0
    for i in range(n):
        j = np.random.randint(0, m + 1)
        if Y[i][j] != 0:
            sum_number += 1
    print(sum_number * m)


def zh_mf(sum11, U, V, ratings, tipss):
    m = np.alen(ratings[0])
    # n = 138493
    n = np.alen(ratings)
    V0 = np.load('D:\V0.npy')
    d = 15
    lu = 10 ** -8
    lv = 10 ** -8
    k = 10

    if tipss == 0 or tipss == 1:
        s = np.load('D:\RR_order_list.npy')
    for it in range(k):
        rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        common_function.show_time()
        dU = np.zeros((n, d))
        dV = np.zeros((m, d))
        for i in range(n):
            if tipss == 3 or tipss == 2:
                s = np.random.randint(0, m, 5000)
            for o in range(5000):
                j = int(s[o])
                if ratings[i][j] != 0:
                    T = np.dot(U[i], V[j].T)
                    dV[j] = dV[j] + (-2) * U[i] * (ratings[i][j] - T) / n
                    dU[i] = dU[i] + (-2) * V[j] * (ratings[i][j] - T)
            if tipss == 1 or tipss == 3:
                laplace_noisy = np.random.laplace(0, 10 * 15 * 5000 / (0.2 * (2 * tipss + 1)), (5000, d))
                for number_tipsss in range(5000):
                    dV[int(s[number_tipsss])] += laplace_noisy[number_tipsss] / n
        # dV /= n
        if tipss == 0 or tipss == 2:
            laplace_noisy = np.zeros((5000, d))
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        common_function.show_time()
        print(np.linalg.norm(V - V0, np.inf))
    print('\n' + 'I have done it!')
    np.save(f'D:\\U{tipss}' + f'_{sum11}', U)
    np.save(f'D:\V{tipss}' + f'_{sum11}', V)
    # V2 = load('D:\Labor\V2.npy')
    # print(np.linalg.norm(V - V2, ord=np.inf))


def zh_rmse(ratings, U, V):
    print(np.shape(ratings))
    n = np.alen(ratings)
    m = np.alen(ratings[0])
    click = 0
    rmse = 0
    for i in range(n):
        for j in range(m):
            if ratings[i][j] != 0:
                click += 1
                rmse += ((ratings[i][j] - np.dot(U[i], V[j].T)) ** 2)
    rmse = (rmse / click) ** 0.5
    print(rmse, click)


def get_top_k():
    n = 138493
    m = 131262
    data = readfile(138493, 131262, 'D:\Labor\\ratings.txt')
    top_k = []
    for i in range(n):
        t = np.random.randint(0, 131262)
        if data[i][t] != 0:
            top_k.append(t)
            print(t)
    np.save('D:\\top_k', top_k)
    print(top_k)


def top_k_order():
    top_k = np.load('D:\\RR_top_k.npy')
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
    for i in range(5000):
        sum_top_1000 += save_order_array[1][i]
    print(sum_top_1000 / len(top_k))
    list_in_order3 = save_order_array[0]
    list_in_order4 = sorted(list_in_order3)
    np.save('D:\RR_order_list', list_in_order3)
    np.save('D:\RR_top_array', save_order_array)
    print(list_in_order3)
    print(list_in_order4)


def get_RR_top_k(eta, ratings, rrs):
    n = np.alen(ratings)
    p = 2 / (1 + math.exp(eta))
    top_k = []
    for i in range(n):
        l = 0
        t = np.random.randint(0, 131262, rrs)
        for j in range(rrs):
            if ratings[i][t[j]] != 0:
                pro1 = np.random.random()
                if pro1 < (p/2):
                    l = 0
                else:
                    l = 1
            else:
                if np.random.random() < (p/2):
                    l = 1
                else:
                    l = 0
            if l == 1:
                top_k.append(t[j])
    np.save('D:\RR_top_k', top_k)
    print(len(top_k), rrs)


def get_percent(data):
    # order_list = load('D:\RR_order_list.npy')
    n = np.alen(data)
    # m = len(order_list)
    m = 1000
    sum_number = 20000000
    part_sum = 0
    for s in range(10):
        part_sum = 0
        for i in range(n):
            order_list = np.random.randint(0, 100000, 1000 * (s + 1))
            order_list = np.sort(order_list)
            for j in range(len(order_list)):
                # temp_j = s * m + j
                list_j = int(order_list[j])
                if data[i][list_j] != 0:
                    part_sum += 1
        print(part_sum / sum_number)
    print(part_sum / sum_number)


def select_the_frency(eta):
    r = np.load('D:\Labor\\n.npy')
    c = np.load('D:\Labor\m.npy')
    # ratings = np.load('D:\Labor\\r.npy')
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
    sum1 = 0
    i = 0
    all_list = []
    user_list = []
    while sum1 < sum:
        j = c[sum1]

        if sum1 < sum:
            try:
                test = r[sum1]
            except Exception as er:
                print(er)
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
    np.save('D:\RR_top_k', all_list)


def get_the_heavy_hitter():
    top_k = np.load('D:\RR_top_k.npy')
    order_list = np.load('D:\RR_order_list.npy')
    n = len(order_list)
    m = len(top_k)
    order_array = np.zeros((2, n))
    top_k_sorted = sorted(top_k)
    ssr = 0
    for i in range(n):
        order_array[0][i] = order_list[i]
    for j in range(m):
        if order_array[0][ssr] != top_k_sorted[j]:
            ssr += 1
            continue
        else:
            order_array[1][ssr] += 1


def zh_mf_without_third_part(sum11, U, V, ratings, tipss):
    m = np.alen(ratings[0])
    # n = 138493
    n = np.alen(ratings)
    d = 15
    lu = 10 ** -8
    lv = 10 ** -8
    k = 10
    if tipss == 0 or tipss == 1:
        s = np.load('D:\RR_order_list.npy')
    # print(s)
    # can = 15000 * (15 ** 0.5)
    # n_number = 1000
    # rr_order_list = load('D:\RR_order_list.npy')
    # lens = len(rr_order_list)
    # U = load("D:\\U8.npy")
    # print(U[:3], V[:3])
    # V = load('D:\V8.npy')
    for it in range(1):
        rt = 1 / (it + 1)
        print(str(it) + '次迭代！')
        common_function.show_time()
        dU = np.zeros((n, d))
        dV = np.zeros((m, d))
        for i in range(n):
            if tipss == 3 or tipss == 2:
                s = np.random.randint(0, m, 5000)
            if tipss == 1 or tipss == 3:
                laplace_noisy = np.random.laplace(0, 10 * 15 * 5000 / (0.2 * (2 * tipss + 1)), (5000, d))
                for number_tipsss in range(5000):
                    dV[int(s[number_tipsss])] += laplace_noisy[number_tipsss]
            for o in range(5000):
                j = int(s[o])
                if ratings[i][j] != 0:
                    T = np.dot(U[i], V[j].T)
                    dV[j] = dV[j] + (-2) * U[i] * (ratings[i][j] - T)
                    dU[i] = dU[i] + (-2) * V[j] * (ratings[i][j] - T)

        dV /= n
        if tipss == 0 or tipss == 2:
            laplace_noisy = np.zeros((5000, d))
        dU /= n
        U = U - rt * (dU + 2 * lu * U)
        V = V - rt * (dV + 2 * lv * V)
        common_function.show_time()
    print('\n' + 'I have done it!')
    np.save(f'D:\\U{tipss}' + f'_{sum11}', U)
    np.save(f'D:\V{tipss}' + f'_{sum11}', V)
    # V2 = load('D:\Labor\V2.npy')
    # print(np.linalg.norm(V - V2, ord=np.inf))


def percent_of_differency(U, V):
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
    sum1 = 0
    i = 0
    all_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    select_list = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    while sum1 < sum:
        j = c[sum1]

        if sum1 < sum:
            try:
                percent_number = abs(ratings[sum1] - np.dot(U[i], V[j].T))
                for ss in range(12):
                    if percent_number < 0.5 * (ss + 1):
                        all_list[ss] += 1
                        break

            except Exception as er:
                print(er)
                break
            if r[sum1] != i:
                i += 1
        sum1 += 1
    temp = 0
    for sst in range(12):
        temp += all_list[sst]
        select_list[sst] = temp / sum
    print(select_list)


if __name__ == '__main__':
    # select_the_frency()
    # top_k_order()
    filename = 'E:\labor\\'
    # U = np.random.rand(138493, 15)
    # V = np.random.rand(131262, 15)
    # save('D:\Labor\\U', U)
    # save('D:\Labor\V', V)
    n = 138493
    m = 131262
    d = 15

    for i in range(4):
        filename_U = filename + f'U7_{i}.npy'
        filename_V = filename + f'V7_{i}.npy'
        get_rmse(filename_U, filename_V)

    '''
    for ssl in range(3):
        if ssl == 0:
            U = load('D:\\U0.npy')
            V = load('D:\V0.npy')
            print('The NP prediction!')
            percent_of_differency(U, V)
        elif ssl == 1:
            for sslt in range(3):
                tep = sslt + 1
                U = load(f'D:\\U0_{tep}.npy')
                V = load(f'D:\V0_{tep}.npy')
                print('The epsilion is 0.2, 0.6, 1.0 zh')
                percent_of_differency(U, V)
        else:
            for ssltt in range(3):
                tepp = ssltt + 1
                U = load(f'D:\\U2_{tepp}.npy')
                V = load(f'D:\V2_{tepp}.npy')
                print('The epsilion is 0.2, 0.6, 1.0 RS')
                percent_of_differency(U, V)
                
                
    '''
    # data = readfile(138493, 131262, 'D:\Labor\\ratings.txt')
    # U = np.load('D:\Labor\\U.npy')
    # V = np.load('D:\Labor\V.npy')
    # U = np.zeros((138493, 15))
    # for i in range(n):
    # U[i] = U1[i]

    # V = np.zeros((m, d))
    # for i in range(m):
    # V[i] = V1[i]

    '''
    U = load('D:\\U_double.npy')
    V = load('D:\V_double.npy')
    list_Vmax = []
    V0 = np.load('D:\V0.npy')
    for i in range(5):
        # j = i + 4
        machine_learning(i, U, V)
        # get_rmse(U, V)
        V1 = load(f'D:\V7_{i}.npy')
        tep = np.linalg.norm(V0 - V1, np.inf)
        list_Vmax.append(tep)
    print(list_Vmax)
    # get_percent(data)
    '''

    '''
    for number_tip in range(1):
        number_tipss = number_tip + 3
        for number in range(3):
            number += 2
            if number_tipss == 0:
                select_the_frency(((2 * number_tipss) + 1) / 10)
                top_k_order()
                zh_mf(number + 1, U, V, data, number_tipss)
            elif number_tipss == 1:
                select_the_frency(((2 * number_tipss) + 1) / 5)
                top_k_order()
                zh_mf(number + 1, U, V, data, number_tipss)
            else:
                zh_mf(number + 1, U, V, data, number_tipss)
    '''

    '''
    for number in range(3):
        filename1 = 'D:\V0_' + str(number + 1) + '.npy'
        filename2 = 'D:\\U0_' + str(number + 1) + '.npy'

        V1 = load(filename1)
        U1 = load(filename2)
        print(np.linalg.norm(V0 - V1, np.inf))
    '''

    '''
    for number in range(1):
        select_the_frency(10 * 10)
        top_k_order()
        # number_k = number + 11
        get_percent(data)
        # zh_mf((number + 1), U, V, data)
        # machine_learning(number + 1, U, V)
    '''


        # filename3 = 'D:\V0.npy'
        # filename4 = 'D:\\U0.npy'
        # get_rmse(filename2, filename1)

    # top_k_order()
    # machine_learning(0, U, V)

    # zh_mf(1, U, V, data)
    # U0 = np.load('D:\\U0.npy')
    '''
    V0 = np.load('D:\V0.npy')
    # zh_rmse(U0, V0)
    rmse_list = []
    Vmax_list = []
    for i in range(6):
        filename1 = 'D:\\U5_' + f'{i}.npy'
        filename2 = 'D:\V5_' + f'{i}.npy'
        tem = get_rmse(filename1, filename2)
        rmse_list.append(tem)
        V1 = load(filename2)
        tem1 = np.linalg.norm(V0 - V1, np.inf)
        Vmax_list.append(tem1)
    print(rmse_list)
    print(Vmax_list)
    '''

    # machine_learning(19820562, U, V)
    # np.save('D:\\U', u)
    # np.save('D:\V', v)
    # ratings = load('D:\Labor\\ratings.npy')
    # local_differencial_privacy(0.01, 1)
    # local_differencial_privacy(0.1, 2)
    # local_differencial_privacy(1, 3)
    # for number in range(1):
    # get_rmse(f'D:\\U{2}.npy', f'D:\V{2}.npy')
    # get_rmse('D:\\U138492.npy', 'D:\V138492.npy')
    # get_rmse('D:\\U19820562.npy', 'D:\V19820562.npy')
    # get_all_remse(ratings, 10)
    # getNM('D:\Labor\\ratings.txt')
    # ratings = readfile(671, 163949, 'D:\\ratings.txt')
    # save('D:\\ratings', ratings)
    # get_rmse(ratings, filename1, filename2)
    # get_rmse(ratings, filename3, filename4)
    # 当采用抽样检查获得前一千个用户的近似评分总数：150629，以下为学习结果
    '''
        epsilon = [0.01, 0.1, 1]
        rmse_np = [1.4315024934913296, 1.3855245310951287]
        rmse_zh = [1.4310252492420061, 1.4310252492420061, 1.4310252492420061]
        rmse = {'iteration': 10, 'rmse': 1.8764717760746124, 'm': 131262, 'n': 138493, 'n1': 1000, 'number': 150629, 'random_number': 131262}
        dict1 = {'ratings_number': 20000263, 'random_number': 19820562}
        # 当采用所有138492个用户的评分进行机器学习时，以下为学习结果：
        epsilon = [0.01, 0.1, 1]
        rmse_q = [1.3710723357794279]
        rmse_np = [1.3714400341684307, 1.367946042691678]
        rmse = {'V': 1.3714400341684307, 'V0': 1.367946042691678, 'V1': 1.3710723357794279, 
        'V2': 1.3264490477880087, 'V8': 1.3568300899598797, 'V9': 1.3705171729140744}
        rmse = {'eta = 10;number = 150': 1.3568300899598797,'eta = 9;number = 169': 1.3556879499507695,
         'eta = 7;number = 312': 1.3576327418265601,'eta = 5;1091':1.3523195975138098, 'eta = 6;494': 1.3574081571207508
         eta = '7;284': 1.3537251025648394, 'eta = 8;203': 1.355827704358222}
        Vmax = {}
        
    '''
    '''
    data = readfile(138492, 131262, 'D:\Labor\\ratings.txt')
    for i in range(4):
        print(f'eta = {i+5}')
        get_RR_top_k(i+5, data)
        top_k_order()

        U = load('D:\Labor\\U.npy')
        # print(shape(U))
        V = load('D:\Labor\V.npy')
        zh_mf(i+5, U, V, data)
        # top_k = load('D:\order_list.npy')
        # zh_mf(9, U, V, data)
        # np.save('D:\Labor\\1000ratings', data[:1000][:1000])
        # get_top_k()
        # top_k_order()
        get_rmse(f'D:\\U{i+5}.npy', f'D:\\V{i+5}.npy')
        '''
    eta = [11, 12, 13, 14, 15, 16]
    Vmax_random_select = [20.387821226183668, 18.75416671700819, 17.21558649708432, 16.506450206425114, 15.9792887782977, 13.863720971517717]
    Vmax_frency_select = [520.3349216675277, 512.2784811728112, 483.29434433329897, 401.474858868435, 401.702152713404, 360.0512776711606]
    Vmax_random_sub = [3.7252867028330567, 3.6495745190411233, 3.677822577050982, 3.5697825509893275]
    Vmax_number = {'507': 1}

    hitter_pencent = {'NP,7300,V00': 0.96356765}

    Vmax_random = [0.482060, 2.043970, 2.043970, 2.043970, 1.369870, 0.824813, 0.610801, 0.547482, 0.572333, 0.708466]
    Vmax_one = [0.052222, 0.026184, 0.015611, 0.012269, 0.009356, 0.008046, 0.006663, 0.006522, 0.005360, 0.005202]
    Rmse_random = [1.168776, 1.183729, 1.183851, 1.180520, 1.174393, 1.171997, 1.171280, 1.170965, 1.170950, 1.170909]
    Rmse_one = [1.167425, 1.167417, 1.167429, 1.167431, 1.167431, 1.167426, 1.167428, 1.167425, 1.167426, 1.167426]

    # U3/V3 eta = [0.01, 0.1, 1]  number = 5000
    # VmaxRandom3 = [1.4589328020305923, 1.4000514078197601, 1.5338089800959858]
    # Rmse_random3 = [1.187680532946816, 1.1893944644922307, 1.1895439286344838]
    # Rmse_NP = [1.1674273606475294]

    # U0/V0 eta = [0.2, 0.6, 1] number = 5000   加第三方
    Vmax_random0 = [1.3516330202296076, 1.4224023537477741, 1.5338089800959858]
    Rmse_random0 = [1.1862244838086082, 1.1856310925094316, 1.1854331021659894]

    # U1/V1 eta = [0.2, 0.6, 1] number = 5000  不加第三方
    Vmax_random1 = [111622,  110708, 106457]
    Rmse_random1 = [1.1874350707807253, 1.188895074871264, 1.1893552342248943]

    # U2/V2 eta = [0.2, 0.6, 1] number =5000  加第三方 随机选择
    Vmax_random2 = [3.9014927315240433, 3.9050503242219277, 3.895731279364963]
    Rmse_random2 = [1.3507068190190155, 1.3507081159766927, 1.3506705008700235]

    # U3/V3 eta = [0.2, 0.6, 1] number = 5000 不加第三方 随机选择
    Vmax_random3 = [9490]
    Rmse_random3 = []

    # U5/V5 eta = [10, 12, 14, 16, 18, 20] * 2.5 number = all 加第三方
    Vmax_5 = [1.293804, 1.326500, 1.3609815968915122, 1.4218918188363094, 1.4973061087810735, 1.6304191503588146]
    Rmse_5 = [9.013175, 10.83782, 10.8376745611863, 13.093554037962319, 15.107029094515351, 19.862034294960758]

    # U6/V6 eta = [10, 12, 14, 16, 18, 20] number = all 加第三方
    Vmax_6 = [24.834135695369763, 25.351933824623263, 27.123895715338413, 30.477977785870607]
    
    # U7/V7 eta = [10, 12, 14, 16, 18, 20] / 10 number = all 加第三方

    # percent eta = 0.1 , 1 , 10  and NP top-k from 1000 to 10000
    eta_0_1 = [0.63124885, 0.7907007, 0.84842425, 0.85655335, 0.8624762, 0.88454875, 0.88468025, 0.8853961, 0.8861833, 0.88738235, 0.88738235]
    eta_1 = [0.6317457, 0.79112805, 0.8464938, 0.8549998, 0.86097345, 0.88326715, 0.88336345, 0.8838681, 0.8848709, 0.8861775, 0.8861775]
    eta_10 = [0.63317425, 0.79075645, 0.8479411, 0.8544186, 0.8646958, 0.88374055, 0.88400235, 0.8848213, 0.88619865, 0.88784605, 0.88784605]
    frency_NP = [0.6333224, 0.7936918, 0.86221175, 0.88106805, 0.89377145, 0.89527535, 0.89752025, 0.90220665, 0.9046658, 0.92555175, 0.92555175]

    # U1/V1 eta = [0.2, 0.6, 1] number = 5000  不加第三方
    Vmax_random1 = [106457, 1.4589328020305923, 1.3516330202296076]
    Rmse_random1 = [1.1874350707807253, 1.188895074871264, 1.1893552342248943]