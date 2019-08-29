import numpy as np
import math
import common_function as cf


# 思路：因每次迭代每个用户只提交一个物品梯度，故最多提交k个物品梯度
# 现不妨从所有物品中筛选k个满足差分隐私的物品编号，每次迭代随机提交其中一个物品的梯度值
# 不放会抽样
def sample_list(li, num):
    total = len(li)
    res = []
    for i in range(num):
        t = np.random.randint(i, total-1)
        res.append(li[t])
        li[t], li[i] = li[i], li[t]
    return(res)


# 第一步：得到与迭代次数相等的物品编号
def get_items(r, c, ratings, m, n, sum11, k, index, eta):
    index = index + "iR//"
    sum_ratings = len(ratings)
    print('计算相关物品编号...')
    cf.show_time()
    eta_k = eta/k
    pr = math.exp(eta_k) / (math.exp(eta_k) + 1)
    sum1 = 0
    i = 0
    # 首先选择k个用户相关物品：1若相关物品大于k，则随机从中选k个；
    # 2若相关物品小于k，则不足的数量由非相关物品编号替换
    user_list = []
    other_list = []
    sum_list = []
    user_array = np.zeros((n, k))
    for number_i in range(m):
        sum_list.append(number_i)
    other_list = sum_list[:]
    while sum1 < sum_ratings:
        j = c[sum1]
        user_list.append(j)
        other_list.remove(j)
        sum2 = sum1 + 1
        if sum2 < sum_ratings:
            try:
                test = r[sum2]
            except Exception as er:
                print(er, test)
                break
            if r[sum2] != i:
                user_list_len = len(user_list)
                user_list_select = []
                if user_list_len > k:
                    user_list = sample_list(user_list, k)
                user_list_len = len(user_list)
                for user_list_number in range(user_list_len):
                    p = np.random.random()
                    if p > pr or p == pr:
                        user_list_select.append(user_list[user_list_number])
                user_list_len = len(user_list_select)
                if user_list_len < k:
                    t = k - user_list_len
                    other_list_select = sample_list(other_list, t)
                user_list_select += other_list_select
                user_list_len = len(user_list_select)
                if user_list_len > k:
                    user_list_select = sample_list(user_list_select, k)
                # print(len(user_list_select))
                # print(i)
                user_array[i] = user_list_select[:]
                user_list = []
                other_list = sum_list[:]
                i += 1
        sum1 += 1
    print(f'完成物品编号选择！! {sum11}')
    cf.show_time()
    np.save(index + f'array_list{sum11}', user_array)
    return user_array


# 第二步：根据相应的物品编号计算相应的梯度值
def get_granted(r, c, ratings, m, n, d, sum11, U, V, k, index, eta, rt, sum111):
    index = index + "iR//"
    # user_array = np.load(index + f"array_list{sum11}.npy")
    user_array = np.load(index + f"array_list0.npy")
    sum_ratings = len(ratings)
    lu = 10 ** -8
    lv = 10 ** -8
    t_sum = k * d * ((math.exp(eta/k)+1)/(math.exp(eta/k)-1))
    t_sum_1 = math.exp(eta/k)
    # 考虑迭代次数的影响
    # k = 10
    # k = 1
    for it in range(k):
        # 考虑不同的学习速率的影响
        rt_u = 1 / (it + 1) / rt
        rt_v = 1 / (it + 1) / rt
        # rt_v = 1 / (it + 1) / (k ** 2)
        # 验证实验
        # rt = 1 / (it+1) / (k ** 2)
        print(str(it) + '次迭代！')
        cf.show_time()
        dU = np.zeros((n, d))
        dV = np.zeros((m, d))
        sum1 = 0
        i = 0
        item_number = np.random.randint(0, 10)
        item_index = int(user_array[i][item_number])
        click = 0
        while sum1 < sum_ratings:
            j = c[sum1]
            T = np.dot(U[i], V[j].T)
            if j == item_index:
                click = 1
                granted_vetor = -2 * U[i] * (ratings[sum1] - T)
                item_number_g = np.random.randint(0, d)
                granted_number = granted_vetor[item_number_g]
                # 若选择的物品为相关物品，则进行如下梯度值计算
                t = granted_number
                if t > 1:
                    t = 1
                elif t < -1:
                    t = -1
                T = (t * (t_sum_1 - 1) + t_sum_1 + 1) / (2 * (t_sum_1 + 1))
                random_t = np.random.random()
                if random_t <= T:
                    dV[j][item_number_g] += t_sum
                else:
                    dV[j][item_number_g] -= t_sum
            # 遍历所有评分，计算并更新个人侧面向量
            dU[i] += (-2) * V[j] * (ratings[sum1] - T)
            sum2 = sum1 + 1
            if sum2 < sum_ratings:
                try:
                    test = r[sum2]
                except Exception as er:
                    print(er, test)
                    break
                if r[sum2] != i:
                    # 若选择的物品为不相关物品，则进行如下梯度值计算
                    if click == 0:
                        item_number_g = np.random.randint(0, d)
                        T = 1 / 2
                        random_t = np.random.random()
                        if random_t <= T:
                            dV[item_index][item_number_g] += t_sum
                        else:
                            dV[item_index][item_number_g] -= t_sum
                    # 选择下次的物品编号，更新标记值
                    item_number = np.random.randint(0, 10)
                    item_index = int(user_array[i][item_number])
                    click = 0
                    # 更新用户编号
                    i += 1
            sum1 += 1
        dV = dV / n
        dU /= n
        # 一次迭代结束，更新侧面矩阵
        U = U - rt_u * (dU + 2 * lu * U)
        V = V - rt_v * (dV + 2 * lv * V)
        cf.show_time()
    print('\n' + f'I have done it! {sum11}')
    # 进行k次迭代后，将侧面矩阵写入磁盘
    # 0为学习率为1/i，1为学习率1/k/i，2为学习率1/i/k*eta
    np.save(index + f'U_{sum111}{sum11}', U)
    np.save(index + f'V_{sum111}{sum11}', V)


if __name__ == "__main__":
    # 进入数据集目录
    index = 'E:\\movielens\\'
    r = np.load(index + 'n.npy')
    c = np.load(index + 'm.npy')
    # c = np.load('E:\\movielens\column_list.npy')
    ratings = np.load(index + 'r.npy')
    u = np.load(index + 'u.npy')
    v = np.load(index + 'v.npy')
    v1 = np.load(index + 'v_00.npy')

    # 重新定义文件夹目录，进入iteration算法目录
    # index = index + 'iR\\'
    k = 10
    eta_list = [0.00625, 0.0125, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8]
    m = 26744
    n = 138493
    d = 15
    # for i in range(6):
    # get_items(r, c, ratings, m, n, i, k, index1, eta_list[i])
    list_vmax = []
    k_list = [1, 2, 3, 4, 5, 20, 50]
    for ii in range(2):
        i = ii
        eta = eta_list[i]
        # user_array = np.load(index + "array_list1.npy")
        get_granted(r, c, ratings, m, n, d, ii+6, u, v, k, index, eta, 1, 0)
        # v2 = np.load(index + f'v_0{i}.npy')
        # list_vmax.append(np.linalg.norm(v1-v2, np.inf))
    print(list_vmax)
