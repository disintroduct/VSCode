import numpy as np
import part1000
import improve
import spares
'''
* get top-k
* compute the gradient
* randomize the gradient
* recovery the gradient
* compress the data set
'''


def improve_LDP(eta1, number_top_k, filename_number):
    # get top-k
    eta1 = 0.1
    number_top_k = 5000
    all_list = improve.select_the_frency(eta1)
    list_in_order4 = improve.top_k_order(all_list, number_top_k)

    # get gradient
    U = np.load('E:\Labor\\U.npy')
    V = np.load('E:\Labor\V.npy')
    improve.machine_learning(list_in_order4, filename_number, U, V)


def get_percent(U, V):
    r = np.load('E:\\Labor\\n.npy')
    c = np.load('E:\\Labor\\m.npy')
    # c = np.load('E:\Labor\column_list.npy')
    ratings = np.load('E:\\Labor\\r.npy')
    error_list = [0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
    sum_ratings = 20000000
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


def get_gaossin():
    np.random.seed(0)
    s = 1/(500 ** 0.5) * np.random.randn(500, 5000)
    np.save('E:\Labor\D4', s)
    ss = np.dot(s, s.T)
    sss = np.linalg.inv(ss)
    ssss = np.dot(s.T, sss)
    np.save('E:\Labor\D5', ssss)
   

if __name__ == '__main__':

    # improve_LDP(0.1, 5000, 8)
    '''
    U = np.load('E:\Labor\\U.npy')
    V = np.load('E:\Labor\V.npy')
    improve.machine_learning_LDP(0, U, V)
    '''
    
    # U8 是top5000的无损实验，U9 是全部的无损实验
    filename = 'E:\\limbi\\'
    filename1 = 'E:\\'

    # v2 = np.load(filename + 'V1000_21.npy')
    v1 = np.load(filename + 'V1000_10.npy')
    v3 = np.load(filename + 'V1000_41.npy')
    u2 = np.load(filename + 'U1000_31.npy')
    v2 = np.load(filename + 'V1000_31.npy')
    # u3 = np.load(filename + 'U1000_30.npy')
    u1 = np.load(filename + 'U1000_10.npy')
    u3 = np.load(filename + 'U1000_41.npy')
    u4 = np.load(filename + 'U1000_51.npy')
    v4 = np.load(filename + 'V1000_51.npy')
    print(np.linalg.norm(v1-v3, np.inf), np.linalg.norm(v1-v2, np.inf), np.linalg.norm(v1 - v4))
    l1 = spares.get_rmse(filename + 'U1000_31.npy', filename + 'V1000_31.npy')
    l2 = spares.get_rmse(filename + 'U1000_41.npy', filename + 'V1000_41.npy')
    l3 = spares.get_rmse(filename + 'U1000_51.npy', filename + 'V1000_51.npy')
    print(l1, l2, l3)

    '''
    filename = 'E:\\'
    v2 = np.load(filename + 'V1000_20.npy')
    v1 = np.load(filename + 'V1000_10.npy')
    v3 = np.load(filename + 'V1000_30.npy')
    u2 = np.load(filename + 'U1000_20.npy')
    # v3 = np.load(filename + 'V1000_30.npy')
    # u3 = np.load(filename + 'U1000_30.npy')
    u1 = np.load(filename + 'U1000_10.npy')
    u3 = np.load(filename + 'U1000_30.npy')

    get_percent(u1, v1)
    get_percent(u2, v2)
    get_percent(u3, v3)
    # part1000.machine_learning_LDP(11, u1, v1)
    '''
    # get_gaossin()