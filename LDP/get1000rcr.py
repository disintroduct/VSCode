import numpy as np

filename = 'E:\Labor\\'

column = np.load(filename + 'm.npy')
column_list = np.load(filename + 'list_in_order4.npy')
m = len(column)

for j in range(m):
    if column[j] in column_list:
        t = 1
    else:
        column[j] = -1

np.save(filename + 'column_list', column)
print(column)
