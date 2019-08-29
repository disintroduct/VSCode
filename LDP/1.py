list_o = input('list')
list_1 = list_o[:]
list_o += list_1
l = len(list_o)
sum_max = 0
for i in range(l):
    sum_l = 0
    if list_o[i] == '1':
        for j in range(l-i):
            if list_o[i + j] == '1':
                sum_l += 1
            else:
                i = i + j
                if sum_l > sum_max:
                    sum_max = sum_l
                break
print(sum_max)


s = int(input())
ss = input()
ss_list = []
click = 0
l = 0
j = s - 1
while j > 0:
    if ss_list[j] > ss_list[j-1]：
        click += 1
        j -= 1
    else:
        break
print(s - click)
