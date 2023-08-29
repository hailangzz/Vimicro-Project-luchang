def dizeng(list,diff_number = 7):
    l1 = []
    l2 = []
    for i in range(0,len(list)-1):
        if list[i]+diff_number==list[i+1]:
            l2.append(list[i])
            l2.append(list[i+1])
            if i==len(list)-2:
                l1.append(l2)
        else:
            l1.append(l2)
            l2=[]
            continue

    l1_1 = [i for i in l1 if i]
    l1_2 = []
    for a in l1_1:
        list2 = []
        [list2.append(i) for i in a if not i in list2]
        l1_2.append(list2)
    return(l1_2)


list = [7,14,21,28,35,49,56,63,147,154,161,167]

print(dizeng(list))