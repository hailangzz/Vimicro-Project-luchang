import os
point_number=98
index_list=[]
for i in range(point_number):
    index_list.append(1+i*2)
str_index = "".join(str(index_list))

print(str_index)

import os
point_number=98
index_list=[]
for i in range(point_number*2):
    index_list.append(8+i)
str_index = "".join(str(index_list))

print(str_index,len(index_list)+8)


point_number=98
index_list=[]
for i in range(point_number*2):
    if i%2==0:
        index_list.append(3)
    else:
        index_list.append(2)
str_index = "".join(str(index_list))

print(str_index,len(index_list)+8)