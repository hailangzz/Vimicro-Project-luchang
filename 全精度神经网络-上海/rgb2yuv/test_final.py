import cv2
import os
from scipy.special import logit
import pandas as pd

dir='beijing_08_17_24-4-0/'


log_path='log/'+dir
path='/home/public/2.0.8/'

flag='bai'
cnt=4
t1=17
t2=35
t3=60
save_apth1='result/t2/'+dir
t=0
f_w=open('result/result.txt','a')
log_list=os.listdir(log_path)

def crop(x,y):
    if x<0:
        x=0
    if x>y:
        x=y
    return x



def expland(rect,ratio,shape):
    xmin,ymin,xmax,ymax=rect
    w=xmax-xmin
    h=ymax-ymin
    w_e=int(w*ratio)//2
    h_e=int(h*ratio)//2
    x=crop(xmin-w_e,shape[1])
    x1=crop(xmax+w_e,shape[1])
    y=crop(ymin-h_e,shape[0])
    y1=crop(ymax+h_e,shape[0])
    return [x,y,x1,y1]

def InverseSigmoid( x):
    x = logit(x)
    return x


for log in log_list:
    path_f=log_path+log_f+'/'
    
    
    right_17=0
    right_35=0
    
    print(log)
    f=open(path_f+log,'r')
    if log[:-4].replace('result_','') in ['04-qujingyoudeng','04-qujingyoudeng_ori','05-qujingwudeng','05-qujingwudeng_ori','06-heibai']:
        t=t1
    else:

        t=t2
    id_q={}
    lines=f.readlines()
    
    for i in range(len(lines)):
        if 'save to' in lines[i]:
            img_name=lines[i].split(',')[0].split('/')[-1]
            name=str(int(img_name[:-4]))+img_name[-4:]
            num=int(lines[i+1].strip().split(':')[1])
            if 'ori' in log[:-4]:
                print(path_ori+log[:-4].replace('result_','')+'/'+name)
                img=cv2.imread(path_ori+log[:-4].replace('result_','')+'/'+name)
            else:
                img=cv2.imread(path+log[:-4].replace('result_','')+'/'+name)
            for j in range(num):
                list_=lines[i+j+2].split(':')[1].split(' ')
                #print(list_)
                id_=list_[0]
                q_=float(list_[6])
                #q_=pow(float(list_[6])/100,2)*100
                conf=float(list_[5])
                #if conf>0.58:
                rect1=list(map(int,list_[1:5]))



                #print(list_[1:5],rect)
                # if 'tongdao' in log[:-4]:
                #     rect1[0]=rect1[0]*3
                #     rect1[1]=rect1[1]*1080/384
                #     rect1[2]=rect1[2]*3
                #     rect1[3]=rect1[3]*1080/384
                rect=expland(rect1,0.5,img.shape[:2])
                face=img[rect[1]:rect[3],rect[0]:rect[2]]
                if id_ not in id_q.keys():
                    id_q[id_]=[[q_],[conf],[name],[face]]
                else:
                    list_q_r=id_q[id_]
                    list_q,list_c,list_n,list_r=list_q_r[0],list_q_r[1],list_q_r[2],list_q_r[3]
                    list_q.append(q_)
                    list_c.append(conf)
                    list_n.append(name)
                    list_r.append(face)
                    id_q[id_]=[list_q,list_c,list_n,list_r]
    for id in id_q.keys():
        list_q_r=id_q[id]
        list_q,list_c,list_n,list_r=list_q_r[0],list_q_r[1],list_q_r[2],list_q_r[3]
        if sum(i>=t2 for i in list_q)>cnt:
            right_35+=1
        if sum(i>=t1 for i in list_q)>cnt:
            right_17+=1
        max_=max(list_q)
        if max_<t:
            continue
        
        if max_>=t :
            index=list_q.index(max_)
            face=list_r[index]
            conf = list_c[index]
            name=list_n[index]
            save_apth=save_apth1+log[:-4]+'/{}/'.format(t)
            if not os.path.exists(save_apth):
                os.makedirs(save_apth)
            cv2.imwrite(save_apth+str(id)+'_'+str(conf)+'_'+str(max_)+'_'+name,face)
    
    result=path_f+log+' '+str(right_35)+' '+str(right_17)+'\n'
    f_w.write(result)