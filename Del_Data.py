

import numpy as np


def data_screen(file):
    f=np.loadtxt(file,dtype=np.str_)
    #print('原先数据集大小',len(f))
    train=[]
    for j in range(2,4):
        count=0
        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        for i in range(len(f)):
            a=f[i].split(sep='|', maxsplit=-1)
            #print(a)
            if float(a[j])==0:
                count+=1
                if count<1000:#0类型选择1000个
                    train.append(f[i])
            elif float(a[j])==1:
                count1+=1
                if count1<1000:#2类型选择1000个
                    train.append(f[i])
            elif float(a[j])==2:
                count2+=1
                if count2<1000:#2类型选择1000个
                    train.append(f[i])
            elif float(a[j])==3:
                count3+=1
                 #train.append(f[i])#此处数量太小全用了
                if count1<1000:#2类型选择1000个
                    train.append(f[i])
            elif float(a[j])==4:
                count4+=1
                train.append(f[i])
                # if count1<100000:#2类型选择1000个
                #  train.append(f[i])
            elif float(a[j])==5:
                count5+=1
                # train.append(f[i])#此处数量太小全用了
                if count1<30000:#2类型选择1000个
                    train.append(f[i])
    np.savetxt("./data/face_test2.txt", train,fmt='%s',delimiter=' ')


#统计各类别的数量
def cal_class_num(file):
    f=np.loadtxt(file,dtype=np.str_)
    print('数据集大小',len(f))
    count_clear=0
    count_blur=0
    for k in range (len(f)):
        a=f[k].split(sep='|', maxsplit=-1)
        if float(a[1])<0.7:
            count_clear+=1
        else:
            count_blur+=1
    print('模糊各类别数',count_clear,count_blur)
    for j in range(2,6):
        count=0
        count1=0
        count2=0
        count3=0
        count4=0
        count5=0
        for i in range(len(f)):
            a=f[i].split(sep='|', maxsplit=-1)
            #print(a)
            if float(a[j])==0:
                count+=1
            elif float(a[j])==1:
                count1+=1
            elif float(a[j])==2:
                count2+=1
            elif float(a[j])==3:
                count3+=1
            elif float(a[j])==4:
                count4+=1
            elif float(a[j])==5:
                count5+=1
        if j==2:
            print('左眼各类别数:',count+count2,count3+count1,count4,count5)
        elif j==3:
            print('右眼各类别数:',count+count2,count3+count1,count4,count5)
        elif j==4:
            print('嘴巴各类别数:',count,count1+count2+count3)
            print('*'*70)


#数据集划分
def Data_division(file):
    f=np.loadtxt(file,dtype=np.str_)
    train=[]
    val=[]
    #数据集划分为7：3
    for i in range(len(f)):
        if i%4==0:
            val.append(f[i])
        else:
            train.append(f[i])
    np.savetxt("./data/val_new.txt", val,fmt='%s',delimiter=' ')
    np.savetxt("./data/train_new.txt", train,fmt='%s',delimiter=' ')
    print('训练集大小：',np.array(train).shape,'测试集大小：',np.array(val).shape)



if __name__ == '__main__':
    #统计筛选后的各类别数大小，根据自己数据调整
    file='./data/face_attributes.txt'
    cal_class_num(file)

    #首先筛选数据，大致保证各类别数相差不要太大，需根据自己数据调整
    file='./data/face_attributes_new.txt'#筛选后保存在face_test1.txt中
    data_screen(file)

    #统计筛选后的各类别数大小，根据自己数据调整
    file='./data/face_test2.txt'
    cal_class_num(file)

    #数据集划分，分为测试集和训练集，保存在对应train.txt和val.txt中
    file='./data/face_test2.txt'
    Data_division(file)
