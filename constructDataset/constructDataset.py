import os 
import shutil
import numpy as np 

# for root,_,files in os.walk('dataset/searchHiddenSimilarCode/leetcode'):
#     for file in files:
#         path1=os.path.join(root,file)
#         label=path1.split(os.path.sep)[-2]
#         if file.startswith('1'):
#             path2=os.path.join('dataset/searchHiddenSimilarCode/positive1',str(label)+'.cpp')
#         elif file.startswith('2'):
#             path2=os.path.join('dataset/searchHiddenSimilarCode/positive2',str(label)+'.cpp')  

#         shutil.copy(path1,path2)

# cnt=1
# for root,_,files in os.walk('/home/cheng/Desktop/104_code'):
#     for file in files:
#         path1=os.path.join(root,file)
#         path2=os.path.join('/home/cheng/Desktop/negative',str(cnt)+'.cpp')
#         cnt+=1
#         shutil.copy(path1,path2)
# print(cnt)

# index=np.random.randint(1,40000,size=100)


# cnt=1
# for i in index:
#     path1=os.path.join('/home/cheng/Desktop/negative',str(i)+'.cpp')
#     path2=os.path.join('/home/cheng/Desktop/negative2',str(cnt)+'.cpp')
#     shutil.move(path1,path2)
#     cnt+=1


# positive_path=[]
# negative_path=[]
# for root,_,files in os.walk('dataset/searchHiddenSimilarCode/positive2'):
#     for file in files:
#         positive_path.append(os.path.join(root,file))


# for root,_,files in os.walk('dataset/searchHiddenSimilarCode/negative2'):
#     for file in files:
#         negative_path.append(os.path.join(root,file))

# cnt=1
# for i in range(100):
#     f1=open(positive_path[i],'r')
#     f2=open(negative_path[i],'r')
#     merge_string=f1.read()+"\n"+f2.read()
    
#     path_w=os.path.join('dataset/searchHiddenSimilarCode/mergePositive',str(cnt)+'.cpp')
#     with open(path_w,'w') as f:
#         f.write(merge_string)
#     cnt+=1

cnt=101
for root,_,files in os.walk('/home/cheng/Desktop/negative'):
    for file in files:
        path1=os.path.join(root,file)
        path2=os.path.join(root,str(cnt)+'.cpp')
        os.rename(path1,path2)
        cnt+=1
print(cnt-100)
