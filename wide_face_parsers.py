import os
import tqdm
import pickle
import argparse
import numpy as np
from scipy.io import loadmat
# from bbox import bbox_overlaps
#%%

def get_gt_boxes(gt_dir):
    """ gt dir: (wider_face_val.mat, wider_easy_val.mat, wider_medium_val.mat, wider_hard_val.mat)"""

    gt_mat = loadmat(os.path.join(gt_dir, 'wider_face_val.mat'))
    hard_mat = loadmat(os.path.join(gt_dir, 'wider_hard_val.mat'))
    medium_mat = loadmat(os.path.join(gt_dir, 'wider_medium_val.mat'))
    easy_mat = loadmat(os.path.join(gt_dir, 'wider_easy_val.mat'))

    facebox_list = gt_mat['face_bbx_list']
    event_list = gt_mat['event_list']
    file_list = gt_mat['file_list']

    hard_gt_list = hard_mat['gt_list']
    medium_gt_list = medium_mat['gt_list']
    easy_gt_list = easy_mat['gt_list']

    return facebox_list, event_list, file_list, hard_gt_list, medium_gt_list, easy_gt_list
path=r'C:\Users\isaac\Downloads\wider_face_split\wider_face_split'
files= os.listdir(path)
#%%
gt_mat = loadmat(os.path.join(path, 'wider_face_val.mat'))
gt_mat1= loadmat(os.path.join(path, 'wider_face_train.mat'))
gt_mat2= loadmat(os.path.join(path, 'wider_face_test.mat'))
#%%
# gt_mat['file_list']
#%%
def get_gt_boxes_from_txt(gt_path, cache_dir):

    cache_file = os.path.join(cache_dir, 'gt_cache.pkl')
    if os.path.exists(cache_file):
        f = open(cache_file, 'rb')
        boxes = pickle.load(f)
        f.close()
        return boxes

    f = open(gt_path, 'r')
    state = 0
    lines = f.readlines()
    lines = list(map(lambda x: x.rstrip('\r\n'), lines))
    boxes = {}
    print(len(lines))
    f.close()
    current_boxes = []
    current_name = None
    for line in lines:
        if state == 0 and '--' in line:
            state = 1
            current_name = line
            continue
        if state == 1:
            state = 2
            continue

        if state == 2 and '--' in line:
            state = 1
            boxes[current_name] = np.array(current_boxes).astype('float32')
            current_name = line
            current_boxes = []
            continue

        if state == 2:
            box = [float(x) for x in line.split(' ')[:4]]
            current_boxes.append(box)
            continue

    f = open(cache_file, 'wb')
    pickle.dump(boxes, f)
    f.close()
    return boxes
data=get_gt_boxes_from_txt(os.path.join(path, 'wider_face_val_bbx_gt.txt'),os.path.join(path,'val'))
data1=get_gt_boxes_from_txt(os.path.join(path, 'wider_face_test_filelist.txt'),os.path.join(path,'test'))
data2=get_gt_boxes_from_txt(os.path.join(path, 'wider_face_train_bbx_gt.txt'),os.path.join(path,'train'))


#%%
#%%
from PIL import Image
incial_path=r'C:\Users\isaac\Downloads\WIDER_test\WIDER_test\images'
paths=[]
tops=[]
lefts=[]
width=[]
height=[]
zero=[]
one=[]
many=[]
pic_width=[]
pic_height=[]
count=0
dict_count=0
dic_paths={0:r"val",
            1:r"test",
            2:r"train"}
dictionaries=[data,data1,data2]
#%%
for dictionary in dictionaries:
    for x in dictionary.keys():
        try:
            line=dictionary[x]
            if len(line)==1:
                img=Image.open(os.path.join(dic_paths[dict_count],x))
                paths.append(x)

                top,left,w,h=line[0]
                tops.append(top)
                lefts.append(left)
                width.append(w)
                height.append(h)
                zero.append(0)
                one.append(1)
                many.append(0)
            elif len(line)==0:
                paths.append(x)
                tops.append(0)
                lefts.append(0)
                width.append(0)
                height.append(0)
                zero.append(1)
                one.append(0)
                many.append(0)
            else :
                paths.append(x)
                tops.append(0)
                lefts.append(0)
                width.append(0)
                height.append(0)
                zero.append(0)
                one.append(0)
                many.append(1)
        except:
            print(x)
            count+=1
            print(count)
            pass
    dict_count+=1
#%%
import pandas as pd
df_train=pd.DataFrame({'path':paths,'top':tops,'left':lefts,'width':width,'height':height,'zero':zero,'one':one,'many':many})
df_train.to_csv(r'wider_train_dataset.csv',index=False)
