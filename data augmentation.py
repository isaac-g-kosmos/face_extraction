import tensorflow as tf
import numpy as np
import albumentations as alb
from PIL import Image
import matplotlib.pyplot as plt
import os
import pandas as pd
import json
from uuid import uuid4
output_path = r'C:\Users\isaac\PycharmProjects\face_exctraction\augmented_pics'
df = json.load(open(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset.json'))
augmentor = alb.Compose([alb.RandomCrop(width=180, height=180),
                        alb.HorizontalFlip(p=0.5),
                         alb.augmentations.geometric.rotate.Rotate(limit=10, p=0.5),
                         alb.RandomBrightnessContrast(p=0.2),
                         alb.RandomGamma(p=0.2),
                         alb.RGBShift(p=0.2),
                         alb.Flip(p=0.2),
                         alb.VerticalFlip(p=0.5)],
                        bbox_params=alb.BboxParams(format='coco',
                                                  label_fields=['class_labels']))
#%%
path=os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',"Data/CLS-LOC/test/ILSVRC2012_test_00045844.JPEG_RESAMPLED_fc924a80820eb0ce.png")
image=Image.open(path)
image=np.array(image)
image=augmentor(image=image,bboxes=[],class_labels=[])
#%%
#path,pic_width,pic_height,top,left,width,height,zero,one,many
maxes=[]
paths=[]
pic_height=[]
pic_width=[]
top=[]
left=[]
zero=[]
one=[]
many=[]
final_dict={}
count=7213
for x in range(2):
    for key in df.keys():
        print(count)
        try:
            new_dict={}
            inner_dict=df[key]
            path=os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',
                             inner_dict['path'].replace('Data/','data/'))
            paths.append(path)
            img=Image.open(path)
            img=np.array(img)
            face=[]
            boxes=[]
            if 'boxes' in df[key].keys():
                face=['face'+str(i) for i in range(len(df[key]['boxes']))]
                for box in df[key]['boxes']:
                    boxes.append(np.array(box))


            augmented=augmentor(image=img,bboxes=boxes,class_labels=face)
            augmented_img=augmented['image']
            augmented_boxes=augmented['bboxes']
            name=str(uuid4())
            output_name=os.path.join(output_path, name + '.jpg')
            Image.fromarray(augmented_img).save(output_name)

            new_dict['path']=output_name
            new_dict['boxes']=augmented_boxes
            new_dict['width'] = (df[key]['width'])
            new_dict['height'] = (df[key]['height'])
            final_dict[count]=new_dict
            count+=1
        except:
            pass
#%%
with open(r'C:\Users\isaac\PycharmProjects\face_exctraction\augmented_dataset.json', 'w') as f:
    json.dump(final_dict, f)