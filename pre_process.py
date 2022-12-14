import numpy as np

from PIL import Image
import tensorflow as tf


def augmented_cut(pic_width, pic_height, top, left, width, height, leeway=0):
    top = top / pic_height
    left = left / pic_width
    width = width / pic_width
    height = height / pic_height
    if leeway != 0:
        veritcal_leeway = height * leeway
        top = top - veritcal_leeway/2
        height = height + veritcal_leeway
        horizontal_leeway = width * leeway
        left = left - horizontal_leeway/2
        width = width + horizontal_leeway
    return top, left, width, height



# %%
#
# import matplotlib.pyplot as plt
#
# #%%
# # %%
# from PIL import Image
# import pandas as pd
# import numpy as np
# import random
# import os
# import matplotlib.pyplot as plt
# df=pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset1.csv')
# idx=random.randint(0,len(df)-1)
# sample_path=df['path'][idx]
# sample_path=os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',sample_path)
# sample_image=Image.open(sample_path)
# sample_image=np.array(sample_image)
# # plt.imshow(sample_image)
# # plt.show()
#
# points=(df['top'][idx],df['left'][idx],df['width'][idx],df['height'][idx])
#
# ##aply augmented cut to all the boxes
# top,left,width,height=augmented_cut(sample_image.shape[0],sample_image.shape[1],df['top'][idx],df['left'][idx],df['width'][idx],df['height'][idx],.1)
# fig,ax=plt.subplots(1)
# ax.imshow(sample_image)
# ax.add_patch(plt.Rectangle((points[1],points[0]),points[2],points[3],fill=False,edgecolor='red',linewidth=3))
# ax.add_patch(plt.Rectangle((left*sample_image.shape[0],top*sample_image.shape[1]),width*sample_image.shape[0],height*sample_image.shape[1],fill=False,edgecolor='green',linewidth=3))
# fig.show()

#%%
def bb_box_metrics(pred,label):
    #cordinates x1,y1,y2,y2
    #calculate the intesection of abounding box
    x1=max(pred[0],label[0])
    y1=max(pred[1],label[1])
    x2=min(pred[2],label[2])
    y2=min(pred[3],label[3])
    if x2<x1 or y2<y1:
        intersection=0
    else:
        intersection=(x2-x1)*(y2-y1)

    #calculate the union of abounding box
    union=(pred[0]-pred[2])*(pred[1]-pred[3])+(label[0]-label[2])*(label[1]-label[3])-intersection
    #calculate the IoU
    accuracy=intersection/union
    presicion=intersection/((pred[2]-pred[0])*(pred[3]-pred[1]))
    recall=intersection/((label[2]-label[0])*(label[3]-label[1]))
    if (presicion+recall)==0:
        f1_score=0
    else:
        f1_score=2*presicion*recall/(presicion+recall)
    IoU=intersection/union
    return accuracy,presicion,recall,f1_score, IoU
def bb_box_metrics_extension(pred,label):
    #calculate the intesection of abounding box
    x1=max(pred[0],label[0])
    y1=max(pred[1],label[1])
    x2=min(pred[0]+pred[2],label[0]+label[2])
    y2=min(pred[1]+pred[3],label[1]+label[3])
    if x2<x1 or y2<y1:
        intersection=0
    else:
        intersection=(x2-x1)*(y2-y1)

    #calculate the union of abounding box
    union=(pred[2]*pred[3])+(label[2]*label[3])-intersection
    #calculate the IoU
    accuracy=intersection/union
    presicion=intersection/(pred[2]*pred[3])
    recall=intersection/(label[2]*label[3])
    if (presicion + recall )== 0:
        f1_score = 0
    else:
        f1_score = 2 * presicion * recall / (presicion + recall)
    IoU=intersection/union
    return accuracy,presicion,recall,f1_score, IoU
#%%
class BB_Metrics:
    def __init__(self):
        self.accuracy = 0
        self.recall = 0
        self.Iou = 0
        self.presicion = 0
        self.f1 = 0
        self.count = 0
    def update(self,pred,label,wh=False):

        for x in range(len(pred)):
            if wh:
                accuracy, presicion, recall, f1_score, IoU = bb_box_metrics_extension(pred[x], label[x])
            else:
                accuracy,presicion,recall,f1_score, IoU = bb_box_metrics(pred[x],label[x])
            self.accuracy += accuracy
            self.presicion += presicion
            self.recall += recall
            self.f1 += f1_score
            self.Iou += IoU
        self.count += len(pred)

    def get_metrics(self,pre_fix=''):
        #return a dict of metrics with the appropiate pre-fix
        return {pre_fix+'_accuracy':self.accuracy/self.count,
                pre_fix+'_presicion':self.presicion/self.count,
                pre_fix+'_recall':self.recall/self.count,
                pre_fix+'_f1':self.f1/self.count,
                pre_fix+'_Iou':self.Iou/self.count}
    def clear_data(self):
        self.accuracy = 0
        self.presicion = 0
        self.recall = 0
        self.f1 = 0
        self.Iou = 0
        self.count = 0
#%%
# met=BB_Metrics()
# met.update(train_landmarks,train_slice)
# #%%
# met.get_metrics('BB_train')
# #%%
# met.clear_data()
# #%%
# met.get_metrics()