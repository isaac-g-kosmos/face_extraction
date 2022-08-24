
def augmented_cut(pic_width,pic_height,top,left,width,height,leeway=0):
    top=top/pic_height
    left=left/pic_width
    width=width/pic_width
    height=height/pic_height
    if leeway!=0:
        leeway=height*leeway
        top=top-leeway
        height=height+leeway
    return top,left,width,height





#%%

# import pandas as pd
# import matplotlib.pyplot as plt
#
# #%%
# df=pd.read_csv(r'C:\Users\isaac\PycharmProjects\face_exctraction\dataset1.csv')
#%%
from PIL import Image
import numpy as np
# import random
# import os
# sample_path=df['path'][1]
# sample_path=os.path.join(r'C:\Users\isaac\PycharmProjects\face_exctraction\dlib_face_detection_dataset',sample_path)
# sample_image=Image.open(sample_path)
# sample_image=np.array(sample_image)
# plt.imshow(sample_image)
# plt.show()
# #%%
# points=(df['top'][1],df['left'][1],df['width'][1],df['height'][1])
# fig,ax=plt.subplots(1)
# ax.imshow(sample_image)
# ax.add_patch(plt.Rectangle((points[1],points[0]),points[2],points[3],fill=False,edgecolor='red',linewidth=3))
# plt.show()
# #%%
# ##aply augmented cut to all the boxes
# top,left,width,height=augmented_cut(sample_image.shape[0],sample_image.shape[1],df['top'][1],df['left'][1],df['width'][1],df['height'][1],.2)
# fig,ax=plt.subplots(1)
# ax.imshow(sample_image)
# ax.scatter(left*sample_image.shape[0],top*sample_image.shape[1],s=10,c='red')
# ax.scatter((left+width)*sample_image.shape[0],(top+height)*sample_image.shape[1],s=10,c='red')
#
# plt.show()
# #%%
#
# top,left,width,height=augmented_cut(sample_image.shape[0],sample_image.shape[1],df['top'][1],df['left'][1],df['width'][1],df['height'][1],0)
# fig,ax=plt.subplots(1)
# ax.imshow(sample_image)
# ax.scatter(left*sample_image.shape[0],top*sample_image.shape[1],s=10,c='red')
# ax.scatter((left+width)*sample_image.shape[0],(top+height)*sample_image.shape[1],s=10,c='red')
#
# plt.show()