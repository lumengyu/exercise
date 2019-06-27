import numpy as np
# 导入图片处理的API
from PIL import Image

im = Image.open("../data/phone.jpg")
print(im,type(im))
# 图--> np.array类型
im =np.array(im)
#  0 255
print('原始矩阵:',type(im),im.shape,im.dtype)
# np.arange(498*610*3).reshape(498, 610, 3)
im = [255,255,255] - im
print('新矩阵',type(im),im.shape,im.dtype)
im = Image.fromarray(im.astype(np.uint8))
im.save("../data/phone2.jpg")