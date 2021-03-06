# Opencv(2)_核心操作

[toc]

## 1.基础操作

### 1.1 pixel&attributes

- 使用array.item()获取像素，array.itemset()改变像素

```python
import cv2
import numpy as np
img = cv2.imread('data/messi5.jpg')
#pixel
print(img.item(10, 10, 2))
img.itemset((10, 10, 2), 100)
print(img.item(10, 10, 2))
#attributes
print(img.shape, img.dtype, img.size)
```

### 1.2 ROI

```python
roi = img[a:b,c:d]
```

### 1.3 channels

```python
b=img[;,;,0]
g=img[;.;,1]
r=img[;,;,2]
```

### 1.4 Border

- cv2.copyMarkBorder(img,top=1,bottom=1,left=1, right=1, [borderType=](#ps:borderType))

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img1 = cv2.imread('data/opencv_logo.png')
replicate = cv2.copyMakeBorder(img1, top=100, bottom=100, left=100, right=100, borderType=cv2.BORDER_REPLICATE)
reflect = cv2.copyMakeBorder(img1, 100, 100, 100, 100, cv2.BORDER_REFLECT)
reflect101 = cv2.copyMakeBorder(img1, 100, 100, 100, 100, cv2.BORDER_REFLECT_101)
wrap = cv2.copyMakeBorder(img1, 100, 100, 100, 100,cv2.BORDER_WRAP)
constant = cv2.copyMakeBorder(img1, 100, 100, 100, 100, cv2.BORDER_CONSTANT, value=[0,255,0])  # value 边界颜色

plt.subplot(231), plt.imshow(img1, 'gray'), plt.title('ORIGINAL')
plt.subplot(232), plt.imshow(replicate, 'gray'), plt.title('REPLICATE')
plt.subplot(233), plt.imshow(reflect, 'gray'), plt.title('REFLECT')
plt.subplot(234), plt.imshow(reflect101, 'gray'), plt.title('REFLECT_101')
plt.subplot(235), plt.imshow(wrap, 'gray'), plt.title('WRAP')
plt.subplot(236), plt.imshow(constant, 'gray'), plt.title('CONSTANT')
plt.savefig('saved_img/make_border')
plt.show()
```

## 2.运算

### 2.1 add

- cv2.add()-饱和运算

```python
x = np.uint8([250])
y = np.uint8([10])
print(cv2.add(x, y))  # 250+10 = 260 => 255
print(x + y)  # 250+10=260%256=4
```

### 2.2 bitwise

- cv2.bitwise_and()
- cv2.bitwise_not()

```python
import cv2
import numpy as np

img1 = cv2.imread('data/messi5.jpg')  
img2 = cv2.imread('data/opencv-logo.png')  
img2 = cv2.resize(img2, (0, 0), fx=0.2, fy=0.2, interpolation=cv2.INTER_CUBIC)
rows, cols, channels = img2.shape
roi = img1[:rows, :cols]

img2gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
ret, mask = cv2.threshold(img2gray, 245, 255, cv2.THRESH_BINARY)

mask_inv = cv2.bitwise_not(mask)
img1_bg = cv2.bitwise_and(roi, roi, mask=mask)
img2_fg = cv2.bitwise_and(img2, img2, mask=mask_inv)
img1[:rows, :cols] = cv2.add(img1_bg, img2_fg)

cv2.imshow('img1', img1)
cv2.waitKey(0)&0xFF == ord('q')
cv2.destroyAllWindows()
```

### 3.性能优化

- [Python performance tips](https://wiki.python.org/moin/PythonSpeed/PerformanceTips 'office website')

- [scipy- advanced np](http://scipy-lectures.org/advanced/advanced_numpy/index.html#advanced-numpy 'scipy_lectures_org')







###### ps:borderType

[back](#1.4 Border)

1. cv2.BORDER_CONSTANT： 添加有颜色的常数值边界，需要value params
2. cv2.BORDER_REFLECT ：边界元素的镜像。比如: fedcba|abcdef
3. cv2.BORDER_REFLECT_101 | cv2.BORDER_DEFAULT： almost same as 2
4. cv2.BORDER_REPLICATE： 重复最后一个元素
5. cv2.BORDER_WRAP：辅助平铺

![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\make_border.png)