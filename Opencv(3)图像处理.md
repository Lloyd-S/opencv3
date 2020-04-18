# Opencv(3)图像处理

 [toc]

## 1.颜色空间转换

### 1.1转换

- cv2中有150+转换，由flag确定，但常用的是BGR2GREY 和BGR2HSV.
- cv2.cvtColor(img, cv2.COLOR_BGR2GREY/cv2.BGR2HSV)

```python
import cv2 
img = cv2.imread('data/messi5.jpg')
grey = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
hsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
cv2.imshow('g',grey)
cv2.imshow('h',hsv)
cv2.waitKey(0) | 0xFF ==ord('q')
cv2.destroyAllWindows

```

### 1.2物体追踪

- HSV更容易表示特定颜色

- cv2.inRange()

```python
import cv2
import numpy as np
img = cv2.imread('data/ppang.jpg')

lower = np.array([20, 100, 100])
upper = np.array([34, 255, 255])
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(hsv, lower, upper)
res = cv2.bitwise_and(img, img, mask=mask)

cv2.imshow('res', res)
cv2.waitKey(0) or 0xFF==ord('q')
cv2.destroyAllWindows()
```

[RGB code & HSV convertor](https://www.rapidtables.com/web/color/RGB_Color.html#color-picker)

## 2.几何变换

- 图形变换原理：

  将（x,y）变换成$（\mu,\nu）$,其中：

$$
\mu = a_1x+b_1y+c_1;
\qquad\nu = a_2x+b_2y+c_2
$$

​		矩阵形式为
$$
\begin{bmatrix}\mu\\\nu\end{bmatrix}=\begin{bmatrix}a_1&b_1&c_1\\a_2&b_2&c_2\end{bmatrix}\cdot\begin{bmatrix}x\\y\\1\end{bmatrix}
$$
​		Let $M =\begin{bmatrix}a_1&b_1&c_1\\a_2&b_2&c_2\end{bmatrix}$, 则可用M表示图像变换

### 2.1放大/缩小、平移、旋转

- cv2.resize()

```python
import cv2
import numpy as np
img = cv2.imread('data/ppang.jpg')

res = cv2.resize(img, None, fx=1/2, fy=1/2) 
#使用缩放因子，因此有None；Default：interpolation=cv2.INTER_LINEAR

cv2.imshow('resize', res)
cv2.imshow('src img', img)
v2.waitKey(0)
cv2.destroyAllWindows()
```

- 平移：$M =\begin{bmatrix}1&0&\Delta x\\0&1&\Delta y\end{bmatrix}$，cv2.wrapAffine()

```python
import cv2
import numpy as np

# 移动了100,50 个像素。
img = cv2.imread('data/messi5.jpg', 0)

rows, cols = img.shape
M = np.float32([[1, 0, 100], [0, 1, 50]])
dst = cv2.warpAffine(img, M, (cols, rows))#由于row表示y，col表示x，因此（cols，rows）

cv2.imshow('img1', img)
cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

- 旋转：cv2.getRotationMatrix2D(旋转中心，旋转角度，缩放因子)， cv2.wrapAffine()

```python
import cv2
import numpy as np

img = cv2.imread('data/messi5.jpg', 0)
rows, cols = img.shape

# 第一个参数为旋转中心 第二个为旋转角度,第三个为旋转后的缩放因子
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 2)
dst = cv2.warpAffine(img, M, (2 * cols, 2 * rows))

cv2.imshow('img', dst)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### 2.2 Affine变换

- [2.1](#2.1放大/缩小、平移、旋转)本质为Affine变换特例
- 原图中平行的直线，在变换后的图像中，仍平行。否则使用[透视变换](2.3透视变换)
- 用两点即可,使用cv2.wrapAffineTransfrom()得到M矩阵

```python
import cv2
import numpy as np
img = cv2.imread('data/drawing.png')
rows, cols, ch = img.shape

pts1 = np.float32([[50, 50], [200, 50], [50, 200]])
pts2 = np.float32([[10, 100], [200, 50], [100, 250]])
M = cv2.getAffineTransform(pts1, pts2)
dst = cv2.warpAffine(img, M, (cols, rows))
cv2.imshow('img', dst)

cv2.waitKey(0)  # & 0xFF
cv2.destroyAllWindows()
```

### 2.3透视变换

- 三点，使用cv2.getPerspectiveTransform()获得M矩阵，cv2.wrapPerspective()

```python
import cv2
import numpy as np

img = cv2.imread('data/screen_shot.png')
rows, cols, ch = img.shape

pts1 = np.float32([[68, 44], [223, 104], [45,458], [239,438]])
pts2 = np.float32([[0, 0], [cols, 0], [0, rows], [cols, rows]])
M = cv2.getPerspectiveTransform(pts1, pts2)
res = cv2.warpPerspective(img, M, (cols, rows))

cv2.imshow('result', res)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

## 3.Threshold

([top](#Opencv(3)图像处理))

### 3.1简单阈值

- ret,img_1 = cv2.threshold(img,a,b,[flags](#ps1.cv2.threshold )), a=threshold, b=values

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('data/grey-gradient.jpg', 0)

ret, thresh1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
ret, thresh2 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
ret, thresh3 = cv2.threshold(img, 127, 255, cv2.THRESH_TRUNC)
ret, thresh4 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO)
ret, thresh5 = cv2.threshold(img, 127, 255, cv2.THRESH_TOZERO_INV)

titles = ['Original', 'BINARY', 'BINARY_INV', 'TRUNC', 'TOZERO', 'TOZERO_INV']
images = [img, thresh1, thresh2, thresh3, thresh4, thresh5]

for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
```

### 3.2Adaptive阈值

- 在global threshold基础上增加[三个参数](#ps2.cv2.adaptiveThreshold())

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img_source = cv2.imread('data/sudoku.jpg', 0)
# 中值滤波
img = cv2.medianBlur(img_source, 5)

ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
th2 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 2)
th3 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
th4 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 11, 20)
th5 = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 20)


titles = ['Original', 'Global Threshold',
          'Adaptive Mean', 'Adaptive Gaussian','LAdaptive Mean', 'LAdaptive Gaussian']
images = [img, th1, th2, th3, th4, th5]
for i in range(6):
    plt.subplot(2, 3, i + 1), plt.imshow(images[i], 'gray')
    plt.title(titles[i])
    plt.xticks([]), plt.yticks([])
plt.show()
```

### 3.3 Otsu's二值化: 选择合理阈值

- retval, eg: ret, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY),  ret =127

  pass the param '**cv2.THRESH_OTSU**' to function '**cv2.thresholding()**', retVal is the best threshold value

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('data/noisy2.png', 0)

# global thresholding
ret1, th1 = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)
# Otsu's thresholding
ret2, th2 = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# Otsu's thresholding after Gaussian filtering
blur = cv2.GaussianBlur(img, (5, 5), 0)
ret3, th3 = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)


images = [img, 0, th1,img, 0, th2,blur, 0, th3]
titles = ['Original Noisy Image', 'Histogram', 'Global Thresholding (v=127)',
          'Original Noisy Image', 'Histogram', "Otsu's Thresholding",
          'Gaussian filtered Image', 'Histogram', "Otsu's Thresholding"]
for i in range(3):
    plt.subplot(3, 3, i * 3 + 1), plt.imshow(images[i * 3], 'gray')
    plt.title(titles[i * 3]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 2), plt.hist(images[i * 3].ravel(), 256)
    plt.title(titles[i * 3 + 1]), plt.xticks([]), plt.yticks([])
    plt.subplot(3, 3, i * 3 + 3), plt.imshow(images[i * 3 + 2], 'gray')
    plt.title(titles[i * 3 + 2]), plt.xticks([]), plt.yticks([])
plt.show()
```

![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\Otsu.png)

refrence: [大津算法](https://zh.wikipedia.org/wiki/%E5%A4%A7%E6%B4%A5%E7%AE%97%E6%B3%95 'wiki')

## 4.Blur

[top](#Opencv(3)图像处理)

### 4.1平均

- cv2.blur()

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('data/opencv_logo.png')

blur =cv2.blur(img,(5,5))

plt.subplot(121), plt.imshow(img), plt.title('Original')
plt.xticks([]), plt.yticks([])
plt.subplot(122), plt.imshow(blur), plt.title('Blurred')
plt.xticks([]), plt.yticks([])
plt.show()
```

### 4.2Gaussian

- cv2.GaussianBlur()

```python
blur = cv2.GaussianBlur(img, (5, 5), 0) # 0 是Gasussian函数标准差
```

### 4.3Median

- 椒盐噪声
- cv2.medianBlur

```python
median = cv2.medianBlur(img, 5)
```

### 4.4双边滤波

- 不模糊边界
- cv2.bilateralFilter()

```
blur = cv2.bilateralFilter(img, 9, 75, 75)
# 9是直径，两个75，分别是Gaussian函数标准差、灰度值相似性Gaussian函数标准差
```

## 5.形态学转换

[top](#Opencv(3)图像处理)

### 5.1基本操作：erode, dilate

[Result](#ps3.erode&dilate)

#### 5.1.1 Erode

- 原理：卷积核移动，若卷积为1，则保持原值；otherwise，则变为0.

  因此可以去除白噪声

- erosion = cv2.erode(img, kernel, iterations=1)

```python
import cv2
import numpy as np
img = cv2.imread('data/j.png', 0)

kernel = np.ones((5, 5), np.uint8)
erosion = cv2.erode(img, kernel, iterations=1)

cv2.imshow('erode', erosion)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 5.1.2 Dilate

- 原理：与erode相反，卷积中有一个为1，则不变。

  因此会增加白色区域

- 去除噪声时，一般先erode，再dilate

```python
dilation = cv2.dilate(img, kernel, iterations=1)
```

### 5.2其他操作

[Result](#ps4.open, close, gradient, tophat, blackhat)

- cv2.morphologyEx()
- [flags](#ps5.cv2.morphology()-flags:)

#### 5.2.1开运算

- erode, then dilate
- 去除噪声

```python
opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)		
```

#### 5.2.2闭运算

- dilate, then erode

-  先膨胀再腐 。它经常 用来填充前景物体中的小洞 或者前景物体上的小黑点。

```
closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
```

#### 5.2.3梯度

- dilate - erode

```
gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel1)
```

#### 5.2.4礼帽

- original - opening

```
tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
```

#### 5.2.5黑帽

- closing - original

```python
blackhat = cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
```

### 5.3其他形状的Kernel

- cv2.getStructuringElement()

```python
# Rectangular Kernel
>>> cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))
array([[1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1]], dtype=uint8)
# Elliptical Kernel
>>> cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
array([[0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0]], dtype=uint8)
# Cross-shaped Kernel
>>> cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
array([[0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0],
       [1, 1, 1, 1, 1],
       [0, 0, 1, 0, 0],
       [0, 0, 1, 0, 0]], dtype=uint8)
```

## 6.ImageGradient & EdgeDetect

[top](#Opencv(3)图像处理)

### 6.1derivatives or gradient

- 高通滤波器，检测边缘

- cv2.Scharr()/cv2.Sobel(): Scharr(3*3 matrix) is a optimized version of Sobel, for calculating 1st or 2nd derivatives
- cv2.Laplacian(): for 2nd derivatives, and: $kernel = \begin{bmatrix}0&1&0\\1&-4&1\\0&1&0\end{bmatrix}$

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('data/sudoku.jpg', 0)

# cv2.CV_64F输出图像的深度（若使用 -1, 与原图像保持一致，i.e.:np.uint8）
#若使用-1（而非cv2.cv_64F）,会丢失导数为负数的边界（从白到黑） 
laplacian = cv2.Laplacian(img, cv2.CV_64F)
# 参数1，0表示x方向（最大可以求2nd导数）
sobelx = cv2.Sobel(img, cv2.CV_64F, 1, 0, ksize=5)
# 参数0,1表示y方向（最大可以求2nd导数）
sobely = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=5)

plt.subplot(2, 2, 1), plt.imshow(img, cmap='gray')
plt.title('Original'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 2), plt.imshow(laplacian, cmap='gray')
plt.title('Laplacian'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 3), plt.imshow(sobelx, cmap='gray')
plt.title('Sobel X'), plt.xticks([]), plt.yticks([])
plt.subplot(2, 2, 4), plt.imshow(sobely, cmap='gray')
plt.title('Sobel Y'), plt.xticks([]), plt.yticks([])
plt.show()
```

ps：depth of image--bits

### 6.2 Edge detection--Canny

#### 6.2.1 原理

1. Noise reduction: using 5 by 5 Gaussian filter
2. With Sobel(), get Gx andGy--the 1st derivatives on x and y axes. Using them to caculate the graident and angle of detected edge, and $Angle(\theta)=\tan^{-1}(\frac{Gx}{Gy})$, $(G)=\sqrt{(Gx^2+Gy^2)}$

3. Scan the whole image, and ingore the points that are not on edges, by the following idea: if one pixel is on the edge, **the gradient of this point should be the largest** among its neighboors **with the same angle.**

4. Find the 'true' edge: derivatives>maxVal$\rightarrow$ture

- derivatives>maxVal$\rightarrow$ture
- derivatives<minVal$\rightarrow$false
- derivatives$\in$(minVal,maxVal): depend on weather it is connected w/ true edge

![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\CannyAlgo.png)

#### 6.2.2 cv.canny()

```python
import cv2
import numpy as np
from matplotlib import pyplot as plt
img = cv2.imread('data/messi5.jpg',-1)

edges = cv2.Canny(img, 230, 240)
#Canny(image, minVal, maxVal,kenerl=3_by_3,LaGradient=False)
#L2Gradient=False: Edge−Gradient(G)=|Gx^2|+|Gy^2|
#L2Gradient=True: Edge−Gradient(G)=(Gx^2+G2y^2)
cv2.imshow('Edges',edges)
cv2.waitKey(0)
cv2.destroyAllWindows

```

ps: 

L2Gradient=False$\rightarrow$ Edge−Gradient(G)=$|{Gx}^2|+|{Gy}^2|$

L2Gradient=True$\rightarrow$ Edge−Gradient(G)=$\sqrt{(Gx^2+Gy^2)}$

## 7.图像金字塔

[top](#Opencv(3)图像处理)

#### 7.1 高斯金字塔

- 不同分辨率的子图集合: 去除row和col，每一层由M*N$\rightarrow$M/2\*N/2
- cv2.pyrUp() / cv2.pyrDown()

```python
import cv2
import numpy as np
higher_reso = cv2.imread('data/messi5.jpg')

# 尺寸变小 分辨率降低 。
lower_reso = cv2.pyrDown(higher_reso)
# 尺寸变大，分辨率不变（丢失信息）
higher_reso2 = cv2.pyrUp(lower_reso)

cv2.imshow('lower_reso', lower_reso)
cv2.imshow('higher_reso2', higher_reso2)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 7.2 拉普拉斯金字塔

- 丢失的信息即为拉普拉斯金字塔：$L_i=G_i-PryUp(G_{i+1})$

- 用于image blending

```python
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
    
# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    w,h,_=gpA[i - 1].shape
    w1,h1,_=GE.shape
    min_w=min(w,w1)
    min_h=min(h,h1)
    L = cv2.subtract(gpA[i - 1][0:min_w,0:min_h,:], GE[0:min_w,0:min_h,:])
    lpA.append(L)
```

ps: [完整code](#ps6.image blending example code)

## 8.Contours

[top](#Opencv(3)图像处理)

### 8.1Intro

- 需要输入二值化图像：先灰度图，再黑白（threshhold）
- cv2.findContours(); cv2.drawContours()

``` python
findContours(image, mode, method[, contours[, hierarchy[, offset]]]) -> contours, hierarchy
```

```python
drawContours(image, contours, contourIdx, color[, thickness[, lineType[, hierarchy[, maxLevel[, offset]]]]]) -> image
#其中，contourIdx = -1：all
```

```python
import numpy as np
import cv2
im = cv2.imread('data/star.jpg')
imgray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(src=imgray, thresh=127, maxval=255, type=cv2.THRESH_BINARY)#src, thresh, maxval, type

contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
img = cv2.drawContours(im, contours, -1, (0, 255, 0), 2)

cv2.imshow("thresh", thresh)
cv2.imshow("imgray", imgray)
cv2.imshow("contours", img)
cv2.waitKey(0)
cv2.destroyAllWindows
```

### 8.2轮廓特征

#### 8.2.1Moment

[[wiki](https://en.wikipedia.org/wiki/Image_moment)]

- 具有不变性：平移、缩放、旋转不变性； 和唯一性：M(p,q)由f(x,y)唯一确定

- 公式(连续)

$$
M_{pd}=\int_{-\infty}^\infty\int_{-\infty}^\infty x^py^qf(x,y)dxdy
$$

​		二值图像面积：$M_{00}$

​		几何中心：
$$
\{\bar x\,\bar y\}= \{\frac{M_{10}}{M_{00}},\frac{M_{01}}{M_{00}}\}
$$

```python
import cv2
import numpy as np
from pprint import pprint #pprint打印dictionary更好看一点，与print功能相同

img = cv2.imread('data/star.jpg')
gray_img=cv2.cvtColor(img,cv2.COLOR_BGRA2GRAY)
ret, thresh = cv2.threshold(gray_img, 127, 255, 0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

cnt = contours[0] #first contour
M = cv2.moments(cnt)#返回dictionary

pprint(M)
```

#### 8.2.2面积

- cv2.contourArea(), or M['m00']

```
area=cv2.contourArea(cnt)
```

#### 8.2.3周长

- cv2.arcLength()

```python
perimeter=cv2.arcLength(cnt,True) #True fro closed contour
```

#### 8.2.4轮廓近似

- [Douglas-Peucker algorithm](https://zh.wikipedia.org/wiki/%E9%81%93%E6%A0%BC%E6%8B%89%E6%96%AF-%E6%99%AE%E5%85%8B%E7%AE%97%E6%B3%95)：是将曲线近似表示为一系列点，并减少点的数量的一种算法。
- cv2.approxPolyDP(): output approx contour rather than exact one

```python
epsilon = 0.1*perimeter
approx = cv2.approxPolyDP(cnt,epsilon,True)
image=cv2.drawContours(img,[approx],-1,(255,0,0),2)

cv2.imshow('approxPolyDP',image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

#### 8.2.5Convex hull

![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\convex.png)

- to get the convex

```
hull = cv2.convexHull(cnt)
```

- convexity_defects

```python
cv2.isContourConvex(cnt)-->True/False
or
cv2.defects = cv2.convexityDefects(cnt, hull)-->4-element integer vector,(start_index, end_index, farthest_pt_index, fixpt_depth)
```

#### 8.2.6边界矩形

- 直边界矩形-cv2.boundingRect(cnt) --> (w,y,w,h), (x,y)左上角坐标,(w,h)宽和高
  - 面积不是最小的

```python
w,y,w,h = cv2.boudingRect()
img_1 = cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),255)
```



- 旋转的边界矩形-cv2.minAreaRect() -->(x,y),(w,h),($\phi$)-->Box2D
  - 面积最小矩阵

```python
rect = cv2.minAreaRect(cnt)
box = cv2.boxPoints(rect)
box = np.int0(box)
img_2 = cv2.drawContours(img,[box],0,(0,255,0)
```

#### 8.2.7最小外接圆

- cv2.minEnclosingCircle()-->(x,y),radius

```python
(x,y),r = cv2.minEnclosingCircle(cnt)
img = cv2.circle(img,(int(x),int(y)),int(r),(255.0,0),2)
```

#### 8.2.8椭圆拟合

- cv2.fitEllipse()-->[旋转边界矩形](#8.2.6边界矩形)的内切圆

```python
ellipse = cv2.fitEllipse(cnt)
im = cv2.ellipse(im,ellipse,(0,255,0),2)
```

#### 8.2.9直线拟合

- cv2.fitLine()

```python
rows, cols = img.shape[:2]
[vx, vy, x, y] = cv2.fitLine(cnt, cv2.DIST_L2, 0, 0.01, 0.01)
lefty = int((-x * vy / vx) + y)
righty = int(((cols - x) * vy / vx) + y)
cv2.line(img, (cols - 1, righty), (0, lefty), (0, 255, 0), 2)
```

### 8.3轮廓的性质（8.2 extended）

[top](#Opencv(3)图像处理)

#### 8.3.1宽高比

$$
wh\_ratio =\frac{Width}{Height}
$$



```python
w,y,w,h = cv2.boudingRect()
aspect_ratio = float(w/h)
```

#### 8.3.2 extent 

$$
extent = \frac{object\_area}{bounding\_rec\_area}
$$



```python
area = cv2.contourArea(cnt)
w,y,w,h = cv2.boudingRect()
extent = float(area/(w*y))
```

#### 8.3.3 Solidity

$$
solidity = \frac{cnt\_area}{hull\_area}
$$



```python
area = cv2.contourArea(cnt)
hull = cv2.convexHull(cnt)
hull_area = cv2.contourArea(hull)
solidity = float(area/hull_area)
```

#### 8.3.4 Equivalent diameter

$$
ed = \sqrt{\frac{4*cnt\_area}{\pi}}
$$

```python
area = cv2.contourArea(cnt)
ed = np.sqrt(4*area/np.pi)
```

#### 8.3.5 方向、长短轴

```python
(x,y),(l,s),angle = cv2.fitEllipse(cnt)
```

#### 8.3.6 Mask -- 像素点、最值和平均值

- mask & all pixel points

```python
import cv2
import numpy as np

img = cv2.imread('data/star.jpg',0)
ret, thresh = cv2.threshold(img, 127, 255, 0)
contours,hierarchy = cv2.findContours(thresh,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cnt = contours[0] #first contour
mask = np.zeros(img.shape,np.uint8)
mask = cv2.drawContours(mask,[cnt],0,255,-1) #must be '-1' -绘制填充的轮廓

pixel_points = np.transpose(np.nonzero(mask))#np计算出(row,col)-->(y,x)
print(pixel_points)
```

- max&min&their position

```python
min_val,max_val,min_loc,mac_loc = cv2.minMaxLoc(imgrey,mask=mask)
```

- 平均颜色|平均灰度

```
mean_val = cv2.mean(imgrey|img, mask=mask)
```

#### 8.3.7极点

```
left = tuple(cnt[cnt[:,:,0].argmin()][0])
right = tuple(cnt[cnt[:,:,0].argmax()][0])
top = tuple(cnt[cnt[:,:,1].argmin()][0])
bottom = tuple(cnt[cnt[:,:,1].argmax()][0])
```

### 8.4 Other functions

[top](#Opencv(3)图像处理)

























**PS**: [top](#Opencv(3)图像处理)

###### ps1.cv2.threshold 

[back](#3.1简单阈值)

- flags：

  cv2.THRESH_BINARY
  cv2.THRESH_BINARY_INV 
  cv2.THRESH_TRUNC
  cv2.THRESH_TOZERO
  cv2.THRESH_TOZERO_INV

  ![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\Threshold.png)

###### ps2.cv2.adaptiveThreshold()

[back](#3.2Adaptive阈值)

```python
adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])
```

- Adaptive Method- 指定 算阈值的方法。
  – cv2.ADPTIVE_THRESH_MEAN_C  值取自相邻区域的平均值
  – cv2.ADPTIVE_THRESH_GAUSSIAN_C  值取值相邻区域 的加权和 ，权重为一个高斯窗口

- 11 为 Block size 邻域大小 用来计算阈值的区域大小 
- 2 为 C值，常数， 阈值就等于的平均值或者加权平均值减去这个常数

![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\adaptiveThreshold.png)

###### ps3.erode&dilate

[back](#5.1基本操作：erode, dilate)

![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\Erode_dilate.png)

###### ps4.open, close, gradient, tophat, blackhat

[back](#5.2其他操作)

![](C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\imageMorphology.png)

###### ps5.cv2.morphology()-flags:

[back](#5.2其他操作)

- cv2.MORPH_OPEN

- cv2.MORPH_CLOSE

- cv2.MORPH_GRADIENT

- cv2.MORPH_TOPHAT

- cv2.MORPH_BLACKHAT

###### ps6.image blending example code

[back](#7.2 拉普拉斯金字塔)

```python
import cv2
import numpy as np, sys
a = np.asarray(range(0,24)).reshape((2,3,4))
b = np.asarray(range(25,49)).reshape((2,3,4))
c = [4,5,6,7,8]
zipped = zip(a,b)     
for ai,bi in zip(a,b):
    print("#"*100)
    print(f"{ai},{bi}")
[(1, 4), (2, 5), (3, 6)]
A = cv2.imread('data/apple.jpg')
B = cv2.imread('data/orange.jpg')
 
# generate Gaussian pyramid for A
G = A.copy()
gpA = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpA.append(G)
# generate Gaussian pyramid for B
G = B.copy()
gpB = [G]
for i in range(6):
    G = cv2.pyrDown(G)
    gpB.append(G)

# generate Laplacian Pyramid for A
lpA = [gpA[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpA[i])
    w,h,_=gpA[i - 1].shape
    w1,h1,_=GE.shape
    min_w=min(w,w1)
    min_h=min(h,h1)
    L = cv2.subtract(gpA[i - 1][0:min_w,0:min_h,:], GE[0:min_w,0:min_h,:])#TODO error
    lpA.append(L)
# generate Laplacian Pyramid for B
lpB = [gpB[5]]
for i in range(5, 0, -1):
    GE = cv2.pyrUp(gpB[i])
    w,h,_=gpB[i - 1].shape
    w1,h1,_=GE.shape
    min_w=min(w,w1)
    min_h=min(h,h1)
    L = cv2.subtract(gpB[i - 1][0:min_w,0:min_h,:], GE[0:min_w,0:min_h,:])#TODO error    
    #L = cv2.subtract(gpB[i - 1], GE)
    lpB.append(L)

# Now add left and right halves of images in each level
# numpy.hstack(tup)
# Take a sequence of arrays and stack them horizontally
# to make a single array.
LS = []
for la, lb in zip(lpA, lpB):
    rows, cols, dpt = la.shape
    ls = np.hstack((la[:, 0:cols // 2], lb[:, cols // 2:]))
    LS.append(ls)

# now reconstruct
ls_ = LS[0]
for i in range(1, 6):
    ls_ = cv2.pyrUp(ls_)
    w,h,_=ls_.shape
    w1,h1,_=LS[i].shape
    min_w=min(w,w1)
    min_h=min(h,h1) 
    ls_, LS[i]=ls_[0:min_w,0:min_h,:], LS[i][0:min_w,0:min_h,:]
    ls_ = cv2.add(ls_, LS[i])
# image with direct connecting each half
real = np.hstack((A[:, :cols // 2], B[:, cols // 2:]))

cv2.imwrite('Pyramid_blending2.jpg', ls_)
cv2.imwrite('Direct_blending.jpg', real)

```

<img src="C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\Direct_blending.jpg" style="zoom:25%;" />

### <img src="C:\Users\Lloyd\Desktop\笔记\笔记使用的图片\Pyramid_blending2.jpg" style="zoom:25%;" />