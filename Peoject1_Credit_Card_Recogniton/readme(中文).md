# Opencv项目——信用卡识别

## 1. 简介

本项目使用OpenCV（Python接口，版本4.2），识别8张不同信用卡上的卡号。使用的信用卡图片和模板均存放在“img”文件夹中。

已成功识别前七张图片，如下所示：

![avatar](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/results/Results1to7.png)

第八章图片经过调试后，最多成功识别（可以成功定位每个数字的位置）5个数字，如下所示：

![avatar](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/results/res8.png)

## 2.程序的设计

程序的实现可以分成五个部分：

1. 创建数字0-9的模板

   - 8张卡共使用两种模板，其中卡1、3、4使用模板1（ref1.png），其余五张使用模板2（ref2.png）

2. 对图片处理，以定位每个数字的位置：

   - ROI处的背景是”暗“的，使用tophat处理，包括：卡1、3-6
   - ROI处的背景是”亮“的，使用blackhat处理，包括：卡2

   - ROI处的背景是”亮“、”暗“混合的，使用blackhat处理，包括：卡7

3. 模板匹配：

   - 使用1、2中获得的模板和数字位置，进行模板匹配

4. 可视化

   - 使用cv2.rectangle()和cv2.putText()，将结果显示出来

5. 封装和传参

   - 将程序简单封装，并利用argparse模块实现传参功能

## 3.分析

Case 1： 卡1，3-6

- ROI处背景都是”暗“的，使用tophat处理
- 卡1中，不需要多次tophat，卡3-6中，需要迭代2次tophat才能准确识别
- 卡3-6中其他参数可完全相同

Case 2： 卡2

- 使用blackhat迭代两次，即可成功识别

Case 3： 卡7

- 因为这张卡卡号背景上，亮暗部分都有，因此将他分成三个ROI，其中roi_2为亮的部分，之后再分别处理

```python
    roi_1 = gray[135:165,:125]
    roi_2 = gray[135:165,125:180]
    roi_3 = gray[135:165,180:]
```

- 对于roi_1和roi_3，使用case1中的处理方式即可
- 对于roi_2，使用case2中的处理方式，不能准确识别，因此使用Canny并将得到的轮廓画到原图上再识别

Case 4: 卡8

-  定位数字：将每四个数字看成一组。先用canny获得边框，之后膨胀，然后获得包围矩形边框。发现前两组位置准确，并且每组之间、组内数字之间间隔是固定的，据此计算出每个数字的包围矩形边框，如上图所示。
- 尝试各种基本形态学变换和轮廓检测后，发现由blackhat处理后的图像更清晰
- 进行参数调整，包括：卷积和的大小和形状，blackhat迭代次数，阈值，image的尺寸等，最多正确匹配5个数字。

## 4.更新1

原来的程序中，将原图resize后，手动选取了ROI（ROI = gray[135:165,:]。因此在第一个更新中，通过轮廓的外接矩形，来筛选数字所在位置。

步骤如下：

- 使用高斯模糊，消除图片中噪声的影响
- 使用Canny边缘检测后，再使用膨胀操作
- 找到外轮廓后（cv2.RETE_EXTERNAL）及其外接矩形后，通过矩形的（x,y,w,h）来找到数字所在的位置（每四个数字一组）
- 筛选条件为 w$\in$(60,77), h$\in$(22,27)

代码如下（py文件中158-179行）

```python
def find_roi(image,a=3):
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    blur = cv2.GaussianBlur(image, (5, 5), 0)
    canny = cv2.Canny(blur,200,250)
    dilation = cv2.dilate(canny, rectKernel)
    cnt = cv2.findContours(dilation,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)[0]
    locs = []
    for i in cnt:
        x,y,w,h = cv2.boundingRect(i)
        if w>60 and w<77 and h>22 and h<27:
            locs.append((x,y,w,h))
        locs = sorted(locs, key=lambda x: x[0], reverse=False)
    height = 0
    y_axis = 0
    for (x,y,w,h) in locs:
        height +=h
        y_axis += y
    height = round(height/len(locs)) 
    y_start = round(y_axis/len(locs))-a
    y_end = height +y_start+2*a
    roi = image[y_start:y_end,:]
    return roi
```

## 5.更新2

原程序中，手动选择模板（本项目使用两个不同的数字模板）。更新后，匹配时分别用两种模板进行匹配，然后选择相关系数（cv2.TM_CCOEFF）更大（更接近1）的模板匹配结果，作为最终结果。

步骤如下：

- 分别用两种模板进行匹配，匹配结果分别是result_1和result_2
- 分别对每个数字（共16个），比较两种模板进行匹配得到的相关系数（cv2.TM_CCOEFF_NORMED）大小，模板1较大的次数为a，模板2为b
- 若a>b，则选择模板1（result_1）;反之选择模板2（result_2）

代码如下（py文件中60-94行，或95-129行）

```python
def match_template(locs,digits1,digits2,iters=1):
    results = []
    results_1 = []
    results_2 = []
    a=0
    b=0
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    for i,(x,y,w,h) in enumerate(locs):
        image = gray[y-2:y+h+2,x-2:x+w+2]
        hat_image = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,rectKernel,iterations=iters)
        dst = cv2.resize(hat_image,(48,64))

        sorts_1 = []
        sorts_2 = []
        
        for j,temp in enumerate(digits1.values()):
            result_temp1 = cv2.matchTemplate(dst,temp,cv2.TM_CCOEFF_NORMED)
            min_val, max_val1, min_loc, max_loc = cv2.minMaxLoc(result_temp1)
            sorts_1.append(max_val1)
        results_1.append(str(np.argmax(sorts_1)))

        for j2,temp2 in enumerate(digits2.values()):
            result_temp2 = cv2.matchTemplate(dst,temp2,cv2.TM_CCOEFF_NORMED)
            min_val2,max_val2,min_loc2,max_loc2=cv2.minMaxLoc(result_temp2)
            sorts_2.append(max_val2)
        results_2.append(str(np.argmax(sorts_2)))
        if (np.max(sorts_1) >= np.max(sorts_2)):
            a+=1
        elif (np.max(sorts_1) < np.max(sorts_2)):
            b+=1
    if a>b:
        results = results_1
    else:
        results = results_2
    return results
```

ps:原来此部分代码

```python
#previous
def match_template(locs,digits,iters=1):
    results = []
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    for i,(x,y,w,h) in enumerate(locs):
        image = gray[y-2:y+h+2,x-2:x+w+2]
        hat_image = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,rectKernel,iterations=iters)
        dst = cv2.resize(hat_image,(48,64))
        sorts = []
        for j,temp in enumerate(digits.values()):
            result = cv2.matchTemplate(dst,temp,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            sorts.append(max_val)
        results.append(str(np.argmax(sorts)))
    return results
```



