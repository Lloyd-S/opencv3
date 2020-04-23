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
