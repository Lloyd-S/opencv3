# opencv3

## 0.Description of This Repository

This repo contains information about the basic functions and skills of opencv(version 4.2). 

There are two main references used throughout the whole project.

1. The official Opencv user guide 
2. Learning Opencv3 , by A.k. & G.B.

The main purpose of the repository is to record where I was during my own learning path. Thus, the language I use is Chinese or English (sometimes, a combination of both), which largely depends on the language that my reference uses.

And since the main purpose is to organize my own notes and make it convenient for future reviews, there might be some grammar/typing mistakes or lack of basic information, which I prefer not to spend my time in solving.

## 1.Fundamentals - GUI basis and core operations

| Topics                                                       | Contents                            | Important?(Y/N) |
| ------------------------------------------------------------ | ----------------------------------- | --------------- |
| [GUI](https://github.com/Lloyd-S/opencv3/blob/master/Notes/1.GUI%E7%89%B9%E6%80%A7.md) | 图像/视频读取，drawing，slide bar   | N               |
| [核心操作](https://github.com/Lloyd-S/opencv3/blob/master/Notes/2.%E6%A0%B8%E5%BF%83%E6%93%8D%E4%BD%9C.md) | ROI,channels, border\| and, bitwise | N\|Y            |

## 2.Next step - Image processing

| Topics                                                       | Contents                                    | Important?(Y/N) |
| :----------------------------------------------------------- | :------------------------------------------ | :-------------: |
| [颜色空间转换](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.1%E9%A2%9C%E8%89%B2%E7%A9%BA%E9%97%B4%E8%BD%AC%E6%8D%A2.md) | BGR2GRAY 和BGR2HSV，简单物体追踪            |        N        |
| [几何变换](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.2%E5%87%A0%E4%BD%95%E5%8F%98%E6%8D%A2.pdf) | 仿射变换（放大缩小，平移旋转），透视变化    |        Y        |
| [Threshold](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.3.Threshold.md) | Global, adaptive, otsu's threshold          |        Y        |
| [Blur](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.4.Blur.md) | mean, Gaussian, median, 双边滤波            |        Y        |
| [形态学转换](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.5.%E5%BD%A2%E6%80%81%E5%AD%A6%E8%BD%AC%E6%8D%A2.md) | erode, dilate, open,close, tophat,blackhat  |        Y        |
| [梯度与边缘检测](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.6.ImageGradient%20%26%20EdgeDetect.pdf) | gradient(Sobel), Canny edge detector        |        Y        |
| [图像金字塔](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.7.%E5%9B%BE%E5%83%8F%E9%87%91%E5%AD%97%E5%A1%94.md) | Gaussian, 拉普拉斯                          |        N        |
| [Contours](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.8.Contours.pdf) | find/draw contours, hierarchy, features,etc |       YY        |
| [Histogram](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.9.Histogram.md) | 均衡化，clahe， 2D hist，反向投影           |        Y        |
| [模板匹配](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.10.%E6%A8%A1%E6%9D%BF%E5%8C%B9%E9%85%8D.md) |                                             |        Y        |
| [霍夫变换](https://github.com/Lloyd-S/opencv3/blob/master/Notes/3.11.%E9%9C%8D%E5%A4%AB%E5%8F%98%E6%8D%A2.md) | 直线/圆环检测，probabilistic                |        N        |
| [原理1](https://github.com/Lloyd-S/opencv3/blob/master/Notes/Convolution%26HoghTransform.pdf) | 霍夫变换                                    |                 |
| [原理2](https://github.com/Lloyd-S/opencv3/blob/master/Notes/Convolution%26HoghTransform.pdf) | 卷积                                        |                 |
| [原理3](https://zhuanlan.zhihu.com/p/19759362)               | 傅里叶变换                                  |                 |

## 3.Projects

#### Project 1: OCR- Credit Card Recognition

- [readme](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/readme.md)
- [read me\(cn\)](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/readme(%E4%B8%AD%E6%96%87).md)

- [code](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/card_recog_final.py)
- [results](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/results/Results1to7.png)

