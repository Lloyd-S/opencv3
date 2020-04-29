## Credit Card Recognition Project

### 1.Brief Introduction

This project, which was guided by my supervisor Aopu, aims to recognize the 16-digit card number of 8 different credit card (card 1 to 8 as file name indicates) with opencv-python(cv2 verision:4.2.0.34) . You can find the image of each card in the folder named 'img'.

For card 1-7, this project successfully detect all 16 digits on each card. Below shows all the results:

![avatar](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/results/Results1to7.png)

For card 8, this project fails to recognize all the digits. The best one correctly detects  5 digits out of 16. The corresponding code is in file **'unsloved_card8.py'**. Below shows the 'best' result:

![avatar](https://github.com/Lloyd-S/opencv3/blob/master/Peoject1_Credit_Card_Recogniton/results/res8.png)

This project is a good start for someone(like me) who just begin their learning path in opencv and computer vision, and only requires some basic understanding of image operations, including:

- Image morphology: dilation, erosion, opening, closing, tophat, blackhat
- Filter and convolution: threshold operation(otus's),  kerme;s, convolution
- Edge detector and template matching: Sobel, Canny, template matching
- Contours: find/draw, features
- Drawing and annotating

Here lists some useful reference both on the project and opencv:

- OCR project:
  - [github](https://github.com/pmathur5k10/Credit-Card-Recognition)
  - [csdn](https://blog.csdn.net/weixin_44678052/article/details/104076451)

- opencv
  - [MyNote](https://github.com/Lloyd-S/opencv3/tree/master/Notes)
  - "Learning OpenCV3" by A.k. & G.B.: Chapter6, 10, 12, 14.

### 2.Project Design

The whole project can be narrowed down into 5 parts:

1. Create templates for all 10 digits(0-9):
   - note that, there are 2 different using in these credit card. Specifically, the 1st, 3rd and 4th card use template 1('ref1.png'), and the rest 5 cards use template 2('ref2.png').
2. Process the image in order to locate each digit on the card: 
   - to process the card with 'dark' background  by tophat, e.g. card 1,3-6
   - to process the card with 'bright' background by blackhat, e.g. card 2
   - for the the card with combined background, e.g. card 7, dividing the card into dark ROI and bright ROI, then eliminate background noise by tophat and blackhat, respectively
3. Template matching:
   - use the locations of each digits found in part2, and  the template created in part 1, to match the template.
4. Visualization
   - to draw the result by cv2.rectangle() and cv2.putText()
5. Passing parameters' function and packaging:
   - to use argparse module to build the parameter-passing function
   - to aggregate the results for all 8 cards by packaging

### 3.Cases

**Case 1: Card 1, 3-6**

- They all have 'dark' background, and it turns out that the tophat technique fulfill image processing need.
- Regarding iteration: card 1 do not require any iteration, while the rest four cards require 2 iterations 
- Card 3-6 can share exactly the same parameters/codes, no need to tune.

**Case 2: Card 2**

- The blackhat with 2 iterations would successfully detect the digit on it.

**Case 3: Card 7**

- It has both bright and dark part, so I divide the card into dark ROI(roi_2) and bright ROI(roi_1&3) as below:

```python
    roi_1 = gray[135:165,:125]
    roi_2 = gray[135:165,125:180]
    roi_3 = gray[135:165,180:]
```

- For roi_1&3, I dealt with them in the same way as in case 1.
- For roi_2. I tried blackhat it, but it failed to show a clear contours. Thus, I used Canny edge detector, which results in a good match

Case 4: The unsolved case - card 8

- This project succeed in locating all the 16 digits on this card. By Canny-->dilation,  the digits are evenly divided into for groups, and the first 2 groups have pretty clear shapes, so that I can locate all the individual digits by assuming(which is also the truth) groups, and the digits in each group, are evenly separated. The result is shown above.
- This project fails in template matching in this case. I've tried all the above methods in case 1-3 and found the best contours/image come from blackhat. Then I tried to tune several key parameters. including: the kernel size and type, the roi size, the iteration times, the threshold values. In the end, I could only correctly recognize 5 digits out of 16 as the best result.
- If anyone can provide a better solution for this case, pls contact/at me on GiuHub. Thx in advance. 

### 4.Update 1

In the previous program, I manually selected the ROI (ROI = gray[135:165,:]). Thus, this update uses cv2.boundingRect to find where the digits locate.

Below shows the design:

- Using Gaussian Blur to reduce the noise
- Using Canny edge detector and then dilate it
- Find the external contours (cv2.RETE_EXTERNAL) and the corresponding bonding rectangles. Locate the digit groups(4 digits form a group) by (x,y,w,h) of the bonding rectangles.
- Filtering the bonding rectangles by  w$\in$(60,77), h$\in$(22,27)

Below show the code (line 158-179 from the py file):

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

### 5.Update 2

The previous program manually selected the digit template (2 templates used in total). Now, I use both of the templates to do the matching, then select the one with larger  correlation coefficients (TM_CCOEFF_NORMED) as the final results.

Below shows the design:

- Do the matching with both 2 templates, and the results are saved in result_1 and result_2
- Compare the correlation coefficients gained by 2 templates of every digit (16 digits in total). The times of template 1 has larger CC is a, and template 2 is b
- if a>b, use template 1(result_1); Vice versa

Below show the code (line 60-94 or 95-129 from the py file):

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

ps: previous code

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

