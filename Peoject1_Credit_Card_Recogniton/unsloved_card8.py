import cv2
import numpy as np
# #Part One: Reference
ref =cv2.imread('img/ref2.png')
# # ref =cv2.imread('reference.png')
ref = cv2.resize(ref,(400,64))

ref_gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
ref_gray = cv2.threshold(ref_gray,100,255,cv2.THRESH_BINARY)[1]
ref_cont = cv2.findContours(ref_gray,cv2.CV_8UC1,cv2.RETR_CCOMP)[0]
point = []
for i in ref_cont:
    x,y,w,h = cv2.boundingRect(i)   
    point.append((x,y,w,h))
point = sorted(point,key=lambda x:x[0], reverse=False)

digit = {}
for i,(x,y,w,h) in enumerate(point):
    digit[i] = cv2.resize(ref_gray[y-2:y+h+2,x-2:x+w+2],(48,64))


img = cv2.imread('img/8.jpg')
img = cv2.resize(img,(400,250))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
roi = gray[135:165,:]
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))

canny = cv2.Canny(roi,200,250)
# adp_thresh = cv2.adaptiveThreshold(canny, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
    #  cv2.THRESH_BINARY_INV, 11, 20)
dilate = cv2.dilate(canny,sqKernel,iterations=2)
cnt = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]

loc = []
for i in cnt:
    x,y,w,h = cv2.boundingRect(i)
    if w*h>1000:
        loc.append((x,y+135,w,h))
    
loc = sorted(loc, key=lambda x: x[0], reverse=False)
print(loc,len(loc))
(x_0,y_0,w_0,h_0) = loc[0]
(x_1,y_1,w_1,h_1) = loc[1]

locs = []
for i in range(4):
    for j in range(4):
        temp_tuple = (int(x_0+j*w_0/4+i*(x_1-x_0)),y_0,int(w_0/4),h_0)
        locs.append(temp_tuple)


results = []
#exp
rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))
sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
for i,(x,y,w,h) in enumerate(locs):
    image = gray[y-1:y+h+1,x-1:x+w+1]
    hat2 = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,rectKernel,iterations=2)
    thresh2 = cv2.threshold(hat2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    dst = thresh2
    dst = cv2.resize(dst,(48,64))

    
    sorts = []
    for j,temp in enumerate(digit.values()):
        result = cv2.matchTemplate(dst,temp,cv2.TM_CCORR)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        sorts.append(max_val)
    results.append(str(np.argmax(sorts)))
print(results)

#part4:
count = []
for (i, (x, y, w, h)) in enumerate(locs):    
    cv2.rectangle(img, (x, y),(x + w, y + h), (0, 0, 255), 2)
    
    # conditon = i%4
    # if conditon == 0:
    #     x_begin = x
    #     cv2.rectangle(img, (x_begin, y-5),(x_begin + 4*w+10, y + h+10), (0, 0, 255), 2)
    cv2.putText(img, "".join(results[i:(i+1)]), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
cv2.imshow('res',img)
cv2.waitKey(0) 