import cv2
import numpy as np
import argparse

#Part One: Arguments passing
parser = argparse.ArgumentParser(description='Recognise credit cards number (OCR) by Opencv')
parser.add_argument('-i','--image', help = 'Path to the image of credit cards') 
parser.add_argument('-r1','--template_1', help='Path to the 1st template(card 1,3 and 4),default=img/ref1.png')
parser.add_argument('-r2','--template_2', help='Path to the 2nd template(the rest cards),default=img/ref2.png')
args = parser.parse_args()
#Example of args settings: 
#"args":["-i","img/1.png","-r1","img/ref1.png","-r2","img/ref2.png"]
#PS: card1-3-->.png file; card 4-8-->.jpg file
file_name = args.image

#Part Two: Packaging useful functions in each step
#Stage One: Create template
def template(ref):
    ref = cv2.resize(ref,(400,64))
    ref_gray = cv2.cvtColor(ref,cv2.COLOR_BGR2GRAY)
    ref_gray = cv2.threshold(ref_gray,100,255,cv2.THRESH_BINARY)[1]
    ref_cont = cv2.findContours(ref_gray,cv2.CV_8UC1,cv2.RETR_CCOMP)[0]
    point = []
    for i in ref_cont:
        x,y,w,h = cv2.boundingRect(i)   
        point.append((x,y,w,h))
    point = sorted(point,key=lambda x:x[0], reverse=False)

    digits = {}
    for i,(x,y,w,h) in enumerate(point):
        digits[i] = cv2.resize(ref_gray[y-2:y+h+2,x-2:x+w+2],(48,64))
    return digits

#Stage Two: Image processing and finding the location of each digits
#Using tophat to remove dark background(for card 1,3-6)
def tophat_thresh(image, iters=1):
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    hat = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,rectKernel,iterations=iters)
    thresh = cv2.threshold(hat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return cnt
#Using blackhat to remove bright background(for 2nd card)
def blackhat(image,iters=2):
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    hat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,rectKernel,iterations=iters)
    thresh = cv2.threshold(hat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
    cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
    return cnt
#Using boundingRect() function to get the locations of each dight(for card 1-6)
def get_locs(cnt):
    locs = []
    for i in cnt:
        x,y,w,h = cv2.boundingRect(i)
        if w*h>200:
            locs.append((x,y+135,w,h))
    locs = sorted(locs, key=lambda x: x[0], reverse=False)
    return locs

#Stage Three: Template match(for card 1-6)
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
def match_template_blackhat(locs,digits,iters=2):
    results = []
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    for i,(x,y,w,h) in enumerate(locs):
        image = gray[y-2:y+h+2,x-2:x+w+2]
        hat_image = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,rectKernel,iterations=iters)
        dst = cv2.resize(hat_image,(48,64))
        sorts = []
        for j,temp in enumerate(digits.values()):
            result = cv2.matchTemplate(dst,temp,cv2.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
            sorts.append(max_val)
        results.append(str(np.argmax(sorts)))
    return results

#Stage Four: Visualize the results
def draw_digits(image,results,locs): 
    count = []
    for (i, (x, y, w, h)) in enumerate(locs):    
        conditon = i%4
        if conditon == 0:
            x_begin = x
            cv2.rectangle(image, (x_begin, y-5),(x_begin + 4*w+10, y + h+10), (0, 0, 255), 2)
        cv2.putText(img, "".join(results[i:(i+1)]), (x, y - 15),cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
    cv2.imshow('res',img)
    cv2.waitKey(0)
    cv2.destroyAllWindows
    return 

#Part THree: The main part
#step 1: selecting the correct reference and generating template for each digit 
ref1 = cv2.imread(args.template_1)
ref2 = cv2.imread(args.template_2)
if ('1.png' in file_name) or ('3.png' in file_name) or ('4.jpg' in file_name):
    ref = ref1
else:
    ref = ref2
digits = template(ref)

#step 2: image processing and template matching
img = cv2.imread(file_name)
img = cv2.resize(img,(400,250))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
roi = gray[135:165,:]

#image  for card 1
cnt = tophat_thresh(roi)
locs = get_locs(cnt)
results = match_template(locs,digits)
#using the lenth of locs(the number of recognised digits) to judge which card is being processed
if len(locs) != 16: #for card 3,4,5,6
    cnt = tophat_thresh(roi,2)
    locs = get_locs(cnt)
    results = match_template(locs,digits,2)
if len(locs) != 16: #for card 2
    cnt = blackhat(roi)
    locs = get_locs(cnt)
    results = match_template_blackhat(locs,digits)
if len(locs) != 16: #for card 7
    #devide the images into 3 the ROI: 
    #roi_1 and roi_3 have dark background, which can be dealt with tophat; while roi_2 have bright background(), which is processed by canny edge detection.
    #  
    roi_1 = gray[135:165,:125]
    roi_2 = gray[135:165,125:180]
    roi_3 = gray[135:165,180:]
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,9))
    #process roi_2 by canny edge detection
    def hat_canny(image,iters=2):
        hat = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,rectKernel,iterations=iters)
        canny = cv2.Canny(hat,230,250)
        cnt = cv2.findContours(canny,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        cont = cv2.drawContours(image,cnt,-1,(0,0,0),2)
        thresh = cv2.threshold(cont,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)[1]
        cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        return cnt
    #process roi1 and 3
    def hat_thresh(image, iters=2):
        hat = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,rectKernel,iterations=iters)
        thresh = cv2.threshold(hat,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
        cnt = cv2.findContours(thresh,cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)[0]
        return cnt
    #find the locations of each digit in roi1,2 and 3.
    def get_locs_pic7(roi1,roi2,roi3):
        cnt_1 = hat_thresh(roi1)
        cnt_2 = hat_canny(roi2)
        cnt_3 = hat_thresh(roi3)
        cnts = [cnt_1,cnt_2,cnt_3]
        locs = []
        i = 0
        for cnt in cnts:
            i+=1
            for j in cnt:
                x,y,w,h = cv2.boundingRect(j)
                if w*h>150 and w*h<200:
                    locs.append((x,y+135,w,h+10))

                if w*h>200:
                    if i==1:
                        locs.append((x,y+135,w,h))
                    elif i==2:
                        locs.append((x+125,y+135,w,h))
                    else:
                        locs.append((x+180,y+135,w,h))
        locs = sorted(locs, key=lambda x: x[0], reverse=False)
        return locs
    locs = get_locs_pic7(roi_1,roi_2,roi_3)
    #template matching
    def match_template_pic7(locs,digits):
        results = []
        for i,(x,y,w,h) in enumerate(locs):
            if i<4:
                image = gray[y-3:y+h+3,x-3:x+w+3]
                dst= cv2.morphologyEx(image,cv2.MORPH_TOPHAT,rectKernel,iterations=2)
            elif i>6:
                image = gray[y:y+h+2,x-1:x+w+2]
                dst = cv2.morphologyEx(image,cv2.MORPH_TOPHAT,rectKernel,iterations=2)
            else:
                image = gray[y-3:y+h+3,x-3:x+w+3]
                dst = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,rectKernel,iterations=3)
            dst = cv2.resize(dst,(48,64))
            sorts = []
            for j,temp in enumerate(digits.values()):
                result = cv2.matchTemplate(dst,temp,cv2.TM_CCOEFF)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                sorts.append(max_val)
            results.append(str(np.argmax(sorts)))
        return results
    results = match_template_pic7(locs,digits)
if len(locs) != 16: #for card 8
    pass
#unsloved. Best experiment recogise 5 digits out of 16. See 'unsloved_card8.py'.

#step3: result visualization
draw_digits(img,results,locs)