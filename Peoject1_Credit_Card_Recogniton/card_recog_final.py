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
def match_template_blackhat(locs,digits1,digits2,iters=2):
    results = []
    results_1 = []
    results_2 = []
    a=0
    b=0
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(9,3))
    for i,(x,y,w,h) in enumerate(locs):
        image = gray[y-2:y+h+2,x-2:x+w+2]
        hat_image = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,rectKernel,iterations=iters)
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

#Stage Four: Visualize the results
def draw_digits(image,results,locs,individual=False): 
    count = []
    for (i, (x, y, w, h)) in enumerate(locs):
        if individual:
            cv2.rectangle(img, (x, y),(x + w, y + h), (0, 0, 255), 2)    
        else:    
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
digits1 = template(ref1)
digits2 = template(ref2)
#step 2: image processing and template matching
img = cv2.imread(file_name)
img = cv2.resize(img,(400,250))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
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

#for card 8-- the unsloved case.
if ('8.jpg' in file_name) or ('unsloved.jpg' in file_name): #for card 8--unsloved. Best experiment recogise 5 digits out of 16.
    roi=gray[135:165,:]
    rectKernel=cv2.getStructuringElement(cv2.MORPH_RECT,(7,7))

    sqKernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7,7))
    def canny_dilate(roi):
        canny = cv2.Canny(roi,200,250)
        dilate = cv2.dilate(canny,sqKernel,iterations=2)
        cnt = cv2.findContours(dilate,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0]
        return cnt
    def find_locs(cnt):
        loc = []
        for i in cnt:   
            x,y,w,h = cv2.boundingRect(i)
            if w*h>1000:
                loc.append((x,y+135,w,h))
        loc = sorted(loc, key=lambda x: x[0], reverse=False)
        (x_0,y_0,w_0,h_0) = loc[0]
        (x_1,y_1,w_1,h_1) = loc[1]
        locs = []
        for i in range(4):
            for j in range(4):
                temp_tuple = (int(x_0+j*w_0/4+i*(x_1-x_0)),y_0,int(w_0/4),h_0)
                locs.append(temp_tuple)
        return locs
    def template_match_pic8(locs,digits):
        results = []
        for i,(x,y,w,h) in enumerate(locs):
            image = gray[y-1:y+h+1,x-1:x+w+1]
            hat2 = cv2.morphologyEx(image,cv2.MORPH_BLACKHAT,rectKernel,iterations=2)
            thresh2 = cv2.threshold(hat2,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            dst = thresh2
            dst = cv2.resize(dst,(48,64))       
            sorts = []
            for j,temp in enumerate(digits.values()):
                result = cv2.matchTemplate(dst,temp,cv2.TM_CCORR)
                min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                sorts.append(max_val)
            results.append(str(np.argmax(sorts)))    
        return results

    cnt = canny_dilate(roi)
    locs = find_locs(cnt)
    results = template_match_pic8(locs,digits2)    
    draw_digits(img,results,locs,True)

#for successful cases
else:
    #image  for card 1
    roi = find_roi(gray)
    cnt = tophat_thresh(roi)
    locs = get_locs(cnt)
    results = match_template(locs,digits1,digits2)
    #using the lenth of locs(the number of recognised digits) to judge which card is being processed
    if len(locs) != 16: #for card 4-6
        roi = find_roi(gray,2)
        cnt = tophat_thresh(roi,2)
        locs = get_locs(cnt)
        results = match_template(locs,digits1,digits2,2)
    if len(locs) != 16: #for card 3
        roi = find_roi(gray,1)
        cnt = tophat_thresh(roi,2)
        locs = get_locs(cnt)
        results = match_template(locs,digits1,digits2,2)
    if  len(locs) != 16: #for card 2
        roi = find_roi(gray,2)
        cnt = blackhat(roi)
        locs = get_locs(cnt)
        results = match_template_blackhat(locs,digits1,digits2)
    if (len(locs) != 16) and ('8.jpg' not in file_name): #for card 7
        #devide the images into 3 the ROI: 
        #roi_1 and roi_3 have dark background, which can be dealt with tophat; while roi_2 have bright background(), which is processed by canny edge detection.
        roi = find_roi(gray,2)
        roi_1 = roi[:,:125]
        roi_2 = roi[:,125:180]
        roi_3 = roi[:,180:]
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
        # template matching
        def match_template_pic71(locs,digits1,digits2):
            results = []
            results_1 = []
            results_2 = []
            a=0
            b=0
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
                
                sorts_1 = []
                sorts_2 = []
                for j,temp in enumerate(digits1.values()):
                    result_temp1 = cv2.matchTemplate(dst,temp,cv2.TM_CCOEFF)
                    min_val, max_val1, min_loc, max_loc = cv2.minMaxLoc(result_temp1)
                    sorts_1.append(max_val1)
                results_1.append(str(np.argmax(sorts_1)))

                for j2,temp2 in enumerate(digits2.values()):
                    result_temp2 = cv2.matchTemplate(dst,temp2,cv2.TM_CCOEFF)
                    min_val2,max_val2,min_loc2,max_loc2=cv2.minMaxLoc(result_temp2)
                    sorts_2.append(max_val2)
                results_2.append(str(np.argmax(sorts_2)))
                if (np.max(sorts_1) >= np.max(sorts_2)):
                    a+=1
                elif (np.max(sorts_1) < np.max(sorts_2)):
                    b+=1
            print(results_1,results_2,a,b)
            if a>b:
                results = results_1
            else:
                results = results_2
            return results
        
        results = match_template_pic71(locs,digits1,digits2)

    #step3: result visualization
    draw_digits(img,results,locs)