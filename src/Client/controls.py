#!/home/jeet/anaconda3/bin/python3

#chane the shebang line or else the script won't work
import pyautogui
import time
import cv2
from yolo import YOLO

def scrollV(cap,network,device,size,confidence):
    if network == "normal":
        print("loading yolo...")
        yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
    elif network == "prn":
        print("loading yolo-tiny-prn...")
        yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
    else:
        print("loading yolo-tiny...")
        yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
	
    yolo.size = size
    yolo.confidence = confidence
    
    cnt = 0
    curr = 0
    prev = 0
    exit = 0
    
    rval, frame = cap.read()
    
    while True:
        width, height, inference_time, results = yolo.inference(frame)
    
        if len(results)==1:
            exit = 0
            cnt += 1 
            
            id, name, confidence, x, y, w, h = results[0]
            cx = x + (w // 2)
            cy = y + (h // 2)
            
            if cnt <= 5:
                curr = cy
            
            color = (0, 255, 255)
            cv2.circle(frame, (cx,cy),10,color,-1)
            #print("Cy: ", cy)
                       
            if cnt%10 == 0 and cnt>5:
                prev = curr
                curr = cy  
                #print("Prev: ",prev)
                #print("Curr: ", curr)              
                clicks = prev-curr
                #print(clicks)
                #if clicks>30 and clicks<170:
                clicks = clicks//2
                
                if abs(clicks) > 10:
                    pyautogui.scroll(clicks)
             
        else:
            exit +=1
            if exit>50:
                print(exit)
                break
         
        cv2.imshow("preview", frame)
        rval, frame = cap.read()

        key = cv2.waitKey(1) 
        if key == 27:  # exit on ESC
            break
            
    cv2.destroyWindow("preview")
	
	
def scrollH(cap,network,device,size,confidence):
    if network == "normal":
        print("loading yolo...")
        yolo = YOLO("models/cross-hands.cfg", "models/cross-hands.weights", ["hand"])
    elif network == "prn":
        print("loading yolo-tiny-prn...")
        yolo = YOLO("models/cross-hands-tiny-prn.cfg", "models/cross-hands-tiny-prn.weights", ["hand"])
    else:
        print("loading yolo-tiny...")
        yolo = YOLO("models/cross-hands-tiny.cfg", "models/cross-hands-tiny.weights", ["hand"])
	
    yolo.size = size
    yolo.confidence = confidence
    
    cnt = 0
    curr = 0
    prev = 0
    exit = 0
    
    rval, frame = cap.read()
    
    while True:
        width, height, inference_time, results = yolo.inference(frame)
    
        if len(results)==1:
            exit = 0
            cnt += 1 
            
            id, name, confidence, x, y, w, h = results[0]
            cx = x + (w // 2)
            cy = y + (h // 2)
            
            if cnt <= 5:
                curr = cx
            
            color = (0, 255, 255)
            cv2.circle(frame, (cx,cy),10,color,-1)
            #print("Cy: ", cy)
                       
            if cnt%10 == 0 and cnt>5:
                prev = curr
                curr = cx  
                #print("Prev: ",prev)
                #print("Curr: ", curr)              
                clicks = prev-curr
                #print(clicks)
                #if clicks>30 and clicks<170:
                #clicks = clicks//2
                
                if abs(clicks) > 10:
                    pyautogui.hscroll(clicks)
             
        else:
            exit +=1
            if exit>50:
                print(exit)
                break
         
        cv2.imshow("preview", frame)
        rval, frame = cap.read()

        key = cv2.waitKey(20)
        if key == 27:  # exit on ESC
            break
        
    cv2.destroyWindow("preview")
		
def zoomIn():
	pyautogui.click()
	pyautogui.hotkey('ctrl','+')

def zoomOut():
	pyautogui.click()
	pyautogui.hotkey('ctrl','-')
	
def selectText():
	pyautogui.moveTo(416,196)
	pyautogui.dragTo(1051,297,button='left')

def selectFile():
	#pyautogui.moveTo(x1,x2) #x1 and x2 to be provided by object tracking
	pyautogui.click()
	
def copy():
	pyautogui.moveTo(416,196)
	pyautogui.dragTo(1051,297,button='left')
	pyautogui.hotkey('ctrl','c')

def paste():
	pyautogui.click()
	pyautogui.hotkey('ctrl','v')

def switchApplication():
	pyautogui.hotkey('alt','tab')
	
def nextSlide():
	pyautogui.click()
	pyautogui.press('right')
	
def prevSlide():
	pyautogui.click()
	pyautogui.press('left')
	
def exitFullScreen():
	time.sleep(5)
	pyautogui.press('esc')	
	
def presentModeDemo():
	time.sleep(7)
	pyautogui.press('right')
	time.sleep(1)
	pyautogui.press('left')
	time.sleep(1)
	pyautogui.press('esc')


def performAction(action,cap):
	if action == 'scrollUp':
	    scrollV(cap,'normal',0,256,0.2)
	elif action == 'scrollDown':
	    scrollV(cap,'normal',0,256,0.2)
	elif action == 'scrollRight':
	    scrollH(cap,'normal',0,256,0.2)
	elif action == 'scrollLeft':
	    scrollH(cap,'normal',0,256,0.2)
	elif action == 'zoomIn':
	    zoomIn()
	elif action == 'zoomOut':
	    zoomOut()
	elif action == 'nextSlide':
	    nextSlide()
	elif action == 'prevSlide':
	    prevSlide()
	elif action == 'exitFS':
	    exitFullScreen()
	elif action == 'selectFile':
	    selectFile()
	elif action == 'selectText':
	    selectText()
	elif action == 'copy':
	    copy()
	elif action == 'paste':
	    paste()
	elif action == 'switchApplication':
	    switchApplication()
	elif action == 'present':
	    presentModeDemo()
	elif action == 'doNothing':
		pass
	else:
	    print("Invalid Action Passed!!")
	

if __name__ == "__main__":
	#The following code is just to check all the function individually. Do not include it in integration	
	val = True
	cv2.namedWindow("preview")
	cap = cv2.VideoCapture(0)
	cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
	cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

	while(val):
		action = input('Enter action: ')	
		performAction(action,cap)
		if action == 'exit':
			val = False

	cap.release()		


