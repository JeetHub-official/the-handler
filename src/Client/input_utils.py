import os
import shutil
import sys
import cv2
import time
import pandas as pd

# define the countdown func. 
def countdown(t): 
    
    while t: 
        mins, secs = divmod(t, 60) 
        timer = '{:02d}:{:02d}'.format(mins, secs) 
        print('The camera will record in - ',timer, end="\r") 
        time.sleep(1) 
        t -= 1
    print('The camera will record in - ',timer)
      

def record_gesture(class_name, no_of_samples=10, sample_duration=2, clip_duration=16, save_path='./recorded_gestures',warn_time=2):
    
    no_of_samples = int(no_of_samples)
    sample_duration = int(sample_duration)
    warn_time = int(warn_time)
    clip_duration = int(clip_duration)
    
    print(f'We will take {no_of_samples} samples of {sample_duration} seconds each.')
    
    for i in range(no_of_samples):
        countdown(warn_time) 
        
        vid = cv2.VideoCapture(0)
        fps = vid.get(cv2.CAP_PROP_FPS) #for opencv version > 3
        no_of_frames = int(fps*sample_duration)
        downsample = int(no_of_frames/clip_duration)
        
        list_of_frames = []*no_of_frames
        for j in range(no_of_frames):
        
            # Capture the video frame by frame 
            ret, frame = vid.read()
            #frame = cv2.flip(frame,1) 

            # Display the resulting frame 
            cv2.imshow(f'{class_name} {i}', frame) 
            #print(frame.shape)
            
            list_of_frames.append(frame)
            #save frame
            
        
            if cv2.waitKey(1) & 0xFF == ord('q'): 
                break
           
        
        # After the loop release the cap object 
        vid.release() 
        # Destroy all the windows 
        cv2.destroyAllWindows() 
        
        #convert size to clip_duration
        selected_frames = [list_of_frames[j] for j in range(0, no_of_frames, downsample)]
        selected_frames = selected_frames[0:clip_duration]
        for index, frame in enumerate(selected_frames):
            cv2.imwrite(f'{save_path}/{class_name}-sample{i}-frame{index}.jpg',frame)

save_path = './recorded_gestures'
if not os.path.exists(save_path):
    os.makedirs(save_path)
labels = ['scrollUp', 'scrollDown', 'scrollRight', 'scrollLeft', 'zoomIn','doNothing']

df = pd.DataFrame(labels)
df.to_csv(f'{save_path}/class_labels.csv',header=False,index=False)

for class_name in labels:
    print(f'Now for {class_name}')
    countdown(5)
    record_gesture(class_name,no_of_samples=5,save_path=save_path)
    



