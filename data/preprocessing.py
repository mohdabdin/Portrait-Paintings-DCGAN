import pandas as pd
import numpy as np
from PIL import Image
import os
import cv2
import face_recognition

#rename images numerically 0.jpg, 1.jpg, 2.jpg...
def rename(path):
    files = os.listdir(path)
    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '.jpg'])))
        
        
#crop faces with offset
def crop_faces(in_path, out_path, offset):
    files = os.listdir(in_path)
    for index, file in enumerate(files):
        try:
            img = Image.open(in_path+file)
            img_arr = np.array(img)
            top, right, bottom, left = face_recognition.face_locations(img_arr)[0]
            face_image = img_arr[top-offset:bottom+offset, left-offset:right+offset]
            img = Image.fromarray(face_image)
            img.save(out_path+str(index)+'.jpg')
        except:
            pass
        
        
#resize images
def resize(in_path, out_path, dim):
    files = os.listdir(in_path)
    for index, file in enumerate(files):
        img = Image.open(in_path+file)
        if img.size[0]<img.size[1]:
            ratio = dim/float(img.size[0])
    
            new_x = dim
            new_y = int(ratio * img.size[1])
            
            img = img.resize((new_x, new_y))
            val = int((new_y-dim)/2.0)
            ltrb = (0,val,img.size[0],dim+val)
            img = img.crop(ltrb)
            
        elif img.size[0]>img.size[1]:
            ratio = dim/float(img.size[1])
    
            new_x = int(ratio * img.size[0])
            new_y = dim
            
            img = img.resize((new_x, new_y))
            
            val = int((new_x-dim)/2.0)
            ltrb = (val,0,dim+val,img.size[1])
                
            #cropping
            img = img.crop(ltrb)
        
        #if image is square    
        else:
            img = img.resize((dim, dim))
            
        img = img.convert('RGB') 
        img.save(out_path+str(index)+'.jpg')
            

imgs_path = r'downloaded images file path'            
out_path = r'output of processed images'            
#crop_faces(imgs_path, out_path, offset=50)
#resize(imgs_path, out_path, dim=128)
#rename(imgs_path)
