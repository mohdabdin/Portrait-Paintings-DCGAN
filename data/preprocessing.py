import pandas as pd
import numpy as np
from PIL import Image
import os
import imagehash
import cv2

imgs_path = r'C:\Users\mohd_\OneDrive\Desktop\ML\[project] Picasso\data\archive\celeb_dataset\images'

#rename images numerically 0.jpg, 1.jpg, 2.jpg...
def rename_images(path):
    files = os.listdir(path)

    for index, file in enumerate(files):
        os.rename(os.path.join(path, file), os.path.join(path, ''.join([str(index), '_.jpg'])))

#resize images
def resize(path, dim, aspect_ratio=True):
    if aspect_ratio:
        files = os.listdir(path)
        for index, file in enumerate(files):
            dirname = '128_imgs/'+str(index)+'.jpg'
            img = Image.open(dirname)
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
            img.save('output_dump/'+str(index)+'.jpg')
        
    else:
        files = os.listdir(path)
        for index, file in enumerate(files):
            try:
                dirname = '128_imgs/'+str(index)+'.jpg'
                img = Image.open(dirname)
                new_img = img.resize((dim,dim))
    
                new_img.save(str(dim)+'_imgs/'+str(index)+'.jpg')
                print('e')
                
            except:
                pass

#helper function to check similarity between two images
#def check_similarity(img1, img2):
    
    
#return the 'colorfulness' of an image
def image_colorfulness(image):
	# split the image into its respective RGB components
	(B, G, R) = cv2.split(image.astype("float"))
	# compute rg = R - G
	rg = np.absolute(R - G)
	# compute yb = 0.5 * (R + G) - B
	yb = np.absolute(0.5 * (R + G) - B)
	# compute the mean and standard deviation of both `rg` and `yb`
	(rbMean, rbStd) = (np.mean(rg), np.std(rg))
	(ybMean, ybStd) = (np.mean(yb), np.std(yb))
	# combine the mean and standard deviations
	stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
	meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))
	# derive the "colorfulness" metric and return it
	return stdRoot + (0.3 * meanRoot)

def filter_colorless(path):
    files = os.listdir(path)
    num_deleted = 0
    for index, file in enumerate(files):
        try:
            dirname = 'cubism_imgs_duplicate/'+str(index)+'.jpg'
            img = Image.open(dirname)
            img_arr = np.array(img)
            
            colorfulness = image_colorfulness(img_arr)
            if colorfulness<16:
                os.remove(dirname)
                num_deleted +=1
                print(num_deleted)
        except:
            pass

#helper function to check similarity between two images
def check_similarity(img1, img2):
    hash0 = imagehash.average_hash(img1) 
    hash1 = imagehash.average_hash(img2) 
    
    return hash0-hash1            


def filter_duplicates(path):
    files = os.listdir(path)
    num_imgs = len(files)
    num_deleted = 0
    for i in range(num_imgs):
        j=i+1
        print('on: '+str(i))
        while j<num_imgs:
            try:
                img1 = Image.open('64_imgs/'+str(i)+'.jpg')
                img2 = Image.open('64_imgs/'+str(j)+'.jpg')
                sim = check_similarity(img1, img2)
                if sim==0:
                    os.remove('64_imgs/'+str(j)+'.jpg')
                    num_deleted+=1
                    print(num_deleted)
            except:
                pass
            j+=1
    
    
    
    
    
if __name__ == '__main__':
    rename_images(imgs_path)
    #filter_colorless(imgs_path)
    #filter_duplicates(r'C:\Users\mohd_\OneDrive\Desktop\ML\[project] Picasso\data\64_imgs')
    #resize(imgs_path, 64, aspect_ratio=False)
    
    
 
