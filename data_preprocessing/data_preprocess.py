#! /bin/python3

"""
Given an image the function/file does the following
1. For the predefined set of classes of image to be operated on
    1. Read the image; Resize the image
    2. Draw bounding boxes
    3. Add additional data preprocessing;  Resizing; Image augmentation methods etc
    4. Save the label 

Image dimension here is 640 x 640; The intention is to allow
the model to detect on different lighting conditions and different
Image augmentations
    
Following set of augmentation are performed for the same
1. Horizontal filipping
    Rotates the image by a scale of 180 degree
        The new coordinates are now
        x: - DIM/2 + (DIM/2 - x_original)
        y: y
        h: h
        w: w
2. Vertical fillipping: This causes image to mirror with respect to x axis
    x: x
    y: DIM/2 + (DIM/2 - y_original)
    h: h
    w: w
3. Scaling: Positive affine Scaling
4. Negative affine scaling: done on 2 levels 
    (reason being negative scaling does not make the edges to disappear)
    x: DIM_x/2 + scale * (DIM_x/2 - old_DIM_x)
    y: DIM_y/2 + scale * (DIM_y/2 - old_DIM_y)
    h: scale * h_old
    w: scale * w_old
5. Gaussian Blurring
6. Contrastive Normalization
7. AdditiveGaussianNoise
8. Multiply
9. AdditiveGaussianNoise
"""
import os
import cv2
import numpy as np
import tkinter as tk
import math
import random
import pathlib
import imgaug as ia
import imgaug.augmenters as iaa


LABEL_CLASS = {"watermarks":0}
WORKDIR = ""
DIM = 640

# Common Functions
#For finding the height and width
def euclidean(x1,y1,x2,y2):
    return math.sqrt(sum([(x1-x2)**2,(y1-y2)**2]))

label = 0
label_name = "watermarks"
# Class for Labels
class App(tk.Frame):
    def __init__(self, master=None):
        super().__init__(master)
        self.pack()
        self.entrythingy = tk.Entry()
        self.entrythingy.pack()
        self.contents = tk.StringVar()
        self.contents.set(label_name)
        self.entrythingy["textvariable"] = self.contents
        self.entrythingy.bind("<Key-Return>", self.print_contents)
    
    def print_content(self, event):
        global label
        label = self.contents.get()
        if label.lower().strip() in LABEL_CLASS:
            label = LABEL_CLASS[label.lower().strip()]
            self.entrythingy.destroy()
            self.master.destroy()
        else:
            self.contents.set("")

# Class for bounding box creation
class DrawBounding:
    def __init__(self):
        self.drawing = False
        self.mode = True
        self.ix = -1
        self.iy = -1
        self.img = None
        self.img_c = None
        self.labels = []
        self.img_c1 = None
        self.class_labels = {
            'normal': [],
            'lr': [],
            'ud': [],
            'as10': [],
            'ad10':[],
            'ad20': [],
            'gb': [],
            'cn': [],
            'ag1': [],
            'm': [],
            'ag2': []
        }
        self.m = []

    def reset_image(self):
        self.img = self.img_c.copy()

    def set_images(self,img):
        self.img = img.copy()
        self.img_c = img.copy()
        self.img_c1 = img.copy()
    
    def reset_bounding(self):
        if len(self.labels)>1:
            self.labels.pop()
            for key in self.class_labels:
                if len(self.class_labels[key])>1:
                    self.class_labels[key].pop()
            self.img = self.img_c1.copy()
            if len(self.labels)>0:
                for label in self.labels:
                    x,y,x1,y1 = label
                    cv2.rectange(self.img,(x,y),(x1,y1),(0.255,0))
    
    def reset_trans(self):
        if len(self.m)>1:
            d = self.m.pop()
            x,y,x1,y1 = d
            cv2.rectangle(self.img_c,(x,y),(x1,y1),(0,255,0))
            self.img = self.img - self.img_c
            self.img_c = np.zeros((DIM,DIM,3), np.uint8)
    
    # mouse click events
    def draw_label(self,event,x,y,flags,params):
        print(event, cv2.EVENT_LBUTTONDOWN,cv2.EVENT_MOUSEMOVE, cv2.EVENT_LBUTTONUP,cv2.EVENT_RBUTTONDBLCLK)
        if event == 0:
            if self.drawing:
                if self.mode:
                    print("Drawing mode is on")
                    self.img_c = self.img.copy()
                    cv2.rectangle(self.img_c, (self.ix, self.iy),(x,y), (0,255,0))
                    cv2.namedWindow('Image2')
                    while (1):
                        cv2.imshow('Image2',self.img_c)
                        k = cv2.waitKey(2) & 0xFF
                        if k == ord('q'):
                            break
                    cv2.destroyWindow('Image2')
                    print((self.ix, self.iy),(x,y))
        elif event == cv2.EVENT_LBUTTONDOWN:
                self.m = []
                self.drawing = True
                self.ix = x
                self.iy = y
        elif event == cv2.EVENT_MOUSEMOVE:
                if self.drawing:
                    if self.mode:
                        self.img_c = self.img.copy()
                        cv2.rectangle(self.img_c, (self.ix, self.iy),(x,y),(0,255,0))
        elif event == cv2.EVENT_LBUTTONUP:
                self.drawing = False
                if self.mode:
                    cv2.rectangle(self.img, (self.ix, self.iy),(x,y),(0,255,0))
                    self.labels.append([self.ix,self.iy,x,y])
                    #root = tk.Tk()
                    #app = App(master=root)
                    #app.mainloop()
                    width = np.round(euclidean(self.ix,self.iy, x,self.iy))/DIM
                    height = np.round(euclidean(self.ix,self.iy, self.ix,y))/DIM
                    centre_x = (x+self.ix)/(2*DIM)
                    centre_y = (y+self.iy)/(2*DIM)
                    self.class_labels['normal'].extend([label, centre_x,centre_y, width, height])
                    # lr
                    x_new = (DIM/2+ (DIM/2-centre_x*DIM))/DIM
                    self.class_labels['lr'].extend([label, x_new, centre_y, width, height])
                    self.class_labels['ud'].extend([label, centre_x, (DIM/2+(DIM/2-centre_y*DIM))/DIM,width, height])
                    self.class_labels['as10'].extend([label,(DIM/2+1.1*(DIM/2 - centre_x*DIM))/DIM,(DIM/2 + 1.1*(DIM/2-centre_y*DIM))/DIM,1.1*width,1.1*height])
                    self.class_labels['ad10'].extend([label,(DIM/2+0.9*(DIM/2 - centre_x*DIM))/DIM,(DIM/2 + 0.9*(DIM/2-centre_y*DIM))/DIM,0.9*width,0.9*height])
                    self.class_labels['ad20'].extend([label,(DIM/2+0.8*(DIM/2 - centre_x*DIM))/DIM,(DIM/2 + 0.8*(DIM/2-centre_y*DIM))/DIM,0.8*width,0.8*height])
                    self.class_labels['gb'].extend([label, centre_x,centre_y,width,height])
                    self.class_labels['cn'].extend([label, centre_x,centre_y,width,height])
                    self.class_labels['ag1'].extend([label, centre_x,centre_y,width,height])
                    self.class_labels['m'].extend([label, centre_x,centre_y,width,height])
                    self.class_labels['ag2'].extend([label, centre_x,centre_y,width,height])
                    print(self.class_labels)     
        elif event == cv2.EVENT_RBUTTONDBLCLK:
            print("Double button clicked")
            self.reset_bounding()
            cv2.imshow('Image', self.img)
    
    def get_labels(self):
        return self.class_labels.copy()

class ImageAugumenter():
    def __init__(self, source_folder:str=None,file_save_dir:str=None, train_test_split=0.8):
        self.source_folder = source_folder
        self.file_save_dir = file_save_dir
        self.train_test_split = train_test_split
        self.current_dir = os.getcwd()
        if os.path.exists(os.path.join(self.current_dir, self.source_folder)):
            self.image_path = os.path.join(self.current_dir, self.source_folder)
        else:
            raise FileNotFoundError("File/Folder does not exists")
        self.train_image = os.path.join(self.current_dir, self.file_save_dir, 'train','images')
        self.train_image_label = os.path.join(self.current_dir, self.file_save_dir, 'train','labels')
        self.val_image = os.path.join(self.current_dir, self.file_save_dir,'val','images')
        self.val_image_label = os.path.join(self.current_dir, self.file_save_dir, 'val','labels')
        pathlib.Path(self.train_image).mkdir(parents=True,exist_ok=True)
        pathlib.Path(self.train_image_label).mkdir(parents=True,exist_ok=True)
        pathlib.Path(self.val_image).mkdir(parents=True,exist_ok=True)
        pathlib.Path(self.val_image_label).mkdir(parents=True,exist_ok=True)

    def save_image_label(self, img,labels, img_path, label_path):
        cv2.imwrite(img_path, img)
        with open(label_path,"w+") as f:
            print(" ".join([str(l) for l in labels]))
            f.write(" ".join([str(l) for l in labels]))

    def process_image(self):
        # Process image
        for file in os.listdir(self.source_folder):
            if file.endswith('.jpg'):
                img = cv2.imread(os.path.join(self.source_folder,file))
                img = cv2.resize(img, (DIM, DIM))
                d = DrawBounding()
                d.set_images(img)
                cv2.namedWindow("Image")
                cv2.setMouseCallback('Image',d.draw_label)
                while(1):
                    cv2.imshow('Image', img)
                    k = cv2.waitKey(1) & 0xFF
                    if k == 27:
                        break
                cv2.destroyAllWindows()
                img = cv2.imread(os.path.join(self.source_folder,file))
                img = cv2.resize(img, (DIM, DIM))
                train = random.randint(1,10)<(10*self.train_test_split)
                img_dir = self.train_image if train else self.val_image
                label_dir = self.train_image_label if train else self.val_image_label
                labels = d.get_labels()
                print(labels)
                # normal image
                image = img.copy()
                image_path =  os.path.join(img_dir, file)
                label_path = os.path.join(label_dir, file.split('.')[0]+'.txt')
                label = labels["normal"]
                self.save_image_label(image,label,image_path,label_path)
                # flip lr
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.Fliplr(1)])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_lr.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_lr.txt')
                label = labels["lr"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # flip up
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.Flipud(1)])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_ud.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_ud.txt')
                label = labels["ud"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # as10
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.Affine(
                scale= {"x":(1.1),"y":(1.1)}
                )])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_as10.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_as10.txt')
                label = labels["as10"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # ad10
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.Affine(
                scale= {"x":(0.9),"y":(0.9)}
                )])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_ad10.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_ad10.txt')
                label = labels["ad10"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # ad20
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.Affine(
                scale= {"x":(0.8),"y":(0.8)}
                )])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_ad20.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_ad20.txt')
                label = labels["ad20"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # gb
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.GaussianBlur(sigma=(0,0.5))])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_gb.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_gb.txt')
                label = labels["gb"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # cn
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.ContrastNormalization((1,2))])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_cn.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_cn.txt')
                label = labels["cn"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # m
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.Multiply((1.8,1.2),per_channel=0.6)])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_m.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_m.txt')
                label = labels["m"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # ag1
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0,scale=(0.0,0.05*255),per_channel=0.5)])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_ag1.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_ag1.txt')
                label = labels["ag1"]
                self.save_image_label(image_aug[0],label,image_path,label_path)
                # ag2
                image = np.expand_dims(img, 0)
                seq = iaa.Sequential([iaa.AdditiveGaussianNoise(loc=0,scale=(0.05*255,0.0),per_channel=0.5)])
                image_aug = seq(images=image)
                image_path =  os.path.join(img_dir, file.split('.')[0]+'_ag2.jpg')
                label_path = os.path.join(label_dir, file.split('.')[0]+'_ag2.txt')
                label = labels["ag2"]
                self.save_image_label(image_aug[0],label,image_path,label_path)

if __name__ == '__main__':
    data_augmenter = ImageAugumenter(source_folder='r123-watermark',file_save_dir='datasets')
    data_augmenter.process_image()












                

                
                    




