"""
The function contains the helper functions for allowing 
1. preprocessing the data before inference
2. Non maximum suppression
3. Predict the output
4. Draw bounding box with the labels
5. Convert image to to base64 for serving if the prediction exists else send the image itself.
"""
import cv2
import numpy as np
import torch, torchvision
from PIL import Image
from copy import deepcopy

# Class Names
NAMES = {0: "watermark"}

def preprocess(img, new_shape=(640,640)):
    """
    Prepare image for onnx model to be processed
    """
    shape = img.shape[:2]
    r = min(new_shape[1]/shape[1],new_shape[0]/shape[0])
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw,dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    dw/=2
    dh/=2
    img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(
            img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114)
    )
    img = np.rollaxis(img,-1,0)
    img = img.astype(np.float32)
    img /= 255
    return img

def xywh2xyxy(x):
    y = torch.empty_like(x)
    xy = x[...,:2]
    wh = x[...,2:]/2
    y[...,:2] = xy - wh
    y[...,2:] = xy + wh
    return y

def non_max_suppression(
      prediction,
      conf_thresh=0.25,
      iou_thresh=0.45,
      max_det=300,
      nc=None, # number of classes
      max_nms=30000,
      max_wh=7680
    ):
    import torchvision    
    if isinstance(prediction,(tuple, list)):
        prediction = prediction[0]
    if not isinstance(prediction, torch.Tensor):
        prediction = torch.tensor(prediction)
    bs = prediction.shape[0] # batch size
    nc = nc or (prediction.shape[1] - 4) # number of classes
    nm = prediction.shape[1] - nc - 4
    mi = 4+nc # mask start index
    xc = prediction[:, 4:mi].amax(1) > conf_thresh
    prediction = prediction.transpose(-1,-2) # shape (1,84,6300) to shape(1,6300,84)
    prediction[..., :4] = xywh2xyxy(prediction[..., :4])
    output = [torch.zeros((0,6+nm), device=prediction.device)]*bs
    for xi, x in enumerate(prediction):
        # Apply constraints
        x = x[xc[xi]]
        if not x.shape[0]:
            continue
        box, cls, mask = x.split((4,nc,nm),1)
        conf,j = cls.max(1, keepdim=True)
        x = torch.cat((box, conf, j.float(), mask),1)[conf.view(-1)>conf_thresh]
        n = x.shape[0]
        if not n:
            continue
        if n > max_nms:
            x = x[x[:,4].argsort(descending=True)[:max_nms]]
        # Batch NMS
        c = x[:,5:6] * (max_wh)
        scores = x[:,4]
        boxes = x[:,:4] + c # boxes
        i = torchvision.ops.nms(boxes=boxes, scores=scores, iou_threshold=iou_thresh)
        print(i)
        i = i[:max_det]
        output[xi] = x[i]
    return output

def clip_boxes(boxes, shape):
    if isinstance(boxes, torch.Tensor):
        boxes[...,0] = boxes[...,0].clamp(0, shape[1])
        boxes[...,1] = boxes[...,1].clamp(1, shape[0])
        boxes[...,2] = boxes[...,2].clamp(0, shape[1])
        boxes[...,3] = boxes[...,3].clamp(0, shape[0])
    else:
        boxes[...,[0.2]] = boxes[...,[0,2]].clip(0,shape[1]) # x1,x2
        boxes[...,[1,3]] = boxes[...,[1,3]].clip(0, shape[0]) # y1, y2
    return boxes

def scale_boxes(img1_shape, boxes, img0_shape, ratio_pad=None, padding=True, xywh=False):
    """
    Rescales bounding boxes (in the format of xyxy by default) from the shape of the image they were originally
    specified in (img1_shape) to the shape of a different image (img0_shape).

    Args:
        img1_shape (tuple): The shape of the image that the bounding boxes are for, in the format of (height, width).
        boxes (torch.Tensor): the bounding boxes of the objects in the image, in the format of (x1, y1, x2, y2)
        img0_shape (tuple): the shape of the target image, in the format of (height, width).
        ratio_pad (tuple): a tuple of (ratio, pad) for scaling the boxes. If not provided, the ratio and pad will be
            calculated based on the size difference between the two images.
        padding (bool): If True, assuming the boxes is based on image augmented by yolo style. If False then do regular
            rescaling.
        xywh (bool): The box format is xywh or not, default=False.

    Returns:
        boxes (torch.Tensor): The scaled bounding boxes, in the format of (x1, y1, x2, y2)
    """
    if ratio_pad is None:  # calculate from img0_shape
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain  = old / new
        pad = (
            round((img1_shape[1] - img0_shape[1] * gain) / 2 - 0.1),
            round((img1_shape[0] - img0_shape[0] * gain) / 2 - 0.1),
        )  # wh padding
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    if padding:
        boxes[..., 0] -= pad[0]  # x padding
        boxes[..., 1] -= pad[1]  # y padding
        if not xywh:
            boxes[..., 2] -= pad[0]  # x padding
            boxes[..., 3] -= pad[1]  # y padding
    boxes[..., :4] /= gain
    return clip_boxes(boxes, img0_shape)

class BaseTensor:
    """
    Main class for simple tensor based operation
    """
    def __init__(self, data, orig_shape):
        self.data = data
        self.orig_shape = orig_shape
    
    @property
    def shape(self):
        return self.data.shape
    
    def cpu(self):
        return self if isinstance(self.data, np.ndarray) else self.__class__(self.data.numpy(), self.orig_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.__class__(self.data[idx], self.orig_shape)

def xyxy2xywh(x):
    """
    Convert bounding box coordinates from (x1,y1,x2,y2) format to (x,y, width, height) format where (x1, y1) is 
    the top-left corner and (x2,y2)
    """
    y = torch.empty_like(x)
    y[...,0] = (x[...,0] + x[...,2])/2
    y[...,1] = (x[...,1] + x[...,3])/2
    y[...,2] = x[...,2] - x[...,0]
    y[...,3] = x[...,3] - x[...,1]
    return y

class Boxes(BaseTensor):
    """
    A class for managing and manipulating detection boxes
    """
    def __init__(self, boxes, orig_shape):
        super().__init__(boxes,orig_shape)
        if boxes.ndim == 1:
            boxes = boxes[None,:]
        n = boxes.shape[-1]
        assert n in {6,7}, f"Expected 6 or 7 values but got {n}"
        self.orig_shape = orig_shape
    
    @property
    def xyxy(self):
        """
        Returns bounding boxes in [x1,y1,x2,y2]
        """
        return self.data[:4]

    @property
    def conf(self):
        return self.data[-2]

    @property
    def cls(self):
        return self.data[-1]
    
    @property
    def xywh(self):
        return xyxy2xywh(self.xyxy)

    @property
    def xyxyn(self):
        xyxy = self.xyxy.clone() if isinstance(self.xyxy, torch.Tensor) else np.copy(self.xyxy)
        xyxy[...,[0,2]] /= self.orig_shape[1]
        xyxy[...,[1,3]] /= self.orig_shape[0]
        return xyxy

class Annotator:
    def __init__(self, im, line_width=None, font_size=None, font="Arial.tff", example="abc"):
        self.lw = line_width or max(round(sum(im.shape)/2*0.003),2)
        self.im = im if im.flags.writeable else im.copy()
        self.tf = max(self.lw-1,1) # font thickness
        self.sf = self.lw / 3 # font scale
        self.dark_colors = {
            (235, 219, 11),
            (243, 243, 243),
            (183, 223, 0),
            (221, 111, 255),
            (0, 237, 204),
            (68, 243, 0),
            (255, 255, 0),
            (179, 255, 1),
            (11, 255, 162),
        }
        self.light_colors = {
            (255, 42, 4),
            (79, 68, 255),
            (255, 0, 189),
            (255, 180, 0),
            (186, 0, 221),
            (0, 192, 38),
            (255, 36, 125),
            (104, 0, 123),
            (108, 27, 255),
            (47, 109, 252),
            (104, 31, 17),
        }

    def get_txt_color(self, color=(128,128,128), txt_color=(255,255,255)):
        """
        Assign a text color based on the background color
        """
        if color in self.dark_colors:
            return 104,31,17
        elif color in self.light_colors:
            return 255,255,255
        else:
            return txt_color
        
    
    def text_label(self, box, label="",color=(128,128,128), txt_color=(255,255,255), margin=5):
        """
        Draws a label with a background rectangle centered within a given bounding box.
        Args:
            box (tuple): The bounding box coordinate
            label: The text label to be labeled
            color: The background color of the rectangle
            txt_color: The color of the txt
            margin: The margin between the text and the rectangle border
        """
        x_center,y_center = int((box[0]+box[2])/2), int((box[1]+box[3])/2)
        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, self.sf - 0.1, self.tf)[0]
        text_x = x_center - text_size[0] // 2
        text_y = y_center - text_size[1] // 2
        rect_x1 = text_x - margin
        rect_y1 = text_y - text_size[1] - margin
        rect_x2 = text_x + text_size[1] - margin
        rect_y2 = text_y + margin
        cv2.rectangle(self.im, (rect_x1, rect_y1), (rect_x2, rect_y2), color, -1)
        # Draw the text on top of the rectangle
        cv2.putText(
            self.im,
            label,
            (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            self.sf - 0.1,
            self.get_txt_color(color, txt_color),
            self.tf,
            lineType=cv2.LINE_AA,
        )

    
    def box_label(self, box, label="", color=(128,128,128), txt_color=(255,255,255), rotated=False):
        """
        Draws a bounding box to image with label
        """
        txt_color = self.get_txt_color(color, txt_color)
        if isinstance(box,torch.Tensor):
            box = box.tolist()
        if rotated:
            p1 = [int(b) for b in box[0]]
            cv2.polylines(self.im, [np.asarray(box,dtype=int)],True, color, self.lw) 
        else:
            p1,p2 = (int(box[0]), int(box[1])),(int(box[2]), int(box[3]))
            cv2.rectangle(self.im, p1,p2, color, thickness=self.lw, lineType=cv2.LINE_AA)
        if label:
            w,h = cv2.getTextSize(label,0, fontScale=self.sf, thickness=self.tf)[0]
            h += 3
            outside = p1[1] >= h
            if p1[0] > self.im.shape[1] - w:
                p1 = self.im.shape[1] - w,p1[1]
            p2 = p1[0] + w, p1[1] - h  if outside else p1[1] + h
            cv2.rectangle(self.im,p1,p2, color,-1, cv2.LINE_AA)
            cv2.putText(
                self.im,
                label,
                (p1[0],p1[1]-2 if outside else p1[1]+h-1),
                0,
                self.sf,
                txt_color,
                thickness=self.tf,
                lineType=cv2.LINE_AA
            )
    
    def rectangle(self, xy, fill=None, outline=None, width=1):
        self.draw.rectangle(xy, fill, outline, width)
    
    def text(self, xy, text, txt_color=(255,255,255), anchor="top",box_style=False):
        if anchor == "botton":
            w,h = self.font.getsize(text)
        if box_style:
            w,h = cv2.getTextSize(text,0, fontScale=self.sf, thickness=self.tf)[0]
            h+=3
            outside =  xy[1]>= h
            p2 = xy[0] + w, xy[1] - h if outside else xy[1] + h
            cv2.rectangle(self.im,xy, p2, txt_color, -1, cv2.LINE_AA)
            txt_color = (255,255,255)
        cv2.putText(self.im, text,xy,0,self.sf, txt_color, thickness=self.tf, lineType=cv2.LINE_AA)
    
    def result(self):
        return np.asarray(self.im)
    
    def show(self,title=None):
        im = Image.fromarray(np.asarray(self.im)[...,::-1])
        im.show(title=title)
    
    @staticmethod
    def get_bbox_dimension(bbox=None):
        """
        Calculate the area of a bounding area
        """
        x_min, y_min, x_max, y_max = bbox
        width = x_max - x_min
        height = y_max - y_min
        return width, height, width*height

def construct_result(pred, img, orig_image):
    """
    Constructs the result
    """
    pred = pred[0]
    pred[:,:4] = scale_boxes((640,640), pred[:,:4], orig_image.shape)
    boxes = Boxes(pred[:,:6], orig_shape=orig_image.shape[:2])
    annotator = Annotator(orig_image,example=NAMES[0])
    for  id,d in enumerate(reversed(boxes)):
        print(d.data)
        c,d_conf,id = int(d.cls),float(d.conf), id
        name = ("" if id is None else f"id:{id} ") + NAMES[c]
        label = f"{name} {d_conf:.2f}"
        box = d.xyxy.squeeze()
        annotator.box_label(box, label)
    return orig_image
    



