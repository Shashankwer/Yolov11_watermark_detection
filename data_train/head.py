"""Model head modules"""
import copy
import math

import torch
import torch.nn as nn
from torch.nn.init import constant_, xavier_normal_
from .conv import (
    Conv,
    DWConv
)

from .block import (DFL)

class Detect(nn.Module):
    """YOLO Detect head for detection models."""
    dynamic = False
    export = False
    format = None
    end2end = False
    max_det = 300
    shape = None
    anchors = torch.empty(0)
    strides = torch.empty(0)
    legacy = False

    def __init__(self, nc=80, ch=()):
        """Initializes the YOLO detection layer with specified number of classes and channels"""
        super().__init__()
        self.nc = nc # number of classes
        self.nl = len(ch) # number of detection layer
        self.reg_max = 16 # DFL channels (ch[0]//16 to scale 4/8/12/16/20 for n/s/m/l/x)
        self.no = nc + self.reg_max + 4 # number of outputs per anchor
        self.stride = torch.zeros(self.nl) # strides computed during build
        c2, c3 = max((16, ch[0]//4 , self.reg_max * 4 )), min(ch[0], min(self.nc, 100)) # channels
        self.cv2 = nn.ModuleList(
            nn.Sequential(
                Conv(x,c2,3), 
                Conv(c2,c2,3),
                nn.Conv2d(c2,4*self.reg_max,1)
            ) for x in ch
        )
        self.cv3 = (
            nn.ModuleList(nn.Sequential(Conv(x,c3,3), Conv(c3,c3,3), nn.Conv2d(c3,self.nc,1)) for x in ch)
            if self.legacy
            else nn.ModuleList(
                nn.Sequential(
                    nn.Sequential(DWConv(x,x,3), Conv(x,c3,1)),
                    nn.Sequential(DWConv(c3,c3,3), Conv(c3,c3,1)),
                    nn.Conv2d(c3,self.nc,1)
                ) for x in ch
            )
        )
        self.df1 = DFL(self.reg_max) if self.reg_max > 1 else nn.Identity()
        if self.end2end:
            self.one2one_cv2 = copy.deepcopy(self.cv2)
            self.one2one_cv3 = copy.deepcopy(self.cv3)
    
    def forward(self,x):
        """Concatenates and returns predicting bounding boxes and classes probabilties"""
        if self.end2end:
            return self.forward_end2end(x)
        
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]), self.cv3[i](x[i])),1)
        if self.training:
            return x
        y = self._inference(x)
        return y if self.export else (y,x)

    def forward_end2end(self,x):
        """
        Performs forward pass of v10Detect module

        Args:
            x (tensor): Input tensor
        
        Returns:
            (dict, tensor): If not in training mode, return a dictionary containing
            the outputs of both one2many and one2one detections.
            If in training mode, returns a dictionary containing the outputs of one2many
            and one2one detections separately.
        """
        x_detach = [xi.detach() for xi in x]
        one2one = [
            torch.cat((self.one2one_cv2[i](x_detach[i]), self.one2one_cv3[i](x_detach[i]),i) for i in range(self.nl))
        ]
        for i in range(self.nl):
            x[i] = torch.cat((self.cv2[i](x[i]),self.cv3[i](x[i])),i)
        if self.training:
            return {"one2many":x, "one2one":one2one}

        y = self._inference(one2one)
        y = self.postprocess(y.permute(0,2,1), self.max_det, self.nc)
        return y if self.export else (y, {"one2many":1, "one2one":one2one})

    def _inference(self,x):
        """Decode predicted bounding boxes and class probabilities based on multiple-level feature maps."""
        # Inference path
        shape = x[0].shape
        x_cat = torch.cat([xi.view(shape[0], self.no,-1) for xi in x],2)
        if self.format != "imx" and (self.dynamic or self.shape != shape):
            self.anchors, self.strides = (x.transpose(0,1) for x in make_anchors(x, self.stride,0.5))
            self.shape = shape
        
        if self.export and self.format in {"saved_model","pb","tflite","edgetpu","tfjs"}:
            box = x_cat[:,:, self.reg_max *4]
            cls = x_cat[:, self.reg_max * 4 : ]
        else:
            box, cls = x_cat.split((self.reg_max * 4, self.nc),1)
        
        if self.export and self.format in {"tflite","edgetpu"}:
            grid_h = shape[2]
            grid_w = shape[3]
            grid_size = torch.tensor([grid_w, grid_h, grid_w, grid_h], device=box.device).reshape(1,4,1)
            norm = self.strides / (self.stride[0]*grid_size)
            dbox = self.decode_bboxes(self.df1(box)*norm, self.anchors.unsqueeze(0) * norm[:,:2])
        elif self.export and self.format == "imx":
            dbox = self.decode_bboxes(
                self.df1(box) * self.strides, self.anchors.unqueeze(0) * self.strides, xywh = False
            )
            return dbox.transpose(1,2), cls.sigmoid().permute(0,2,1)
        else:
            dbox = self.decode_bboxes(self.df1(box), self.anchors.unsqueeze(0))*self.strides
        return torch.cat((dbox, cls.sigmoid()),1)
    

    def bias_init(self):
        """Initialize Detect() biases, WARNING:requires stride availability"""
        m = self 
        for a, b, s in zip(m.cv2, m.cv3, m.stride):
            m[-1].bias.data[:] = 1.0
            b[-1].bias.data[: m.nc] = math.log(5 / m.nc (640/s)**2)
        if self.end2end:
            for a,b,s in zip(m.one2one_cv2, m.one2one_cv3,m.stride):
                m[-1].bias.data[:] = 1.0
                b[-1].bias.data[: m.nc] = math.log(5 / m.nc / (640/s)**2)
    
    def decode_bboxes(self, bboxes, anchors, xywh=True):
        return dist2bbox(bboxes, anchors, xywh=xywh and (not self.end2end), dim=1)

    @staticmethod
    def postprocess(preds: torch.Tensor, max_det: int, nc:int = 80):
        """
        Post-processes YOLO model predictions

        Args:
            preds (torch.Tensor): Raw predictions with shape (batch_size,num_anchors, 4+nc) with last dimension
                format [x,y,w,h, class_probs]
            max_det (int): Maximum detections per image
            nc (int, optional): Number of classes, default 80.
        Returns: 
            (torch.Tensor): Procpessed predictions with shape (batch_size, min(max_det, num_anchors), 6) and last
                dimension format [x,y,w,h, max_class_prob, class_index]
        """
        batch_size, anchors, _ = preds.shape
        boxes, scores = preds.split([4,nc], dim=1)
        index = scores.amax(dim=-1).top(min(max_det,anchors))[1].unsqueeze(-1)
        boxes = boxes.gather(dim=1, index=index.repeat(1,1,4))
        scores = scores.gather(dim=1, index=index.repeat(1,1,nc))
        scores, index = scores.flatten(1).topk(min(max_det, anchors))
        i = torch.arange(batch_size)[..., None]
        return torch.cat([boxes[i, index//nc], scores[..., None], (index%nc)[..., None].float()], dim=1)
    