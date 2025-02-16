import torch 
from typing import Union
from conv import Conv
from block import C3k2, SPPF,C2PSA
from head import Detect

class Yolov11(torch.nn.Module):
    def __init__(self,nc):
        super().__init__()
        self.model = None
        self.traininer = None
        self.ckpt = {}
        self.ckpt_path = None
        self.overrides = {}
        self.metrics = {}
        self.updates = {}
        self.session = None
        self.model_name = 'yolov11'
        self.nc = nc if nc is not None else 80 # default coco
        self._modules()
        del self.training
        
    
    def __call__(self, source=None,**_kwargs):
        """
        Used for making prediction
        """
        return self.predict(source)

    def load(self, weights):
        self.model.load(weights)
        return self

    def _modules(self):
        self.backbone = torch.nn.Sequential(
            Conv(640,64, k=3,s=1),
            Conv(128,256,k=3,s=2),
            C3k2(256,256,n=0.25, c3k=False),
            Conv(256,512,k=3,s=2),
            C3k2(512,512,k=0.25, C3k=False),
            Conv(512,512,k=3,s=2),
            C3k2(512,1024,c3k=True),
            Conv(1024,1024, k=3,s=2),
            C3k2(1024,1024,c3k=True),
            C3k2(1024,1024,c3k=True),
            SPPF(1024,1024,k=5),
            C2PSA(1024)   
        )  
        self.head = torch.nn.Sequential(
            torch.nn.Upsample(None, 2, "nearest"),
            torch.concat(dim=(1)),
            C3k2(512,1024, c3k=False),
            torch.concat([1]),
            C3k2(256,256,False),
            C3k2(256,256,False),
            Conv(256,3,2),
            C3k2(256, False),
            Conv(512,512, 3,2),
            torch.concat([1]),
            Conv(1024,1024, True),
            Conv(1024,1024, True),
            Detect(self.nc)
        )
        self.model_network = torch.nn.Sequential(
            self.backbone,
            self.head
        )
    
    def forward(self, data):
        return self.model_network(data)

        
