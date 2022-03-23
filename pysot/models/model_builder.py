# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import torch as t
import torch.nn as nn
import torch.nn.functional as F
from pysot.core.config import cfg
from pysot.models.loss import select_cross_entropy_loss,IOULoss
from pysot.models.backbone.newalexnet import AlexNet
from pysot.models.utile.utile import HiFT
import numpy as np



class ModelBuilder(nn.Module):
    def __init__(self):
        super(ModelBuilder, self).__init__()

        self.backbone = AlexNet().cuda()
        self.grader=HiFT(cfg).cuda()
        self.cls2loss=nn.BCEWithLogitsLoss()
        self.IOULoss=IOULoss()          
        
    def template(self, z):
        with t.no_grad():
            zf = self.backbone(z)
    
            self.zf=zf
            
    
    def track(self, x):
        with t.no_grad():
            
            xf = self.backbone(x)  
            loc,cls1,cls2=self.grader(xf,self.zf)

            return {

                'cls1': cls1,
                'cls2': cls2,
                'loc': loc
               }

    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)

        return cls



    def getcentercuda(self,mapp):

        def con(x):
            return x*143
        def dcon(x):
           x[t.where(x<=-1)]=-0.99
           x[t.where(x>=1)]=0.99
           return (t.log(1+x)-t.log(1-x))/2 
        
        size=mapp.size()[3]
        #location 
        x=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        y=t.Tensor(np.tile((16*(np.linspace(0,size-1,size))+63).reshape(-1,1)-cfg.TRAIN.SEARCH_SIZE//2,size).reshape(-1)).cuda()
        
        shap=dcon(mapp)*143
        
        xx=np.int16(np.tile(np.linspace(0,size-1,size),size).reshape(-1))
        yy=np.int16(np.tile(np.linspace(0,size-1,size).reshape(-1,1),size).reshape(-1))


        w=shap[:,0,yy,xx]+shap[:,1,yy,xx]
        h=shap[:,2,yy,xx]+shap[:,3,yy,xx]
        x=x-shap[:,0,yy,xx]+w/2+cfg.TRAIN.SEARCH_SIZE//2
        y=y-shap[:,2,yy,xx]+h/2+cfg.TRAIN.SEARCH_SIZE//2

        anchor=t.zeros((cfg.TRAIN.BATCH_SIZE//cfg.TRAIN.NUM_GPU,size**2,4)).cuda()

        anchor[:,:,0]=x-w/2
        anchor[:,:,1]=y-h/2
        anchor[:,:,2]=x+w/2
        anchor[:,:,3]=y+h/2
        return anchor
    

    def forward(self,data):
        """ only used in training
        """
                
        template = data['template'].cuda()
        search =data['search'].cuda()
        bbox=data['bbox'].cuda()
        labelcls1=data['label_cls1'].cuda()
        labelxff=data['labelxff'].cuda()
        labelcls2=data['labelcls2'].cuda()
        weightxff=data['weightxff'].cuda()
        

        
        zf = self.backbone(template)
        xf = self.backbone(search)
        loc,cls1,cls2=self.grader(xf,zf)
       
        cls1 = self.log_softmax(cls1) 

        
 
        cls_loss1 = select_cross_entropy_loss(cls1, labelcls1)
        cls_loss2 = self.cls2loss(cls2, labelcls2)  
        
        pre_bbox=self.getcentercuda(loc) 
        bbo=self.getcentercuda(labelxff) 
        
        loc_loss=cfg.TRAIN.w3*self.IOULoss(pre_bbox,bbo,weightxff) 
       
        cls_loss=cfg.TRAIN.w4*cls_loss1+cfg.TRAIN.w5*cls_loss2

        
        

        outputs = {}
        outputs['total_loss'] =\
            cfg.TRAIN.LOC_WEIGHT*loc_loss\
                +cfg.TRAIN.CLS_WEIGHT*cls_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss

        return outputs
