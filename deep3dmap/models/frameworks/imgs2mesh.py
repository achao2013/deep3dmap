import os
import math
from copy import deepcopy
from turtle import forward

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.nn.parallel import DataParallel as DP
import torch.distributed as dist
import torchvision
import numpy as np
import scipy.io as sio
from ..builder import MODELS, build_backbone
from deep3dmap.models.frameworks.custom import CustomFramework
from deep3dmap.datasets.pipelines.formating import to_tensor
from deep3dmap.core.utils.device_transfer import to_cuda
from deep3dmap.core.utils.fileio import read_obj
from deep3dmap.core.renderer.renderer_pt3d import Pt3dRenderer
from ..builder import BACKBONES, build_backbone
from deep3dmap.core.all3dmm.bfm_tools import param2points_bfm
from pytorch3d.transforms import euler_angles_to_matrix, matrix_to_euler_angles,Rotate

@MODELS.register_module()
class imgs2mesh(CustomFramework):
    def __init__(self, model_cfgs, train_cfg=None, test_cfg=None):
        super(imgs2mesh, self).__init__()
        # basic parameters
        self.model_name = model_cfgs.get('model_name', self.__class__.__name__)
        self.checkpoint_dir = model_cfgs.get('checkpoint_dir', 'results')
        self.category = model_cfgs.get('category','face')
        self.rank = dist.get_rank() if self.distributed else 0
        self.world_size = dist.get_world_size() if self.distributed else 1
        self.device = torch.device('cuda')
        self.use_sampling=model_cfgs.get('use_sampling', False)
        if self.use_sampling:
            v,f,normals = read_obj(model_cfgs.get('template_normal_path',"data/template_normal.obj"))
            if np.mean(normals[:,2])<0:
                normals = -normals
            self.template_normals=np.array(normals)
            self.template_normals=to_cuda(to_tensor(self.template_normals))
        self.template_uvs = np.load(model_cfgs.get('template_uvs_path', "data/uvs.npy"))
        self.template_uvs = to_cuda(to_tensor(self.template_uvs))
        self.template_uvs3d = torch.cat([self.template_uvs, torch.ones(self.template_uvs.shape[0],1).to(self.device)],1)
        self.template_uvs3d-=0.5
        self.template_uvs3d*=2
        self.shape_param = sio.loadmat(model_cfgs.get('shape_param_path','data/Model_Shape.mat'))
        self.lm68idx=to_cuda(to_tensor(self.shape_param['keypoints'][0].astype(np.int32)))
        self.shape_param['sigma']=to_cuda(to_tensor(self.model_shape['sigma']))
        self.shape_param['w']=to_cuda(to_tensor(self.shape_param['w']))
        self.shape_param['mu_shape']=to_cuda(to_tensor(self.shape_param['mu_shape']))
        self.triangles=to_cuda(to_tensor(self.shape_param['tri'])).transpose() - 1
        self.expression_param = sio.loadmat(model_cfgs.get('exp_param_path','data/Model_Expression.mat'))
        self.expression_param['w_exp']=to_cuda(to_tensor(self.expression_param['w_exp']))
        self.other_param=sio.loadmat(model_cfgs.get('other_param_path','data/sigma_exp.mat'))
        self.other_param['sigma_exp']=to_cuda(to_tensor(self.other_param['sigma_exp']))
        #self.lookview=to_cuda(to_tensor(np.array([0,0,1])))
        self.image_size=model_cfgs.get('image_size',256)
        self.texture_size=model_cfgs.get('texture_size',256)
        self.renderer=Pt3dRenderer(self.device, self.texture_size, lookview=to_cuda(to_tensor(np.array([0,0,1]))))
        self.model_shape=build_backbone(model_cfgs.get('model_shape','Shape3dmmEncoder'))
        self.tuplesize=model_cfgs.get('tuplesize',3)
        self.l1loss = torch.nn.L1Loss(reduction='mean')
        self.l1loss_keepdim = torch.nn.L1Loss(reduction='none')

    def forward(self, input, state):
        assert len(input['imgs']) == self.tuplesize
        inchannel=int(input['imgs'][0].shape[0]//self.tuplesize)
        output=dict()
        outpts_list=[]
        outpose_list=[]
        for i in range(self.tuplesize):
            outparam = self.model_shape(input['imgs'][:,i*inchannel:(i+1)*inchannel])
            outres=param2points_bfm(self.shape_param, self.expression_param, self.other_param, outparam)
            outpts,outpose=outres[0],outres[1]
            outpts=torch.clamp(outpts, min=-125000, max=125000)
            outpts_list.append(outpts)
            outpose_list.append(outpose)
        output['outpts_list']=outpts_list
        output['outpose_list']=outpose_list

        if self.use_sampling:
            batchsize=outpts[0].shape[0]
            sampleimgs=torch.zeros(batchsize,self.tuplesize*3,self.texture_size,self.texture_size).to(self.device)
            weightedimg=torch.zeros(batchsize,3,self.texture_size,self.texture_size).to(self.device)
            rendered_coefs=torch.zeros(batchsize,3,self.texture_size,self.texture_size).to(self.device)
            sumedcoefs=torch.zeros(batchsize,3,self.texture_size,self.texture_size).to(self.device)
            template_normals=self.template_normals.repeat(batchsize,1,1)
            triangles=self.triangles.repeat(batchsize,1,1)
            
            gtaux=input['gtaux']
            gtobj=input['gtobj']
            
            uvimg_list=[]
            uvmask_list=[]
            for k in range(self.tuplesize):
                outangles=torch.clamp(outpose_list[k][:,1:4], min=-3.1415, max=3.1415)

                refAngle = gtaux[:,k,149:152]

                s=gtaux[:,k,136]
                R=gtaux[:,k,137:146].reshape(-1,3,3)
                T=gtaux[:,k,146:149]


                #print("k:",k," s:",s[0], " gt s:",gtaux[0,k,136], " angle:",outangles[0], " refangle:",refAngle[0], " T:",T[0], " gt T:",gtaux[0,k,146:149])
                face_project = (s.reshape(-1,1,1)*torch.matmul(R, gtobj.permute(0,2,1))+T.reshape(-1,3,1)*self.image_size).permute(0,2,1)[:,:,:2]
                face_project/=self.image_size
                face_project[:,:,1]=1-face_project[:,:,1]
                
                uvimg,uvmask=self.renderer.sample(self.template_normals, outangles, self.template_uvs3d, face_project)
                uvimg_list.append(uvimg)
                uvmask_list.append(uvmask)
            output['uvimg_list']=uvimg_list
            output['uvmask_list']=uvmask_list
            
        return output

    def cal_loss(self,output, input, state):
        losses=dict()
        if 'sup' in state and 'unsup' not in state:
            outpts_list=output['outpts_list']
            outpose_list=output['outpose_list']
            gtaux=input['gtaux']
            gtobj=input['gtobj']
            ptsloss=0
            for k in range(self.tuplesize):
                ptsloss+=0.1*self.l1loss(outpts_list[k], gtobj)
            losses['ptsloss']=ptsloss
            
            poseloss=0
            lm68loss=0
            for k in range(self.tuplesize):
                s = outpose_list[k][:,0]
                T = outpose_list[k][:,4:7]
                #print('k:',k,' ',gtaux.shape)
                reflm68 = gtaux[:,k,:136].reshape(-1,68,2)
                #print(reflm68.shape)
                refs = gtaux[:,k,136]
                #print(refs.shape)
                #print(gtaux[:,k, :])
                refR = gtaux[:,k,137:146].reshape(gtaux.shape[0],3,3)
                refT = gtaux[:,k,146:149]
                #refAngle = matrix_to_euler_angles(refR, "XYZ")
                refAngle = gtaux[:,k,149:152]
                #print("outpts[0,:2]:",outpts[0,:2].detach().cpu().numpy())
                #print("s:", refs)
                poseloss += 200*self.l1loss(s, refs)+self.l1loss(outpose_list[k][:,1:4], refAngle)+self.l1loss(T[:,:2], refT[:,:2])
                
                outangle=torch.clamp(outpose_list[k][:,1:4], min=-3.1415, max=3.1415)
                R = euler_angles_to_matrix(outangle, "XYZ")
                lm68 = (s.reshape(-1,1,1)*torch.matmul(R,outpts_list[k].permute(0,2,1))+
                            T.reshape(-1,3,1)*self.image_size).permute(0,2,1)[:,self.lm68idx,:2]
                lm68loss += 0.1*self.l1loss(lm68, reflm68)                
            losses['poseloss']=poseloss
            losses['lm68loss']=lm68loss


            if self.use_sampling:
                uvimg_list=output['uvimg_list']
                uvmask_list=output['uvmask_list']
                uvtex=input['uvtex']
                texloss=0
                for k in range(self.tuplesize):
                    uvimg=uvimg_list[k]
                    uvmask=uvmask_list[k]
                    texloss_k = self.l1loss_keepdim(uvimg, uvtex)
                    texloss_k *= uvmask
                    texloss +=10*texloss_k
                losses['texloss']=texloss
        if 'unsup' in state:
            outpts_list=output['outpts_list']
            outpose_list=output['outpose_list']
            pts_consistent_loss=0
            for k in range(self.tuplesize-1):
                pts_consistent_loss+=0.1*self.l1loss(outpts_list[k], outpts_list[k+1])
            losses['pts_consistent_loss']=pts_consistent_loss

            scale_consistent_loss=0
            for k in range(self.tuplesize-1):
                s1 = outpose_list[k][:,0]
                s2 = outpose_list[k+1][:,0]
                scale_consistent_loss+=200*self.l1loss(s1, s2)
            losses['scale_consistent_loss']=scale_consistent_loss

            losses['ptsloss']=ptsloss
            if self.use_sampling:
                uvimg_list=output['uvimg_list']
                uvmask_list=output['uvmask_list']
                tex_consistent_loss=0
                for k in range(self.tuplesize-1):
                    uvimg1=uvimg_list[k]
                    uvmask1=uvmask_list[k]
                    uvimg2=uvimg_list[k+1]
                    uvmask2=uvmask_list[k+1]
                    tex_consistent_loss_k=self.l1loss_keepdim(uvimg1, uvimg2)
                    tex_consistent_loss_k *= uvmask1*uvmask2
                    tex_consistent_loss+=10*tex_consistent_loss_k
                losses['tex_consistent_loss']=tex_consistent_loss

        
        loss, log_vars = self._parse_losses(losses)
        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(input['imgs']))


