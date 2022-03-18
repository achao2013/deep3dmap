from torchvision import transforms
from torch.utils.data import Dataset
import cv2
import random
import numpy as np
import os
import time, datetime
from multiprocessing import Process,Queue,Array
import ctypes
from skimage.transform import estimate_transform, warp, SimilarityTransform, rotate
import scipy.io as sio
import pickle
from deep3dmap.core.all3dtrans.lmk2angle import R2radangle
from .builder import DATASETS
import torch

def dt():
    return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

def numpy_to_share_supervise(index,image,uvtex,gtobj,gtaux,nparrimage,nparruvtex,nparrgtobj,nparrgtaux):
    nparrimage[index,0:image.size] = image.reshape(-1)[0:image.size]
    nparruvtex[index,0:uvtex.size] = uvtex.reshape(-1)[0:uvtex.size]
    nparrgtobj[index,0:gtobj.size] = gtobj.reshape(-1)[0:gtobj.size]
    nparrgtaux[index,0:gtaux.size] = gtaux.reshape(-1)[0:gtaux.size]

def return_batchdata_supervise(result,imagelist,uvtexlist, gtobjlist, gtauxlist, freearr,nparrimage,nparruvtex,nparrgtobj, nparrgtaux):
    index = freearr.get() # wait for free index
    
    images = np.asarray(imagelist, dtype=np.float32)
    uvtexs = np.asarray(uvtexlist, dtype=np.float32)
    gtobjs = np.asarray(gtobjlist, dtype=np.float32)
    gtauxs = np.asarray(gtauxlist, dtype=np.float32)
    numpy_to_share_supervise(index,images,uvtexs,gtobjs,gtauxs,nparrimage,nparruvtex,nparrgtobj,nparrgtaux)
    result.put((index,images.shape,uvtexs.shape,gtobjs.shape, gtauxs.shape))
    del imagelist[:]
    del uvtexlist[:]
    del gtobjlist[:]
    del gtauxlist[:]

def read_obj(objpath):
    v=[]
    with open(objpath) as file:
        for line in file:
            linelist=line.strip().split()
            if len(linelist) < 1:
                continue
            flag=linelist[0]
            if flag == 'v':
                tmp=list(map(float, linelist[1:4]))
                v.append(tmp)
            else:
                continue
    return v



def get_batch_supervise(datainfo, imgpath2auxinfo, fix_pose, maxnum,patchid, uvtex_width, uvtex_height, outsize, tuplesize, tuplenum, auxsize, batchsize,result,freearr,arrimage,arruvtex,arrgtobj, arrgtaux, is_train, seed):
    np.random.seed(seed)
    imagelist = []
    uvtexlist = []
    gtobjlist = []
    gtauxlist = []
    nparrimage = np.frombuffer(arrimage.get_obj(),np.float32).reshape(tuplenum,int(len(arrimage)/tuplenum))
    nparruvtex = np.frombuffer(arruvtex.get_obj(),np.float32).reshape(tuplenum,int(len(arruvtex)/tuplenum))
    nparrgtobj = np.frombuffer(arrgtobj.get_obj(),np.float32).reshape(tuplenum,int(len(arrgtobj)/tuplenum))
    nparrgtaux = np.frombuffer(arrgtaux.get_obj(),np.float32).reshape(tuplenum,int(len(arrgtaux)/tuplenum))
    while True:
        if maxnum==None:
            maxnum = len(datainfo.keys())
        uvtexnames = list(datainfo.keys())
        uvtexname = uvtexnames[np.random.randint(maxnum)]


        inimgs=[]
        gtaux = np.zeros((tuplesize,auxsize))
        pose2imgpaths = datainfo[uvtexname]
        uvtex = cv2.imread(uvtexname)
        if uvtex is None:
            print('uvtex read failed', uvtexname)
            continue
        else:
            uvtex = cv2.resize(uvtex,(uvtex_width, uvtex_height), interpolation=cv2.INTER_CUBIC)
            uvtex = uvtex.astype(float)
            uvtex = uvtex/255.0
            #uvtex = (uvtex-127.5)/127.5
            uvtex=uvtex.transpose(2,0,1)
            uvtex=uvtex.astype(float)
            if fix_pose:
                if tuplesize ==3:
                          
                    if "05_1" not in pose2imgpaths or "08_0" not in pose2imgpaths or "19_0" not in pose2imgpaths:
                        continue
                    idx = np.random.randint(len(pose2imgpaths["05_1"]))
                    inimgs.append(cv2.imread(pose2imgpaths["05_1"][idx]))
                    idx = np.random.randint(len(pose2imgpaths["19_0"]))
                    inimgs.append(cv2.imread(pose2imgpaths["19_0"][idx]))
                    idx = np.random.randint(len(pose2imgpaths["08_0"]))
                    inimgs.append(cv2.imread(pose2imgpaths["08_0"][idx]))

                    auxinfo=imgpath2auxinfo[pose2imgpaths["05_1"][idx]]
                    if isinstance(auxinfo['lm68'], np.ndarray):
                        gtaux[0,0:136] = np.array(auxinfo['lm68']).flatten()
                        gtaux[0,136:137] = np.array(auxinfo['s'])
                        gtaux[0,137:146] = np.array(auxinfo['R']).flatten()
                        gtaux[0,146:149] = np.array(auxinfo['t']).flatten()
                        if auxsize==152:
                            gtaux[0,149:152],flag, newR = R2radangle(np.array(auxinfo['R']))
                            gtaux[0,137:146] = newR.flatten()
                            if not flag:
                                continue
                    else:
                        continue
                    auxinfo=imgpath2auxinfo[pose2imgpaths["19_0"][idx]]
                    if isinstance(auxinfo['lm68'], np.ndarray):
                        gtaux[1,0:136] = np.array(auxinfo['lm68']).flatten()
                        gtaux[1,136:137] = np.array(auxinfo['s'])
                        gtaux[1,137:146] = np.array(auxinfo['R']).flatten()
                        gtaux[1,146:149] = np.array(auxinfo['t']).flatten()
                        if auxsize==152:
                            gtaux[1,149:152],flag, newR = R2radangle(np.array(auxinfo['R']))
                            gtaux[1,137:146] = newR.flatten()
                            if not flag:
                                continue
                    else:
                        continue
                    auxinfo=imgpath2auxinfo[pose2imgpaths["08_0"][idx]]
                    if isinstance(auxinfo['lm68'], np.ndarray):
                        gtaux[2,0:136] = np.array(auxinfo['lm68']).flatten()
                        gtaux[2,136:137] = np.array(auxinfo['s'])
                        gtaux[2,137:146] = np.array(auxinfo['R']).flatten()
                        gtaux[2,146:149] = np.array(auxinfo['t']).flatten()
                        if auxsize==152:
                            gtaux[2,149:152],flag, newR = R2radangle(np.array(auxinfo['R']))
                            gtaux[2,137:146] = newR.flatten()
                            if not flag:
                                continue
                    else:
                        continue



                elif tuplesize ==5:
                    pass
                else:
                    print("tuplesize error")
                    exit(0)
            else:
                front_poses=["05_1","05_0","14_0"]
                left_poses=["24_0","01_0","20_0","19_0","04_1"]
                right_poses=["11_0","12_0","09_0","08_0","13_0"]
                if tuplesize ==3:
                    i=0
                    resample=False
                    front_pose = "05_1"
                    while front_pose not in pose2imgpaths:
                        front_pose = front_poses[np.random.randint(len(left_poses))]
                        i=i+1
                        if i>20:
                            resample=True
                            break
                    if resample:
                        continue
                    inimgs.append(cv2.imread(pose2imgpaths["05_1"]))
                    left_pose = left_poses[np.random.randint(len(left_poses))] 
                    i=0
                    resample=False
                    while left_pose not in pose2imgpaths:
                        left_pose = left_poses[np.random.randint(len(left_poses))]
                        i=i+1
                        if i>20:
                            resample=True
                            break
                    if resample:
                        continue
                    inimgs.append(cv2.imread(pose2imgpaths[left_pose]))
                    right_pose = right_poses[np.random.randint(len(right_poses))] 
                    i=0
                    resample=False
                    while right_pose not in pose2imgpaths:
                        right_pose = left_poses[np.random.randint(len(right_poses))]
                        i=i+1
                        if i>20:
                            resample=True
                            break
                    if resample:
                        continue
                    inimgs.append(cv2.imread(pose2imgpaths[right_pose]))                   
                elif tuplesize ==5:
                    pass
                else:
                    print("tuplesize error")
                    exit(0)

            gtobj = None
            gtobj = read_obj(uvtexname.replace('Completed_UV/good','3dmm_obj').replace('.png','.obj'))
            gtobj = np.array(gtobj)*100000
            if gtobj is None:
                print('obj read failed', uvtexname.replace('Completed_UV/good','3dmm_obj').replace('.png','.obj'))
                continue

            for i in range(len(inimgs)):
                img = inimgs[i].astype(float)

                # RGB jitter
                #img[:, :, 0] *= random.uniform(0.8, 1.2)
                #img[:, :, 1] *= random.uniform(0.8, 1.2)
                #img[:, :, 2] *= random.uniform(0.8, 1.2)
                #if np.random.rand()>0.5:
                #    img = img.clip(0, 255)
                #else:
                #    img = img/np.max(img)*255


                img = img/255.0
                #img = (img-127.5)/127.5
                img=img.astype(float)
                
                #shift,scale
                imgw=img.shape[1]
                imgh=img.shape[0]
                insize = min(imgw,imgh)
                center=np.array([insize/2,insize/2])
                marg = insize*0.02
                t_x = np.random.rand()*marg*2 - marg
                t_y = np.random.rand()*marg*2 - marg
                center[0] = center[0]+t_x; center[1] = center[1]+t_y
                
                #size = (insize-2*max(abs(t_x),abs(t_y))-2)*(np.random.rand()*0.1 + 0.95)
                size = (insize-2*max(abs(t_x),abs(t_y))-2)
                src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
                DST_PTS = np.array([[0,0], [0,outsize - 1], [outsize - 1, 0]])
                tform = estimate_transform('similarity', src_pts, DST_PTS)
                img = warp(img, tform.inverse, output_shape=(outsize, outsize))
                kpt = gtaux[i,:136].reshape(68,2)
                kpt = np.hstack((kpt, np.ones((68,1))))
                kpt = np.dot(tform.params, kpt.T).T
                #saveimg = img[:]*255
                #for k in range(kpt.shape[0]):
                #    cv2.circle(saveimg, (int(kpt[k,0]), int(kpt[k,1])), 1, (0,0,255))
                #cv2.imwrite("test.jpg",saveimg)
                gtaux[i,:136] = kpt[:,:2].flatten()
                gtaux[i,136] *= outsize/insize*0.00001
                gtaux[i,146:148] /= outsize
                gtaux[i,146] += int(np.round(-t_x/insize))
                gtaux[i,147] += int(np.round(-t_y/insize))



                img = img.transpose(2,0,1)
                #img = 2*(img - 0.5)
                inimgs[i]=img
        if uvtex.shape != (3,uvtex_height,uvtex_width):
            print('uvtex shape error', uvtex.shape)
            continue


        if uvtex is not None and gtobj is not None and gtaux is not None and inimgs is not None:
            imgtuple = np.concatenate(tuple(inimgs),axis=0)
            imagelist.append(imgtuple)
            uvtexlist.append(uvtex)
            gtobjlist.append(gtobj)
            gtauxlist.append(gtaux)
        if len(imagelist)==batchsize: 
            return_batchdata_supervise(result,imagelist,uvtexlist,gtobjlist,gtauxlist,freearr,nparrimage,nparruvtex,nparrgtobj,nparrgtaux)

@DATASETS.register_module()
class FaceTexUVAsyncDataset(object):
    def __init__(self,datafile, auxfile, state='sup', image_size=384,fix_pose=True, tuplesize=3, patchid=0,batchsize=1,
                 nthread=4,maxnum=None, is_train=True):
        print((dt()), 'load data info...')
        datainfo = pickle.load(open(datafile,"rb"))
        imgpath2auxinfo = pickle.load(open(auxfile,"rb"))
        print(dt(), 'load finished, %d uvs'%len(datainfo.keys()))
        self.state=state
        self.iter_size = 1+int(len(datainfo.keys())/batchsize)

        
        #tuplesize = 5 #5view
        tuplenum = 4
        #uvtex_width = 512 #595
        #uvtex_height = 512 #377
        uvtex_width = image_size #595
        uvtex_height = image_size #377
        pointsnum=53215
        auxsize=152
        self.arrimage = Array(ctypes.c_float, tuplenum*batchsize*tuplesize*3*image_size*image_size)
        self.arruvtex = Array(ctypes.c_float, tuplenum*batchsize*3*uvtex_width*uvtex_height)
        self.arrgtobj = Array(ctypes.c_float, tuplenum*batchsize*3*pointsnum)
        self.arrgtaux = Array(ctypes.c_float, tuplenum*batchsize*tuplesize*auxsize)
        self.nparrimage = np.frombuffer(self.arrimage.get_obj(),np.float32).reshape(tuplenum,int(len(self.arrimage)/tuplenum))
        self.nparruvtex = np.frombuffer(self.arruvtex.get_obj(),np.float32).reshape(tuplenum,int(len(self.arruvtex)/tuplenum))
        self.nparrgtobj = np.frombuffer(self.arrgtobj.get_obj(),np.float32).reshape(tuplenum,int(len(self.arrgtobj)/tuplenum))
        self.nparrgtaux = np.frombuffer(self.arrgtaux.get_obj(),np.float32).reshape(tuplenum,int(len(self.arrgtaux)/tuplenum))

        self.result   = Queue()
        self.freearr  = Queue()

        for i in range(tuplenum): 
            self.freearr.put(i)

        for i in range(nthread):
            p = Process(target=get_batch_supervise, args=(datainfo, imgpath2auxinfo, fix_pose, maxnum, patchid, uvtex_width,uvtex_height, imagesize, tuplesize,tuplenum,auxsize, batchsize,self.result,
                                                self.freearr,self.arrimage,self.arruvtex,self.arrgtobj, self.arrgtaux, is_train, 123+i))
            p.start()

    def get(self):
        index, imageshape,uvtexshape,gtobjshape,gtauxshape = self.result.get()
        imagesize = np.prod(imageshape)
        uvtexsize = np.prod(uvtexshape)
        gtobjsize = np.prod(gtobjshape)
        gtauxsize = np.prod(gtauxshape)
        image = np.empty(imageshape,np.float32)
        uvtex = np.empty(uvtexshape,np.float32)
        gtobj = np.empty(gtobjshape,np.float32)
        gtaux = np.empty(gtauxshape,np.float32)
        image.reshape(imagesize)[:] = self.nparrimage[index,0:imagesize]
        uvtex.reshape(uvtexsize)[:] = self.nparruvtex[index,0:uvtexsize]
        gtobj.reshape(gtobjsize)[:] = self.nparrgtobj[index,0:gtobjsize]
        gtaux.reshape(gtauxsize)[:] = self.nparrgtaux[index,0:gtauxsize]
        self.freearr.put(index)

        image=torch.from_numpy(image)
        uvtex=torch.from_numpy(uvtex)
        gtobj=torch.from_numpy(gtobj)
        gtaux=torch.from_numpy(gtaux)
        input={'imgs':image,'uvtex':uvtex,'gtobj':gtobj,'gtaux':gtaux}
        return input



##############################unsupervise
def numpy_to_share_unsupervise(index,image,gtaux,nparrimage,nparrgtaux):
    nparrimage[index,0:image.size] = image.reshape(-1)[0:image.size]
    nparrgtaux[index,0:gtaux.size] = gtaux.reshape(-1)[0:gtaux.size]

def return_batchdata_unsupervise(result,imagelist,gtauxlist, freearr,nparrimage, nparrgtaux):
    index = freearr.get() # wait for free index
    
    images = np.asarray(imagelist, dtype=np.float32)
    gtauxs = np.asarray(gtauxlist, dtype=np.float32)
    numpy_to_share_unsupervise(index,images,gtauxs,nparrimage,nparrgtaux)
    result.put((index,images.shape,gtauxs.shape))
    del imagelist[:]
    del gtauxlist[:]


def get_batch_unsupervise(datainfo, imgpath2auxinfo, fix_pose, maxnum,patchid, uvtex_width, uvtex_height, outsize, tuplesize, tuplenum, batchsize,result,freearr,arrimage, arrgtaux, is_train, seed):
    np.random.seed(seed)
    imagelist = []
    gtauxlist = []
    nparrimage = np.frombuffer(arrimage.get_obj(),np.float32).reshape(tuplenum,int(len(arrimage)/tuplenum))
    nparrgtaux = np.frombuffer(arrgtaux.get_obj(),np.float32).reshape(tuplenum,int(len(arrgtaux)/tuplenum))
    while True:
        if maxnum==None:
            maxnum = len(datainfo.keys())
        id = list(datainfo.keys())[np.random.randint(maxnum)]
        illum = list(datainfo[id].keys())[np.random.randint(len(datainfo[id].keys()))]
        express = datainfo[id][illum].keys()[np.random.randint(len(datainfo[id][illum].keys()))]


        inimgs=[]
        gtaux = np.zeros((tuplesize,152))
        pose2imgpaths = datainfo[id][illum][express]
        if fix_pose:
            if tuplesize ==3:
                      
                if "05_1" not in pose2imgpaths or "08_0" not in pose2imgpaths or "19_0" not in pose2imgpaths:
                    continue
                idx = np.random.randint(len(pose2imgpaths["05_1"]))
                inimgs.append(cv2.imread(pose2imgpaths["05_1"][idx]))
                idx = np.random.randint(len(pose2imgpaths["19_0"]))
                inimgs.append(cv2.imread(pose2imgpaths["19_0"][idx]))
                idx = np.random.randint(len(pose2imgpaths["08_0"]))
                inimgs.append(cv2.imread(pose2imgpaths["08_0"][idx]))

                auxinfo=imgpath2auxinfo[pose2imgpaths["05_1"][idx]]
                if isinstance(auxinfo['lm68'], np.ndarray):
                    gtaux[0,0:136] = np.array(auxinfo['lm68']).flatten()
                    gtaux[0,136:137] = np.array(auxinfo['s'])
                    gtaux[0,137:146] = np.array(auxinfo['R']).flatten()
                    gtaux[0,146:149] = np.array(auxinfo['t']).flatten()
                else:
                    continue
                auxinfo=imgpath2auxinfo[pose2imgpaths["19_0"][idx]]
                if isinstance(auxinfo['lm68'], np.ndarray):
                    gtaux[1,0:136] = np.array(auxinfo['lm68']).flatten()
                    gtaux[1,136:137] = np.array(auxinfo['s'])
                    gtaux[1,137:146] = np.array(auxinfo['R']).flatten()
                    gtaux[1,146:149] = np.array(auxinfo['t']).flatten()
                else:
                    continue
                auxinfo=imgpath2auxinfo[pose2imgpaths["08_0"][idx]]
                if isinstance(auxinfo['lm68'], np.ndarray):
                    gtaux[2,0:136] = np.array(auxinfo['lm68']).flatten()
                    gtaux[2,136:137] = np.array(auxinfo['s'])
                    gtaux[2,137:146] = np.array(auxinfo['R']).flatten()
                    gtaux[2,146:149] = np.array(auxinfo['t']).flatten()
                else:
                    continue



            elif tuplesize ==5:
                pass
            else:
                print("tuplesize error")
                exit(0)
        else:
            front_poses=["05_1","05_0","14_0"]
            left_poses=["24_0","01_0","20_0","19_0","04_1"]
            right_poses=["11_0","12_0","09_0","08_0","13_0"]
            if tuplesize ==3:
                i=0
                resample=False
                front_pose = "05_1"
                while front_pose not in pose2imgpaths:
                    front_pose = front_poses[np.random.randint(len(left_poses))]
                    i=i+1
                    if i>20:
                        resample=True
                        break
                if resample:
                    continue
                inimgs.append(cv2.imread(pose2imgpaths["05_1"]))
                left_pose = left_poses[np.random.randint(len(left_poses))] 
                i=0
                resample=False
                while left_pose not in pose2imgpaths:
                    left_pose = left_poses[np.random.randint(len(left_poses))]
                    i=i+1
                    if i>20:
                        resample=True
                        break
                if resample:
                    continue
                inimgs.append(cv2.imread(pose2imgpaths[left_pose]))
                right_pose = right_poses[np.random.randint(len(right_poses))] 
                i=0
                resample=False
                while right_pose not in pose2imgpaths:
                    right_pose = left_poses[np.random.randint(len(right_poses))]
                    i=i+1
                    if i>20:
                        resample=True
                        break
                if resample:
                    continue
                inimgs.append(cv2.imread(pose2imgpaths[right_pose]))                   
            elif tuplesize ==5:
                pass
            else:
                print("tuplesize error")
                exit(0)


        for i in range(len(inimgs)):
            img = inimgs[i].astype(float)

            # RGB jitter
            #img[:, :, 0] *= random.uniform(0.8, 1.2)
            #img[:, :, 1] *= random.uniform(0.8, 1.2)
            #img[:, :, 2] *= random.uniform(0.8, 1.2)
            #if np.random.rand()>0.5:
            #    img = img.clip(0, 255)
            #else:
            #    img = img/np.max(img)*255


            img = img/255.0
            #img = (img-127.5)/127.5
            img=img.astype(float)
            
            #shift,scale
            imgw=img.shape[1]
            imgh=img.shape[0]
            insize = min(imgw,imgh)
            center=np.array([insize/2,insize/2])
            marg = insize*0.02
            t_x = np.random.rand()*marg*2 - marg
            t_y = np.random.rand()*marg*2 - marg
            center[0] = center[0]+t_x; center[1] = center[1]+t_y
            
            #size = (insize-2*max(abs(t_x),abs(t_y))-2)*(np.random.rand()*0.1 + 0.95)
            size = (insize-2*max(abs(t_x),abs(t_y))-2)
            src_pts = np.array([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
            DST_PTS = np.array([[0,0], [0,outsize - 1], [outsize - 1, 0]])
            tform = estimate_transform('similarity', src_pts, DST_PTS)
            img = warp(img, tform.inverse, output_shape=(outsize, outsize))
            kpt = gtaux[i,:136].reshape(68,2)
            kpt = np.hstack((kpt, np.ones((68,1))))
            kpt = np.dot(tform.params, kpt.T).T
            gtaux[i,:136] = kpt[:,:2].flatten()
            gtaux[i,136] *= outsize/insize*0.00001
            gtaux[i,146:148] /= outsize
            gtaux[i,146] += int(np.round(-t_x/insize))
            gtaux[i,147] += int(np.round(-t_y/insize))



            img = img.transpose(2,0,1)
            #img = 2*(img - 0.5)
            inimgs[i]=img


        if gtaux is not None and inimgs is not None:
            imgtuple = np.concatenate(tuple(inimgs),axis=0)
            imagelist.append(imgtuple)
            gtauxlist.append(gtaux)
        if len(imagelist)==batchsize: 
            return_batchdata_supervise(result,imagelist,gtauxlist,freearr,nparrimage,nparrgtaux)

@DATASETS.register_module()
class FaceImagesAsyncDataset(object):
    def __init__(self,datafile, auxfile, state='unsup', image_size=384, fix_pose=True, tuplesize=3, patchid=0,batchsize=1,
                 nthread=12,maxnum=None, is_train=True):
        print((dt()), 'load data info...')
        datainfo = pickle.load(open(datafile,"rb"))
        imgpath2auxinfo = pickle.load(open(auxfile,"rb"))
        print(dt(), 'load finished, %d ids'%len(datainfo.keys()))
        self.state=state
        self.iter_size = 1+int(len(datainfo.keys())/batchsize)

        
        #tuplesize = 5 #5view
        tuplenum = 4
        uvtex_width = image_size #595
        uvtex_height = image_size #377
        pointsnum=53215
        auxsize=149
        self.arrimage = Array(ctypes.c_float, tuplenum*batchsize*tuplesize*3*image_size*image_size)
        self.arrgtaux = Array(ctypes.c_float, tuplenum*batchsize*tuplesize*auxsize)
        self.nparrimage = np.frombuffer(self.arrimage.get_obj(),np.float32).reshape(tuplenum,int(len(self.arrimage)/tuplenum))
        self.nparrgtaux = np.frombuffer(self.arrgtaux.get_obj(),np.float32).reshape(tuplenum,int(len(self.arrgtaux)/tuplenum))

        self.result   = Queue()
        self.freearr  = Queue()

        for i in range(tuplenum): 
            self.freearr.put(i)

        for i in range(nthread):
            p = Process(target=get_batch_unsupervise, args=(datainfo, imgpath2auxinfo, fix_pose, maxnum, patchid, uvtex_width,uvtex_height, imagesize, tuplesize,tuplenum, batchsize,self.result,
                                                self.freearr,self.arrimage,self.arrgtaux, is_train, 123+i))
            p.start()

    def get(self):
        index, imageshape,gtauxshape = self.result.get()
        imagesize = np.prod(imageshape)
        gtauxsize = np.prod(gtauxshape)
        image = np.empty(imageshape,np.float32)
        gtaux = np.empty(gtauxshape,np.float32)
        image.reshape(imagesize)[:] = self.nparrimage[index,0:imagesize]
        gtaux.reshape(gtauxsize)[:] = self.nparrgtaux[index,0:gtauxsize]
        self.freearr.put(index)

        image=torch.from_numpy(image)
        gtaux=torch.from_numpy(gtaux)
        input={'imgs':image,'gtaux':gtaux}
        return input
        



class FaceTexUVDataset(Dataset):

    #  assume images  as B x C x H x W  numpy array
    def __init__(self, txt_path, use_sigmoid=True,  transform=None):
        self.IDs = []
        self.imgpaths = []
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            random.shuffle(lines)
            for line in lines:
                imgpath = line.strip()
                if os.path.exists(imgpath) and os.path.exists(imgpath.replace('_inp.jpg','.npy')):
                    self.imgpaths.append(imgpath)
        print(len(self.imgpaths))
        self.transform = transform
        self.use_sigmoid=use_sigmoid

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, idx):

        image = cv2.imread(self.imgpaths[idx])
        if image is None:
            print("read ",self.imgpaths[idx], " error")
        else:
            image=image.astype(float)
            gt=np.load(self.imgpaths[idx].replace('_inp.jpg','.npy'))
            if self.use_sigmoid:
                gt=gt.transpose(2,0,1)/256.0
            else:
                gt=gt.transpose(2,0,1)
            #gt=gt.transpose(2,0,1)/np.max(gt)
            #gt[np.where(gt<0)]=0
            gt=gt.astype(float)
        image = image.transpose(2,0,1)
        image = (image-127.5)/127.5
        if self.transform is not None:
            image = self.transform(image)
            gt = self.transform(gt)
        return (image, gt, self.imgpaths[idx])

class FaceTexUVNewDataset(Dataset):

    #  assume images  as B x C x H x W  numpy array
    def __init__(self, txt_path, use_sigmoid=True,  transform=None):
        self.IDs = []
        self.imgpaths = []
        with open(txt_path) as input_file:
            lines = input_file.readlines()
            random.shuffle(lines)
            for line in lines:
                imgpath = line.strip()
                if os.path.exists(imgpath) and os.path.exists(imgpath.replace('.jpg','.mat')):
                    self.imgpaths.append(imgpath)
        print(len(self.imgpaths))
        self.transform = transform
        self.use_sigmoid=use_sigmoid

    def __len__(self):
        return len(self.imgpaths)

    def __getitem__(self, idx):

        image = cv2.imread(self.imgpaths[idx])
        if image is None:
            print("read ",self.imgpaths[idx], " error")
        else:
            image=image.astype(float)
            gt = sio.loadmat(self.imgpaths[idx].replace('jpg','mat'))
            kpt = gt['pt3d_68']
            kpt = kpt.astype(float)
        if image.ndim < 3:
            image = np.tile(image[:,:,np.newaxis], [1,1,3])
                
        if kpt is not None:
            if np.max(kpt.shape) > 4: # key points to get bounding box                                         
                kpt = kpt
                if kpt.shape[0] > 3:
                    kpt = kpt.T
                left = np.min(kpt[0, :]); right = np.max(kpt[0, :]);                                                  
                top = np.min(kpt[1,:]); bottom = np.max(kpt[1,:])
            else:  # bounding box
                bbox = kpt
                left = bbox[0]; right = bbox[1]; top = bbox[2]; bottom = bbox[3]                                      
            old_size = (right - left + bottom - top)/2                                                                
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])                          
            #size = int(old_size*1.6)
            size = int(old_size*1.5)
 
        resolution_inp=256
        src_pts = np.float32([[center[0]-size/2, center[1]-size/2], [center[0] - size/2, center[1]+size/2], [center[0]+size/2, center[1]-size/2]])
        dst_pts = np.float32([[0,0], [0, resolution_inp - 1], [resolution_inp - 1, 0]])                        
        tform = estimate_transform('similarity', src_pts, dst_pts)
        
        image = image[:,:,[2,1,0]]/255.0
        cropped_image = warp(image, tform.inverse, output_shape=(resolution_inp, resolution_inp))
        cropped_image = cropped_image[:,:,[2,1,0]]
        cropped_image = cropped_image.transpose(2,0,1)
        if self.transform is not None:
            cropped_image = self.transform(cropped_image)
            gt = self.transform(gt)
        if self.use_sigmoid:
            kpt = kpt/256.0
        return (cropped_image, kpt[:2,:], tform.params)



class cvresize(object):

    #  assume image  as C x H x W  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, (tuple))
        self.output_size = output_size

    def __call__(self, image):
        resized_image = cv2.resize(image,self.output_size)

        return resized_image

class RandCrop(object):

    #  assume image  as H x W x C  numpy array

    def __init__(self, output_size):
        assert isinstance(output_size, tuple)
        assert len(output_size) == 2
        self.output_size = output_size

    def __call__(self, image):
        h, w = image.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        cropped_image = image[top:top+new_h, left:left+new_w]

        return cropped_image, [left,top,left+new_w,top+new_h]
