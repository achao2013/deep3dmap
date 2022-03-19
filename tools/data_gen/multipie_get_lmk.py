#!/usr/bin/env python
# -*- coding: utf-8 -*-

import caffe
from pnpmodules.face-alignment import face_alignment
import cv2
import numpy as np
#from python_wrapper import *
import os
import math


def drawBoxes(im, boxes):
    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]
    for i in range(x1.shape[0]):
        cv2.rectangle(im, (int(x1[i]), int(y1[i])), (int(x2[i]), int(y2[i])), (0,255,0), 1)
    return im

from time import time
_tstart_stack = []
def tic():
    _tstart_stack.append(time())
def toc(fmt="Elapsed: %s s"):
    print(fmt % (time()-_tstart_stack.pop()))



def crop_rect(img, rect, enlarge_rato=1.5, input_size=(128, 128)):
    (x, y, w, h) = rect
    # coordinates of center point
    # coordinates of center point
    mins_ = [x, y]
    maxs_ = [x + w, y + w]
    # c = np.array([x  + w / 2.0, y + h / 2.0])  # center
    c = np.array([maxs_[0] - (maxs_[0] - mins_[0]) / 2, maxs_[1] - (maxs_[1] - mins_[1]) / 2])  # center
    max_wh = max((maxs_[0] - mins_[0]) / 2, (maxs_[1] - mins_[1]) / 2)
    scale = enlarge_rato
    rot = 0

    M = cv2.getRotationMatrix2D((c[0], c[1]), rot, input_size[0] / (2 * max_wh * scale))
    M[0, 2] = M[0, 2] - (c[0] - input_size[0] / 2.0)
    M[1, 2] = M[1, 2] - (c[1] - input_size[0] / 2.0)

    img = cv2.warpAffine(img, M, input_size)

    return img, M, get_inv_M(M)

def color_normalize(x, mean):
    if x.shape[-1] == 1:
        x = np.repeat(x, axis=2)
    h, w, c = x.shape
    x = np.transpose(x, (2, 0, 1))
    x = np.subtract(x.reshape(c, -1), mean[:, np.newaxis]).reshape(-1, h, w)
    x = np.transpose(x, (1, 2, 0))

    return x

def img_to_caffe(img, mean = np.array([0.5, 0.5, 0.5])):
    img_ndrry = img.astype(np.float32)
    if img_ndrry.max() > 1:
        img_ndrry /= 255
    img_ndrry = color_normalize(img_ndrry, mean)
    img_ndrry = np.transpose(img_ndrry, (2, 0, 1))

    return img_ndrry

def get_inv_M(M):
    inv_M = np.empty_like(M)
    d = M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0]
    if d != 0:
        d = 1/d
    else:
        d = 0
    inv_M[0, 0] = M[1, 1]* d
    inv_M[0, 1] = M[0, 1]* (-d)
    inv_M[1, 0] = M[1, 0]* (-d)
    inv_M[1, 1] = M[0, 0]* d

    inv_M[0, 2] = -inv_M[0, 0]* M[0, 2] - inv_M[0, 1]* M[1, 2];
    inv_M[1, 2] = -inv_M[1, 0]* M[0, 2] - inv_M[1, 1]* M[1, 2];
    return inv_M

def transform_lan(pts, M):
    M_inv = get_inv_M(M)
    pseudo_pts = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    pseudo_pts = np.matrix(pseudo_pts) * np.matrix(M_inv.transpose())
    return pseudo_pts

def crop_landmarks(img, pts, sh = 0, enlarge_rato = 1.5, rot = 0, input_size=(128, 128)):

    assert pts.shape[0] == 72 or pts.shape[0] == 68 
    # coordinates of both eyes
    if pts.shape[0] == 72:
        idx1 = 13
        idx2 = 34
    else:
        idx1 = 36
        idx2 = 45

    # angle between the eyes
    alpha = 0
    if pts[idx2, 0] != -1 and pts[idx2, 1] != -1 and pts[idx1, 0] != -1 and pts[idx1, 1] != -1:
        alpha = math.atan2(pts[idx2, 1] - pts[idx1, 1], pts[idx2, 0] - pts[idx1, 0]) * 180 / math.pi

    pts[pts == -1] = np.inf
    mins_ = np.min(pts, 0).tolist()[0]  # min vals
    pts[pts == np.inf] = -1
    maxs_ = np.max(pts, 0).tolist()[0]  # max vals
    
    print(mins_, maxs_)

    # coordinates of center point
    c = np.array([maxs_[0] - (maxs_[0] - mins_[0]) / 2, maxs_[1] - (maxs_[1] - mins_[1]) / 2])  # center
    max_wh = max((maxs_[0] - mins_[0]) / 2, (maxs_[1] - mins_[1]) / 2)

    # Shift the center point, rot add eyes angle
    c += sh * max_wh
    rot += alpha

    M = cv2.getRotationMatrix2D((c[0], c[1]), rot, input_size[0] / (2 * max_wh * enlarge_rato))
    M[0, 2] = M[0, 2] - (c[0] - input_size[0] / 2.0)
    M[1, 2] = M[1, 2] - (c[1] - input_size[0] / 2.0)

    face_img = cv2.warpAffine(img, M, input_size, flags=cv2.INTER_CUBIC)

    return face_img, M, get_inv_M(M)

def guard(x, N):
    x=np.array(x)
    x[np.where(x<0)]=0
    print('x[x<0]:',x[x<0])
    x[np.where(x>N)]=N
    r = x.tolist()
    return r
def transform(x, y, ang, s0, s1):
    # x,y position
    # ang angle
    # s0 size of original image
    # s1 size of target image
    
    x0 = x - s0[1]/2
    y0 = y - s0[0]/2
    xx = x0*math.cos(ang) - y0*math.sin(ang) + s1[1]/2
    yy = x0*math.sin(ang) + y0*math.cos(ang) + s1[0]/2
    return [xx,yy]
def casia_align(img, f5pt, crop_size, ec_mc_y, ec_y):
    ang_tan = (f5pt[0][1]-f5pt[1][1])/(f5pt[0][0]-f5pt[1][0])
    ang = math.atan(ang_tan) / math.pi * 180
    h = img.shape[0]
    w = img.shape[1]
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, 1)
    img_rot = cv2.warpAffine(img, M, (w, h))
    
    # eye center
    x = (f5pt[0][0]+f5pt[1][0])/2
    y = (f5pt[0][1]+f5pt[1][1])/2
    
    ang = -ang/180*math.pi
    [xx, yy] = transform(x, y, ang, img.shape, img_rot.shape)
    eyec = np.round([xx,yy])
    x = (f5pt[3][0]+f5pt[4][0])/2
    y = (f5pt[3][1]+f5pt[4][1])/2
    [xx, yy] = transform(x, y, ang, img.shape, img_rot.shape)
    mouthc = np.round([xx,yy])
     
    resize_scale = ec_mc_y/(mouthc[1]-eyec[1])
    if resize_scale<=0:
        return [img, eyec, img_rot, resize_scale]
    try:
        img_resize = cv2.resize(img_rot, (int(w*resize_scale),int(h*resize_scale)))
    except:
        return [None, eyec, img_rot, -1]  
    res = img_resize
    eyec2 = (eyec - [img_rot.shape[1]/2,img_rot.shape[0]/2]) * resize_scale + [img_resize.shape[1]/2,img_resize.shape[0]/2]
    eyec2 = np.round(eyec2)
    img_crop = np.zeros((crop_size, crop_size, img_rot.shape[2]))
    crop_y = int(eyec2[1] - ec_y)
    crop_y_end = int(crop_y + crop_size - 1)
    crop_x = int(eyec2[0]-int(np.floor(crop_size/2)))
    crop_x_end = int(crop_x + crop_size - 1)
    
    #print [crop_x,crop_x_end,crop_y,crop_y_end]
    #print 'img_resize.shape:',img_resize.shape
    box_x = guard(map(int,[crop_x,crop_x_end]), img_resize.shape[1]-1)
    box_y = guard(map(int,[crop_y,crop_y_end]), img_resize.shape[0]-1)
    box = box_x + box_y
    #print 'box:',box
    img_crop[box[2]-crop_y+1:box[3]-crop_y+1, box[0]-crop_x+1:box[1]-crop_x+1] = img_resize[box[2]:box[3],box[0]:box[1]]
    
    return [res, eyec2, img_crop, resize_scale]

def bbox2lmk(src_dir,dst_dir):


    minsize = 20
    
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    

    #error = []
    f_bbox = open('data/multipie/multipie_detect_result_img.txt', 'r')
    bboxlines = f_bbox.readlines()
    path2bbox = {}
    f_lm = open('data/multipie/multipie_pyramidbox_face_400x400_casia_align.txt', 'w')
    lm_err = open('data/multipie/multipie_lm_v2_error.txt', 'w')
    align_err = open('data/multipie/multipie_casia_align_v2_error.txt', 'w')
    for i in range(len(bboxlines)/3):
        linepath = bboxlines[3*i]
        linebbox = bboxlines[3*i+2]
        path2bbox[linepath.strip()] = [float(box) for box in linebbox.strip().split()]

    for imgpath in path2bbox.keys():
        
        #print("######\n", src_dir+imgpath)
        if os.path.exists(dst_dir+imgpath):
            continue
        img = cv2.imread(src_dir+imgpath)
        if img is None:
            continue
        camstr=imgpath.split('/')[-1].split('_')[-2]
        if camstr=='081' or camstr=='191':
            M180 = cv2.getRotationMatrix2D((img.shape[1]/2,img.shape[0]/2), 180, 1)
            img = cv2.warpAffine(img, M180, (img.shape[1], img.shape[0]))
        img_matlab = img.copy()
        tmp = img_matlab[:,:,2].copy()
        img_matlab[:,:,2] = img_matlab[:,:,0]
        img_matlab[:,:,0] = tmp


        [xmin, ymin, xmax, ymax, score]= path2bbox[imgpath] 


        rect = [xmin, ymin,xmax-xmin,ymax-ymin]
        img_crop, M, M_inv = crop_rect(img, rect, enlarge_rato=1.7, input_size=(256, 256))

        pts = fa.get_landmarks(img_crop)
        

        #check stage 2 img_crop and landmark
        #ptslist = pts.flatten().tolist()
        #for i in range(len(ptslist)/2):
        #    cv2.circle(img_crop2, (int(ptslist[2*i]),int(ptslist[2*i+1])), 2, (255,0,0))
        #cv2.imwrite(dst_dir+imgpath,img_crop2)

        if not pts:
            lm_err.write('{:s}'.format(imgpath))
            lm_err.write(' error')
            lm_err.write('\n')
            continue
        landmark = transform_lan(pts.reshape(68,2), M)


        img_crop_, M_, M_inv_ = crop_landmarks(img, landmark, enlarge_rato = 1.5, input_size=(256, 256))
        pts_ = fa.get_landmarks(img_crop_)
        landmark_ = transform_lan(pts_.reshape(68,2), M_)
         
            
        #align

        f5pt = [(landmark_[36]+landmark_[39])/2,(landmark_[38]+landmark_[45])/2,landmark_[33],landmark_[48],landmark_[54]]
        print(f5pt)
        if camstr=='110':
            f5pt[0][0]=f5pt[1][0]+0.00000001 
            f5pt[0][1]=f5pt[1][1] 
            f5pt[3]=f5pt[4] 
        elif camstr=='240':
            f5pt[1][0]=f5pt[0][0]+0.00000001 
            f5pt[1][1]=f5pt[0][1] 
            f5pt[4]=f5pt[3] 
        elif camstr=='120':
            f5pt[0]=list((np.array(f5pt[0])+np.array(f5pt[1]))/2) 
            f5pt[3]=list((np.array(f5pt[3])+np.array(f5pt[4]))/2) 
        elif camstr=='010':
            f5pt[1]=list((np.array(f5pt[0])+np.array(f5pt[1]))/2) 
            f5pt[4]=list((np.array(f5pt[3])+np.array(f5pt[4]))/2) 
        [res, eyec2, cropped, resize_scale] = casia_align(img, f5pt, 400, 96, 160)
        if resize_scale <= 0:
           align_err.write('{:s}\n'.format(imgpath))
           continue


        #record
        f_lm.write('{:s}'.format(imgpath))
        lmlist = landmark_.flatten().tolist()
        for p in lmlist[0]:
            f_lm.write(' {:s}'.format(str(p)))
        f_lm.write('\n')
        
        
        #print boundingboxes.shape
        #print("*****\n", dst_dir+imgpath)
        if not os.path.exists(os.path.dirname(dst_dir+imgpath)):
            os.makedirs(os.path.dirname(dst_dir+imgpath))
        #cv2.circle(cropped, (int(eyec2[0]),int(eyec2[1])), 4, (255,0,0))
        cv2.imwrite(dst_dir+imgpath, cropped)
        #cv2.imwrite(dst_dir+imgpath,img[int(ymin):int(ymax),int(xmin):int(xmax)])
        
        #draw for check
        #cv2.rectangle(img,(int(xmin),int(ymin)),(int(xmax),int(ymax)),(0,255,0))
        #for i in range(len(lmlist[0])/2):
        #    cv2.circle(img, (int(lmlist[0][2*i]),int(lmlist[0][2*i+1])), 2, (255,0,0))
        #cv2.imwrite(dst_dir+imgpath,img)




        #if boundingboxes.shape[0] > 0:
        #    error.append[imgpath]
    #print(error)
    f_bbox.close()
    f_lm.close()
    lm_err.close()
    align_err.close()

if __name__ == "__main__":
    src_dir = "/home/data/multi-pie/"
    dst_dir = "/home/data/multi-pie-pyramidbox-casia-align-face-400x400/"
    bbox2lmk(src_dir,dst_dir)
