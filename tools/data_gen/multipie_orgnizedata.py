import pickle
import torch
import numpy as np
from deep3dmap.core.all3dtrans.lmk2angle import draw_landmark,project_param
from PIL import Image
import scipy.io as sio
from pnpmodules.face-alignment import face_alignment

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

def get_expression(session, recordid):
    expression = 0
    if session == "01":
       if recordid == "01":
           expression = 0
       elif recordid == "02":
           expression = 1
    elif session == "02":
        if recordid == "01":
            expression = 0
        elif recordid == "02":
            expression = 2
        elif recordid == "03":
            expression = 3
    elif session == "03":
        if recordid == "01":
            expression = 0
        elif recordid == "02":
            expression = 1
        elif recordid == "03":
            expression = 4
    elif session == "04":
        if recordid == "01":
            expression = 0
        elif recordid == "02":
            expression = 0
        elif recordid == "03":
            expression = 5

    return expression

def split_data():

    infile = open("data/multipie/multipie_pyramidbox_face_400x400_casia_align.txt")
    outtrain = open("data/multipie/multipie_pyramidbox_face_400x400_casia_align_set1_train_label.txt", 'w')
    outtest = open("data/multipie/multipie_pyramidbox_face_400x400_casia_align_set1_test_label.txt", 'w')
    outtrain1 = open("data/multipie/multipie_pyramidbox_face_400x400_casia_align_set1_train_label_60_noexp.txt", 'w')
    outtest1 = open("data/multipie/multipie_pyramidbox_face_400x400_casia_align_set1_test_label_60_noexp.txt", 'w')
    lines = infile.readlines()
    cam2pose = {'11_0':0,'12_0':1,'09_0':2,'08_0':3,'13_0':4,'14_0':5,'05_1':6,'05_0':7, '04_1':8,'19_0':9,'20_0':10,'01_0':11,'24_0':12, '08_1':13, '19_1':14}

    for line in lines:
        line=line.split(' ')[0]
        linelist = line.strip().split('/')
        imgpath=line.strip()
        session = imgpath.split('/')[-1].split('_')[1]
        recordid = imgpath.split('/')[-1].split('_')[2]
        expression = get_expression(session, recordid)
        idstr = linelist[-4]
        if len(idstr) != 3:
            print(line,"id error")
        id = int(idstr[0])*100+int(idstr[1])*10+int(idstr[2])-1
        pose_str = linelist[-2]
        pose = cam2pose[pose_str]
        if id<=199 and pose<=12:
            outtrain.write(line.strip()+' '+str(id)+' '+str(pose)+'\n')
        elif id >199 and pose<=12:
            outtest.write(line.strip() +' '+str(id)+' '+str(pose)+'\n')
        if id<=199 and pose<=10 and pose>=2 and expression==0:
            outtrain1.write(line.strip()+' '+str(id)+' '+str(pose)+'\n')
        elif id >199 and pose<=10 and pose>=2 and expression==0:
            outtest1.write(line.strip() +' '+str(id)+' '+str(pose)+'\n')
    infile.close()
    outtrain.close()
    outtest.close()
    outtrain1.close()
    outtest1.close()

def package_data():
    traintxt="data/multipie/multipie_pyramidbox_face_400x400_casia_align_set1_train_label.txt"
    idillumexp2poseimgpaths = {}
    pose2cam = ['11_0','12_0','09_0','08_0','13_0','14_0','05_1','05_0', '04_1','19_0','20_0','01_0','24_0']
    with open(traintxt) as infile:
        lines = infile.readlines()
        for line in lines:
            linelist = line.strip().split()
            imgpath = linelist[0]
            session = imgpath.split('/')[-1].split('_')[1]
            recordid = imgpath.split('/')[-1].split('_')[2]
            expression = get_expression(session, recordid)
            illum = imgpath.split('/')[-1].split('_')[-1].split('.')[0]
            id = int(linelist[1])
            pose = pose2cam[int(linelist[2])]
            #if illum=="10" and id==55:
            #    print("pose:",pose, " expression:",expression)
            if id in idillumexp2poseimgpaths:
                if illum in  idillumexp2poseimgpaths[id]:
                    if expression in idillumexp2poseimgpaths[id][illum]:
                        if pose in idillumexp2poseimgpaths[id][illum][expression]:
                            idillumexp2poseimgpaths[id][illum][expression][pose].append(imgpath)
                        else:
                            idillumexp2poseimgpaths[id][illum][expression][pose]=[imgpath]
                    else:
                        idillumexp2poseimgpaths[id][illum][expression]={pose:[imgpath]}
                else:
                    idillumexp2poseimgpaths[id][illum]={expression:{pose:[imgpath]}}
            else:
                idillumexp2poseimgpaths[id]={illum:{expression:{pose:[imgpath]}}}
            #if illum=="10" and id==55:
            #    print(idillumexp2poseimgpaths[55]["10"])
    pickle.dump(idillumexp2poseimgpaths, open("data/multipie/multipie_idillumexp2poseimgpaths.pkl","wb"))
    print('ids:',idillumexp2poseimgpaths.keys())
    train_uvtxt="data/multipie/multipie_3dmm_uvtex.txt"
    uvtex2poseimgs={}
    with open(train_uvtxt, "r") as f:
        for line in f:
            filename = line.strip()
            id = int(filename.split('/')[-1].split('_')[0])-1
            session = filename.split('/')[-1].split('_')[1]
            recordid = filename.split('/')[-1].split('_')[2].split('.')[0]
            expression = get_expression(session, recordid)
            #print(filename,id,session,recordid,expression)
            if (id not in idillumexp2poseimgpaths) or ("10" not in idillumexp2poseimgpaths[id]) or (expression not in idillumexp2poseimgpaths[id]["10"]):
                continue
            uvtex2poseimgs[filename] = idillumexp2poseimgpaths[id]["10"][expression]
            #if id==55:
            #    print("id:",id," pose:",idillumexp2poseimgpaths[id]["10"][expression].keys(), " expression:",expression)
    pickle.dump(uvtex2poseimgs, open("data/multipie/multipie_uvtex2poseimgs.pkl","wb"))
    print("uvtex:",len(uvtex2poseimgs.keys()))

    device = torch.device("cuda:0")
    torch.cuda.set_device(device)

    objtxt="data/multipie/multipie_3dmm_gtobj.txt"
    model_shape = sio.loadmat('../Model_Shape.mat')
    name2objpath={}
    id2objpath={}
    with open(objtxt) as f:
        for line in f:
            filename = line.strip().split('/')[-1]
            id = filename.split('_')[0]
            if id in id2objpath:
                id2objpath[id].append(line.strip())
            else:
                id2objpath[id]=[line.strip()]
            name2objpath[filename.split('.')[0]] = line.strip()
    #print(name2objpath)
    #print(id2objpath)
    model_shape = sio.loadmat('../Model_Shape.mat')
    traintxt="data/multipie/multipie_pyramidbox_face_400x400_casia_align_set1_train_label.txt"
    fa = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D, flip_input=False)
    imgpath2auxinfo={}
    with open(traintxt) as infile:
        lines = infile.readlines()
        for line in lines:
            linelist=line.strip().split(' ')
            filename = linelist[0].split('/')[-1]
            print("process ",filename)
            id = filename.split('_')[0]
            session = filename.split('_')[1]
            recordid = filename.split('_')[2]
            pts = fa.get_landmarks(np.array(Image.open(linelist[0]).convert('RGB')))
            if not pts:
                imgpath2auxinfo[linelist[0]]={'lm68':-1, "s":-1, "R":-1, "t":-1}
                print('lm68:',-1, " s:",-1, " R:",-1, " t:",-1)
            else:
                lm2d68=pts[0]
                if str(id+"_"+session+"_"+recordid) in name2objpath:
                    shape=read_obj(name2objpath[id+"_"+session+"_"+recordid])
                else:
                    shape=read_obj(id2objpath[id][np.random.randint(len(id2objpath[id]))])  
                shape=np.array(shape)
                templete3d68=shape[model_shape['keypoints'][0].astype(np.int64),:]
                s,R,t=project_param(lm2d68, templete3d68)
                print('lm68:',lm2d68, " s:",s, " R:",R, " t:",t)
                imgpath2auxinfo[linelist[0]]={'lm68':lm2d68, "s":s, "R":R, "t":t}

    pickle.dump(imgpath2auxinfo,open("multipie_imgpath2auxinfo.pkl","wb"))

if __name__ == "__main__":
    split_data()
    package_data()
