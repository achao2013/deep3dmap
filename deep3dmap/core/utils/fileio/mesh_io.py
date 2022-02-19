
import os

def read_obj(objpath):
    v=[]
    f=[]
    n=[]
    with open(objpath) as file:
        for line in file:
            linelist=line.strip().split()
            if len(linelist) < 1:
                continue
            flag=linelist[0]
            if flag == 'v':
                tmp=list(map(float, linelist[1:4]))
                v.append(tmp)
            elif flag == 'vn':
                tmp=list(map(float, linelist[1:4]))
                n.append(tmp)
            elif flag == 'f':
                tmp=[t.split('/')[0] for t in linelist[1:]]
                tmp=list(map(int,tmp))
                f.append(tmp)
            else:
                continue
    return v,f,n
