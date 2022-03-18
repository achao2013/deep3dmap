import cv2
import math
import numpy as np
def isRotationMatrix(R):
    ''' checks if a matrix is a valid rotation matrix(whether orthogonal or not)
    '''
    Rt = np.transpose(R)
    shouldBeIdentity = np.dot(Rt, R)
    I = np.identity(3, dtype = R.dtype)
    n = np.linalg.norm(I - shouldBeIdentity)
    return n < 1e-6
def matrix2angle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    assert(isRotationMatrix)
    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    return rx, ry, rz
def R2radangle(R):
    ''' get three Euler angles from Rotation Matrix
    Args:
        R: (3,3). rotation matrix
    Returns:
        x: pitch
        y: yaw
        z: roll
    '''
    num=10
    while (not isRotationMatrix(R)) and num>0:
        x=R[0,:]
        y=R[1,:]
        error=np.dot(x,y)
        x_ort=x-(error/2)*y
        y_ort=y-(error/2)*x
        z_ort=np.cross(x_ort, y_ort)
        x_new = 1/2*(3-np.dot(x_ort,x_ort))*x_ort
        y_new = 1/2*(3-np.dot(y_ort,y_ort))*y_ort
        z_new = 1/2*(3-np.dot(z_ort,z_ort))*z_ort
        R[0,:]=x_new
        R[1,:]=y_new
        R[2,:]=z_new
        num-=1

    sy = math.sqrt(R[0,0] * R[0,0] +  R[1,0] * R[1,0])
     
    singular = sy < 1e-6
 
    if  not singular :
        x = math.atan2(R[2,1] , R[2,2])
        y = math.atan2(-R[2,0], sy)
        z = math.atan2(R[1,0], R[0,0])
    else :
        x = math.atan2(-R[1,2], R[1,1])
        y = math.atan2(-R[2,0], sy)
        z = 0

    return np.array([x, y, z]),isRotationMatrix(R),R
    # rx, ry, rz = np.rad2deg(x), np.rad2deg(y), np.rad2deg(z)
    #rx, ry, rz = x*180/np.pi, y*180/np.pi, z*180/np.pi
    #return rx, ry, rz


def P2sRt(P):
    ''' decompositing camera matrix P
    Args: 
        P: (3, 4). Affine Camera Matrix.
    Returns:
        s: scale factor.
        R: (3, 3). rotation matrix.
        t: (3,). translation. 
    '''
    t = P[:, 3]
    R1 = P[0:1, :3]
    R2 = P[1:2, :3]
    s = (np.linalg.norm(R1) + np.linalg.norm(R2))/2.0
    r1 = R1/np.linalg.norm(R1)
    r2 = R2/np.linalg.norm(R2)
    r3 = np.cross(r1, r2)

    R = np.concatenate((r1, r2, r3), 0)
    return s, R, t


def estimate_affine_matrix_3d22d(X, x):
    ''' Using Golden Standard Algorithm for estimating an affine camera
        matrix P from world to image correspondences.
        See Alg.7.2. in MVGCV 
        Code Ref: https://github.com/patrikhuber/eos/blob/master/include/eos/fitting/affine_camera_estimation.hpp
        x_homo = X_homo.dot(P_Affine)
    Args:
        X: [n, 3]. corresponding 3d points(fixed)
        x: [n, 2]. n>=4. 2d points(moving). x = PX
    Returns:
        P_Affine: [3, 4]. Affine camera matrix
    '''
    X = X.T; x = x.T
    assert(x.shape[1] == X.shape[1])
    n = x.shape[1]
    assert(n >= 4)

    #--- 1. normalization
    # 2d points
    mean = np.mean(x, 1) # (2,)
    x = x - np.tile(mean[:, np.newaxis], [1, n])
    average_norm = np.mean(np.sqrt(np.sum(x**2, 0)))
    scale = np.sqrt(2) / average_norm
    x = scale * x

    T = np.zeros((3,3), dtype = np.float32)
    T[0, 0] = T[1, 1] = scale
    T[:2, 2] = -mean*scale
    T[2, 2] = 1

    # 3d points
    X_homo = np.vstack((X, np.ones((1, n))))
    mean = np.mean(X, 1) # (3,)
    X = X - np.tile(mean[:, np.newaxis], [1, n])
    m = X_homo[:3,:] - X
    average_norm = np.mean(np.sqrt(np.sum(X**2, 0)))
    scale = np.sqrt(3) / average_norm
    X = scale * X

    U = np.zeros((4,4), dtype = np.float32)
    U[0, 0] = U[1, 1] = U[2, 2] = scale
    U[:3, 3] = -mean*scale
    U[3, 3] = 1

    # --- 2. equations
    A = np.zeros((n*2, 8), dtype = np.float32);
    X_homo = np.vstack((X, np.ones((1, n)))).T
    A[:n, :4] = X_homo
    A[n:, 4:] = X_homo
    b = np.reshape(x, [-1, 1])
 
    # --- 3. solution
    p_8 = np.linalg.pinv(A).dot(b)
    P = np.zeros((3, 4), dtype = np.float32)
    P[0, :] = p_8[:4, 0]
    P[1, :] = p_8[4:, 0]
    P[-1, -1] = 1

    #print("A:",A)
    #print("np.linalg.pinv(A):",np.linalg.pinv(A))
    #print("b:",b)
    #print("p_8:",p_8)
    #print("P:",P)

    # --- 4. denormalization
    P_Affine = np.linalg.inv(T).dot(P.dot(U))
    return P_Affine


def draw_landmark(img, pts, color = (0, 0, 255)):
    '''Example function with types documented in the docstring.'''

    for j in range(pts.shape[0]):
        cv2.circle(img, (int(pts[j, 0]), int(pts[j, 1])), 2, color, -1)
    return img
def face_orientation(landmarks):#(frame, landmarks):
    #size = frame.shape #(height, width, color_channel)
    landmarks = [x *2 for x in landmarks]

    image_points = np.array([
                            (landmarks[4], -landmarks[5]),     # Nose tip
                            #(landmarks[10], landmarks[11]),   # Chin
                            (landmarks[0], -landmarks[1]),     # Left eye left corner
                            (landmarks[2], -landmarks[3]),     # Right eye right corne
                            (landmarks[6], -landmarks[7]),     # Left Mouth corner
                            (landmarks[8], -landmarks[9])      # Right mouth corner
                        ], dtype="double")

    model_points = np.array([
                            (0.0, 0.0, 0.0),             # Nose tip
                            #(0.0, -330.0, -65.0),        # Chin
                            (-165.0, 170.0, -135.0),     # Left eye left corner
                            (165.0, 170.0, -135.0),      # Right eye right corne
                            (-150.0, -150.0, -125.0),    # Left Mouth corner
                            (150.0, -150.0, -125.0)      # Right mouth corner                         
                        ])
    P_Affine = estimate_affine_matrix_3d22d(model_points, image_points)
    s, R, t = P2sRt(P_Affine)
    rx, ry, rz = matrix2angle(R)

    pitch, yaw, roll = rx, ry, rz
    imgpts, modelpts = None, None
    return imgpts, modelpts, (str(int(roll)), str(int(pitch)), str(int(yaw))), (landmarks[4], landmarks[5])

def project_param(landmarks, templete_points):#(frame, landmarks):
    #print("landmarks:",landmarks)
    print("landmarks.shape:",landmarks.shape)
    #print("templete_points:",templete_points)
    print("templete_points.shape:",templete_points.shape)

    #image_points = landmarks[[30,36,39,42,45,48,51,54,57],:]
    image_points = landmarks[[30,36,45,48,54],:]
    #image_points = landmarks
    image_points[:,1]=224-image_points[:,1]
                          
    #model_points = templete_points[[30,36,39,42,45,48,51,54,57],:]
    model_points = templete_points[[30,36,45,48,54],:]
    #model_points = templete_points

    print("image_points:",image_points)
    print("model_points:",model_points)
    P_Affine = estimate_affine_matrix_3d22d(model_points, image_points)
    print("P_Affine",P_Affine)
    s, R, t = P2sRt(P_Affine)
    rx, ry, rz = matrix2angle(R)
    print("angle:",rx, ry, rz)
    return s,R,t


if __name__=="__main__":

    with open('box_and_landmark.txt','r') as fin:
        with open('output.txt','w') as fout:
            line = fin.readline().strip()
            while(line):
                imgpath = line
                face_num = fin.readline().strip()
                value = []
                fout.write(imgpath)
                fout.write('\n'+face_num)
                for i in range(int(face_num)):
                    line = fin.readline().strip()
                    line_split = line.split()
                    if(int(float(line_split[4]))==-1):
                        angleline = line_split[0]+' '+line_split[1]+' '+line_split[2]+' '+line_split[3]+' '+'/ / /'
                    else:
                        landmarks=[line_split[4],line_split[5],line_split[7],line_split[8],line_split[10],line_split[11],line_split[13],line_split[14],line_split[16],line_split[17]]
                        landmarks = list(map(float,landmarks))
                        landmarks = list(map(int,landmarks))
                        img = cv2.imread(imgpath)
                        imgpts, modelpts, rotate_degree, nose = face_orientation(landmarks)#(img, landmarks)
                        angleline = line_split[0]+' '+line_split[1]+' '+line_split[2]+' '+line_split[3]+' '+str(rotate_degree[0])+' '+str(rotate_degree[1])+' '+str(rotate_degree[2])
                    fout.write('\n'+angleline)
                line = fin.readline().strip()
                if(line):
                    fout.write('\n')

