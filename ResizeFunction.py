import numpy as  np
import cv2
import scipy.misc as misc
###################################################################################
def ResizeImg(Imgs,Rate):
    N,h,w,d=Imgs.shape
    h=int(Rate*h)
    w=int(Rate*w)
    ResImgs=np.zeros([N,h,w,d],dtype=Imgs.dtype)
    for i in range(N):
         ResImgs[i]=cv2.resize(Imgs[i],(w,h),interpolation=cv2.INTER_LINEAR).astype(Imgs.dtype)
    return ResImgs
###################################################################################
def ResizeMask(Imgs,Rate):
    N,h,w=Imgs.shape
    h=int(Rate*h)
    w=int(Rate*w)
    ResImgs=np.zeros([N,h,w],dtype=Imgs.dtype)
    for i in range(N):
         ResImgs[i]=cv2.resize(Imgs[i],(w,h),interpolation=cv2.INTER_NEAREST).astype(Imgs.dtype)
    return ResImgs
##################################################################
def ResizeProb(Prob,h,w):
    N,httt,wttt=Prob.shape
    ProbRes=np.zeros([N,h,w],dtype=Prob.dtype)
    for i in range(N):
         ProbRes[i]=cv2.resize(Prob[i].astype(np.float),(w,h),interpolation=cv2.INTER_LINEAR)
    MaskRes=(ProbRes>0.5).astype(np.uint8)
    return MaskRes,ProbRes
####################################################################################
def GeneratePointerMask(x,y,h,w,Rate):
    N=len(x)
    #print("h"+str(h)+"   w"+str(w)+" Rate="+str(Rate))
    ResImgs=np.zeros([N,h,w],dtype=np.float)
    for i in range(N):
        ResImgs[i,np.min([int(y[i]*Rate),h-1]),np.min([int(np.floor(x[i]*Rate)),w-1])]=1
    return ResImgs
####################################################################################
