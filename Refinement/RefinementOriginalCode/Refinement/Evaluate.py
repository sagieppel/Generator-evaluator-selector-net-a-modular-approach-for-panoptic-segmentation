# Train the  net (in train.py)
# 1) Download COCO panoptic dataset and train images  from [here](http://cocodataset.org/#download)
# 2) Set the path to COCO train images folder in the ImageDir parameter
# 3) Set the path to COCO panoptic train annotations folder in the AnnotationDir parameter
# 4) Set the path to COCO panoptic data .json file in the DataFile parameter
# 5) Run script.
# Trained model weight and data will appear in the path given by the TrainedModelWeightDir parameter
#...............................Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import Reader
import FCN_NetModel as NET_FCN # The net Class
import matplotlib.pyplot as plt
import cv2
import scipy.misc as misc
os.environ["CUDA_VISIBLE_DEVICES"]="1"
#########################################################################################33
def getIOU(a,b):
    return((a*b).sum()/(a.sum()+b.sum()-(a*b).sum()))
##################################Input paramaters#########################################################################################
ImageDir="/home/sagi/DataZoo/val2017/"
GTDir="/media/sagi/2T/Data_zoo/SegmentAccuracyPredicitor/BEval/OutGT/"
PredDir="/media/sagi/2T/Data_zoo/SegmentAccuracyPredicitor/BEval/OutPred/"
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
#...............Other training paramters..............................................................................
Trained_model_path="logs/860000.torch" # Path of trained model weights If you want to return to trained model, else should be =""
MaxSize=1000
MaxPixels=340000*4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=Reader.Reader(ImageDir=ImageDir,GTDir=GTDir,PredDir=PredDir,MaxSize=MaxSize,MaxPixels=MaxPixels, AnnotationFileType="png", ImageFileType="jpg", TrainingMode=True)
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
Net.eval()
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
#..............Start Training loop: Main Training....................................................................
AVGLoss=-1# running average loss
DifIOU=[]
PredIOU=[]
RefinIOU=[]
for itr in range(1, np.min([9000, Reader.FileList.__len__()])):
    Imgs, GTMask, PredMask, IOU, IsThing = Reader.LoadSingleClean()
    print(IOU)
  #  if GTMask.mean()<0.05:continue
    if IOU[0]<0.25: continue
    Prob, Lb = Net.forward(Images=Imgs, InMask=PredMask)  # Run net inference and get prediction
    RefinePred=Prob[:,1,:,:].data.cpu().numpy()
    RefinePred=(RefinePred>0.5).astype(np.float32)
    InIOU=getIOU(GTMask,PredMask)
    OutIOU = getIOU(GTMask, RefinePred)
    RefinIOU.append(OutIOU)
    PredIOU.append(InIOU)
    DifIOU.append(OutIOU-InIOU)
    print("InMean=" + str(np.mean(PredIOU)) + " OutMean=" + str(np.mean(RefinIOU)) + str("  DifMean=") + str(np.mean(DifIOU)))
    print("InMedian=" + str(np.median(PredIOU)) + " OutMedian=" + str(np.median(RefinIOU)) + str("  DifMedian=") + str(np.median(DifIOU)))
#--------------------------
    # Imgs[:,:,:,0] *=1 - PredMask
    # Imgs[:, :, :, 1] *= 1 - RefinePred
    # misc.imshow(Imgs[0])
    # cv2.imshow("4",Imgs[0])
    # cv2.waitKey(3000)
#-----------------------------

print(str("  DifMean=") + str(np.mean(DifIOU))+ str("  DifMedian=") + str(np.median(DifIOU)))
plt.plot(PredIOU,DifIOU, 'ro')
plt.axis([0, 1, 0, 1])
plt.show()


