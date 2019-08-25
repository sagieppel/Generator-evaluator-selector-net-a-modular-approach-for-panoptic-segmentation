# Evaluate net performance
#...............................Imports..................................................................
import os
import torch
import numpy as np
import Reader
import NetModel as NET_FCN # The net Class
import scipy.misc as misc

Trained_model_path = "logs/Defult.torch"
##################################Input folders#########################################################################################
ImageDir="../SampleData/PointerNetTrainigData/Image/"
ClassEqualDataDirs=[
   "../SampleData/TrainningDataForeEvalClassifyRefineNetsClassEquivalent/Pred/"
]





#########################Params unused######################################################################33
NumClasses=205
MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
MaxPixels=340000*4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
MinPrecision=0.0
#=========================Load Paramters====================================================================================================================

#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net() # Create net and load pretrained encoder path
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
Net.eval()
Net.half()

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
TReader=Reader.Reader(ImageDir=ImageDir,MaskDirs=ClassEqualDataDirs,NumClasses=NumClasses,ClassBalance=True,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,MinPrecision=MinPrecision,MaxBatchSize=MaxBatchSize,AugmentImage=False)

#.............. Evaluating....................................................................
print("Start Training")
DifIOU=[]
for itr in range(0,1000): # Mainevaluation loop
    TReader.ClassBalance=True
    Images, SegmentMask, GtIOU = TReader.LoadSingle()



    # Images[0,:,:,0] *=1 - SegmentMask[0,:,:]
    # Images[0, :, :, 1] *= 1 - SegmentMask[0, :, :]
    # print(GtIOU)
    # misc.imshow(Images[0])

    PredIOU = Net.forward(Images, SegmentMask, TrainMode=False)  # Run net inference and get prediction

    DifIOU.append(abs(PredIOU.cpu().detach().numpy()-GtIOU))
    print(str(itr)+") Prediction-GT   Mean="+str(np.mean(DifIOU))+"   Median="+str(np.median(DifIOU)))

