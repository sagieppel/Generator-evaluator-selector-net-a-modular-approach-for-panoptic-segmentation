
print("Evaluate Classification")
import numpy as np
import NetModel as Net
import Reader as data_Reader
#import CocoPanoptic_Reader as ReaderGT
import os
import cv2
import scipy.misc as misc
import torch
# ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/"
# DataDirs=[
#    "/scratch/gobi2/seppel/GeneratedPredictions/101MultiTestClassEquivalent/Pred//"
# ]
ImageDir="../SampleData/PointerNetTrainigData/Image/"
DataDirs=[
   "../SampleData/TrainningDataForeEvalClassifyRefineNetsClassEquivalent/Pred/"
]


NumClasses=205

Trained_model_path="logs/Defult.torch" # Model path # Path of trained model weights If you want to return to trained model
#...............Other training paramters..............................................................................



Learning_Rate_Init=1e-5 # Initial learning rate
Learning_Rate=1e-5 # learning rate
#Learning_Rate_Decay=Learning_Rate[0]/40 # Used for standart
Learning_Rate_Decay=Learning_Rate/20
StartLRDecayAfterSteps=170000
MaxBatchSize=9 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
MaxPixels=340000*8 # Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.

MinPrecision=0.2



#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader=data_Reader.Reader(ImageDir=ImageDir,MaskDirs=DataDirs,NumClasses=NumClasses,ClassBalance=True,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,MinPrecision=MinPrecision,MaxBatchSize=MaxBatchSize,AugmentImage=False)

#---------------------Create and Initiate net -----------------------------------------------------------------------------------

Net=Net.Net(NumClasses=NumClasses)
Net.load_state_dict(torch.load(Trained_model_path))
Net.half()
Net.eval()

#..............Start evaluation loop.............................................................................................

print("Start Evaluation")
Correct=0
Total=0
for itr in range(10000):  # Main evaluation  loop
    Images, SegmentMask, Labels = Reader.LoadSingleClean()

   #***********************Display***********************************************************************************************************    for ii in range(Labels.shape[0]):

    # Images[0, :, :, 1] *= 1 - SegmentMask[0]
    # Images[0, :, :, 0] *= 1 - SegmentMask[0]
    # misc.imshow(Images[0].astype(np.uint8))
#**************************Run Trainin cycle***************************************************************************************
    with torch.no_grad():
                Prob, Lb=Net.forward(Images,ROI=SegmentMask,EvalMode=True) # Run net inference and get prediction

    Correct+=Lb.cpu().numpy()[0]==Labels
    Total+=1
    print(str(itr)+") average accuracy rate="+str(100*Correct/Total)+"%")