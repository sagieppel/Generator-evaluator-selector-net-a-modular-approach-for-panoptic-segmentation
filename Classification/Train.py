
print("Training classification with resnet 101")
import numpy as np
import NetModel as Net
import Reader as data_Reader
#import CocoPanoptic_Reader as ReaderGT
import os
import cv2
import scipy.misc as misc
import torch
# ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/"
# ClassEqualDataDirs=[
#    "/scratch/gobi2/seppel/GeneratedPredictions/101MultiTestClassEquivalent/Pred//",
#    "/scratch/gobi2/seppel/GeneratedPredictions/101AddClassEquivalent/Pred//"  ,
#    "/scratch/gobi2/seppel/GeneratedPredictions/51MultiTestClassEquivalent/Pred//"      ,
#    "/scratch/gobi2/seppel/GeneratedPredictions/51AddClassEquivalent/Pred//"
# ]
# AllFilesDataDirs=[
#   "/scratch/gobi2/seppel/GeneratedPredictions/101MultiTestAllFiles/Pred//",
#     "/scratch/gobi2/seppel/GeneratedPredictions/51MultiTestAllFiles/Pred//"    ,
#    "/scratch/gobi2/seppel/GeneratedPredictions/51AddAllfiles/Pred//"
# ]
#==============================Training data folders==================================================================
ImageDir="../SampleData/PointerNetTrainigData/Image/"
ClassEqualDataDirs=[
   "../SampleData/TrainningDataForeEvalClassifyRefineNetsClassEquivalent/Pred/"
]
AllFilesDataDirs=[
  "../SampleData/TrainningDataForeEvalClassifyRefineNetsAll/Pred/"
]

#====================================================Other input parameters=========================================================
NumClasses=205
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
if not os.path.exists(TrainedModelWeightDir): os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""
#...............Other training paramters..............................................................................

InitStep=0
Learning_Rate_Init=1e-5 # Initial learning rate
Learning_Rate=1e-5 # learning rate
#Learning_Rate_Decay=Learning_Rate[0]/40 # Used for standart
Learning_Rate_Decay=Learning_Rate/20
StartLRDecayAfterSteps=170000
MaxBatchSize=9 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
MaxPixels=340000*8 # Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(10000000010) # Max  number of training iteration

MinPrecision=0.2

#=========================Load Paramters====================================================================================================================
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"):
    Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate_Init.npy"):
    Learning_Rate_Init=np.load(TrainedModelWeightDir+"/Learning_Rate_Init.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
ClassEqualReader=data_Reader.Reader(ImageDir=ImageDir,MaskDirs=ClassEqualDataDirs,NumClasses=NumClasses,ClassBalance=True,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,MinPrecision=MinPrecision,MaxBatchSize=MaxBatchSize,AugmentImage=True)
AllFileReader=data_Reader.Reader(ImageDir=ImageDir,MaskDirs=AllFilesDataDirs,NumClasses=NumClasses,ClassBalance=True,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,MinPrecision=MinPrecision,MaxBatchSize=MaxBatchSize,AugmentImage=True)

#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------


Net=Net.Net(NumClasses=NumClasses)
if Trained_model_path!="":
    Net.load_state_dict(torch.load(Trained_model_path))

#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay)


f = open(TrainLossTxtFile, "a")
f.write("Iteration\tloss\t Learning Rate")
f.close()
AVGLoss=0
#..............Start Training loop: Main Training.............................................................................................

print("Start Training")
for itr in range(InitStep, MAX_ITERATION):  # Main training loop
    if np.random.rand() < 0.5:
        Images, SegmentMask, Labels = ClassEqualReader.LoadBatch()
    else:
        Images, SegmentMask, Labels = AllFileReader.LoadBatch()
   #***********************Display***********************************************************************************************************
    # for ii in range(Labels.shape[0]):
    #     print("label="+str(Labels[ii]))
    #     print(ClassEqualReader.GetCategoryData(Labels[ii]))
    #     Images[ii, :, :, 1] *= 1-SegmentMask[ii]
    #     Images[ii, :, :, 0] *= 1 - SegmentMask[ii]
    #     misc.imshow(Images[ii].astype(np.uint8))
    #     # cv2.imshow(ClassEqualReader.GetCategoryData(Labels[ii])[0],Images[ii].astype(np.uint8)); cv2.waitKey()
    #     # cv2.destroyAllWindows()
#**************************Run Trainin cycle***************************************************************************************
    Prob, Lb=Net.forward(Images,ROI=SegmentMask) # Run net inference and get prediction
    Net.zero_grad()
    LabelsOneHot=np.zeros([Labels.shape[0],NumClasses])
    for f in range(Labels.shape[0]):
        LabelsOneHot[f,int(Labels[f])]=1
    OneHotLabels=torch.autograd.Variable(torch.from_numpy(LabelsOneHot.astype(np.float32)).cuda(), requires_grad=False)
    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))*7  # Calculate cross entropy loss
    if AVGLoss==0:  AVGLoss=float(Loss.data.cpu().numpy()) #Caclculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(Loss.data.cpu().numpy())
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient decend change weight
 #   torch.cuda.empty_cache()
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 2000 == 0:# and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/Learning_Rate_Init.npy",Learning_Rate_Init)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 50000 == 0 and itr>0: #Save model weight once every 50k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 40==0: # Display train loss

        txt="\n"+str(itr)+"\tLoss="+str(float(Loss.data))+"\tAverage losss="+str(AVGLoss)+"\tLearning Rate="+str(Learning_Rate)+" Init_LR"+str(Learning_Rate_Init)
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr%10000==0 and itr>=StartLRDecayAfterSteps:
        Learning_Rate-= Learning_Rate_Decay
        if Learning_Rate<=1e-6:

            Learning_Rate_Init-=1e-6
            if Learning_Rate_Init<1e-7: Learning_Rate_Init<2e-6
            Learning_Rate=Learning_Rate_Init*1.0000001
            Learning_Rate_Decay=Learning_Rate/20
        print("Learning Rate="+str(Learning_Rate)+"   Learning_Rate_Init="+str(Learning_Rate_Init))
        print("======================================================================================================================")
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer
        torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
