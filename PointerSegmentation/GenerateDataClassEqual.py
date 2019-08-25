
#Apply the pointer net to generate data for other nets (generate data with equal probability per class  (no need to wait this script to finish, but the more its run the more traininig data you have)
# Generate datacleaner refiner should be appplied aftert  this script
#...............................Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import cv2
import scipy.misc as misc
#import CocoPanoptic_Reader as Data_Reader
import Reader as Data_Reader
#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class
modelDir="logs/"
##################################Input paramaters#########################################################################################
#.................................Main Input Data location..........................................................................................
ImageDir="../SampleData/PointerNetTrainigData/Image/"
GTMaskDir = {"../SampleData/PointerNetTrainigData/SegmentMask/"}
FullSegDir = "../SampleData/PointerNetTrainigData/SegMap/"



#-----------------------------OutPut dir-----------------------------------------------------------------------
OutDir="../SampleData/TrainningDataForeEvalClassifyRefineNetsClassEquivalent/"
OutPredDir=OutDir+"/Pred/"
OutGTDir=OutDir+"/GT/"
if not os.path.exists(OutDir): os.mkdir(OutDir)
if not os.path.exists(OutPredDir): os.mkdir(OutPredDir)
if not os.path.exists(OutGTDir): os.mkdir(OutGTDir)

#-------------------create list of net model-------------------------------------------------------------------------------------------------------------


Trained_model_files=[]
for Name in os.listdir(modelDir):
    if ".torch" in Name:
        Trained_model_files.append(Name)

Trained_model_files.sort()




# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------

Reader=Data_Reader.Reader(ImageDir,GTMaskDir,FullSegDir,NumClasses= 205,TrainingMode=False)# MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,TrainingMode=False)
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
Trained_model=Trained_model_files[np.random.randint(len(Trained_model_files))]
Reader.Reset()



Net.load_state_dict(torch.load(modelDir+"/"+Trained_model))
Net.eval()
Net.half()
fif=0

#--------------------------------------------main loop for generating data--------------------------------------------------------------------------------------------------------------------------------------
while (True):
    fif+=1
    #if fif>1200000: break
    print(fif)
    x=open(OutDir+"/Count.txt","w")
    x.write(str(fif))
    x.close()
    if (fif%1000==0): # Load random net
        Trained_model = Trained_model_files[np.random.randint(len(Trained_model_files))]
        Net.load_state_dict(torch.load(modelDir + "/" + Trained_model))
        print("Loading model "+Trained_model)
        Net.eval()
        Net.half()

#while (Reader.Epoch<1):
    print(Reader.Clepoch.min())
    Img, GTMask, PointerMask, ROIMask, CatID, ImName, sy, sx = Reader.LoadSingleForGeneration(ByClass=True,Augment=True)
#       Img, GTMask, PointerMask, ROIMask, CatID,ImName, sy, sx = Reader.LoadSingleForGeneration(ByClass=False,Augment=False)
#***********************************************************************************************************************************************
    # for f in range(Img.shape[0]):
    #     misc.imshow(Img[f])
    #     Img[f, :, :, 0] *= 1 - GTMask[f]
    #     Img[f, :, :, 1] *= 1 - PointerMask[f]
    #     Img[f, :, :, 2] *= 1 - PointerMask[f]
    #
    #     misc.imshow((ROIMask[f] + GTMask[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
    #     misc.imshow(Img[f])
    # print(ROIMask.shape)
#*************************************Make prediction to generate data*************************************************************************************************************


    with torch.no_grad():
        Prob, Lb=Net.forward(Images=Img,Pointer=PointerMask,ROI=ROIMask,TrainMode=False) # Run net inference and get prediction
    Pred=Lb.cpu().data.numpy()
    Inter=(Pred*GTMask).sum()
    Gs=GTMask.sum()
    Ps=Pred.sum()
    IOU=Inter/(Gs+Ps-Inter)
    Precision=Inter/Ps
    Recall=Inter/Gs
    fname=ImName.replace(".","#")+"#IOU#"+str(IOU)+"#Precision#"+str(Precision)+"#Recall#"+str(Recall)+"#CatID#"+str(CatID)+"#RandID#"+str(np.random.randint(0,1000000000))+".png"
#    print(fname)
    cv2.imwrite(OutGTDir+"/"+fname,GTMask[0].astype(np.uint8))
    cv2.imwrite(OutPredDir + "/" + fname, Pred[0].astype(np.uint8))

x = open(OutDir + "/Finished.txt", "w")
x.close()
#******************************************************************************************************************************************************
    # print()
    # print("Precision=" + str(Precision))
    # print("Recall=" + str(Recall))
    #
    # Img[0, :, :, 0] *= 1-GTMask[0]
    # Img[0, :, :, 1] *= 1-Pred[0]
    # misc.imshow(Img[0])

#******************************************************************************************************************************************************












