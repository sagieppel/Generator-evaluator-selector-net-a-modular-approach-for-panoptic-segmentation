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
import FCN_NetModel as NET_FCN # The net Class
import scipy.misc as misc
import cv2
import matplotlib.pyplot as plt
os.environ["CUDA_VISIBLE_DEVICES"]="0"
#########################################################################################33
def getIOU(a,b):
    return((a*b).sum()/(a.sum()+b.sum()-(a*b).sum()))
##################################Input paramaters#########################################################################################
ImageDir="/home/sagi/DataZoo/train2017/"
GTDir="//home/sagi/DataZoo/PointerGenTrain/AllPred/"
PredDir="/home/sagi/DataZoo/PointerGenTrain/AllGT/"
OutDir="/home/sagi/DataZoo/Train_Refined/"
OutPred=OutDir+"/Pred/"
OutGt=OutDir+"/GT/"

if not os.path.exists(OutDir): os.mkdir(OutDir)
if not os.path.exists(OutGt): os.mkdir(OutGt)
if not os.path.exists(OutPred): os.mkdir(OutPred)
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
#...............Other training paramters..............................................................................
Trained_model_path="logs/860000.torch" # Path of trained model weights If you want to return to trained model, else should be =""
MaxSize=1000
MaxPixels=340000*4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
AnnotationList=[]
for Name in os.listdir(PredDir):
    s = {}
    s["FileName"] = Name
    s["IOU"] = float(Name[Name.find("#IOU#") + 5:Name.find("#Precision#")])
    s["AreaFract"] = float(Name[Name.find("#FractImSize#") + 13: Name.find('#Num#')])
    s["IsThing"] = Name[Name.find("#TYPE#") + 6: Name.find('#IOU#')] == "thing"
    s["ImageName"] = Name[:Name.find('#TYPE#')].replace("##jpg", ".jpg")
    s["CatID"] = int(Name[Name.find("#CatID#") + 7: Name.find('#END#')])
    AnnotationList.append(s)
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
#..............Start Training loop: Main Training....................................................................
AVGLoss=-1# running average loss
DifIOU=[]
PredIOU=[]
RefinIOU=[]
for itr in range(0,len(AnnotationList)):

   #print(itr)
    Imgs=cv2.imread(ImageDir+"/"+AnnotationList[itr]["ImageName"])
    Imgs = Imgs[..., :: -1]
    if (Imgs.ndim == 2):  # If grayscale turn to rgb
        Imgs = np.expand_dims(Imgs, 3)
        Imgs = np.concatenate([Imgs, Imgs, Imgs], axis=2)
    Imgs = Imgs[:, :, 0:3]  # Get first 3 channels incase there are more
    PredMask = cv2.imread(PredDir+"/"+ AnnotationList[itr]["FileName"],0 )
    GTMask = cv2.imread(GTDir+ "/" + AnnotationList[itr]["FileName"], 0)
    #print( AnnotationList[itr]["FileName"])
    Imgs=np.expand_dims(Imgs,axis=0)
    PredMask=np.expand_dims(PredMask,axis=0)
    GTMask = np.expand_dims(GTMask, axis=0)

    GT = AnnotationList[itr]["CatID"]
    IOU = AnnotationList[itr]["IOU"]
    print(IOU)
    # if IOU[0]<0.4: continue
    Prob, Lb = Net.forward(Images=Imgs, InMask=PredMask)  # Run net inference and get prediction
    RefinePred=Prob[:,1,:,:].data.cpu().numpy()
    RefinePred=(RefinePred>0.5).astype(np.float32)
    Name= AnnotationList[itr]["FileName"]
    print(Name)
    IOUtext=Name[Name.find("#IOU#")+5:Name.find("#Precision#")]
    NewIOU=OutIOU = getIOU(GTMask, RefinePred)
    Name=Name.replace(IOUtext,str(NewIOU))
    print(Name)

    print("Old IOU="+IOUtext+"     NewIOU="+str(NewIOU))
    misc.imsave(OutPred+"/"+Name,RefinePred[0].astype(np.uint8))
    misc.imsave(OutGt + "/" + Name, GTMask[0].astype(np.uint8))
    # Imgs[0,:,:,0]*=1-RefinePred[0].astype(np.uint8)
    # Imgs[0, :, :, 1] *= 1 - PredMask[0].astype(np.uint8)
    # misc.imshow(Imgs[0])



