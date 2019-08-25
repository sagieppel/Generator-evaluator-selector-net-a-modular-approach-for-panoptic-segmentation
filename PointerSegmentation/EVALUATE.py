#..............Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import scipy.misc as misc
#import CocoPanoptic_Reader as Data_Reader
import Reader as Data_Reader
#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class
modelDir="logs/"
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
# ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/val2017/"
# MaskDir = {"/scratch/gobi2/seppel/Data_ForPointerNet_Train_Eval/Eval_CocoGenerated/All/"}
# FullSegDir = "/scratch/gobi2/seppel/Data_ForPointerNet_Train_Eval/Eval_CocoGenerated/SegMapDir/"
ImageDir="../SampleData/PointerNetTrainigData/Image/"
MaskDir = {"../SampleData/PointerNetTrainigData/SegmentMask/"}
FullSegDir = "../SampleData/PointerNetTrainigData/SegMap/"
fl=open("EvalResults.txt","w")
fl.write("")
fl.close()
##-----------------------------------List of model to evaluae----------------------------------------------------------------------------
Trained_model_paths=[]
for Name in os.listdir(modelDir):
    if ".torch" in Name:
        Trained_model_paths.append(modelDir+"/"+Name)


# MaxBatchSize=7 # Max images in batch
# MinSize=250 # Min image Height/Width
# MaxSize=1000# Max image Height/Width



#---------------------Create and Initiate net-----------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
for Trained_model_path in Trained_model_paths: # Evaluate all models in the model folder
    Net.load_state_dict(torch.load(Trained_model_path))
    Net.eval()
    Net.half()
    #----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    Reader=Data_Reader.Reader(ImageDir,MaskDir,FullSegDir,NumClasses= 205,TrainingMode=False)# MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,TrainingMode=False)


    print("Start Evaluating")
    Siou=np.zeros([205],dtype=float)
    Sprec=np.zeros([205],dtype=float)
    Srecall=np.zeros([205],dtype=float)
    Sn=np.zeros([205],dtype=float)




#------------------------------------Evaluation loop---------------------------------------------------------------------------------
    for i in range(50000):
        Img, Mask, PointerMask, ROIMask, CatID, sy, sx = Reader.LoadSingle(ByClass=True)
    #***********************************************************************************************************************************************
        # for f in range(Img.shape[0]):
        #     Img[f, :, :, 0] *= 1-Mask[f]
        #     Img[f, :, :, 1] *= ROIMask[f]
        #
        #     misc.imshow((ROIMask[f] + Mask[f] * 2 + PointerMask[f] * 3).astype(np.uint8)*40)
        #     misc.imshow(Img[f])
        # print(ROIMask.shape)
    #**************************************************************************************************************************************************


        with torch.no_grad():
            Prob, Lb=Net.forward(Images=Img,Pointer=PointerMask,ROI=ROIMask,TrainMode=False) # Run net inference and get prediction
        Pred=Lb.cpu().data.numpy()
        Inter=(Pred*Mask).sum()
        Gs=Mask.sum()
        Ps=Pred.sum()
        IOU=Inter/(Gs+Ps-Inter)
        Precision=Inter/Ps
        Recall=Inter/Gs

        Siou[CatID] += IOU
        Sprec[CatID] += Precision
        Srecall[CatID] += Recall
        Sn[CatID]+=1
    #******************************************************************************************************************************************************
        print("IOU="+str(IOU))
        # print("Precision=" + str(Precision))
        # print("Recall=" + str(Recall))
        #
        # Img[0, :, :, 0] *= 1-Mask[0]
        # Img[0, :, :, 1] *= 1-Pred[0]
        # misc.imshow(Img[0])

    #******************************************************************************************************************************************************

        Iou=(Siou/(Sn+0.000001)).sum()/(Sn>0).sum()
        Precision=(Sprec/(Sn+0.000001)).sum()/(Sn>0).sum()
        Recall=(Srecall/(Sn+ 0.000001)).sum()/(Sn>0).sum()
        txt="\n"+Trained_model_path+"\tNum="+str(Sn.sum())+"\tIOU="+str(Iou)+"\tPrecission="+str(Precision)+"\tRecall="+str(Recall)
        print(txt)
    fl=open("logs/EvalResults.txt","a")
    fl.write(txt)
    fl.close()









