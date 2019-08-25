#..............Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import scipy.misc as misc

import Reader as DataReader
#import DeepLab_FCN_NetModel as NET_FCN
import FCN_NetModel as NET_FCN # The net Class
modelDir="logs/"
##################################Input paramaters#########################################################################################
#.................................Main Input parametrs...........................................................................................
# ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/val2017/"
# DataDirs=["/scratch/gobi2/seppel/GeneratedPredictions_EVAL/"]


#-------------------Input folders------------------------------------------------------------
ImageDir="../SampleData/PointerNetTrainigData/Image/"
DataDirs=[
   "../SampleData/TrainningDataForeEvalClassifyRefineNetsClassEquivalent/"
]


#-------------others-----------------------------------------------------------------------------
NumClasses=205



Trained_model_paths=[
    "logs/Defult.torch",
    ]




#---------------------Create and Initiate net ------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained
Net=Net.cuda()
for Trained_model_path in Trained_model_paths:
    print(Trained_model_path)
    Net.load_state_dict(torch.load(Trained_model_path))
    Net.eval()
    Net.half()
    #----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
    Reader = DataReader.Reader(ImageDir=ImageDir, MaskDirs=DataDirs, NumClasses=NumClasses, ClassBalance=True,MinPrecision=0.1, AugmentImage=False, TrainingMode=False)
###################################################################################################################################################################
    Siou=np.zeros([205],dtype=float)
    Sprec=np.zeros([205],dtype=float)
    Srecall=np.zeros([205],dtype=float)
    Sn=np.zeros([205],dtype=float)


#----------------------------Run and generate statitics-------------------------------------


    for i in range(50000):
        Img,MaskGT, MaskPred0, CatID = Reader.LoadSingle(ClassBalance=True)
    #***********************************************************************************************************************************************
        # for f in range(Img.shape[0]):
        #     Img[f, :, :, 0] *= 1-MaskGT[f]
        #     Img[f, :, :, 1] *= 1-MaskPred0[f]
        #
        #     misc.imshow((MaskGT[f] + MaskPred0[f] * 2).astype(np.uint8)*40)
        #     misc.imshow(Img[f])
        # print(MaskGT.shape)
    #**************************************************************************************************************************************************

#        misc.imshow((MaskGT[0] + MaskPred0[0] * 2).astype(np.uint8) * 40)
        with torch.no_grad():
            Prob, Lb = Net.forward(Images=Img, InMask=MaskPred0, TrainMode=False)
          #  Prob, Lb=Net.forward(Images=Img,Pointer=PointerMask,ROI=ROIMask,TrainMode=False) # Run net inference and get prediction
        Pred=Lb.cpu().data.numpy()

        BInter = (MaskPred0 * MaskGT).sum()
        BGs = MaskGT.sum()
        BPs = MaskPred0.sum()
        BIOU = BInter / (BGs + BPs - BInter+0.000000001)
        BPrecision = BInter / (BPs+0.000000001)
        BRecall = BInter / (BGs+0.00000000001)


        AInter=(Pred*MaskGT).sum()
        AGs=MaskGT.sum()
        APs=Pred.sum()
        AIOU=AInter/(AGs+APs-AInter+0.00001)
        APrecision=AInter/(APs+0.0000001)
        ARecall=AInter/(AGs+0.000000001)
        # print("After IOU" + str(AIOU))
        # print("Before IOU"+str(BIOU))
        # #misc.imshow(Img[0])
        # misc.imshow((MaskGT[0] + MaskPred0[0] * 2).astype(np.uint8) * 40)
        # misc.imshow((MaskGT[0] + Pred[0] * 2).astype(np.uint8) * 40)
        #



        Siou[CatID] += (AIOU-BIOU)
        Sprec[CatID] += (APrecision-BPrecision)
        Srecall[CatID] += (ARecall- BRecall)
        Sn[CatID]+=1
    #******************************************************************************************************************************************************
        #print(str(i)+") IOU dif="+str(AIOU-BIOU))
        # print("Precision=" + str(APrecision-BPrecision))
        # print("Recall=" + str(ARecall- BRecall))
        #
        # Img[0, :, :, 0] *= 1-Pred[0].astype(np.uint8)
        # Img[0, :, :, 1] *= 1-MaskPred0[0].astype(np.uint8)
        # misc.imshow(Img[0])

    #******************************************************************************************************************************************************

        Iou=(Siou/(Sn+0.000001)).sum()/(Sn>0).sum()
        Precision=(Sprec/(Sn+0.000001)).sum()/(Sn>0).sum()
        Recall=(Srecall/(Sn+ 0.000001)).sum()/(Sn>0).sum()
        txt="\n"+Trained_model_path+"\tNum="+str(Sn.sum())+"\tIOU ="+str(Iou)+"\tPrecission="+str(Precision)+"\tRecall="+str(Recall)
        print(str(i)+")"+txt)
    # fl=open(OutFile,"a")
    # fl.write(txt)
    # fl.close()



