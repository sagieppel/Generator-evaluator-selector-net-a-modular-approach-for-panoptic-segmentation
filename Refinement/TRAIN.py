
#...............................Imports..................................................................
import ConvertLabelToOneHotEncoding
import os
import torch
import numpy as np
import Reader
import FCN_NetModel as NET_FCN # The net Class
import scipy.misc as misc
# os.environ["CUDA_VISIBLE_DEVICES"]="0"
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
if not os.path.exists(TrainedModelWeightDir): os.mkdir(TrainedModelWeightDir)
Trained_model_path="" # Path of trained model weights If you want to return to trained model, else should be =""
##################################Input paramaters#########################################################################################
# ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/"
# ClassEqualDataDirs=[
#     "/scratch/gobi2/seppel/GeneratedPredictions/101MultiTestClassEquivalent/",
#    "/scratch/gobi2/seppel/GeneratedPredictions/101AddClassEquivalent"  ,
#    "/scratch/gobi2/seppel/GeneratedPredictions/51MultiTestClassEquivalent"      ,
#    "/scratch/gobi2/seppel/GeneratedPredictions/51AddClassEquivalent/" ,
#
# ]
# AllFilesDataDirs=[
#   "/scratch/gobi2/seppel/GeneratedPredictions/101MultiTestAllFiles/"       ,
#     "/scratch/gobi2/seppel/GeneratedPredictions/51MultiTestAllFiles/"    ,
#    "/scratch/gobi2/seppel/GeneratedPredictions/51AddAllfiles/"
# ]
#---------------------------------------Training data folders----------------------------------------------------------------------------
ImageDir="../SampleData/PointerNetTrainigData/Image/"
ClassEqualDataDirs=[
   "../SampleData/TrainningDataForeEvalClassifyRefineNetsClassEquivalent/"
]
AllFilesDataDirs=[
  "../SampleData/TrainningDataForeEvalClassifyRefineNetsAll/"
]

#-----------------------------------------Input parameters---------------------------------------------------------------------

NumClasses=205
Learning_Rate_Init=1e-5 # Initial learning rate
Learning_Rate=1e-5 # learning rate
#Learning_Rate_Decay=Learning_Rate[0]/40 # Used for standart
Learning_Rate_Decay=Learning_Rate/20
StartLRDecayAfterSteps=200000
MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
MaxPixels=340000*4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(10000000010) # Max  number of training iteration
InitStep=0
MinPrecision=0.0

#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
ClassEqualReader=Reader.Reader(ImageDir=ImageDir,MaskDirs=ClassEqualDataDirs,NumClasses=NumClasses,ClassBalance=True,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,MinPrecision=MinPrecision,MaxBatchSize=MaxBatchSize,AugmentImage=False)
AllFileReader=Reader.Reader(ImageDir=ImageDir,MaskDirs=AllFilesDataDirs,NumClasses=NumClasses,ClassBalance=False,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels,MinPrecision=MinPrecision,MaxBatchSize=MaxBatchSize,AugmentImage=False)


#=========================Load Paramters====================================================================================================================
if os.path.exists(TrainedModelWeightDir + "/Defult.torch"):
    Trained_model_path=TrainedModelWeightDir + "/Defult.torch"
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate.npy"): Learning_Rate=np.load(TrainedModelWeightDir+"/Learning_Rate.npy")
if os.path.exists(TrainedModelWeightDir+"/Learning_Rate_Init.npy"): Learning_Rate_Init=np.load(TrainedModelWeightDir+"/Learning_Rate_Init.npy")
if os.path.exists(TrainedModelWeightDir+"/itr.npy"): InitStep=int(np.load(TrainedModelWeightDir+"/itr.npy"))
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#..............Start Training loop: Main Training....................................................................
AVGLoss=-1# running average loss
print("Start Training")
for itr in range(InitStep,MAX_ITERATION): # Main training loop
    if  np.random.rand()<0.5:
        Imgs, GTMask, PredMask, IOU = ClassEqualReader.LoadBatch()
    else:
        Imgs, GTMask, PredMask, IOU = AllFileReader.LoadBatch()

   #  for oo in range(PredMask.shape[0]):
   #     # misc.imshow(Imgs[oo])
   #    #  Imgs[oo,:,:,0] *=1 - PredMask[oo,:,:]
   #      im= Imgs[oo].copy()
   #      im[:,:,0] *= 1 - GTMask[oo,:,:]
   #      im[:, :, 1] *= 1 - PredMask[oo,:,:]
   #      print(IOU[oo])
   # #     misc.imshow((PredMask[oo,:,:]*0+GTMask[oo,:,:]))
   #      misc.imshow(np.concatenate([Imgs[oo],im],axis=0))


    Prob, Lb=Net.forward(Images=Imgs,InMask=PredMask,TrainMode=True) # Run net inference and get prediction
    Net.zero_grad()
    OneHotLabels = ConvertLabelToOneHotEncoding.LabelConvert(GTMask,2)  # Convert labels map to one hot encoding pytorch
#    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and ground truth label
  #-----------------------------Calculate loss Give higher weight to high iou---------------------------------------------------------
    TorchGtIOU = torch.autograd.Variable(torch.from_numpy(IOU.astype(np.float32)).cuda(), requires_grad=False)
    Loss = -torch.mean(torch.mean(torch.mean(torch.mean(OneHotLabels * torch.log(Prob + 0.0000001), dim=1), dim=1),dim=1) * TorchGtIOU) * 3
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
    if AVGLoss==-1:  AVGLoss=float(Loss.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(Loss.data.cpu().numpy()) # Intiate runing average loss
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 2000 == 0:# and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/Defult.torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/DefultBack.torch")
        print("model saved")
        np.save(TrainedModelWeightDir+"/Learning_Rate.npy",Learning_Rate)
        np.save(TrainedModelWeightDir+"/Learning_Rate_Init.npy",Learning_Rate_Init)
        np.save(TrainedModelWeightDir+"/itr.npy",itr)
    if itr % 40000 == 0 and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir+"/"+ str(itr) + ".torch")
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 50==0: # Display train loss

        txt="\n"+str(itr)+"\t"+str(float(Loss.data))+"\t"+str(AVGLoss)+"\t"+str(Learning_Rate)+" Init_LR"+str(Learning_Rate_Init)
        print(txt)
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write(txt)
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
    if itr%10000==0 and itr>=StartLRDecayAfterSteps:
        Learning_Rate-= Learning_Rate_Decay
        if Learning_Rate<=1e-7:
            Learning_Rate_Init-=2e-6
            if Learning_Rate_Init<1e-6: Learning_Rate_Init=1e-6
            Learning_Rate=Learning_Rate_Init*1.00001
            Learning_Rate_Decay=Learning_Rate/20
        print("Learning Rate="+str(Learning_Rate)+"   Learning_Rate_Init="+str(Learning_Rate_Init))
        print("======================================================================================================================")
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer
        torch.cuda.empty_cache()  # Empty cuda memory to avoid memory leaks
