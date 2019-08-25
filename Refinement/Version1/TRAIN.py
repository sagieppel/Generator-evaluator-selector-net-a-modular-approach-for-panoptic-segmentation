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
import scipy.misc as misc
os.environ["CUDA_VISIBLE_DEVICES"]="1"
##################################Input paramaters#########################################################################################
ImageDir="/home/sagi/DataZoo/train2017/"
PredDirs=['/home/sagi/DataZoo/PointerGenTrain/AllPred/',"/home/sagi/DataZoo/Train_Refined/Pred/"]#, "/media/sagi/2T/Data_zoo/SegmentAccuracyPredicitor/Train/Refined/OutPred/","/media/sagi/2T/Data_zoo/SegmentAccuracyPredicitor/Train/OutGT/"]
GTDirs=["/home/sagi/DataZoo/PointerGenTrain/AllGT/","/home/sagi/DataZoo/Train_Refined/GT/"]
NumClasses=203
TrainedModelWeightDir="logs/" # Folder where trained model weight and information will be stored"
#...............Other training paramters..............................................................................
Trained_model_path="logs/860000.torch" # Path of trained model weights If you want to return to trained model, else should be =""
MaxBatchSize=7 # Max images in batch
MinSize=250 # Min image Height/Width
MaxSize=1000# Max image Height/Width
Learning_Rate=1e-5 # Initial learning rate
MaxPixels=340000*4# Max pixel in batch can have (to keep oom out of memory problems) if the image larger it will be resized.
TrainLossTxtFile=TrainedModelWeightDir+"TrainLoss.txt" #Where train losses will be writen
Weight_Decay=1e-5# Weight for the weight decay loss function
MAX_ITERATION = int(100000010) # Max  number of training iteration
#----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
Reader= Reader.Reader(ImageDir=ImageDir,PredMaskDirs=PredDirs,GTMaskDirs=GTDirs,NumClasses=203, MaxBatchSize=MaxBatchSize,MinSize=MinSize,MaxSize=MaxSize,MaxPixels=MaxPixels, AnnotationFileType="png", ImageFileType="jpg",TrainingMode=True)
#---------------------Create and Initiate net and create optimizer------------------------------------------------------------------------------------
Net=NET_FCN.Net(NumClasses=2) # Create net and load pretrained encoder path
if Trained_model_path!="": # Optional initiate full net by loading a pretrained net
    Net.load_state_dict(torch.load(Trained_model_path))
Net=Net.cuda()
#optimizer=torch.optim.SGD(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay,momentum=0.5)
optimizer=torch.optim.Adam(params=Net.parameters(),lr=Learning_Rate,weight_decay=Weight_Decay) # Create adam optimizer
#--------------------------- Create logs files for saving loss during training----------------------------------------------------------------------------------------------------------
if not os.path.exists(TrainedModelWeightDir): os.makedirs(TrainedModelWeightDir) # Create folder for trained weight
f = open(TrainLossTxtFile, "w+")# Training loss log file
f.write("Iteration\tloss\t Learning Rate=")
f.close()
#..............Start Training loop: Main Training....................................................................
AVGLoss=-1# running average loss
print("Start Training")
for itr in range(860001,MAX_ITERATION): # Main training loop
    if  np.random.rand()<0.5: Reader.ClassBalance=True
    else: Reader.ClassBalance=False
    Imgs, GTMask, PredMask, IOU, IsThing = Reader.LoadBatch()
    # for oo in range(PredMask.shape[0]):
    #     print(IOU[oo])
    #     Imgs[oo,:,:,0] *=1 - PredMask[oo,:,:]
    #     Imgs[oo, :, :, 1] *= 1 - GTMask[oo,:,:]
    #     misc.imshow(Imgs[oo])


    Prob, Lb=Net.forward(Images=Imgs,InMask=PredMask) # Run net inference and get prediction
    Net.zero_grad()
    OneHotLabels = ConvertLabelToOneHotEncoding.LabelConvert(GTMask,2)  # Convert labels map to one hot encoding pytorch

#    Loss = -torch.mean((OneHotLabels * torch.log(Prob + 0.0000001)))  # Calculate loss between prediction and ground truth label
    TorchGtIOU = torch.autograd.Variable(torch.from_numpy(IOU.astype(np.float32)).cuda(), requires_grad=False)
    Loss = -torch.mean(torch.mean(torch.mean(torch.mean(OneHotLabels * torch.log(Prob + 0.0000001), dim=1), dim=1), dim=1) * TorchGtIOU)*3
    Loss.backward() # Backpropogate loss
    optimizer.step() # Apply gradient descent change to weight
    if AVGLoss==-1:  AVGLoss=float(Loss.data.cpu().numpy()) #Calculate average loss for display
    else: AVGLoss=AVGLoss*0.999+0.001*float(Loss.data.cpu().numpy()) # Intiate runing average loss
# --------------Save trained model------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 10000 == 0 and itr>0: #Save model weight once every 10k steps
        print("Saving Model to file in "+TrainedModelWeightDir)
        torch.save(Net.state_dict(), TrainedModelWeightDir + "/" + str(itr) + ".torch")
        print("model saved")
#......................Write and display train loss..........................................................................
    if itr % 10==0: # Display train loss
        torch.cuda.empty_cache()  #Empty cuda memory to avoid memory leaks
        print("Step "+str(itr)+" Runnig Average Loss="+str(AVGLoss)+" Learning Rate="+str(Learning_Rate))
        #Write train loss to file
        with open(TrainLossTxtFile, "a") as f:
            f.write("\n"+str(itr)+"\t"+str(float(Loss.data))+"\t"+str(AVGLoss)+"\t"+str(Learning_Rate))
            f.close()
#----------------Update learning rate fractal manner-------------------------------------------------------------------------------
        # ------update learning rate---------------------------------------------------------------------------------------------------------------------------------------------
    if itr % 10000 == 0:
        Learning_Rate -= 5e-7
        if Learning_Rate < 5e-7: Learning_Rate = 1e-5  # Initial learning rate
        optimizer = torch.optim.Adam(params=Net.parameters(), lr=Learning_Rate,weight_decay=Weight_Decay)  # Create adam optimizer
