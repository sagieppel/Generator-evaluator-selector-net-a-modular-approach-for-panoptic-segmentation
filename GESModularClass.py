import torch
import numpy as np
import CocoPanoptic_Reader_Statistics as Data_Reader
import Evaluation.NetModel as IOUEvalnet
import PointerSegmentation.FCN_NetModel as PointerSegmentationNet
import Refinement.FCN_NetModel as RefinNet
import Classification.NetModel as ClassificationNet
import os
import cv2
import ResizeFunction as RF
import scipy.misc as misc
#import matplotlib.pyplot

#########################################################################################################################
class ModuNet:
    def __init__(self,OutFolder = "",ImageDir = "/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/val2017/",AnnotationDir = "/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_val2017",DataFile = "panoptic_val2017.json", ClassificationConsistancyThresh = 0.65, GenerateStatics = False):
        self.GenerateStatics = GenerateStatics # Generate Statics during Proccess can be used only if GT annotation avaialble
        self.MainOutFolder = OutFolder
#
        self.OutSegFolder = self.MainOutFolder + "/FinalPredictionsModular/"
        self.Out2ChannelsCOCOFolder = self.MainOutFolder + "/COCO2Channels/"
        self.Out2ChannelsCOCOStuffInstanceFolder = self.MainOutFolder + "/COCO2ChannelsStuffInstance/"
        self.OutVizFolder = self.MainOutFolder + "/FinalPredictionsVizual/"
        if not os.path.exists(self.MainOutFolder): os.mkdir(self.MainOutFolder)
        #if not os.path.exists(self.OutSegFolder): os.mkdir(self.OutSegFolder)
        if not os.path.exists(self.Out2ChannelsCOCOFolder): os.mkdir(self.Out2ChannelsCOCOFolder)
        #if not os.path.exists(self.Out2ChannelsCOCOStuffInstanceFolder): os.mkdir(self.Out2ChannelsCOCOStuffInstanceFolder)
        if not os.path.exists(self.OutVizFolder): os.mkdir(self.OutVizFolder)

        # .....................................Input parametrs..................................................................................................................
        self.ImageDir =ImageDir  # image folder (coco training)  evaluation set
        self.AnnotationDir = AnnotationDir   # annotation maps from coco panoptic evaluation set
        self.DataFile = DataFile  # Json Data file coco panoptic  evaluation set



        # NumPointerSegmentsBeforeIOUCheck=5 # number of segments to generate (using pointer) before checking IOU
        self.NumPointerSegmentsBeforeRfinement = 8  # number of segments to generate (using pointer) before refining
        self.MaxPixelsInBatch = 800 * 800 * 5 # Max batch size in pixels
        self.NumberRetrysPerImage = 6  # 5 if prediction does not hold thresh hold how many retrys are alowed
        self.MustRefine = False # Only use segments that passed refinement


        self.SegResizeRates = [1]  # Sizes of image to use when applying the generator
        self.EvalResizeRates = [1] # Sizes of image to use when applying the evaluator
        self.ThreshAccept = 0.5#35 # Threshold for accepting segments
        self.ThreshRefine = 0.9 # Min threshold to refine segments
        self.ClassificationConsistancyThresh = ClassificationConsistancyThresh # Minimoum consistancy threshold for class prediction  to accept segment (set this to zero if you want class agnostic segmentation)
        self.SegmentConsistanyWeight=0.25 # How much consitancy between predicted segment and other predicted segments should contribute to the segment score (relative to the predicted IOU)

        #########################################################################################################################
        self.TrainedPointerSegmentationModel = "PointerSegmentation/logs/Defult.torch"
        # "SegmentationNEW/logs/DefultLoss56.torch"  # \
        self.TrainedIOUEvaluateModel = "Evaluation/logs/Defult.torch"#"Evaluation2/logs/Defult.torch"
        self.TrainedRefinedModel = "Refinement/logs/Defult.torch"
        self.Trained_Classification_Model = "Classification/logs/Defult.torch"
        ###############################Load nets###########################################################################################################
        # ---------------------Load Pointer Segmentation net------------------------------------------------------------------------------------
        print("Loading Pointer  model")
        self.NetPointerSeg = PointerSegmentationNet.Net(NumClasses=2)  # Create net and load pretrained encoder path
        # self.NetPointerSeg.AddAttententionLayer()  # Load attention layer
        self.NetPointerSeg.load_state_dict(torch.load(self.TrainedPointerSegmentationModel))  # load traine model
        # ---------------------Load EvaluationNet Segmentation net------------------------------------------------------------------------------------
        print("Loading Eval  model")
        self.NetIOUEval = IOUEvalnet.Net()  # Create net and load pretrained encoder path
        self.NetIOUEval.load_state_dict(torch.load(self.TrainedIOUEvaluateModel))  # load traine model
        # ---------------------Load EvaluationNet Segmentation net------------------------------------------------------------------------------------
        print("Loading Seg refining  model")
        self.NetRefine = RefinNet.Net(NumClasses=2)  # Create net and load pretrained encoder path
        self.NetRefine.load_state_dict(torch.load(self.TrainedRefinedModel))
        # ---------------------Load Classification net------------------------------------------------------------------------------------
        print("Loading classification model")
        self.NetClassification = ClassificationNet.Net(NumClasses=205)  # Create net and load pretrained encoder path
        self.NetClassification.load_state_dict(torch.load(self.Trained_Classification_Model))

        # ------------------------------------------------------------------------------------------------------------------------------
        self.NetRefine.cuda()
        self.NetPointerSeg.cuda()
        self.NetIOUEval.cuda()
        self.NetClassification.cuda()

        self.NetRefine.eval()
        self.NetPointerSeg.eval()
        self.NetIOUEval.eval()
        self.NetClassification.eval()

        self.NetRefine.half()
        self.NetPointerSeg.half()
        self.NetIOUEval.half()
        self.NetClassification.half()
        # ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.Reader = Data_Reader.Reader(ImageDir=self.ImageDir, AnnotationDir=self.AnnotationDir, DataFile=self.DataFile)
        #self.Reader = Data_Reader.Reader(ImageDir=self.ImageDir)
#------------statics-------------------------------------------------------------------------------------------------------------------
        if self.GenerateStatics:
            self.St_IOU_GT = []
            self.St_IsThing = []
            self.St_PrecisionGT = []
            self.St_IOU_DIF = []
            self.St_IOU_Pred = []

            self.St_SelectedSg_IsThing = []
            self.St_SelectedSgIOU = []
            self.St_TopSgIOU = []
            self.St_DifIOU_TOP_SELECTED = []
            self.St_DifTop = []
            self.St_DifSelected = []
            self.St_DifIOU_TOP_SELECTED=[]
            self.St_CorrectClass=[]
#########################################################################################################################################################################
#########################################################################################################################################################################
    def RunAllFiles(self):
        iii = 0

        for self.Reader.itr in range(len(self.Reader.FileList)):
            print("Start New Image")
            iii += 1
            print(iii)
            Img_name = self.Reader.FileList[self.Reader.itr]
            if not os.path.exists(self.OutVizFolder + "/" + Img_name.replace(".png",".jpg")):
                       self.RunSingle()
            else: print("already exists")


#==================Generate statistics============================================================================
            if self.GenerateStatics:
                # -------Display Statitics------------------------------
                print(str(iii)+")-------------------all segments generated even those that was not selected---------------------------------------------------")
                print("total number=" + str(len(self.St_IsThing)))
                Thing = np.array(self.St_IsThing) == 1

                Stuff = np.array(self.St_IsThing) == 0
                print("Mean IOU of selected segments Things=" + str(np.mean(np.array(self.St_IOU_GT)[Thing])))
                print("Mean IOU of selected segments Stuff=" + str(np.median(np.array(self.St_IOU_GT)[Stuff])))

                print(" Mean, Diffrence between the predicted and real IOU  of the selected segments. Thing=" + str(np.mean(np.array(self.St_IOU_DIF)[Thing])))
                print("Mean, Diffrence between the predicted and real IOU  of the selected segments. Stuff=" + str(np.mean(np.array(self.St_IOU_DIF)[Stuff])))
                print(str(iii)+")...................Top segments generated in term of IOU (either top real IOU or top predicted IOU (by the evaluator)................................................................")
                Us = np.array(self.St_SelectedSg_IsThing) > 0
                print("IOU of the segment that was by the evaluator.       Median=" + str(np.median(np.abs(np.array(self.St_SelectedSgIOU))[Us]))+" Mean=" + str(np.mean(np.abs(np.array(self.St_SelectedSgIOU))[Us])))

                print("IOU of the generated segment with the highest IOU  (GT iou not evaluated).     Median=" + str(np.median(np.abs(np.array(self.St_TopSgIOU))[Us]))+" Mean=" +  str(np.mean(np.abs(np.array(self.St_TopSgIOU))[Us])))

                print("Difference of IOU of  between best segment generated and the segment that was selected.     Median=" + str(np.median(np.abs(np.array(self.St_DifIOU_TOP_SELECTED))[Us]))+" Mean=" +  str(np.mean(np.abs(np.array(self.St_DifIOU_TOP_SELECTED))[Us])))

                print("Difference between  the predicted and GT IOU of the best segments in each cycle. Median=" + str(np.median(np.abs(np.array(self.St_DifTop))[Us]))+" Mean=" +  str(np.mean(np.abs(np.array(self.St_DifTop))[Us])))

                print("Difference between  the predicted and GT IOU of the selected  by the evaluator (in each cycle).  segments. Median=" + str(np.median(np.abs(np.array(self.St_DifSelected))[Us]))+" Mean=" +  str(np.mean(np.abs(np.array(self.St_DifSelected))[Us])))
                print("Category prediction Accuracy Rate =" + str(np.array(self.St_CorrectClass).mean()))

###############################################################################################################################################################
##################################Run prediction for a single file##################################################################################################################################
    def RunSingle(self):
        RetrysLeft = self.NumberRetrysPerImage
        for ii in range(100):

            self.Px,self.Py, self.Images0, self.ROIMask0=self.Reader.LoadNextGivenROI(NewImg=(ii==0),Npoints=self.NumPointerSegmentsBeforeRfinement, MinDist=150,GenStatistics=self.GenerateStatics) # Read Data
            dd, h, w = self.ROIMask0.shape
            self.h = h
            self.w = w
            if len(self.Px)==0:break
            # for  oo in range(3):
            #     print("reader")
            #     self.Reader.DisplayTrainExample(self.Images0[oo],self.ROIMask0[oo],self.ROIMask0[oo],self.PointerMask[oo])
            if ii==0: self.GenerateImageDataMatrix() # Data matrix that will contain generated segments
            self.GenerateSegmenationDataList()

            self.PointerSegmentation() # Generate segments using pointer net
            self.SegmentToUse = np.ones([self.ListSegment.shape[0]], bool)
 #           self.SegmentToUse[:len(self.Px)]=True
            self.EvaluateSegments() # Evaluate segments and assign score using evaluation net
         #   self.DisplaySegmentAndData()
            self.SegmentConsistancyScore(RelWeight=self.SegmentConsistanyWeight*self.Reader.ROIMap.mean(),MaxWeight=1,UpdateIOU=True)#RelWeight=0.25*self.Reader.ROIMap.mean() # Evaluate score based on consistancy between generated segments
#===============================Refine top segments=========================================================================================================
            Refinethresh=np.min([0.8,self.IOU.max()])
          #  Refinethresh = self.ThreshRefine
            self.SegmentToUse=(self.IOU >= Refinethresh)
            self.RemoveIncossitantSegs()
            while self.SegmentToUse.sum()<1:
                Refinethresh-=0.05
                self.SegmentToUse = (self.IOU >= Refinethresh)

            PrevNum=len(self.ListSegment)
            self.RefineSegments()  # refine top segments using refinement net
            self.SegmentToUse = np.ones([self.ListSegment.shape[0]], bool)
            self.SegmentToUse[:PrevNum] = False
            self.EvaluateSegments() # Evaluate refined net
#------------------------------Statistics generation------------------------------------------------------------------
            if self.GenerateStatics:
                self.GetSegGTData(AddToGeneralStatistics=True)
#--------------------------------------------------------------------------------------------------------
            if self.MustRefine==True: self.IOU[self.ListX>0]=0 # Add only refined segments by setting the none refined segmet score to zero
          #  SegIOU = np.max(self.IOU)
#.............Check if any selected segment pass threshold...............................................

            if (self.IOU>self.ThreshAccept).sum()==0:
                    RetrysLeft -= 1  # reduce number of retry by one
                    if RetrysLeft < 0:  break # if no more retry finish with file
                    continue

#==========================Classify and add all selected segments that pass thresh====================================
 #******************************************************************
            Refinethresh = np.min([0.8, self.IOU.max()])
            #  Refinethresh = self.ThreshRefine
            self.SegmentToUse = (self.IOU >= Refinethresh)
            self.RemoveIncossitantSegs() # Remove segments which are inconsistant with each other
            while self.SegmentToUse.sum() < 1:
                Refinethresh -= 0.05
                self.SegmentToUse = (self.IOU >= Refinethresh)
#************************************************************
            self.RemoveOverlapRegion() # Remove overlaping regions between selected segments
            NumAdded=0
            for iseg in range(0,len(self.IOU)):#(PrevNum,len(self.IOU))
                if  self.IOU[iseg]<self.ThreshAccept or self.SegmentToUse[iseg]==False : continue # Check
                SelectedSeg=self.ListSegment[iseg]
                SegIOU=self.IOU[iseg]

#--------------------Classify-------------------------------------------------------------
                name, isthing, Pred, Consistancy = self.ClassifySingleSegMultiSize(SelectedSeg, ReSizes=[0.6,0.8,1,1.15, 1.3,1.45]) # Classify segment using classification net, in multisize image and return best class and consitancy score between them
             #   print(Consistancy)
                if Consistancy<self.ClassificationConsistancyThresh:
                    self.Reader.ROIMap[SelectedSeg == 1] = 0
                    continue

    # ===================Update maps add segment==================================================================
                #self.SegmentToUse[:] = False
                # self.SegmentToUse[np.argmax(self.IOU)] = True
                self.AddSegmentToMap(SelectedSeg,SegIOU)
                NumAdded+=1
    #--------------------------------------------------------------------------------------------------
                if isthing: self.FinalSegMapOnlyThings[SelectedSeg > 0] = self.FinalSegMapOnlyThings.max() + 1
                self.IsThingMap[SelectedSeg > 0] = isthing
                self.CategoryMap[SelectedSeg > 0] = Pred
    # =======================Statitics Collection===============================================================================
                if self.GenerateStatics:
                    if self.IsLabeledGT[iseg] and (not self.IsCrowdGT[iseg]) and np.max(self.IOU_GT) > -0.5:
                        self.St_SelectedSg_IsThing.append(self.IsThingGT[iseg])
                        self.St_SelectedSgIOU.append(self.IOU_GT[iseg])
                        self.St_TopSgIOU.append(np.max(self.IOU_GT))
                        self.St_DifIOU_TOP_SELECTED.append(np.max(self.IOU_GT) - self.IOU_GT[iseg])
                        self.St_DifTop.append(self.IOU[np.argmax(self.IOU_GT)] - np.max(self.IOU_GT))
                        self.St_DifSelected.append(SegIOU - self.IOU_GT[iseg])
                        self.St_CorrectClass.append(Pred == self.ClassGT[iseg])
    #=========================If no segment pass update number of retry====================================================
            if (NumAdded==0):
                    RetrysLeft -= 1  # reduce number of retry by one
                    if RetrysLeft < 0:  break # if no more retry finish with file
          #  self.DisplaySegmentation()
#==================================================================================================================================
        self.SaveSegmentation()
            # AA=np.ones([PointerMask.shape[1],PointerMask.shape[2]])
            # for ff in range(Images.shape[0]):
            #        Images[ff, :, :, 0]*=1-ROIMask[ff]
            #        misc.imshow(PointerMask[ff])
            #        misc.imshow(Images[ff])
            #        misc.imshow(ROIMask[ff])
            #        AA+=PointerMask[ff]
            #       misc.imshow(AA)

###################################################################################################################################################3
##########################################Generate Data structure for segmentation #################################################################################################################
    def GenerateSegmenationDataList(self):
        h=self.h
        w=self.w
        self.ListSegment = np.zeros([0, h, w], np.uint8)  # List of all segment predictions all predictions will add to this
        self.ListProb = np.zeros([0, h, w], np.float)
        self.ListX = np.zeros([0], np.float)
        self.ListY = np.zeros([0], np.float)
        self.ListSegmentSizes = np.zeros([0], np.float)
        self.ListImages = np.zeros([0, h, w, 3], np.float)
        self.SegmentToUse = np.zeros([1000, h, w, 3], bool)
        self.IOU=[]
#####################################################################################################################################################
###################################################Generate data structure for output annotation##############################################################################################################
    def GenerateImageDataMatrix(self):
        h = self.h
        w = self.w
        self.BatchSize0 = int(np.floor(self.MaxPixelsInBatch / h / w))
        self.IOUMap = np.zeros([h, w], np.float32)  # map of IOU on accepted selected segments in the image
        self.FinalSegMap = np.zeros([h, w], np.uint8)  # Map segmentation instances
        self.FinalSegMapOnlyThings = np.zeros([h, w], np.uint8)  # Map segmentation instances
        self.CategoryMap = np.zeros([h, w], np.uint8)
        self.IsThingMap = np.zeros([h, w], np.uint8)
#######################################################################################################################################################
###################################################Apply pointer net to generate segments#############################################################################################################
    def PointerSegmentation(self):
        # -------------------------Apply predictions in multiscale-----------------------------------------
        for rsz in self.SegResizeRates:
            BatchSize = int(np.floor(self.BatchSize0 / rsz / rsz))
            # -------------------------Generate segment predictions-----------------------------------------------------------------------------------
            Images = RF.ResizeImg(self.Images0, rsz)
            ROIMask = RF.ResizeMask(self.ROIMask0, rsz)
            PointerMask = RF.GeneratePointerMask(self.Px, self.Py, Images.shape[1], Images.shape[2], rsz)
            for uu in range(0, len(self.Py),BatchSize):  # Make prediction for all batch (if batch to big for memory do it in parts)
                Bsize = np.min([self.NumPointerSegmentsBeforeRfinement - uu, BatchSize])
                with torch.autograd.no_grad():
                    Prob, PredLb = self.NetPointerSeg.forward(Images=Images[uu:uu + Bsize],Pointer=PointerMask[uu:uu + Bsize],ROI=ROIMask[uu:uu + Bsize],TrainMode=False)  # Run net inference and get prediction
               # PredSeg = PredLb.data.cpu().numpy()
                MaskRes, ProbRes = RF.ResizeProb(Prob[:, 1].data.cpu().numpy(), self.h,self.w)  # Resize Prediction to original size
                self.ListSegment = np.concatenate([self.ListSegment, MaskRes], axis=0)
                self.ListProb = np.concatenate([self.ListProb, ProbRes], axis=0)
                self.ListX = np.concatenate([self.ListX, self.Px[uu:uu + Bsize]], axis=0)
                self.ListY = np.concatenate([self.ListY, self.Py[uu:uu + Bsize]], axis=0)
                self.ListSegmentSizes = np.concatenate([self.ListSegmentSizes, np.ones([Bsize]) * rsz], axis=0)
                self.ListImages = np.concatenate([self.ListImages, Images[uu:uu + Bsize]], axis=0)
##################################################################################################################################################################################################
###################################################Apply evaluator net to generate IOU score per segment###############################################################################################################################################
    def EvaluateSegments(self):
        IOUsPersSize = np.zeros([self.SegmentToUse.sum(), len(self.EvalResizeRates)])
        for ir, rsz in enumerate(self.EvalResizeRates):
            Images = RF.ResizeImg(self.ListImages[self.SegmentToUse], rsz)
            ListMask, ProbRes = RF.ResizeProb(self.ListProb[self.SegmentToUse], int(self.h * rsz), int(self.w * rsz))

            with torch.autograd.no_grad(): IOU = self.NetIOUEval.forward(Images=Images[:ListMask.shape[0]], Segment=ListMask, TrainMode=False)  # Eval all predictions
            IOU = IOU.data.cpu().numpy()  # List estimated IOU for each segment
            IOUsPersSize[:, ir] = IOU
        # for oo in range(3):
        #     print(IOU[oo])
        #     Reader.DisplayTrainExample(Images[oo], ROIMask[oo], ListSegment[oo], PointerMask[oo])
        IOU = IOUsPersSize
        #      IOU[IOU>1]=1.0
        #      IOU[IOU<0]=0.0
        if len(self.SegmentToUse)>len(self.IOU):
            self.IOU=np.concatenate([self.IOU,-np.ones([len(self.SegmentToUse)-len(self.IOU)])])
        self.IOU[self.SegmentToUse] = np.mean(IOU, axis=1)
 #       Images = self.ListImages
#  print("print IOU before ref:"+str(np.max(IOU)))
#############################################################################################################################################

#############################################Apply refinement net to improve segment########################################################################################
    def RefineSegments(self):
        SegmentToRefine=self.ListSegment[self.SegmentToUse]
        for uu in range(0, SegmentToRefine.shape[0], self.BatchSize0):
            Bsize = np.min([SegmentToRefine.shape[0] - uu, self.BatchSize0])
            SegRefined = SegmentToRefine[uu:uu + Bsize]
            for irf in range(2):
                with torch.autograd.no_grad():
                        Prob, PredLb = self.NetRefine.forward(Images=self.ListImages[uu:uu+Bsize],InMask=SegRefined, TrainMode=False)
                SegRefined = PredLb.data.cpu().numpy()
           # MaskRes, ProbRes = RF.ResizeProb(Prob[:, 1].data.cpu().numpy(), self.h,self.w)  # Resize Prediction to original size
            self.ListSegment = np.concatenate([self.ListSegment, SegRefined], axis=0)
            self.ListProb = np.concatenate([self.ListProb, Prob[:, 1].data.cpu().numpy()], axis=0)
            self.ListX = np.concatenate([self.ListX, -1*np.ones([Bsize])], axis=0)
            self.ListY = np.concatenate([self.ListY, -1*np.ones([Bsize])], axis=0)
            self.ListSegmentSizes = np.concatenate([self.ListSegmentSizes, np.ones([Bsize]) ], axis=0)
            self.ListImages = np.concatenate([self.ListImages, self.Images0[:Bsize]], axis=0)
###################################################################################################################################################
################################################################################################################################################
################################Add segment to annotation mask and remove it from ROI mask################################################################################################################
    def AddSegmentToMap(self,Seg,SegIOU):
        self.Reader.ROIMap[Seg == 1] = 0  # Remove selected segment from ROI mask
        Seg *= (self.IOUMap < SegIOU)  # ignore selected segment in place were its overlap existing segments and have lower IOU values
        self.FinalSegMap[Seg > 0] = self.FinalSegMap.max() + 1  # add selected segment to the final predicted map
        self.IOUMap[Seg > 0] = SegIOU
############################################################################################################################################

##################################Classifiy segment using classification net in multi image size return average class prediction and consistancy of predictions b############################################################################################
    def ClassifySingleSegMultiSize(self,SelectedSeg,ReSizes):
            ProbSum=[]
            AllPred=np.zeros([0])
            for Sz in ReSizes:
                h = int(self.h * Sz)
                w = int(self.w * Sz)
                Im = cv2.resize(self.ListImages[0],(w,h),interpolation=cv2.INTER_LINEAR)
                Seg = cv2.resize(SelectedSeg, (w,h), interpolation=cv2.INTER_NEAREST)

                ImM=np.fliplr(Im)
                SegM=np.fliplr(Seg)

                Im=np.concatenate([np.expand_dims(Im, axis=0),np.expand_dims(ImM, axis=0)],axis=0)
                Seg=np.concatenate([np.expand_dims(Seg, axis=0),np.expand_dims(SegM, axis=0)],axis=0)
                # =====================================================================================================
                # print(Im.shape)
                # for i in range(Im.shape[0]):
                #     Im[i,:,:,0]*=1-Seg[i]
                #     Im[i, :, :, 1] *= 1-Seg[i]
                #     misc.imshow(Im[i])

                # =======================================================================================================
                with torch.autograd.no_grad():
                     Prob, Lb = self.NetClassification.forward(Im.astype(float), ROI=Seg.astype(float), EvalMode=True)
                     #Prob, Lb = self.NetClassification.forward(Im.astype(float), ROI=Seg.astype(float),TrainMode=False)  # Run net inference and get prediction
                     Lb= Lb.data.cpu().numpy()
                     AllPred=np.concatenate([AllPred,Lb],axis=0)
                     Prob = Prob.data.cpu().numpy().sum(axis=0)
                     if len(ProbSum)==0:
                         ProbSum=Prob
                     else:
                         ProbSum+=Prob


            Pred=np.argmax(ProbSum,axis=0)
            Consistancy=(AllPred==Pred).mean()
            name, isthing = self.Reader.GetCategoryData(Pred)
            #print(name)
            return name, isthing, Pred,Consistancy
#################################################################################################################################
#################################################################################################################################
    def CheckIfExists(self):
        return os.path.exists(self.OutVizFolder + "/" + self.Reader.Img_name)
#####################################Save annotation to file##########################################################################################
    def SaveSegmentation(self):
        ################################################################################################################################################################################################
        COCO2Chanels = np.zeros(self.ListImages[0].shape, dtype=np.uint8)
        COCO2Chanels[:, :, 2] = self.CategoryMap
        COCO2Chanels[:, :, 1] = self.FinalSegMapOnlyThings
        cv2.imwrite(self.Out2ChannelsCOCOFolder + "/" + self.Reader.Img_name.replace(".jpg", ".png"), COCO2Chanels)
        ##############################################################################################################
        # cv2.imwrite(self.OutSegFolder + "/" + self.Reader.Img_name.replace(".jpg", ".png"), self.FinalSegMap)
        # ###########################################################################################################################
        # COCO2ChanelsStuffInstance = np.zeros(self.ListImages[0].shape, dtype=np.uint8)
        # COCO2ChanelsStuffInstance[:, :, 2] = self.CategoryMap
        # COCO2ChanelsStuffInstance[:, :, 1] = self.FinalSegMap
        # cv2.imwrite(self.Out2ChannelsCOCOStuffInstanceFolder + "/" + self.Reader.Img_name.replace(".jpg", ".png"),COCO2ChanelsStuffInstance)
        ######################################################################################################################################################################

        SegViz = np.zeros(self.ListImages[0].shape, dtype=np.uint8)
        SegViz[:, :, 0] = np.uint8((self.FinalSegMap) * 21 % 255)
        SegViz[:, :, 1] = np.uint8(((self.FinalSegMap) * 67) % 255)
        SegViz[:, :, 2] = np.uint8(((self.FinalSegMap) * 111) % 255)
        cv2.imwrite(self.OutVizFolder + "/" + self.Reader.Img_name, np.concatenate([SegViz, self.ListImages[0]], axis=0))
##################################################################################################################################
    def DisplaySegmentation(self):
        ################################################################################################################################################################################################
        cv2.destroyAllWindows()
        cv2.imshow("roi", self.Reader.ROIMap)
        SegViz = np.zeros(self.ListImages[0].shape, dtype=np.uint8)
        SegViz[:, :, 0] = np.uint8((self.FinalSegMap) * 21 % 255)
        SegViz[:, :, 1] = np.uint8(((self.FinalSegMap) * 67) % 255)
        SegViz[:, :, 2] = np.uint8(((self.FinalSegMap) * 111) % 255)
        cv2.imshow("vis", np.concatenate([SegViz, self.ListImages[0]], axis=1).astype(np.uint8))
        cv2.waitKey()
        cv2.destroyAllWindows()
#==================================================================================================================================================
    def DisplaySegmentAndData(self):
            cv2.imshow("ROI MAP",self.ROIMask0[0]);cv2.waitKey()
            for i in range(self.ListSegment.shape[0]):
                if self.SegmentToUse[i]==True:
                    if self.IOU.shape[0]>i:
                        print("Prediction IOU = "+str(self.IOU[i]))
                    else: print("No IOU")
                    if self.ListX.shape[0]>i:
                        print("X="+str(self.ListX[i])+",   Y="+str(self.ListY[i]))
                    else: print("ListX")
                    seg=self.ListSegment[i].copy()
                    img=self.ListImages[i].copy()
                    img[:,:,0]*=1-seg
                    img[:,:,2]*=self.ROIMask0[0]
                    img=img[...,::-1]
                    cv2.imshow("Seg",seg*200)
                    cv2.imshow("ROI",self.ROIMask0[0]*200)
                    cv2.imshow("Img",img.astype(np.uint8));cv2.waitKey();
                    cv2.destroyAllWindows()
###############################Get GT data from gt annotation file (coco panoptic style) optional used only when Generate statictics is TRUE##################################################################################################
    def GetSegGTData(self,AddToGeneralStatistics=True):
        self.IOU_GT = []
        self.PrecisionGT = []
        self.IOU_DIF = []
        self.ListSegmentGT = np.zeros([0,self.h,self.w])
        self.IsThingGT = []
        self.ClassGT = []
        self.IsCrowdGT = []
        self.IsLabeledGT = []

        for ii,Seg in enumerate(self.ListSegment):
            IOU,Precision,Recall,IsThing,Mask,CatID,IsCrowd,IsLabeled=self.Reader.FindCorrespondingSegmentMaxIOU(Seg)
            self.IOU_GT.append(IOU)
            self.IsThingGT.append(IsThing)
            self.ClassGT.append(CatID)
            self.PrecisionGT.append(Precision)
            self.IsCrowdGT.append(IsCrowd)
            self.IsLabeledGT.append(IsLabeled)
            self.ListSegmentGT=np.concatenate([self.ListSegmentGT,np.expand_dims(Mask.astype(float),axis=0)],axis=0)
            if self.IOU.shape[0]>ii: self.IOU_DIF.append(np.abs(self.IOU[ii]-IOU))
            if AddToGeneralStatistics and IsLabeled and not IsCrowd:
                self.St_IOU_GT.append(IOU)
                self.St_IsThing.append(IsThing)
                self.St_PrecisionGT.append(Precision)
                if self.IOU.shape[0] > ii:
                    self.St_IOU_DIF.append(np.abs(self.IOU[ii]-IOU))
                    self.St_IOU_Pred.append(self.IOU[ii])
###############################################################################################################################################################
#################################Generate consitancy score for segment based on how much its consitant with other predicted segmens##############################################################################################################################
# Check if segment is consistant with other segment and generate consistantcy score that can be combine
    def SegmentConsistancyScore(self,RelWeight=0.3,MaxWeight=1,UpdateIOU=True): #
          # self.ListSegment[self.SegmentToUse]
           self.WeightSegConsistancy=np.zeros(len(self.ListSegment))
           self.SegConsistancy = np.zeros(len(self.ListSegment))
           for i in range(len(self.ListSegment)):
               isum = self.ListSegment[i].sum()
               for f in range(i+1,len(self.ListSegment)):
                   fsum=self.ListSegment[f].sum()
                   inter=(self.ListSegment[f]*self.ListSegment[i]).sum()
                   IOU=inter/(fsum+isum-inter+0.000000001)
                   # rf=inter/fsum
                   # ri=inter/isum
                   #minr=np.min([rf,ri])
                   self.SegConsistancy[f] += IOU*IOU
                   self.SegConsistancy[i] += IOU*IOU
                   self.WeightSegConsistancy[i] += IOU
                   self.WeightSegConsistancy[f] += IOU
#*****************************************************************************************************
                   # print("*******************************************************************************")
                   # print(IOU)
                   # #print(minr)
                   # Im=self.Images0[0].copy()
                   # Im[:,:,0]*=1-self.ListSegment[f]
                   # Im[:, :, 1] *= 1 - self.ListSegment[i]
                   # misc.imshow(Im)

#****************************************************************************************
           self.NewScore = np.zeros(len(self.ListSegment))
           for i in range(len(self.ListSegment)):
        #         print("first iou="+str(self.IOU[i]))#****************************************
                 Weight=np.min([self.WeightSegConsistancy[i]*RelWeight,MaxWeight])
                 self.NewScore[i]=(self.IOU[i]+Weight*self.SegConsistancy[i]/(self.WeightSegConsistancy[i]+0.0000001))/(1+Weight)
                 if UpdateIOU: self.IOU[i]=self.NewScore[i]
 #*****************************************************************************************
                 # print("weight="+str(Weight)+" consscore="+str(self.SegConsistancy[i]/(self.WeightSegConsistancy[i]+0.000001)) +"  final iou="+str(self.IOU[i]))
                 # Im = self.Images0[0].copy()
                 # Im[:, :, 0] *= 1 - self.ListSegment[i]
                 # Im[:, :, 1] *= 1 - self.ListSegment[i]
                 # misc.imshow(Im)
                 #
################################################################################################################################33333
##################################Remove different segment with HIGH OVERLAP (so the same segment will not be added twice with different variation)#######################################################################################################
    def RemoveIncossitantSegs(self,ThreshIOU=0.1): # Remove from list segments that overlap with Higher IOU segments in given thresh
       # self.SegmentToAdd=self.SegmentToUse.copy()
        SortInd=np.argsort(-self.IOU)
        for i in SortInd:
               if self.SegmentToUse[i]==0: continue
               isum=self.ListSegment[i].sum()
               for f in SortInd:
                   if f==i or self.SegmentToUse[f] == False: continue
              #     print("remove inconsistany")
               #    misc.imshow(self.Images0[0])
               #    misc.imshow(np.concatenate([self.ListSegment[i], self.ListSegment[f]], axis=1) * 200)  # *************
                   fsum = self.ListSegment[f].sum()
                   inter = (self.ListSegment[i]*self.ListSegment[f]).sum()
                   iou=inter/(fsum+isum-inter)
                   if iou>ThreshIOU:
                       self.SegmentToUse[f]=False
                      # print("removed")

##############################################################################################################
##############################Remove overlap regions between accepted segment###########################################################
    def RemoveOverlapRegion(self):
        for i in range(len(self.ListSegment)):
            if self.SegmentToUse[i] == 0: continue
            isum = self.ListSegment[i].sum()
            for f in range(i+1,len(self.ListSegment)):
                if self.SegmentToUse[f] == False: continue

                inter = (self.ListSegment[i] * self.ListSegment[f])
                if inter.sum()>0:
                    fsum = self.ListSegment[f].sum()
                    # if fsum>isum:
                    #     self.ListSegment[f][inter>0] = 0
                    # else:
                    #     self.ListSegment[i][inter > 0] = 0

                    # if self.IOU[f]<self.IOU[i]:
                    #     self.ListSegment[f][inter>0] = 0
                    # else:
                    #     self.ListSegment[i][inter > 0] = 0

                    print("inter="+str(inter.sum()))
      #************************************************************************
                  # # misc.imshow(self.Images0[0])
                  #   I=self.Images0[0]
                  #   I[:,:,0]*=1-self.ListSegment[i]
                  #   I[:,:,1]*=1-self.ListSegment[f]
                  #   cv2.imshow("ee",I.astype(np.uint8))  # ********************
                  #   cv2.imshow("bb",np.concatenate([self.ListSegment[i], self.ListSegment[f]], axis=1).astype(np.uint8)*100 )
                  #   cv2.waitKey()
                  #   cv2.destroyAllWindows()
    #************************************************************************
                    if self.IOU[f]/fsum<self.IOU[i]/isum:
                        self.ListSegment[f][inter>0] = 0
                   #     misc.imshow(self.ListSegment[f])
                    else:
                        self.ListSegment[i][inter>0] = 0
                    #    misc.imshow(self.ListSegment[i])
                    # misc.imshow(self.Images0[0])
                # misc.imshow(np.concatenate([self.ListSegment[i], self.ListSegment[f]], axis=1) * 200)#******************

#########################################################################################
        # #########################################################################################
        # def MergeOverlapRegions(self, ThreshIOU):
        #     # self.SegmentToAdd=self.SegmentToUse.copy()
        #     SortInd = np.argsort(-self.IOU)
        #     Weights = self.IOU.copy()  # *****Weights=np.ones([len[self.IOU]])*************************
        #     for i in SortInd:
        #         if self.SegmentToUse[i] == 0: continue
        #         isum = self.ListSegment[i].sum()
        #         self.ListProb[i] *= Weights[i]
        #         for f in SortInd:
        #             if f <= i or self.SegmentToUse[f] == False: continue
        #             # misc.imshow(self.Images0[0])
        #             # misc.imshow(np.concatenate([self.ListSegment[i], self.ListSegment[f]], axis=1) * 200)  # *************
        #             # misc.imshow(self.ListSegment[i]*70+self.ListSegment[f]*120)
        #             fsum = self.ListSegment[f].sum()
        #             inter = (self.ListSegment[i] * self.ListSegment[f]).sum()
        #             iou = inter / (fsum + isum - inter)
        #
        #             if iou > ThreshIOU:
        #                 self.SegmentToUse[f] = False
        #                 # print(iou)
        #                 # print(str(i)+")"+str(Weights[i]))
        #                 # print(str(f) + ")" + str(Weights[f]))
        #                 # misc.imshow(np.concatenate([self.ListProb[i], self.ListProb[f]], axis=1) )
        #                 # misc.imshow(self.Images0[0])
        #
        #                 self.ListProb[i] += self.ListProb[f] * Weights[f]
        #                 # misc.imshow(self.ListProb[i])
        #                 Weights[i] += Weights[f]
        #
        #     for i in SortInd:
        #         if self.SegmentToUse[i] == 0: continue
        #         self.ListProb[i] /= Weights[i]
        #         self.ListSegment[i] = (self.ListProb[i] > 0.5).astype(float)















