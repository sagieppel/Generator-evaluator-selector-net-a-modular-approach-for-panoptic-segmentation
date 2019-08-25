
#Reader for the coco panoptic data set for pointer based image segmentation
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
############################################################################################################
def rgb2id(color): # Convert annotation map from 3 channel RGB to instance
    if isinstance(color, np.ndarray) and len(color.shape) == 3:
        if color.dtype == np.uint8:
            color = color.astype(np.uint32)
        return color[:, :, 0] + 256 * color[:, :, 1] + 256 * 256 * color[:, :, 2]
    return color[0] + 256 * color[1] + 256 * 256 * color[2]
#########################################################################################################################
#########################################################################################################################
class Matcher:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir,AnnotationDir,DataFile, AnnotationFileType="png", ImageFileType="jpg",UnlabeledTag=0):

        self.ImageDir=ImageDir # Image dir
        self.AnnotationDir=AnnotationDir # File containing image annotation
        self.AnnotationFileType=AnnotationFileType # What is the the type (ending) of the annotation files
        self.ImageFileType=ImageFileType # What is the the type (ending) of the image files
        self.DataFile=DataFile # Json File that contain data on the annotation of each image
        self.UnlabeledTag=UnlabeledTag # Value of unlabled region in the annotation map (usually 0)
        self.ReadStuff = True # Read things that are not instace object (like sky or grass)
        self.SplitThings = False#True # Split instance of things (object) to connected component region and use each connected region as an instance
        self.SplitStuff = True # Split instance of things (object) to connected component region and use each connected region as instance
        self.SplitCrowd = True # Split areas marked as Crowds using connected componennt
        self.IgnoreCrowds = False # Ignore areas marked as crowd
        self.MinSegSize = 400


      #  self.PickBySize = False  # Pick instances of with probablity proportional to their sizes if false all segments are picked with equal probablity
#........................Read data file................................................................................................................
        with open(DataFile) as json_file:
            self.AnnData=json.load(json_file)

#-------------------Get All files in folder--------------------------------------------------------------------------------------
        self.FileList=[]
        for FileName in os.listdir(AnnotationDir):
            if AnnotationFileType in FileName:
                self.FileList.append(FileName)
        # if self.suffle:
        #     random.shuffle(self.FileList)
        # if TrainingMode: self.StartLoadBatch()
##############################################################################################################################################
#Get annotation data for specific nmage from the json file
    def GetAnnnotationData(self, AnnFileName):
            for item in self.AnnData['annotations']:  # Get Annotation Data
                if (item["file_name"] == AnnFileName):
                    return(item['segments_info'])
############################################################################################################################################
#Get information for specific catagory/Class id
    def GetCategoryData(self,ID):
                for item in self.AnnData['categories']:
                    if item["id"]==ID:
                        return item["name"],item["isthing"]
##########################################################################################################################################3333
#Split binary mask correspond to a singele segment into connected components
    def GetConnectedSegment(self, Seg):
            [NumCCmp, CCmpMask, CCompBB, CCmpCntr] = cv2.connectedComponentsWithStats(Seg.astype(np.uint8))  # apply connected component
            Mask=np.zeros([NumCCmp,Seg.shape[0],Seg.shape[1]],dtype=bool)
            BBox=np.zeros([NumCCmp,4])
            Sz=np.zeros([NumCCmp],np.uint32)
            for i in range(1,NumCCmp):
                Mask[i-1] = (CCmpMask == i)
                BBox[i-1] = CCompBB[i][:4]
                Sz[i-1] = CCompBB[i][4] #segment Size
            return Mask,BBox,Sz,NumCCmp-1
#################################################################################################################################################
#############################################################################################################################
#########################################################################################################################################
# # Given predicted segment (SegMask)  and list of GT segments (self.Sgs)
# Find the GT segment with the highest IOU correlation  to  predictedSegMask
# USed for the evaluation of the serial region by region full image segmentation mode
    def FindCorrespondingSegmentMaxIOU(self,SegMask):
        MaxIOU=-10
        TopSeg=0
        MaxPrec=-10
        for seg in self.Sgs:
            IOU=(seg["Mask"] * SegMask).sum() / (seg["Mask"].sum() + SegMask.sum() - (seg["Mask"] * SegMask).sum())
            Prec=(seg["Mask"] * SegMask).sum() / SegMask.sum()
            if IOU>MaxIOU:
                MaxIOU=IOU
                TopSeg=seg
            if Prec>MaxPrec:
                MaxPrec=Prec
        IOU = (TopSeg["Mask"] * SegMask).sum() / (TopSeg["Mask"].sum() + SegMask.sum() - (TopSeg["Mask"] * SegMask).sum())
        Precision = (TopSeg["Mask"] * SegMask).sum() / SegMask.sum()
        Recall = (TopSeg["Mask"] * SegMask).sum() / TopSeg["Mask"].sum()
        if not TopSeg["IsLabeled"]: SegType = "Unlabeled"
        elif TopSeg["IsCrowd"]:SegType = "crowd"
        elif TopSeg["IsThing"]: SegType = "thing"
        else: SegType = "stuff"
        return IOU,Precision,Recall,SegType,TopSeg["Mask"].astype(float), MaxPrec==Precision # The last is check if the segment is consistantly the best match in both iou and precision
###############################################################################################################################

######################################################################################################
#Generate list of all  segments in the image
# Given the annotation map a json data file create list of all segments and instance with info on each segment
#--------------------------Generate list of all segments--------------------------------------------------------------------------------
    def GeneratListOfAllSegments(self,Ann,Ann_name,AddUnLabeled=False,IgnoreSmallSeg=True):
        AnnList = self.GetAnnnotationData(Ann_name)
        Sgs = []  # List of segments and their info
        SumAreas=0 # Sum areas of all segments up to image
        for an in AnnList:
           # misc.imshow((Ann == an['id']).astype(float))
            an["name"], an["isthing"] = self.GetCategoryData(an["category_id"])
            if an["iscrowd"]:
                    Ann[Ann == an['id']] = self.UnlabeledTag
                    continue
            if  (an["isthing"]==False and self.SplitStuff): #Things are objects that have instances
                TMask, TBBox, TSz, TNm = self.GetConnectedSegment(Ann == an['id']) # Split to connected components
                for i in range(TNm):
                    seg={}
                    seg["Mask"]=TMask[i]
                    seg["BBox"]=TBBox[i]
                    seg["Area"]=TSz[i]
                    seg["NumParts"] =TNm
                    seg["IsSplit"]=TNm>1
                    seg["IsThing"]=an["isthing"]
                    seg["Name"]=an["name"]
                    seg["IsCrowd"]=an["iscrowd"]
                    seg["CatId"]=an["category_id"]
                    seg["IsLabeled"] = True
                    SumAreas+=seg["Area"]
                    Sgs.append(seg)
            else: # none object classes such as sky
                    seg = {}
                    seg["Mask"] = (Ann == an['id'])
                    seg["BBox"] = an["bbox"]
                    seg["Area"] = an["area"]
                    seg["NumParts"] = 1
                    seg["IsSplit"] = False
                    seg["IsThing"] = an["isthing"]
                    seg["Name"] = an["name"]
                    seg["IsCrowd"] = an["iscrowd"]
                    seg["CatId"] = an["category_id"]
                    seg["IsLabeled"]=True
                    SumAreas += seg["Area"]
                    Sgs.append(seg)


        return Sgs,(Ann==self.UnlabeledTag).astype(float)
##################################################################################################################################################
    def CleanerSeletor(self,PredDir,GTDir):
           f=0
           Correct=0
           Wrong=0
           Undefined=0
           ListFile=[]
           for name in os.listdir(PredDir):
                if not ".png" in name: continue
                if ("#V#" in name):
                    continue
                if  ("#WRONG#" in name):
                    continue
                ListFile.append(name)
           random.shuffle(ListFile)

           for name in ListFile: # Get label image name
                if (not ".png" in name) or ("#V#" in name) or  ("#WRONG#" in name) or (not os.path.exists(GTDir + "/" + name)): continue
                Ann_name=name[:name.find("#jpg")]+".png"
                f+=1
                print(str(f)+")"+name+"   Correct="+str(Correct)+"   Wrong="+str(Wrong)+"   Undefined="+str(Undefined)+"   PC="+str(Correct/(Wrong+Correct+Undefined+0.0001)))
                Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name)  # Load Annotation
                #misc.imshow(Ann)

                # Img0 = cv2.imread(self.ImageDir + "/" + Ann_name.replace(".png",".jpg"))
                # misc.imshow(Img0)
                Ann = Ann[..., :: -1]
                self.AnnColor=Ann
                Ann=rgb2id(Ann)

                # misc.imshow(Img)
                H,W=Ann.shape
                self.Sgs, undefined = self.GeneratListOfAllSegments(Ann, Ann_name,AddUnLabeled=True,IgnoreSmallSeg=True)
                # misc.imshow(undefined)
                # misc.imshow((Ann==self.UnlabeledTag).astype(float))
                PredAnn= cv2.imread(PredDir + "/" + name,0)

                GtAnn = cv2.imread(GTDir + "/" + name, 0)
                PredAnn=cv2.resize(PredAnn,(Ann.shape[1],Ann.shape[0]),interpolation=cv2.INTER_NEAREST)
                GtAnn = cv2.resize(GtAnn, (Ann.shape[1], Ann.shape[0]), interpolation=cv2.INTER_NEAREST)


                uintersum=(PredAnn*undefined).sum()
                undefinedsum=undefined.sum()
                PredSum=PredAnn.sum()
                uiou=uintersum/(undefinedsum+PredSum-uintersum+0.00000001)
                uprec=uintersum/(PredSum+0.00000001)

                GTinter = (PredAnn * GtAnn).sum()
                GTsum = GtAnn.sum()
                GTiou = GTinter / (GTsum + PredSum - GTinter)
                GTPrec = GTinter / (PredSum+0.0001)
                IOU,Precision,Recall,SegType,Mask, valid=self.FindCorrespondingSegmentMaxIOU(PredAnn)

                # misc.imshow(GtAnn*50+PredAnn*100)
                # misc.imshow(Mask)
                print("iou="+str(IOU)+" gtiou="+str(GTiou)+"  uiou"+str(uiou))
                if  uiou>IOU:# or uprec>Precision or not valid or
                    Undefined+=1
                    print("undefined")
                    os.rename(GTDir + "/" + name, GTDir + "/" + name.replace(".png","#WRONG#.png"))
                    os.rename(PredDir  + "/" + name, PredDir + "/" + name.replace(".png", "#WRONG#.png"))
                elif ((Mask*GtAnn).sum()/(Mask.sum()+GtAnn.sum()-(Mask*GtAnn).sum())).mean()<0.8 or IOU<0.001:
                    Wrong+=1
                    print("wrong")
                    os.rename(GTDir + "/" + name, GTDir + "/" + name.replace(".png", "#WRONG#.png"))
                    os.rename(PredDir + "/" + name, PredDir + "/" + name.replace(".png", "#WRONG#.png"))
                else:
                    Correct+=1
                    print("Correct")
                    os.rename(GTDir + "/" + name, GTDir + "/" + name.replace(".png", "#V#.png"))
                    os.rename(PredDir + "/" + name, PredDir + "/" + name.replace(".png", "#V#.png"))


##################################################################################################################################################
    def Test(self,PredDir,GTDir):
           Remain=0
           Finished=0


           for name in os.listdir(PredDir):
                if not ".png" in name: continue
                if ("#V#" in name) or ("#WRONG#" in name):
                    Finished+=1
                    continue
                else:
                    Remain+=1
               # print(Remain / (Remain + Finished))

           return Remain,Finished








############################################################################################################################################################################################




ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/" # image folder (coco training) train set
AnnotationDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_train2017/" # annotation maps from coco panoptic train set
DataFile="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/panoptic_train2017.json" # Json Data file coco panoptic train set



##############################################################################################################################################################################################
inputMainDir="/scratch/gobi2/seppel/GeneratedPredictions/Test2/"
ListDirs=[]
x = Matcher(ImageDir, AnnotationDir, DataFile)
for InputFolder in os.listdir(inputMainDir):
    if os.path.isdir(inputMainDir+"/"+InputFolder) and os.path.isdir(inputMainDir+"/"+InputFolder+"/Pred/") and os.path.isdir(inputMainDir+"/"+InputFolder+"/GT/"):
        ListDirs.append(inputMainDir+"/"+InputFolder+"/")
random.shuffle(ListDirs)
# for InputFolder in ListDirs:
#     x.CleanerSeletor(InputFolder+"/Pred/",InputFolder+"/GT/")
#-----------------------------------------------------------------------------------
# import torch
# print(torch.__version__)
Remain=0
Finished=0
for InputFolder in ListDirs:
    R,F=x.Test(InputFolder+"/Pred/",InputFolder+"/GT/")
    Remain+=R
    Finished+=F
    print("Remain sum="+str(Remain))
    print("Finished sum=" + str(Finished))
    print(Remain/(Remain+Finished))

