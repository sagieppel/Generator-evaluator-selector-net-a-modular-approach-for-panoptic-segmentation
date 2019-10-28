
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
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir="",AnnotationDir="", DataFile="", AnnotationFileType="png", ImageFileType="jpg",UnlabeledTag=0,Suffle=True):

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
        self.IgnoreCrowds = True # Ignore areas marked as crowd
        self.itr = 0 # Training iteratation
        self.suffle=Suffle # Suffle list of file
        self.AnnData = False
        self.MinSegSize=100
#........................Read data file................................................................................................................
        if not DataFile=="":
           with open(DataFile) as json_file:
               self.AnnData=json.load(json_file)

#-------------------Get All files in folder--------------------------------------------------------------------------------------
        self.FileList=[]
        for FileName in os.listdir(ImageDir):
            if ImageFileType in FileName:
                self.FileList.append(FileName)
        if self.suffle:
            random.shuffle(self.FileList)
##############################################################################################################################################
#Get annotation data for specific nmage from the json file
    def GetAnnnotationData(self, AnnFileName):
            for item in self.AnnData['annotations']:  # Get Annotation Data
                if (item["file_name"] == AnnFileName):
                    return(item['segments_info'])
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
############################################################################################################################
##############################################################################################################################
# Display loaded data on screen (for debuging)
    def DisplayTrainExample(self,Img2,ROI2,Segment2,SelectedPoint2):
        Img=Img2.copy()
        ROI=ROI2.copy()
        Segment=Segment2.copy()
        SelectedPoint=SelectedPoint2.copy()
        misc.imshow(Img)
        SelectedPoint = cv2.dilate(SelectedPoint.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=1)
        Img[SelectedPoint][:]=[255,0,0]
        Img[:, :, 0] = SelectedPoint.astype(np.uint8)*255+ (1-SelectedPoint.astype(np.uint8))*Img[:, :, 0]
        Img[:, :, 1] *= 1-SelectedPoint.astype(np.uint8)
        Img[:, :, 2] *= 1-SelectedPoint.astype(np.uint8)
        Img[ :, :, 0] *= 1-(ROI.astype(np.uint8)-Segment.astype(np.uint8))
        #Img[:, :, 1] += ROI.astype(np.uint8)*40
        Img[ :, :, 2] *= 1 - Segment.astype(np.uint8)

      #  misc.imshow(Img)
        #print(ROI.mean())
        ROI[0,0]=0
        misc.imshow(ROI.astype(float))
        misc.imshow( Segment.astype(float))
        misc.imshow(SelectedPoint.astype(float))
        misc.imshow(Img)
#############################################################################################################################
######################################################################################################
#--------------------Generate list of all segments--------------------------------------------------------------------------------
    def GeneratListOfAllSegments(self,Ann,Ann_name,AddUnLabeled=False,IgnoreSmallSeg=True):
        AnnList = self.GetAnnnotationData(Ann_name)
        Sgs = []  # List of segments and their info
        SumAreas=0 # Sum areas of all segments up to image
        for an in AnnList:
            an["name"], an["isthing"] = self.GetCategoryData(an["category_id"])
           # print(an["iscrowd"])
            if (an["iscrowd"] and self.IgnoreCrowds) or (not an["isthing"] and not self.ReadStuff):
                Ann[Ann == an['id']] = self.UnlabeledTag
                continue
            if (an["isthing"] and self.SplitThings) or (an["isthing"]==False and self.SplitStuff) or (an["iscrowd"] and self.SplitCrowd): #Things are objects that have instances
                TMask, TBBox, TSz, TNm = self.GetConnectedSegment(Ann == an['id']) # Split to connected components
                for i in range(TNm):
                    seg={}
                    seg["Mask"]=TMask[i]
                    seg["BBox"]=TBBox[i]
                    seg["Area"]=TSz[i]
                    if seg["Area"] < self.MinSegSize and IgnoreSmallSeg:
                        Ann[Ann == an['id']] = self.UnlabeledTag
                        continue
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
                    if seg["Area"] < self.MinSegSize and IgnoreSmallSeg: # Ignore very small segments
                        Ann[Ann == an['id']] = self.UnlabeledTag
                        continue
                    seg["NumParts"] = 1
                    seg["IsSplit"] = False
                    seg["IsThing"] = an["isthing"]
                    seg["Name"] = an["name"]
                    seg["IsCrowd"] = an["iscrowd"]
                    seg["CatId"] = an["category_id"]
                    seg["IsLabeled"]=True
                    SumAreas += seg["Area"]
                    Sgs.append(seg)

        if AddUnLabeled: #Add unlabeled region as additional segments
            TMask, TBBox, TSz, TNm = self.GetConnectedSegment(Ann == self.UnlabeledTag)  # Split to connected components
            for i in range(TNm):
                seg = {}
                seg["Mask"] = TMask[i]
                seg["BBox"] = TBBox[i]
                seg["Area"] = TSz[i]
                seg["NumParts"] = TNm
                seg["Name"] ="unlabeled"
                seg["CatId"] = self.UnlabeledTag
                seg["IsLabeled"] = False
                Sgs.append(seg)
        return Sgs,SumAreas
##################################################################################################################################################

##################################################################################################################################################
    def LoadNextGivenROI(self,NewImg=True,Npoints=-1,MinDist=50,MaxBatchPixels=3*800*800,GenStatistics=False):
        # This function is used serially on the same image cascade full image segmentation
        # Pick random point on a given ROI mask
        # return the point the ROI mask and the image
#-------------If new image load the next image and generate annotation mask--------------------------------------------------
            if NewImg:
                Img_name=self.FileList[self.itr]
                self.Img_name=Img_name
                self.ImgName=Img_name
                Img = cv2.imread(self.ImageDir + "/" + Img_name)  # Load Image
                Img = Img[...,:: -1]
                if (Img.ndim == 2):  # If grayscale turn to rgb
                    Img = np.expand_dims(Img, 3)
                    Img = np.concatenate([Img, Img, Img], axis=2)
                Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more


                self.H,self.W=Img.shape[0:2]
                H=self.H
                W=self.W
                self.ROIMap=np.ones([H,W]) # Generate ROI mask that cover the full image
                ROIMap=self.ROIMap
                if Npoints==-1: Npoints=int(np.floor(MaxBatchPixels/H/W))
                self.BImgs =  np.ones([Npoints,H,W,3])
                for i in range(Npoints): self.BImgs[i]=Img
                self.BROIMask = np.ones([Npoints,H,W])
                self.Npoints=Npoints
                # self.BROIMask = np.expand_dims(ROIMap, axis=0).astype(np.float32)
                # --------------------------------For Statitics collection---------------------------------------------------------------------------------
                if GenStatistics:
                    Ann = cv2.imread(self.AnnotationDir + "/" + self.FileList[self.itr].replace(".jpg",".png"))  # Load Annotation
                    Ann = Ann[..., :: -1]
                    Ann = rgb2id(Ann)
                    self.Sgs, SumAreas = self.GeneratListOfAllSegments(Ann, self.FileList[self.itr].replace(".jpg",".png"))
#-----------------Load--------------------------------------------------------------------------------------------------------
            else:
                 ROIMap = cv2.erode(self.ROIMap.astype(np.uint8),np.ones((2,2),np.uint8),iterations=3)
                 H = self.H
                 W = self.W


#------------------------------------------------------------------------------------------------------------

#Select poinre points
            x=[]
            y=[]


            if ROIMap.mean() > 0.01:
                while (len(x)<Npoints):
                    while(True):
                        tx = np.random.randint(W)
                        ty = np.random.randint(H)
                        if (ROIMap[ty, tx]) == 1: break
                    ChkDst=True
                    for tt in range(len(x)):
                        if np.sqrt(np.power(x[tt]-tx,2)+np.power(y[tt]-ty,2))<MinDist:
                            ChkDst = False
                            break
                    if ChkDst:
                        x.append(tx)
                        y.append(ty)
                    else:
                        MinDist-=1
#---------------------------GenerateOutput-----------------------------------------------------------------------------
            Npoints=len(x)
            # PointerMask = np.zeros([Npoints, H, W])
            for i in range(Npoints):
                 self.BROIMask[i] = self.ROIMap
            #     PointerMask[i,y[i],x[i]]=1

            return  x,y,self.BImgs ,self.BROIMask
#########################################################################################################################################
#########################################################################################################################################
# # Given predicted segment (SegMask)  and list of GT segments (self.Sgs)
# Find the GT segment with the highest IOU correlation  to  predictedSegMask
# USed for the evaluation of the serial region by region full image segmentation mode
    def FindCorrespondingSegmentMaxIOU(self,SegMask):
        MaxIOU=-1
        TopSeg=0
        for seg in self.Sgs:
            IOU=(seg["Mask"] * SegMask).sum() / (seg["Mask"].sum() + SegMask.sum() - (seg["Mask"] * SegMask).sum()+0.00001)
            if IOU>MaxIOU:
                MaxIOU=IOU
                TopSeg=seg
        IOU = (TopSeg["Mask"] * SegMask).sum() / (TopSeg["Mask"].sum() + SegMask.sum() - (TopSeg["Mask"] * SegMask).sum()+0.00000000001)
        Precision = (TopSeg["Mask"] * SegMask).sum() / (SegMask.sum()+0.00000000001)
        Recall = (TopSeg["Mask"] * SegMask).sum() / (TopSeg["Mask"].sum()+0.00000000001)
        if not TopSeg["IsLabeled"]: SegType = "Unlabeled"
        elif TopSeg["IsCrowd"]:SegType = "crowd"
        IsLabeled = not (seg["CatId"] == self.UnlabeledTag)

        return IOU,Precision,Recall,TopSeg["IsThing"],TopSeg["Mask"].astype(float),TopSeg["CatId"],TopSeg["IsCrowd"],IsLabeled
#########################################################################################################################################
############################################################################################################################################
#Get information for specific catagory/Class id
    def GetCategoryData(self,ID,DATAfile="panoptic_val2017.json"):
            if self.AnnData==False:
                with open(DATAfile) as json_file:
                      self.AnnData = json.load(json_file)

            for item in self.AnnData['categories']:
                    if item["id"]==ID:
                        return item["name"],item["isthing"]