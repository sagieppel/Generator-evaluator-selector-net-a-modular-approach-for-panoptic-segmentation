
#Generate training data for the Pointer net  using the COCO panoptic 2017 dataset
import numpy as np
import os
#import scipy.misc as misc
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
class Generator:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir,AnnotationDir,OutDir, DataFile, AnnotationFileType="png", ImageFileType="jpg",UnlabeledTag=0):

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
        self.Outdir=OutDir
        self.OutDirByClass=OutDir+"/Class/"
        self.OutDirAll = OutDir + "/SegmentMask/"
        self.SegMapDir = OutDir + "/SegMap/"
        self.OutImageDir = OutDir + "/Image/"
        self.MinSegSize = 400
        if not os.path.exists(OutDir): os.mkdir(OutDir)
       # if not os.path.exists(self.OutDirByClass): os.mkdir(self.OutDirByClass)
        if not os.path.exists(self.OutDirAll): os.mkdir(self.OutDirAll)
        if not os.path.exists(self.SegMapDir): os.mkdir(self.SegMapDir)
        if not os.path.exists(self.OutImageDir): os.mkdir(self.OutImageDir)


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

######################################################################################################
#Generate list of all  segments in the image
# Given the annotation map a json data file create list of all segments and instance with info on each segment
#--------------------------Generate list of all segments--------------------------------------------------------------------------------
    def GeneratListOfAllSegments(self,Ann,Ann_name,AddUnLabeled=False,IgnoreSmallSeg=True):
        AnnList = self.GetAnnnotationData(Ann_name)
        Sgs = []  # List of segments and their info
        SumAreas=0 # Sum areas of all segments up to image
        for an in AnnList:
            an["name"], an["isthing"] = self.GetCategoryData(an["category_id"])
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
                seg["IsCrowd"] = False
                Sgs.append(seg)
        return Sgs,SumAreas
####################################################Generate data##############################################################################################
    def Generate(self):
           ErrorCount=0

           for f,Ann_name in enumerate(self.FileList): # Get label image name

                if os.path.exists(self.SegMapDir + "/" + Ann_name): continue
                print(str(f)+")"+Ann_name)
                Ann = cv2.imread(self.AnnotationDir + "/" + Ann_name)  # Load Annotation


                from shutil import copyfile

                copyfile(self.ImageDir + "/" + Ann_name.replace(".png",".jpg"), self.OutImageDir+ "/" + Ann_name.replace(".png",".jpg"))


                Ann = Ann[..., :: -1]
                self.AnnColor=Ann
                Ann=rgb2id(Ann)
                # misc.imshow((Ann==0).astype(float))
                # misc.imshow(Img)
                H,W=Ann.shape
                Sgs, SumAreas = self.GeneratListOfAllSegments(Ann, Ann_name,AddUnLabeled=True,IgnoreSmallSeg=True) # Generate list of all segments in image

                SgMap=np.zeros([H,W],np.uint8)

                for ii,seg in  enumerate(Sgs): # Go over all segments and save them as as mask with the mask category  and isthing in the mask name
                    SgMap[seg["Mask"]>0]=ii
                    if  not seg["IsCrowd"] and seg["IsLabeled"]:
                        Name=Ann_name.replace(".","__")+"##Class##"+str(seg["CatId"])+"##IsThing##"+str(seg["IsThing"])+"IDNum"+str(ii)+".png"
 #                       ClassDir=self.OutDirByClass+"/"+str(seg["CatId"])+"/"
#                        if not os.path.exists(ClassDir): os.mkdir(ClassDir)


                        # if os.path.exists(self.OutDirAll + "/" + Name) or  os.path.exists(ClassDir + "/" + Name):
                        #        print("Err "+Name)
                        #        ErrorCount+=1

                        #cv2.imwrite(ClassDir+"/"+Name,seg["Mask"].astype(np.uint8))
                        cv2.imwrite(self.OutDirAll + "/" + Name, seg["Mask"].astype(np.uint8))


                        # print((cv2.imread(ClassDir+"/"+Name, 0) - seg["Mask"].astype(np.uint8)).sum())
                        # Img=Img0.copy()
                        # Img[:, :, 1] *= 1 - seg["Mask"].astype(np.uint8)
                        # Img[:,:,0]*=1-seg["Mask"].astype(np.uint8)
                        # print( self.GetCategoryData(seg["CatId"]))
                        # misc.imshow(Img)


                if ii > 255:
                    print("more then 255 Segs")
                    ErrorCount += 1

                cv2.imwrite(self.SegMapDir + "/" + Ann_name, SgMap.astype(np.uint8))
                print("Num Errors "+str(ErrorCount))
                #
                # print((cv2.imread(self.SegMapDir + "/" + Ann_name,0)-SgMap.astype(np.uint8)).sum())
                # misc.imshow(SgMap * 10)




