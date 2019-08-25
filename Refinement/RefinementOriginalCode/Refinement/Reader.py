
#Reader for the coco panoptic data set for pointer based image segmentation
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
############################################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir,PredMaskDirs,GTMaskDirs,NumClasses,ClassBalance=True, MaxBatchSize=100,MinSize=250,MaxSize=900,MaxPixels=800*800*5, AnnotationFileType="png", ImageFileType="jpg",TrainingMode=True):

        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.AnnotationFileType=AnnotationFileType # What is the the type (ending) of the annotation files
        self.ImageFileType=ImageFileType # What is the the type (ending) of the image files
        self.Epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.ClassBalance=ClassBalance
# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.AnnotationList = []
        self.AnnotationByCat = []
        self.NumClasses=NumClasses
        for i in range(NumClasses):
            self.AnnotationByCat.append([])
        uu=0
        for ii,PredDir in enumerate(PredMaskDirs):
            GTDir=GTMaskDirs[ii]
            for Name in os.listdir(PredDir):
                uu+=1
              #  print(uu)
                s = {}
                s["PredMaskFile"] = PredDir + "/" + Name
                if os.path.exists(GTDir+"/"+Name): s["GTMaskFile"] =GTDir+"/"+Name
                else:
                    print("Missing"+GTDir+"/"+Name)
                    continue

                s["IOU"] = float(Name[Name.find("#IOU#") + 5:Name.find("#Precision#")])
                s["AreaFract"] = float(Name[Name.find("#FractImSize#") + 13: Name.find('#Num#')])
                s["IsThing"] = Name[Name.find("#TYPE#") + 6: Name.find('#IOU#')] == "thing"
                s["ImageFile"] = ImageDir+Name[:Name.find('#TYPE#')].replace("##jpg", ".jpg")
                s["CatID"] = int(Name[Name.find("#CatID#") + 7: Name.find('#END#')])
                if not (os.path.exists(s["ImageFile"]) or os.path.exists(s["MaskFile"])):
                                      print("Missing:"+s["MaskFile"])
                                      continue
                self.AnnotationList.append(s)
                self.AnnotationByCat[s["CatID"]].append(s)
         #       if uu>10000: break
        for i in range(NumClasses):
                print(str(i) + ")" + str(len(self.AnnotationByCat[i])))

        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize(self,Img, GTMask,PredMask,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox= cv2.boundingRect(PredMask)
        [h, w, d] = Img.shape
        Rs = np.max((Hb / h, Wb / w))
        Wbox = int(np.floor(bbox[2]))  # Segment Bounding box width
        Hbox = int(np.floor(bbox[3]))  # Segment Bounding box height
        if Wbox==0: Wbox+=1
        if Hbox == 0: Hbox += 1


        Bs = np.min((Hb / Hbox, Wb / Wbox))
        if Rs > 1 or Bs<1 or np.random.rand()<0.3:  # Resize image and mask to batch size if mask is smaller then batch or if segment bounding box larger then batch image size
            h = int(np.max((h * Rs, Hb)))
            w = int(np.max((w * Rs, Wb)))
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)

            PredMask = cv2.resize(PredMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            GTMask = cv2.resize(GTMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            bbox = (np.float32(bbox) * Rs.astype(np.float)).astype(np.int64)

 # =======================Crop image to fit batch size===================================================================================
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height

        if Wb > Wbox:
            Xmax = np.min((w - Wb, x1))
            Xmin = np.max((0, x1 - (Wb - Wbox)-1))
        else:
            Xmin = x1
            Xmax = np.min((w - Wb, x1 + (Wbox - Wb)+1))

        if Hb > Hbox:
            Ymax = np.min((h - Hb, y1))
            Ymin = np.max((0, y1 - (Hb - Hbox)-1))
        else:
            Ymin = y1
            Ymax = np.min((h - Hb, y1 + (Hbox - Hb)+1))

        if Ymax<=Ymin: y0=Ymin
        else: y0 = np.random.randint(low=Ymin, high=Ymax + 1)

        if Xmax<=Xmin: x0=Xmin
        else: x0 = np.random.randint(low=Xmin, high=Xmax + 1)

        # Img[:,:,1]*=Mask
        # misc.imshow(Img)

        Img = Img[y0:y0 + Hb, x0:x0 + Wb, :]
        PredMask = PredMask[y0:y0 + Hb, x0:x0 + Wb]
        GTMask = GTMask[y0:y0 + Hb, x0:x0 + Wb]
#------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)
        if not (GTMask.shape[0] == Hb and GTMask.shape[1] == Wb): GTMask = cv2.resize(GTMask.astype(float), dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
        if not (PredMask.shape[0] == Hb and PredMask.shape[1] == Wb): PredMask = cv2.resize(PredMask.astype(float), dsize=(Wb, Hb),interpolation=cv2.INTER_NEAREST)

        #-----------------------------------------------------------------------------------------------------------------------------------
        return Img,GTMask,PredMask
# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, batch_pos, Hb=-1, Wb=-1):
# -----------------------------------Image and resize-----------------------------------------------------------------------------------------------------
            if self.ClassBalance: # pick with equal class probability
                while (True):
                     CL=np.random.randint(self.NumClasses)
                     l=len(self.AnnotationByCat[CL])
                     if l>0: break
                Nim = np.random.randint(l)
               # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
                Ann=self.AnnotationByCat[CL][Nim]
             #   print("Blance")
            else: # Pick with equal class probabiliry
                Nim = np.random.randint(len(self.AnnotationList))
                Ann=self.AnnotationList[Nim]
              #  print("Nor")

            Img = cv2.imread(Ann["ImageFile"])  # Load Image
            Img = Img[..., :: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
#-----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            MaskGT = cv2.imread(Ann["GTMaskFile"],0)
            MaskPred =  cv2.imread(Ann["PredMaskFile"],0)  # Load segment to refine

            self.LabelFileName = Ann["PredMaskFile"]
            # -----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            if not Hb == -1:
                Img, MaskGT, MaskPred = self.CropResize(Img, MaskGT, MaskPred, Hb, Wb)
            # ---------------------------------------------------------------------------------------------------------------------------------
            self.BImgs[batch_pos] = Img
            self.BGTMask[batch_pos] = MaskGT
            self.BPredMask[batch_pos] = MaskPred
            self.BIsThing[batch_pos] = Ann["IsThing"]
            self.BIOU[batch_pos] = Ann["IOU"]
############################################################################################################################################################
############################################################################################################################################################
# Start load batch of images, segment masks, ROI masks, and pointer points for training MultiThreading s
    def StartLoadBatch(self):
        # =====================Initiate batch=============================================================================================
        while True:
            Hb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # Batch hight
            Wb = np.random.randint(low=self.MinSize, high=self.MaxSize)  # batch  width
            if Hb*Wb<self.MaxPixels: break
        BatchSize =  np.int(np.min((np.floor(self.MaxPixels / (Hb * Wb)), self.MaxBatchSize)))
        self.BImgs = np.zeros((BatchSize, Hb, Wb, 3))  #
        self.BGTMask = np.zeros((BatchSize, Hb, Wb))
        self.BPredMask = np.zeros((BatchSize, Hb, Wb))
        self.BIsThing = np.zeros((BatchSize))
        self.BIOU= np.zeros((BatchSize))
        #====================Start reading data multithreaded===========================================================
        self.thread_list = []
        for pos in range(BatchSize):
            th=threading.Thread(target=self.LoadNext,name="thread"+str(pos),args=(pos,Hb,Wb))
            self.thread_list.append(th)
            th.start()
        self.itr+=BatchSize
 ##################################################################################################################
    def SuffleFileList(self):
            random.shuffle(self.FileList)
            self.itr = 0
###########################################################################################################
#Wait until the data batch loading started at StartLoadBatch is finished
    def WaitLoadBatch(self):
            for th in self.thread_list:
                 th.join()

########################################################################################################################################################################################
    def LoadBatch(self):
# Load batch for training (muti threaded  run in parallel with the training proccess)
# For training
            self.WaitLoadBatch()
            Imgs=self.BImgs
            GTMask=self.BGTMask
            PredMask=self.BPredMask
            IOU=self.BIOU
            IsThing=self.BIsThing
            self.StartLoadBatch()
            return Imgs, GTMask,PredMask,IOU,IsThing
########################################################################################################################################################################################
    def LoadSingleClean(self):
 # Load batch of on image segment and pointer point without croping or resizing
 # for evaluation step
        if self.itr >= len(self.FileList):
            self.itr = 0
            self.Epoch += 1
        Hb, Wb = cv2.imread(self.ImageDir + "/" + self.FileList[self.itr]).shape[0:2]
        self.BImgs = np.zeros((1, Hb, Wb, 3))  #
        self.FileName=self.FileList[self.itr]

        self.BGTMask = np.zeros((1, Hb, Wb))
        self.BPredMask = np.zeros((1, Hb, Wb))
        self.BIsThing = np.zeros((1))
        self.BIOU = np.zeros((1))

        self.LoadNext(0,self.itr, Hb,Wb)

        self.itr += 1
        Imgs = self.BImgs
        GTMask = self.BGTMask
        PredMask = self.BPredMask
        IsThing = self.BIsThing[0]
        IOU = self.BIOU
        return Imgs,  GTMask , PredMask ,IOU,IsThing


############################################################################################################################################
#Get information for specific catagory/Class id
    def GetCategoryData(self,ID,DATAfile="/home/sagi/DataZoo/COCO_panoptic/panoptic_val2017.json"):
            if self.AnnData==False:
                with open(DATAfile) as json_file:
                      self.AnnData = json.load(json_file)

            for item in self.AnnData['categories']:
                    if item["id"]==ID:
                        return item["name"],item["isthing"]