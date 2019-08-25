
#Reader for the coco panoptic data set for pointer based image segmentation
import numpy as np
import os
import scipy.misc as misc
import random
import cv2
import json
import threading
import random
############################################################################################################
#########################################################################################################################
class Reader:
# Initiate reader and define the main parameters for the data reader
    def __init__(self, ImageDir,MaskDir,FullSegDir,NumClasses,ClassBalance=True, MaxBatchSize=100,MinSize=250,MaxSize=1000,MaxPixels=800*800*5, AnnotationFileType="png", ImageFileType="jpg",TrainingMode=True):

        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.AnnotationFileType=AnnotationFileType # What is the the type (ending) of the annotation files
        self.ImageFileType=ImageFileType # What is the the type (ending) of the image files
        self.Epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.ClassBalance=ClassBalance
        self.Findx=None
# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.AnnotationList = []
        self.AnnotationByCat = []
        self.NumClasses=NumClasses

        for i in range(NumClasses):
            self.AnnotationByCat.append([])
        uu=0
        #Ann_name.replace(".","__")+"##Class##"+str(seg["Class"])+"##IsThing##"+str(seg["IsThing"])+"IDNum"+str(ii)+".png"
        print("Creating file list for reader this might take a while")
        for AnnDir in MaskDir:
            for Name in os.listdir(AnnDir):

                uu+=1
                #####if uu>1000: break
                print(uu)
                s = {}
                s["MaskFile"] = AnnDir + "/" + Name
                s["FullAnnFile"] = FullSegDir + "/"+Name[:Name.find("##Class##")].replace("__", ".")
                s["IsThing"] = int(Name[Name.find("##IsThing##") + 11: Name.find('IDNum')])==1
                s["ImageFile"] = ImageDir+"/"+Name[:Name.find("##Class##")].replace("__png", ".jpg")
                s["Class"] = int(Name[Name.find("##Class##") + 9: Name.find("##IsThing##")])
                if not (os.path.exists(s["ImageFile"]) and os.path.exists(s["MaskFile"]) and os.path.exists(s["FullAnnFile"])):
                                      print("Missing:"+s["MaskFile"])
                                      continue
                self.AnnotationList.append(s)
                self.AnnotationByCat[s["Class"]].append(s)

        for i in range(NumClasses):
                np.random.shuffle(self.AnnotationByCat[i])
                print(str(i) + ")" + str(len(self.AnnotationByCat[i])))
        np.random.shuffle(self.AnnotationList)

        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize(self,Img, Mask,AnnMap,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        bbox= cv2.boundingRect(Mask.astype(np.uint8))
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
            Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            AnnMap = cv2.resize(AnnMap.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
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
        Mask = Mask[y0:y0 + Hb, x0:x0 + Wb]
        AnnMap = AnnMap[y0:y0 + Hb, x0:x0 + Wb]
#------------------------------------------Verify shape match the batch shape----------------------------------------------------------------------------------------
        if not (Img.shape[0] == Hb and Img.shape[1] == Wb): Img = cv2.resize(Img, dsize=(Wb, Hb),interpolation=cv2.INTER_LINEAR)
        if not (Mask.shape[0] == Hb and Mask.shape[1] == Wb):Mask = cv2.resize(Mask.astype(float), dsize=(Wb, Hb), interpolation=cv2.INTER_NEAREST)
        if not (AnnMap.shape[0] == Hb and AnnMap.shape[1] == Wb): AnnMap = cv2.resize(AnnMap.astype(float), dsize=(Wb, Hb),interpolation=cv2.INTER_NEAREST)
#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------
        return Img,Mask,AnnMap
        # misc.imshow(Img)
#################################################Generate Annotaton mask#############################################################################################################333
    def GenerateROImask(self, AnnMap,Mask):
         x=np.unique(AnnMap)
         random.shuffle(x)
         ROImask=np.zeros(Mask.shape,dtype=float)
         if np.random.rand()<0.05: # Create by random segments number
              x=x[:np.random.randint(0,len(x))]
              for i in x:
                  ROImask[AnnMap==i]=1
         else: # By size
             r=np.random.rand()
             for i in x:
                 ROImask[AnnMap == i] = 1
                 if ROImask.mean()>r: break
         ROImask[Mask>0]=1
         return ROImask
#################################################Generate Pointer mask#############################################################################################################333
    def GeneratePointermask(self, Mask):
        bbox = cv2.boundingRect(Mask.astype(np.uint8))
        x1 = int(np.floor(bbox[0]))  # Bounding box x position
        Wbox = int(np.floor(bbox[2]))  # Bounding box width
        xmax = np.min([x1 + Wbox+1, Mask.shape[1]])
        y1 = int(np.floor(bbox[1]))  # Bounding box y position
        Hbox = int(np.floor(bbox[3]))  # Bounding box height
        ymax = np.min([y1 + Hbox+1, Mask.shape[0]])
        PointerMask=np.zeros(Mask.shape,dtype=np.float)
        if Mask.max()==0:return PointerMask

        while(True):
            x =np.random.randint(x1,xmax)
            y = np.random.randint(y1, ymax)
            if Mask[y,x]>0:
                PointerMask[y,x]=1
                return(PointerMask)
######################################################Augmented mask##################################################################################################################################
    def Augment(self,Img,Mask,AnnMap,prob):
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            Mask = np.fliplr(Mask)
            AnnMap = np.fliplr(AnnMap)

        if np.random.rand()< prob/5: # flip up down
            Img=np.flipud(Img)
            Mask = np.flipud(Mask)
            AnnMap = np.flipud(AnnMap)

        if np.random.rand() < prob: # resize
            r=r2=(0.6 + np.random.rand() * 0.8)
            if np.random.rand() < prob*0.3:
                r2=(0.65 + np.random.rand() * 0.7)
            h = int(Mask.shape[0] * r)
            w = int(Mask.shape[1] * r2)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            AnnMap = cv2.resize(AnnMap.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        if np.random.rand() < prob:  # Dark light
            Img = Img * (0.6 + np.random.rand() * 0.7)
            Img[Img>255]=255

        if np.random.rand() < prob:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,Mask,AnnMap
########################################################################################################################################################
# ==========================Read image annotation and data===============================================================================================
    def LoadNext(self, batch_pos, Hb=-1, Wb=-1):
# -----------------------------------Image and resize-----------------------------------------------------------------------------------------------------
            if self.ClassBalance: # pick with equal class probability
                while (True):
                     CL=np.random.randint(self.NumClasses)
                     CatSize=len(self.AnnotationByCat[CL])
                     if CatSize>0: break
                Nim = np.random.randint(CatSize)
               # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
                Ann=self.AnnotationByCat[CL][Nim]
            else: # Pick with equal class probabiliry
                Nim = np.random.randint(len(self.AnnotationList))
                Ann=self.AnnotationList[Nim]
                CatSize=100000000

            Img = cv2.imread(Ann["ImageFile"])  # Load Image
            Img = Img[..., :: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
#-------------------------Read annotation--------------------------------------------------------------------------------
            Mask = cv2.imread(Ann["MaskFile"],0)  # Load mask
            AnnMap = cv2.imread(Ann["FullAnnFile"],0)  # Load mask

#-------------------------Augment-----------------------------------------------------------------------------------------------
            Img,Mask,AnnMap=self.Augment(Img,Mask,AnnMap,np.min([float(1000/CatSize)*0.27+0.03,1]))
#-----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            if not Hb==-1:
               Img, Mask,AnnMap = self.CropResize(Img, Mask, AnnMap, Hb, Wb)
#----------------------------------------------------------------------------------------------------------------------------------
            PointerMask=self.GeneratePointermask(Mask)
            if np.random.rand()<0.5: ROIMask=self.GenerateROImask(AnnMap,Mask)
            else: ROIMask=np.ones(Mask.shape,dtype=float)
#---------------------------------------------------------------------------------------------------------------------------------
            self.BPointerMask[batch_pos] =  PointerMask
            self.BROIMask[batch_pos] =  ROIMask
            self.BImgs[batch_pos] = Img
            self.BSegmentMask[batch_pos] = Mask
            self.BIsThing[batch_pos] = Ann["IsThing"]
            self.BCatID[batch_pos] = Ann["Class"]

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
        self.BSegmentMask = np.zeros((BatchSize, Hb, Wb))
        self.BROIMask = np.zeros((BatchSize, Hb, Wb))
        self.BPointerMask = np.zeros((BatchSize, Hb, Wb))
        self.BIsThing = np.zeros((BatchSize))
        self.BCatID = np.zeros((BatchSize))
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
            SegmentMask=self.BSegmentMask
            IsThing=self.BIsThing
            CatID=self.BCatID
            ROIMask = self.BROIMask
            PointerMask = self.BPointerMask
            self.StartLoadBatch()
            return Imgs, SegmentMask,ROIMask,PointerMask
#Imgs, SegmentMask, ROIMask, PointerMap
############################Load single data with no augmentation############################################################################################################################################################
    def LoadSingle(self, ByClass=False):
        print("*************")
        print(self.Findx)
        if self.Findx==None:
                self.Cindx = int(0)
                self.Findx = int(0)
                self.CindList=np.zeros([len(self.AnnotationByCat)],dtype=int)
                self.ClFinished = np.zeros([len(self.AnnotationByCat)],dtype=int)
                self.Epoch = int(0)

        if ByClass:
            while(True):
                if self.Cindx>=len(self.AnnotationByCat): self.Cindx=0
                if self.CindList[self.Cindx]>=len(self.AnnotationByCat[self.Cindx]):
                    self.ClFinished[self.Cindx]=1
                    self.Cindx+=1
                    if np.mean(self.ClFinished)==1:
                        self.Cindx = int(0)
                        self.Findx = int(0)
                        self.CindList = np.zeros([len(self.AnnotationByCat)], dtype=int)
                        self.ClFinished = np.zeros([len(self.AnnotationByCat)], dtype=int)
                        self.Epoch+=1
                else:
                    Ann = self.AnnotationByCat[self.Cindx][self.CindList[self.Cindx]]
                    self.CindList[self.Cindx]+=1
                    self.Cindx+=1
                    break
        else:  # Pick with equal class probabiliry
            if self.Findx>len(self.AnnotationList):
                self.Findx=int(0)
                self.Epoch+=1
            Ann = self.AnnotationList[self.Findx]



        # -------------------------image--------------------------------------------------------------------------------
        Img = cv2.imread(Ann["ImageFile"])  # Load Image
        Img = Img[..., :: -1]
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more

        # -------------------------Read annotation--------------------------------------------------------------------------------
        Mask = cv2.imread(Ann["MaskFile"], 0)  # Load Annotation
        sy,sx=Mask.shape

        CatID= Ann["Class"]
        PointerMask = self.GeneratePointermask(Mask)
        ROIMask = np.ones(Mask.shape, dtype=float)

        #-----------------------Generat ROI map augment -------------------------------------------------------------------------------------
        # AnnMap = cv2.imread(Ann["FullAnnFile"], 0)  # Load mask
        # Img, Mask, AnnMap = self.Augment(Img, Mask, AnnMap, 1)
        # if np.random.rand() < 0.5: ROIMask = self.GenerateROImask(AnnMap, Mask)
        #------------------------------------------------------------------------------------------------------------------------------------
        PointerMask = np.expand_dims(PointerMask, axis=0).astype(np.float)
        Mask = np.expand_dims(Mask, axis=0).astype(np.float)
        Img = np.expand_dims(Img, axis=0).astype(np.float)
        ROIMask = np.expand_dims(ROIMask, axis=0).astype(np.float)

        return Img, Mask,PointerMask,ROIMask, CatID,sy,sx
########################################################################################################################################################################################
######################################################Augmented mask##################################################################################################################################
    def Augment2Gen(self,Img,Mask,prob):

        if np.random.rand() < prob: # resize
            r=r2=(0.6 + np.random.rand() * 0.8)
            if np.random.rand() < prob*0.3:
                r2=(0.65 + np.random.rand() * 0.7)
            h = int(Mask.shape[0] * r)
            w = int(Mask.shape[1] * r2)
            if w > 224 and h > 224:
                Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
                Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
          #      ROImask = cv2.resize(ROImask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)

        if np.random.rand() < prob:  # Dark light
            Img = Img * (0.6 + np.random.rand() * 0.7)
            Img[Img>255]=255

        if np.random.rand() < prob:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,Mask#,ROImask
####################################################################################################################################
    def LoadSingleForGeneration(self, ByClass=False,Augment=False):
        if ByClass:
            while(True):

                # print("##########Epochlist#################################")
                # print(self.Clepoch)
                # print("########ind list##############################")
                # print(self.CindList)
                # print("#######################"+str(self.Cindx))
                if self.Cindx>=len(self.AnnotationByCat):
                             self.Cindx=0

                if self.CindList[self.Cindx]>=len(self.AnnotationByCat[self.Cindx]):
                    self.Clepoch[self.Cindx]+=1
                    self.CindList[self.Cindx]=0
                    if len(self.AnnotationByCat[self.Cindx])==0: self.Cindx+=1
                else:
                    Ann = self.AnnotationByCat[self.Cindx][self.CindList[self.Cindx]]
                    self.CindList[self.Cindx]+=1
                    self.Cindx+=1
                    break
        else:  # Pick with equal class probabiliry
            print("findx " + str(self.Findx))
            if self.Findx>len(self.AnnotationList):
                self.Findx=int(0)
                self.Epoch+=1
            Ann = self.AnnotationList[self.Findx]
            self.Findx+=1


        # -------------------------image--------------------------------------------------------------------------------
        Img = cv2.imread(Ann["ImageFile"])  # Load Image
        Img = Img[..., :: -1]
        if (Img.ndim == 2):  # If grayscale turn to rgb
            Img = np.expand_dims(Img, 3)
            Img = np.concatenate([Img, Img, Img], axis=2)
        Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more

        # -------------------------Read annotation--------------------------------------------------------------------------------
        Mask = cv2.imread(Ann["MaskFile"], 0)  # Load Annotation
        sy,sx=Mask.shape

        CatID= Ann["Class"]

        ROIMask = np.ones(Mask.shape, dtype=float)

        #-----------------------Generat ROI map augment -------------------------------------------------------------------------------------
      #  misc.imshow(Img)
       # print(Img.shape)
        # AnnMap = cv2.imread(Ann["FullAnnFile"], 0)  # Load mask
        if Augment:
            Img, Mask=self.Augment2Gen(Img, Mask,0.95)
#---------------------------------------------------------------------------------------------------------
        h,w=Mask.shape
        print("++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++")
        print(Ann["ImageFile"])
        print("Original size")
        print(Img.shape)
        if w < 225 or h < 225:
            r=230.0/np.min([h,w])
            h = int(h * r)
            w = int(w * r)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR)
            Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
          #  ROIMask = cv2.resize(ROIMask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST)
            print("Changes size")
            print(Img.shape)
       #-----------------------------------------------------------------------
        # if np.random.rand() < 0.5: ROIMask = self.GenerateROImask(AnnMap, Mask)
      #  print(Img.shape)
     #   misc.imshow(Img)
        #------------------------------------------------------------------------------------------------------------------------------------

        PointerMask = self.GeneratePointermask(Mask)
        PointerMask = np.expand_dims(PointerMask, axis=0).astype(np.float)
        Mask = np.expand_dims(Mask, axis=0).astype(np.float)
        Img = np.expand_dims(Img, axis=0).astype(np.float)
        ROIMask = np.ones(Mask.shape, dtype=float)
        fname=Ann["ImageFile"]
        return Img, Mask,PointerMask,ROIMask, CatID,fname[fname.rfind("/")+1:],sy,sx
############################################################################################################################################################
    def Reset(self):
        self.Cindx = int(0)
        self.Findx = int(0)
        self.CindList = np.zeros([len(self.AnnotationByCat)], dtype=int)
        self.Clepoch = np.zeros([len(self.AnnotationByCat)], dtype=int)
        self.Epoch = int(0)#not valid or

############################################################################################################################################
#Get information for specific catagory/Class id
    def GetCategoryData(self,ID,DATAfile="/home/sagi/DataZoo/COCO_panoptic/panoptic_val2017.json"):
            if self.AnnData==False:
                with open(DATAfile) as json_file:
                      self.AnnData = json.load(json_file)

            for item in self.AnnData['categories']:
                    if item["id"]==ID:
                        return item["name"],item["isthing"]