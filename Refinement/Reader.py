
#Reader for the coco panoptic data set
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
    def __init__(self, ImageDir,MaskDirs,NumClasses,ClassBalance=True, MinPrecision=0.0,MaxBatchSize=100,MinSize=250,MaxSize=900,MaxPixels=800*800*5,TrainingMode=True,AugmentImage=False):#,ReadRatio=1.1):
        print("Start")
        self.AugmentImage=AugmentImage
        self.MaxBatchSize=MaxBatchSize # Max number of image in batch
        self.MinSize=MinSize # Min image width and hight in pixels
        self.MaxSize=MaxSize #Max image width and hight in pixels
        self.MaxPixels=MaxPixels # Max number of pixel in all the batch (reduce to solve oom out of memory issues)
        self.Epoch = 0 # Training Epoch
        self.itr = 0 # Training iteratation
        self.ClassBalance=ClassBalance
# ----------------------------------------Create reader for data set--------------------------------------------------------------------------------------------------------------
        self.AnnotationList = [] # List of all annotation
        self.AnnotationByCat = []# All annotations arranged  by class
        self.NumClasses=NumClasses
        for i in range(NumClasses):
            self.AnnotationByCat.append([]) # All annotations arranged  by class
        #-----------------------------------------------------------
        self.ImageByCat = []# List of all images correspond to a specific annotation
        for i in range(NumClasses):
            self.ImageByCat.append([])
        uu=0
        #----------------------------------------------------------
        for ii,MaskDir in enumerate(MaskDirs):
            uu=0
            print("Counting files")

          #  numfiles=int(len([name for name in os.listdir(MaskDir+"/Pred/") if "#V#" in name])/5)
           # print(str(numfiles)+" files to read in "+MaskDir)
            for Name in os.listdir(MaskDir+"/Pred/"):
               # if np.random.rand()>ReadRatio:continue

                if ("#WRONG#" in Name) or (not ".png" in Name): continue
                uu+=1
                #####if uu>1000: break
                #print(uu)
                if uu%50==0:print(uu)
                s = {}
                s["PredMaskFile"] = MaskDir + "/Pred/" + Name
                s["GTMaskFile"] = MaskDir + "/GT/" + Name
                #************************************************************
                # if cv2.imread(s["PredMaskFile"],0).max()==0 or cv2.imread(s["GTMaskFile"],0).max()==0:
                #     os.remove(s["PredMaskFile"])
                #     os.remove(s["GTMaskFile"])
                #     continue
                #     pp+=1
                # gg+=1
                # print(pp/gg)
                #****************************************************************
                #######################################
                # if os.path.exists(s["PredMaskFile"]): s["GTMaskFile"] =GTDir+"/"+Name
                # else:
                #     print("Missing"+GTDir+"/"+Name)
                #     exit("missing gt file")
                #     continue

                s["IOU"] = float(Name[Name.find("#IOU#") + 5:Name.find("#Precision#")])
                s["Precision"] = float(Name[Name.find("#Precision#") + 11:Name.find("#Recall#")])
                if s["Precision"] < MinPrecision: continue
                s["Recall"] = float(Name[Name.find("#Recall#")+8:Name.find("#CatID#")])
                s["ImageFile"] = ImageDir+"/"+Name[:Name.find("#IOU#")].replace("#jpg",".jpg")
                s["CatID"] = int(Name[Name.find("#CatID#")+7:Name.find("#RandID#")])
                if not (os.path.exists(s["PredMaskFile"]) and os.path.exists(s["GTMaskFile"] and os.path.exists(s["ImageFile"]))):
                                      print("Missing:"+s["GtMaskFile"])
                                      x=ddd
                                      continue
                self.AnnotationList.append(s)
                self.AnnotationByCat[s["CatID"]].append(s)
                if not s["ImageFile"] in self.ImageByCat[s["CatID"]]: self.ImageByCat[s["CatID"]].append(s["ImageFile"])

         #       if uu>10000: break
        print("Num annotation="+str(len(self.AnnotationList)))
        for i in range(NumClasses):
                print(str(i) + "anns " + str(len(self.AnnotationByCat[i]))+ " images" + str(len(self.ImageByCat[i])))

        print("done making file list")
        iii=0
        if TrainingMode: self.StartLoadBatch()
        self.AnnData=False
#############################################################################################################################
# Crop and resize image and mask and ROI to feet batch size
    def CropResize(self,Img, GTMask,PredMask,Hb,Wb):
        # ========================resize image if it too small to the batch size==================================================================================
        if PredMask.sum()==0:
            print("545454")
        bbox= cv2.boundingRect(((PredMask+GTMask)>0).astype(np.uint8))
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
#####################################################################################################################################################
######################################################Augmented image##################################################################################################################################
    def Augment(self,Img,Mask,Mask2,prob):
        print("Augmenting")
        if np.random.rand()<0.5: # flip left right
            Img=np.fliplr(Img)
            Mask = np.fliplr(Mask)
            Mask2 = np.fliplr(Mask2)


        if np.random.rand()< prob/9: # flip up down
            Img=np.flipud(Img)
            Mask = np.flipud(Mask)
            Mask2 = np.flipud(Mask2)


        if np.random.rand() < prob: # resize
            r=r2=(0.65 + np.random.rand() * 0.65)
            if np.random.rand() < prob*0.25:  r2=(0.7 + np.random.rand() * 0.6)
            h = int(Mask.shape[1] * r)
            w = int(Mask.shape[0] * r2)
            Img = cv2.resize(Img, dsize=(w, h), interpolation=cv2.INTER_LINEAR).astype(np.uint8)
            Mask = cv2.resize(Mask.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
            Mask2 = cv2.resize(Mask2.astype(float), dsize=(w, h), interpolation=cv2.INTER_NEAREST).astype(np.uint8)
 #           misc.imshow(Mask*50+Mask2*100)

        if np.random.rand() < prob:  # Dark light
            Img = Img * (0.6 + np.random.rand() * 0.6)
            Img[Img>255]=255
            #misc.imshow(Img)

        if np.random.rand() < prob:  # GreyScale
            Gr=Img.mean(axis=2)
            r=np.random.rand()

            Img[:, :, 0] = Img[:, :, 0] * r + Gr * (1 - r)
            Img[:, :, 1] = Img[:, :, 1] * r + Gr * (1 - r)
            Img[:, :, 2] = Img[:, :, 2] * r + Gr * (1 - r)


        return Img,Mask, Mask2
########################################################################################################################################################

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
                CatSize=len(self.ImageByCat[CL])
             #   print("Blance")
            else: # Pick with equal class probabiliry
                Nim = np.random.randint(len(self.AnnotationList))
                Ann=self.AnnotationList[Nim]
                CatSize = 10000000000
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

#----------------------------------------------------------------------------------------------------------------------------
            if not (Img.shape[0] == MaskGT.shape[0] and Img.shape[1] == MaskGT.shape[1]):
                if np.random.rand() < 0.9:
                    MaskGT = cv2.resize(MaskGT, (Img.shape[1], Img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    MaskPred = cv2.resize(MaskPred, (Img.shape[1], Img.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    #print("ResImage")
                    Img = cv2.resize(Img, (MaskGT.shape[1], MaskGT.shape[0]), interpolation=cv2.INTER_LINEAR)
#-----------------------------------------------------------------------------------------------------------------------------
            if (MaskGT is None) or (MaskPred is None) or (Img is None):

                print("Missing "+ Ann["PredMaskFile"])
                self.LoadNext(batch_pos, Hb, Wb)
            else:
#-------------------------Augment-----------------------------------------------------------------------------------------------
                if self.AugmentImage: Img,MaskGT,MaskPred=self.Augment(Img,MaskGT,MaskPred,np.min([float(1000/CatSize)*0.27+0.03,1]))
#-----------------------------------------------------------------------------------------------------------------------------------------
                self.LabelFileName = Ann["PredMaskFile"]
                # -----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
                if not Hb == -1:
                    # print(Img.shape)
                    # print(MaskGT.shape)
                    # print(MaskPred.shape)
                    Img, MaskGT, MaskPred = self.CropResize(Img, MaskGT, MaskPred, Hb, Wb)


            #     Img, MaskGT, MaskPred = self.CropResize(Img2, MaskGT2, MaskPred2, Hb, Wb)
# ---------------------------------------------------------------------------------------------------------------------------------
                self.BImgs[batch_pos] = Img
                self.BGTMask[batch_pos] = MaskGT
                self.BPredMask[batch_pos] = MaskPred
                self.BIOU[batch_pos] = Ann["IOU"]


                if MaskGT.max() == 0 or MaskPred.max() == 0:
                    self.LoadNext(batch_pos,Hb, Wb)
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
        self.BIOU = np.zeros((BatchSize))
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
            self.StartLoadBatch()
            return Imgs, GTMask,PredMask,IOU
########################################################################################################################################################################################
    def LoadSingle(self,ClassBalance):
            # -----------------------------------Image and resize-----------------------------------------------------------------------------------------------------
            if ClassBalance:  # pick with equal class probability
                while (True):
                    CL = np.random.randint(self.NumClasses)
                    l = len(self.AnnotationByCat[CL])
                    if l > 0: break

                Nim = np.random.randint(l)
                # print("nim "+str(Nim)+"CL "+str(CL)+"  length"+str(len(self.AnnotationByCat[CL])))
                Ann = self.AnnotationByCat[CL][Nim]
                CatSize = len(self.ImageByCat[CL])
            #   print("Blance")
            else:  # Pick with equal class probabiliry
                Nim = np.random.randint(len(self.AnnotationList))
                Ann = self.AnnotationList[Nim]
                CatSize = 10000000000
            #  print("Nor")
           # print(Ann["GTMaskFile"])
            Img = cv2.imread(Ann["ImageFile"])  # Load Image
            Img = Img[..., :: -1]
            if (Img.ndim == 2):  # If grayscale turn to rgb
                Img = np.expand_dims(Img, 3)
                Img = np.concatenate([Img, Img, Img], axis=2)
            Img = Img[:, :, 0:3]  # Get first 3 channels incase there are more
            # -----------------------------------Crop and resize-----------------------------------------------------------------------------------------------------
            CatID=Ann["CatID"]
            MaskGT = cv2.imread(Ann["GTMaskFile"], 0)
            MaskPred = cv2.imread(Ann["PredMaskFile"], 0)  # Load segment to refine
         #   print(Ann["GTMaskFile"])
#--------------------------------------------------------------------------------------------------------------------------------
            if not (Img.shape[0] == MaskGT.shape[0] and Img.shape[1] == MaskGT.shape[1]):
                if np.random.rand() < 1:
                    MaskGT = cv2.resize(MaskGT, (Img.shape[1], Img.shape[0]), interpolation=cv2.INTER_NEAREST)
                    MaskPred = cv2.resize(MaskPred, (Img.shape[1], Img.shape[0]), interpolation=cv2.INTER_NEAREST)
                else:
                    Img = cv2.resize(Img, (MaskGT.shape[1], MaskGT.shape[0]), interpolation=cv2.INTER_LINEAR)
  #-------------------------------------------------------------------------------------------------------------------
            MaskGT=np.expand_dims(MaskGT,axis=0)
            MaskPred = np.expand_dims(MaskPred, axis=0)
            Img = np.expand_dims(Img, axis=0)
            return Img,MaskGT, MaskPred, CatID



############################################################################################################################################
#Get information for specific catagory/Class id
    def GetCategoryData(self,ID,DATAfile="/home/sagi/DataZoo/COCO_panoptic/panoptic_val2017.json"):
            if self.AnnData==False:
                with open(DATAfile) as json_file:
                      self.AnnData = json.load(json_file)

            for item in self.AnnData['categories']:
                    if item["id"]==ID:
                        return item["name"],item["isthing"]