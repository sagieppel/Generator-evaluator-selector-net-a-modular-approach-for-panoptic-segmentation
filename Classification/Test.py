import Reader
import cv2
import numpy as  np
MaskDir=["/media/sagi/Elements/GeneratedPredictions_ByPointerNet/51AddAllfiles/Pred/"]
ImageDir="/media/sagi/2T/Data_zoo/COCO/train2017/"
Rd=Reader.Reader(ImageDir=ImageDir,MaskDirs=MaskDir,NumClasses=205,ClassBalance=False,MinPrecision=0.5,MaxBatchSize=1,Augment=False)
Rd.LoadBatch()
NumImages=0
NImZeros=0
NGT=0
NPR=0

for ii in range(1000000):
    print(ii)
    Imgs, Mask, CatID=Rd.LoadBatch()
    for i in range(Imgs.shape[0]):
        print(i)
        Imgs[i, :, :, 0] *= 1 - Mask[i]
        Imgs[i, :, :, 1] *= 1 - Mask[i]
        print(Rd.GetCategoryData(CatID[i]))
        print(Imgs[i].sum())
        cv2.imshow(Rd.GetCategoryData(CatID[i])[0],Imgs[i].astype(np.uint8))
        cv2.waitKey()
        cv2.destroyAllWindows()