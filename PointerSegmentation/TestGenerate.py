import os

import numpy
import scipy.misc as misc
ImDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/"
PredDir="/scratch/gobi2/seppel/GeneratedPredictions/51AddClassEquivalent/Pred/"
GtDir="/scratch/gobi2/seppel/GeneratedPredictions/51AddClassEquivalent/GT/"
for Name in os.listdir(PredDir):
    if ".png" in Name:
         p = misc.imread(PredDir + "/" + Name)
         g = misc.imread(GtDir + "/" + Name)
         i = misc.imread(ImDir+"/"+Name[:Name.find("#jpg")]+".jpg")
         i=misc.imresize(i,g.shape)
         i[:, :, 0] *= 1 - p
         i[:, :, 1] *= 1 - g
         print(Name)
         misc.imshow(p*50+g*100)
         misc.imshow(i)
