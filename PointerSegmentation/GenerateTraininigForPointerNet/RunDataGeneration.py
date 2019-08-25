#Generate training data for the Pointer net  using the COCO panoptic 2017 dataset
import DataGeneratorForPointerSegmentation as Generator
############################################Input and ouput dir location################################################################################################################################################
ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/train2017/" # image folder (coco training) train set
AnnotationDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_train2017/" # annotation maps from coco panoptic train set
DataFile="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_train2017.json" # Json Data file coco panoptic train set


OutDir="../../SampleData/PointerNetTrainigData/" # Output Dir

# ImageDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/train2017/" # image folder (coco training) train set
# AnnotationDir="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/COCO_panoptic/panoptic_train2017/panoptic_train2017/" # annotation maps from coco panoptic train set
# DataFile="/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/COCO_panoptic/panoptic_train2017.json" # Json Data file coco panoptic train set
# OutDir="/scratch/gobi2/seppel/CocoGenerated/"
x=Generator.Generator(ImageDir,AnnotationDir,OutDir, DataFile)
x.Generate()