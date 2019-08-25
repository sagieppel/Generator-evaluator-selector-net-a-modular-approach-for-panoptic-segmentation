#---------------Run system on folder and generate annotations------------------------------------------------------------------
import GESModularClass as ModularClass
#-------------------------input parameters--------------------------------------------------------------------------------

ImageDir = "SampleData/PointerNetTrainigData/Image/" # input folder  must contain images in jpg format
OutFolder = "OutPrediction/" # output folder

# ImageDir = "/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/val2017/" # input folder  must contain images in jpg format
# OutFolder = "/scratch/gobi2/seppel/Results/eval2017OutCheck/" # output folder
ClassificationConsistancyThresh = 0.65 # For restoring  the Coco panoptic results (class prediction must be 65% consistant in order for segment to be accepter)

#ClassificationConsistancyThresh = 0 # For class agnostic segmentation (no segment is rejected for uncertain classification classification)

#----------------------------------------------------------------------------------------------------------------------------------------------------------
#----------------------------For statistics generation optional can be used only if the GT annotation is available---------------------
GenerateStatics=False
AnnotationDir =""# "/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/val2017/" # Ground truth annotation mask
DataFile =""# "/scratch/gobi1/seppel/DataSets/COCO_PANOPTIC/PanopticFull/panoptic_val2017.json" # Ground truth annotation
#--------------------------run prediction-----------------------------------------------------------------
x=ModularClass.ModuNet(OutFolder = OutFolder,ImageDir = ImageDir,GenerateStatics=GenerateStatics)
x.RunAllFiles()