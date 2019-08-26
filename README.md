# Generator evaluator selector modular net for panoptic image segmentation
This code for the paper [Generator evaluator-selector net: a modular approach for
panoptic segmentation](). To download the same code with trained models weights (ready to run )see these links [1]() [2]().

## Basic concept
The system is composed of a Generator (Pointer Net) that generate segments, evaluator (Evaluation net) that rank and select segments to create category independent segmentation map (Figure 1), and a segment classification net that classifies the selected segments. The nets and the weights were trained and tested on the COCO panoptic data set.
See the tutorial section for running/training instructions and the description section for more details on the system.
### Requirements 
The system was run and trained using Python Anacoda 3.7 with pytorch 1.01, and opencv on single Titan XP GPU.


# Tutorial 


# Running the full system 
1. Download trained system form [here]() or [here]() or train from scratch (See training section).
2. Open “RUN.py” in the main dir
3. Set  input image folder path  to “ImageDir” (the images must be  .jpg format)
4. Set output image folder to “OutFolder” 
5. (Optional) Set “ClassificationConsistancyThresh” to 0 if you interesting in class agnostic segmentation, or to 0.65 if you are interested in getting the best results on COCO panoptic.  The “ClassificationConsistancyThresh”  parameter determines how consistent the segment category prediction must be in order for the segment to be accepted.
6. Run “RUN.py”  results will appear in the OutFolder path
(This scripts should run without change with the sample data included.) 
## What does this do
This will generate full annotation to all the images in the input folder.
Two subfolders will be generated in the output folder:
The ‘FinalPredictionsVizual’ folder will contain the predicted annotation for visualization
The ‘COCO2Channels’ folder will contain the predicted annotation  in a format that can be converted to COCO panoptic standard formats (See COCOConvertor RunConvertEval.py) 




# Training 
Each of the subfolders PointerSegmentation/Evaluation/Classification/Refinement
contains a “TRAIN.py” script.  
Running this “TRAIN.py” should train the net.
All training scripts can be run without change with the sample data included in ‘SampleData’ folder. 
To train net with real data you need to first generate training data, see Generating data for training section.




#  Evaluating trained mode
Each of the subfolders PointerSegmentation/Evaluation/Classification/Refinement 
contains a “Evaluate.py” script.  
Running this “Evaluate.py” should generate evaluation statistics .
All evaluation scripts can be run without change with the sample data included. 
To evaluate  the net with real data, you need to first generate data.
See Generating Data section for more instruction.
Note, that you need to either train or [download trained model]() before evaluating. 


#___________________________________________________________________________
# Generating Data for training
# Generating data for Pointer Net training


1. Download and extract COCO panoptic 2017 train/val data and images from [http://cocodataset.org/#download](http://cocodataset.org/#download)
2. Open “/PointerSegmentation/GenerateTraininigForPointerNet/RunDataGeneration.py” script
3. Set the path for the COCO image folder in the “ImageDir” variable
4. Set the path for the COCO panoptic annotation in the “AnnotationDir” variable.
5. Set the path to the COCO panoptic data json file in  the “DataFile” Variable
6. Run the script.
## What does this generate?
Three subfolders will be generated in the output dir (OutDir)
1. The “Image” subfolder will contain the rgb images for training 
2. The “SegMap” subfolder will contain the full annotation map for the image
3. The “SegmentMask” subfolder will contain binary  masks for individual segments. The name of the file of each mask contain the image used to generate this mask and the category ID of this mask (with COCO panoptic 2017 dataset numeration)
This 3 folders are the inputs for Pointer net training, they also be used to generate training data for Evaluation/Classification/Refinement nets.


# Generating data for Evaluation/Classification/Refinement nets.
1. First you need one or more trained  Pointer net models, and the data used for the pointer net training.
2.  In the “PointerSegmentation” folder run the “GenerateDataClassEqual.py” and the 
“GenerateDataAllFiles.py” scripts. These will generate the data. However before data can be used for training it must be cleaned (See cleaning data section) 
### What does this generate?
These scripts takes the all the trained pointer nets models and use them on the Pointer Training net data to generate predicted segmentation mask. The output folder contain subfolders “Pred” and “GT” which contain the predicted segment mask and the matching Ground Truth segment masks. The segments file names contain the name of the image used to generate the masks, the mask category ID (COCO), and the IOU between the predicted and GT segments. 
These folders should be used as inputs for the  Evaluation/Classification/Refinement nets (after the data pass the cleaning process)
## Cleaning training data  for Evaluation/Classification/Refinement nets (important).
   1. Open ‘/PointerSegmentation/CleanGeneratedData/RUNCleaner.py”
   2. Set the path for the COCO image folder to the   “ImageDir” variable
   3. Set the path for the COCO panoptic annotation to the “AnnotationDir” variable.
   4. Set the path to the COCO panoptic data json file to  the “DataFile” Variable
   5. Run the script. (This will add ‘V’ to names of correct files and ‘Wrong’ to  the rest)
### What does this do/clean?
The pairs of predicted and  GT segments generated earlier, might not correspond to each other. Hence, the predicted segment might match a different ground truth segment then the one assign to  it. In this case the cleaner will add the ‘WRONG’ to the file name else it will add ‘V’ readers of the evaluation/classification/refinement nets will not use files that have ‘WRONG’ in their names.




#__________________________________________________
# Description
For detail description see [Generator evaluator-selector net: a modular approach for
panoptic segmentation]().


## Generator evaluator selector method
A schematic for the full modular system is shown in Figure 1. The method is comprised of four
independent networks combined into one modular structure. The first step is generating several
different segments using the pointer net. The segments generated by this net, are restricted to a given region of interest (ROI) which covers the unsegmented image region. The generated segments are then ranked by the evaluator net. This net assigned each segment a score that estimate how well it corresponds to a real segment in the image. The segments which receive the highest scores and are consistent with each other are selected, while low-ranking segments are filtered out. The selected segments are then polished using the refinement net. Each of the selected segments is then classified using the classifier net. Finally, the selected segments are stitched into the segmentation map (Figure 1). The segmentation map is passed to the next cycle which repeats the process in the remaining unsegmented image regions. The process is repeated until either the full image has been segmented or the quality assigned to all of the predicted segments by the evaluator drop below some threshold.
![](/Figure1.png)
### Figure 1: Full system








# Different nets explained
## Pointer net: 
Pointer net  act as the segment generator, which creates proposals for different segments in the image (Figure 2). Pointer net receives an image and a point within this image. The net predicts the mask of the segment that contains the input point (Figure 2). In this work, the pointer point location is chosen randomly within the unsegmented region of the image. The net will predict different segments for different input points, even if the points are located within the same segment (Figure 2). While this feature was not planned, it allows pointer net to act as a random segment generator with the ability to generate a large variability of segments by selecting random input points. Another input of the pointer net is a region of interest (ROI) mask which restricts the region of the predicted segments. The generated output segment region will be confined to the ROI mask. This property prevents newly generated segments from overlapping previously generated segments. In this work, the ROI mask is simply the unsegmented region of the image. 
## Evaluator net:
 The evaluator net is used to check and rank the generated segments. The ranking is done according to how well the input segment fits the best matching real segments in the image. The evaluator net is a simple convolutional net that receives two inputs: an image and a generated segment mask (Figure 2d). The evaluator net predicts the intersection over union (IOU) between the input segment and the closest real segment in the image. 
## Refinement net: 
Refinement net is used to polish the boundaries of the generated segment. The net receives the image and an imperfect segment mask. The net output is a refined version of the input segment (Figure 2e). This approach has been examined in several previous works. 
## Classifier net: 
Determining the segment category is done using a region-specific classification net. The net receives the image and a segment mask. The net predicts the category of the input segment (Figure 2f). This approach has been explored in previous works. 


![](/Figure2.jpg)
### Figure 2: Different nets
