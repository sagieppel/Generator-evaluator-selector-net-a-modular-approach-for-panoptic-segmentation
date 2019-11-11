# Pointer net: 
Pointer net receives an image and a point within this image. The net predicts the mask of the segment that contains the input point. Another input of the pointer net is a region of interest (ROI) mask which restricts the region of the predicted segments. The generated output segment region will be confined to the ROI mask.  Fully trained system can be download from [here](https://drive.google.com/file/d/1hOO4QLQ0NfhvNj_K5VIBJCJ7LspqG0nF/view?usp=sharing) or [here](https://drive.google.com/file/d/1k0mfvLv0QoA88b5CVxq6jf5sZkQFLJd0/view?usp=sharing).


![](/PointerSegmentation/Figure.png)
![](/PointerSegmentation/Figure2.jpg)


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

# Training
After generating the DATA run TRAIN.py. Train model will be written to the log  folder
# Runing
run Run_Segmentation.py.
(first train or download trained model from rom [here](https://drive.google.com/file/d/1hOO4QLQ0NfhvNj_K5VIBJCJ7LspqG0nF/view?usp=sharing) or [here](https://drive.google.com/file/d/1k0mfvLv0QoA88b5CVxq6jf5sZkQFLJd0/view?usp=sharing))
