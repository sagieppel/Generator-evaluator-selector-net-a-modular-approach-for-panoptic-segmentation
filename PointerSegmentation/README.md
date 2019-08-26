# Pointer net:
## See parent folder for usage instructions  
Pointer net receives an image and a point within this image. The net predicts the mask of the segment that contains the input point. Another input of the pointer net is a region of interest (ROI) mask which restricts the region of the predicted segments. The generated output segment region will be confined to the ROI mask. This property prevents newly generated segments from overlapping previously generated segments. In this work, the ROI mask is simply the unsegmented region of the image. 
![](/PointerSegmentation/Figure.png)
![](/PointerSegmentation/Figure2.jpg)
