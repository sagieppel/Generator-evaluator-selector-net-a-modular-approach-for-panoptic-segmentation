#Evaluator net:
## See parent folder for usage instructions  
 The evaluator net is used to check and rank the generated segments. The ranking is done according to how well the input segment fits the best matching real segments in the image. The evaluator net is a simple convolutional net that receives two inputs: an image and a generated segment mask. The evaluator net predicts the intersection over union (IOU) between the input segment and the closest real segment in the image. 


![](/Figure.png)