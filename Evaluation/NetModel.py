#
#  Build resnet 50 neural net with attention mask directed classification (Classiffy only the image region marked in the mask)
#  
#
import scipy.misc as misc
import torchvision.models as models
import torch
import copy
from torch.autograd import Variable
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):# Net for region based segment classification
######################Load main net (resnet 50) class############################################################################################################
        def __init__(self, UseGPU=True): # Load pretrained encoder and prepare net layers
            super(Net, self).__init__()
            self.UseGPU = UseGPU # Use GPu with cuda
# ---------------Load pretrained torchvision resnet (need to be connected to the internet to download model in the first time)----------------------------------------------------------
            self.Net = models.resnet101(pretrained=True)
#----------------Segmentation mask proccessing layer---------------------------------------------------------------------
            self.SegConv=nn.Conv2d(1,  self.Net.conv1.weight.shape[0], stride=2, kernel_size=3, padding=1, bias=True)
#----------------Change final layers to single channal IOU prediction------------------------------------------------------------------------------------------
            self.Net.fc1 = nn.Linear(2048, 512)
            self.Net.fc2 = nn.Linear(512, 1)
            # net.fc.weight=torch.nn.Parameter(net.fc.weight[0:NumClass,:].data)
            # net.fc.bias=torch.nn.Parameter(net.fc.bias[0:NumClass].data)
#==========================================================================================
            if self.UseGPU==True:self=self.cuda()
###############################################Run prediction inference using the net ###########################################################################################################
        def forward(self,Images,Segment,UseGPU=True,TrainMode=True):
                if TrainMode:
                    tp = torch.FloatTensor
                else:
                    tp = torch.HalfTensor
                    self.eval()
                    self.half()
#------------------------------- Convert from numpy to pytorch-------------------------------------------------------
                SegmentMask = torch.autograd.Variable(torch.from_numpy(Segment.astype(np.float)), requires_grad=False).unsqueeze(dim=1).type(tp)
                InpImages = torch.autograd.Variable(torch.from_numpy(Images.astype(float)), requires_grad=False).transpose(2, 3).transpose(1, 2).type(tp)
                if UseGPU == True: # Convert to GPU
                    InpImages = InpImages.cuda()
                    SegmentMask = SegmentMask.cuda()
                    self.cuda()
# -------------------------Normalize image-------------------------------------------------------------------------------------------------------
                RGBMean = [123.68, 116.779, 103.939]
                RGBStd = [65, 65, 65]
                for i in range(len(RGBMean)): InpImages[:, i, :, :]=(InpImages[:, i, :, :]-RGBMean[i])/RGBStd[i] # Normalize image by std and mean
#============================Run net layers===================================================================================================
                x=InpImages
                x = self.Net.conv1(x) # First resnet convulotion layer
                 #----------------Apply Attention layers--------------------------------------------------------------------------------------------------
                x = self.Net.bn1(x)
                x = self.Net.relu(x)
                s =self.SegConv(SegmentMask)
                x =  x + s
                x = self.Net.maxpool(x)
                #------------------------------------------------------------------------------------------------------------------------

                x = self.Net.layer1(x)
                # --------------------Second Resnet 50 Block------------------------------------------------------------------------------------------------
                x = self.Net.layer2(x)
                x = self.Net.layer3(x)
                # -----------------Resnet 50 block 4---------------------------------------------------------------------------------------------------
                x = self.Net.layer4(x)
                # ------------Fully connected final vector--------------------------------------------------------------------------------------------------------
                x = torch.mean(torch.mean(x, dim=2), dim=2)
                #x = x.squeeze()
                x = self.Net.fc1(x)
                x = self.Net.fc2(x)
                return x.squeeze()
                #---------------------------------------------------------------------------------------------------------------------------
                # ProbVec = F.softmax(x,dim=1) # Probability vector for all classes
                # Prob,Pred=ProbVec.max(dim=1) # Top predicted class and probability
                # return ProbVec,Pred
###################################################################################################################################


