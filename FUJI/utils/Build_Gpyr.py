import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn

cuda1 = torch.device('cuda:0')

kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]], device=cuda1), requires_grad=False)
kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]], device=cuda1), requires_grad=False)

kernel = kernelv*kernelh*4
kernel1 = kernelv*kernelh

ker00 = kernel[:,:,0::2,0::2]
ker01 = kernel[:,:,0::2,1::2]
ker10 = kernel[:,:,1::2,0::2]
ker11 = kernel[:,:,1::2,1::2]

def BuildGpyr(im):
    
    g2 = pyrReduce(im)
    g3 = pyrReduce(g2)
    g4 = pyrReduce(g3)
    g5 = pyrReduce(g4)
    g6 = pyrReduce(g5)
    
    return g2, g3, g4, g5, g6
    
def pyrReduce(im):
    
    im_out = torch.zeros(im.size(0),3,im.size(2)/2,im.size(3)/2, device=cuda1)
   
    for k in range(3):
        
        temp = im[:,k,:,:].unsqueeze(dim=1)
        
        im_cp = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
        im_cp = torch.cat((im_cp, im_cp[:,:,:,-1].unsqueeze(dim=3), im_cp[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
        
        im_bp = torch.cat((im_cp[:,:,0,:].unsqueeze(dim=2), im_cp[:,:,0,:].unsqueeze(dim=2), im_cp), dim=2) # padding columns
        im_bp = torch.cat((im_bp, im_bp[:,:,-1,:].unsqueeze(dim=2), im_bp[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im1 = F.conv2d(im_bp, kernel1, padding = [0,0], groups=1)
        im_out[:,k,:,:] = im1[:,:,0::2,0::2]
    
    return im_out                 