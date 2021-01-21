import torch
import torch.nn.functional as F
from torch.autograd import Variable
import torch.nn

cuda1 = torch.device('cuda:0')

kernelh = Variable(torch.tensor([[[[0.0625, 0.2500, 0.3750, 0.2500, 0.0625]]]], device=cuda1), requires_grad=False)
kernelv = Variable(torch.tensor([[[[0.0625], [0.2500], [0.3750], [0.2500], [0.0625]]]], device=cuda1), requires_grad=False)

kernel = kernelv*kernelh*4

ker00 = kernel[:,:,0::2,0::2]
ker01 = kernel[:,:,0::2,1::2]
ker10 = kernel[:,:,1::2,0::2]
ker11 = kernel[:,:,1::2,1::2]

def reconLpyr(sub1, sub2, sub3, sub4, lp):
    
    lpex = pyrExpand(lp)
    gpyr4 = sub4 + lpex[:,:,:,:-1]
    gpyr3 = sub3 + pyrExpand(gpyr4)
    gpyr2 = sub2 + pyrExpand(gpyr3)
    img = sub1 + pyrExpand(gpyr2)

    return img    

def pyrExpand(im):
    
    out = torch.zeros(im.size(0),im.size(1),im.size(2)*2,im.size(3)*2, device=cuda1, dtype=torch.float32)
    
    for k in range(3):
        
        temp = im[:,k,:,:]
        temp = temp.unsqueeze(dim=1)
                       
        im_c1 = torch.cat((temp, temp[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
        im_c1r1 = torch.cat((im_c1, im_c1[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                
        im_r2 = torch.cat((temp[:,:,0,:].unsqueeze(dim=2), temp), dim=2) # padding columns
        im_r2 = torch.cat((im_r2, im_r2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im_r2c1 = torch.cat((im_r2, im_r2[:,:,:,-1].unsqueeze(dim=3)), dim=3) 
                
        im_c2 = torch.cat((temp[:,:,:,0].unsqueeze(dim=3), temp), dim=3) # padding columns
        im_c2 = torch.cat((im_c2, im_c2[:,:,:,-1].unsqueeze(dim=3)), dim=3) # padding columns
        
        im_c2r1 = torch.cat((im_c2, im_c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) 
                
        im_r2c2 = torch.cat((im_c2[:,:,0,:].unsqueeze(dim=2), im_c2), dim=2) # padding columns
        im_r2c2 = torch.cat((im_r2c2, im_r2c2[:,:,-1,:].unsqueeze(dim=2)), dim=2) # padding columns
        
        im_00 = F.conv2d(im_r2c2, ker00, padding = [0,0], groups=1)
        im_01 = F.conv2d(im_r2c1, ker01, padding = [0,0], groups=1)
        im_10 = F.conv2d(im_c2r1, ker10, padding = [0,0], groups=1)
        im_11 = F.conv2d(im_c1r1, ker11, padding = [0,0], groups=1)
                
        out[:,k,0::2,0::2] = im_00[:,0,:,:]
        out[:,k,1::2,0::2] = im_10[:,0,:,:]
        out[:,k,0::2,1::2] = im_01[:,0,:,:]
        out[:,k,1::2,1::2] = im_11[:,0,:,:]
                 
    return out