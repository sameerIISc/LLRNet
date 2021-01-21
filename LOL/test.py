def test():
    
    import sys
    sys.path.insert(0, 'models/')
    sys.path.insert(0, 'utils/')
    
    from Build_Gpyr import BuildGpyr

    import os
    import numpy as np
    from PIL import Image
    import numpy
    from LpyrRecon import reconLpyr
    import matplotlib as mp
    
    import torch
    from torch.autograd import Variable
    
    device = torch.device('cuda:0')
      
    short_dir = '../../LOL_Data/test/short/'
    enh_dir = 'restored/'
    
    if not os.path.exists(enh_dir):
        os.makedirs(enh_dir)
    
    filenames = os.listdir(short_dir)
    pathnames1 = [os.path.join(short_dir, f) for f in filenames]
    
    model_bsub1 = torch.load('checkpoints/model_bsub1.pth')
    model_bsub1.eval()
    model_bsub1 = model_bsub1.cuda(device)
    
    model_bsub2 = torch.load('checkpoints/model_bsub2.pth')
    model_bsub2.eval()
    model_bsub2 = model_bsub2.cuda(device)
    
    model_bsub3 = torch.load('checkpoints/model_bsub3.pth')
    model_bsub3.eval()
    model_bsub3 = model_bsub3.cuda(device)
    
    model_bsub4 = torch.load('checkpoints/model_bsub4.pth')
    model_bsub4.eval()
    model_bsub4 = model_bsub4.cuda(device)
    
    modellp = torch.load('checkpoints/model_lp.pth')
    modellp.eval()
    modellp = modellp.cuda(device)   
    
    model_grad = torch.load('checkpoints/model_lp_grad.pth')
    model_grad.eval()
    model_grad = model_grad.cuda(device)   
    
    for i in range(len(filenames)):
        print i
        
        im_short = Image.open(pathnames1[i])
        im_short = np.array(im_short, dtype=np.float32)
        im_short = im_short/255        
        
        im_short = numpy.moveaxis(im_short, (0,1,2), (1,2,0))
        im_short = torch.tensor(im_short, device=device, dtype=torch.float32)
        im_short = im_short.unsqueeze(dim=0)
        im_short = im_short[:,0:3,:,:]    
        
        g2, g3, g4, g5, _ = BuildGpyr(im_short)
        
        with torch.no_grad():
            s1e = model_bsub1(im_short)
            s2e = model_bsub2(g2)
            s3e = model_bsub3(g3)
            s4e = model_bsub4(g4)
            grad_out = model_grad(g5)
            lpe = modellp(torch.cat((g5, grad_out), dim=1))
    
        im_enh = reconLpyr(s1e, s2e, s3e, s4e, lpe)
        
        im_enh = Variable(im_enh, requires_grad=False).cpu().numpy()
        im_enh = im_enh[0,:,:,:]
        im_enh = numpy.moveaxis(im_enh, (0,1,2), (2,0,1))
        
        im_enh[im_enh>1] = 1
        im_enh[im_enh<0] = 0
        
        im_name = os.path.basename(pathnames1[i])
        im_path = enh_dir+im_name
        
        mp.image.imsave(im_path, im_enh)
        
if __name__ == "__main__":
    test()