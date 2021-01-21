clear all
clc

short_dir = 'restored/';
long_dir = 'long/';

files = dir(strcat(short_dir, '*.png'));

for i = 1:length(files)
    i
    name = files(i).name;

    im_enh = imread(strcat(short_dir, name));
    im_long = imread(strcat(long_dir, name));

    pnr(i) = psnr(im_enh, im_long);
    sim(i) = ssim(im_enh, im_long);
end 
mean(pnr)
mean(sim)