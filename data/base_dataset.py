import torch.utils.data as data
from PIL import Image
import torchvision.transforms as transforms
import numpy
from skimage import color

#Added
import time
from math import exp
import random

class BaseDataset(data.Dataset):
    def __init__(self):
        super(BaseDataset, self).__init__()

    def name(self):
        return 'BaseDataset'

    def initialize(self, opt):
        pass

def get_transform_no(opt):
    transform_list = []
    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_size(opt):
    transform_list = []

    transform_list += [transforms.Resize([480,832], Image.BICUBIC),
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def get_transform_size_sat(opt):
    transform_list = [transforms.Resize([480,832]),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),"original",0.4,1,1,1,time.time())))
    transform_list += [
                       transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)    

def get_transform_filter_sat(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),"random",1,1,1,1,time.time())))
    transform_list += [
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                          ]
    return transforms.Compose(transform_list)

def get_transform_filter_gray(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),"original",0.4,1,1,1,time.time())))
    transform_list += [
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                          ]
    return transforms.Compose(transform_list)

def get_transform_filter_red(opt):
    transform_list = [transforms.Resize(opt.fineSize),]
    transform_list.append(transforms.Lambda(lambda img: Filter_syn(numpy.array(img),"original",1.0,1,1,1,time.time())))
    transform_list += [
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5, 0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
                          ]
    return transforms.Compose(transform_list)


def Filter_syn(I,mode,w,x,y,z,start_time):

    # Preprocessing
    this_time=time.time()
    elapsed_time = this_time - start_time

    if mode == "random":
        w = ((elapsed_time*1000)%1000)*(1.5)/1000 # above 1
        if w > 1.3:
            w = 1.3
        if w <0.4:
            w = 0.4


    # Change Saturation
    hsv = color.rgb2hsv(I)
    h = (hsv[:, :, 0] / 360.0 ) * 255.0
    s = (hsv[:, :, 1] / 100.0 ) * 255.0 * (w)
    v = (hsv[:, :, 2] / 100.0 ) * 255.0
    
    r = (h / 255.0 ) * 360.0
    g = (s / 255.0 ) * 100.0
    b = (v / 255.0 ) * 100.0

    rgb = color.hsv2rgb(numpy.dstack([r, g, b]).astype(numpy.float64))

    # Change Color
    rgb[:,:,0] = rgb[:,:,0] * x
    rgb[:,:,1] = rgb[:,:,1] * y
    rgb[:,:,2] = rgb[:,:,2] * z


    # Post-processing
    rgb[:,:,0] = numpy.clip(rgb[:,:,0],0,1)
    rgb[:,:,1] = numpy.clip(rgb[:,:,1],0,1)
    rgb[:,:,2] = numpy.clip(rgb[:,:,2],0,1)

    return rgb



def get_transform(opt):
    transform_list = []
    if opt.resize_or_crop == 'resize_and_crop':
        osize = [opt.loadSize, opt.loadSize]
        transform_list.append(transforms.Scale(osize, Image.BICUBIC))
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'crop':
        transform_list.append(transforms.RandomCrop(opt.fineSize))
    elif opt.resize_or_crop == 'scale_width':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.fineSize)))
    elif opt.resize_or_crop == 'scale_width_and_crop':
        transform_list.append(transforms.Lambda(
            lambda img: __scale_width(img, opt.loadSize)))
        transform_list.append(transforms.RandomCrop(opt.fineSize))

    if opt.isTrain and not opt.no_flip:
        transform_list.append(transforms.RandomHorizontalFlip())
    

    transform_list += [transforms.ToTensor(),
                       transforms.Normalize((0.5, 0.5, 0.5),
                                            (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

def __scale_width(img, target_width):
    ow, oh = img.size
    if (ow == target_width):
        return img
    w = target_width
    h = int(target_width * oh / ow)
    return img.resize((w, h), Image.BICUBIC)
