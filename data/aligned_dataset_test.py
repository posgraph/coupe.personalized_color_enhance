import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_transform_no, get_transform_size, get_transform_size_sat
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class AlignedDataset_Test(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot
        
        if opt.fast_blind == 'true':
            print(opt.video_folder)
            print("QUACHITCHU")
            self.dir_A = os.path.join(opt.dataroot,opt.video_folder)
        else:
            self.dir_A = os.path.join(opt.dataroot,'testA')
            #self.dir_A = os.path.join(opt.dataroot,'testB_exN_resized')
            #self.dir_A = os.path.join(opt.dataroot,'testB_exN_resized')

        self.A_paths = make_dataset(self.dir_A)

        self.A_paths = sorted(self.A_paths)
        
        self.A_size = len(self.A_paths)

        self.transform_no = get_transform_no(opt)
        self.transform_size = get_transform_size(opt)
        self.transform_size_sat = get_transform_size_sat(opt)


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]
        

        A_img = Image.open(A_path).convert('RGB')
        #A = self.transform_size(A_img)
        A = self.transform_size_sat(A_img)
        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        if input_nc == 1:  # RGB to gray
            tmp = A[0, ...] * 0.299 + A[1, ...] * 0.587 + A[2, ...] * 0.114
            A = tmp.unsqueeze(0)


        return {'A': A,'A_paths': A_path, }

    def __len__(self):
        return max(self.A_size,self.A_size)

    def name(self):
        return 'AlignedDataset_Test'

