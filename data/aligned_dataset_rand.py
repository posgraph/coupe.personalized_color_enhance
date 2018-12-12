import os.path
import torchvision.transforms as transforms
from data.base_dataset import BaseDataset, get_transform, get_transform_no, get_transform_filter_sat, get_transform_filter_gray
from data.image_folder import make_dataset
from PIL import Image
import PIL
import random

class AlignedDataset_Rand(BaseDataset):
    def initialize(self, opt):
        self.opt = opt
        self.root = opt.dataroot

        if opt.phase == 'train':
            self.dir_A = os.path.join(opt.dataroot, 'trainA')
            self.dir_B = os.path.join(opt.dataroot, 'trainB_exC_resized') # trainB_exM_resized

            self.dir_Adv = os.path.join(opt.dataroot, 'trainB_exC_resized') # trainB_exN_resized
        else:
            print(not_on_test_mode) 



        self.A_paths = make_dataset(self.dir_A)
        self.B_paths = make_dataset(self.dir_B)
        self.Adv_paths = make_dataset(self.dir_Adv)

        self.A_paths = sorted(self.A_paths)
        self.B_paths = sorted(self.B_paths)
        self.Adv_paths = sorted(self.Adv_paths)
        
        self.A_size = len(self.A_paths)
        self.B_size = len(self.B_paths)
        self.Adv_size = len(self.Adv_paths)

        #self.transform = get_transform_lab(opt)
        self.transform_no = get_transform_no(opt)
        self.transform_sat = get_transform_filter_sat(opt)
        self.transform_gray = get_transform_filter_gray(opt)


    def __getitem__(self, index):
        A_path = self.A_paths[index % self.A_size]

        #if self.opt.serial_batches:
        #    index_B = index % self.B_size
        #else:
        #    index_B = random.randint(0, self.B_size - 1)
        index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]

        Adv_path = self.Adv_paths[index % self.Adv_size]

        # print('(A, B) = (%d, %d)' % (index_A, index_B))
        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')
        Adv_img = Image.open(Adv_path).convert('RGB')

        #### Input A(source) <- Original, Red, Blue
        #### Input B(target) <- Blue
        ran = random.random()
        #if ran > 0.9:
        #    A = self.transform_sat(B_img)
        #    A_gray = self.transform_gray(B_img)
        #    mode = 1
        #elif ran > 0.45 and ran <= 0.9:
        #    A = self.transform_sat(A_img)
        #    A_gray = self.transform_gray(A_img)
        #    mode = 0
        #else:
        #    A = self.transform_sat(Adv_img)
        #    A_gray = self.transform_gray(A_img)
        #    mode = 0


        #### Input A(source) <- Original, C
        #### Input B(target) <- C
        if random.random()>0.1:
            A = self.transform_sat(A_img)
            A_gray = self.transform_gray(A_img)
            mode = 0
        else:
            A = self.transform_sat(B_img) 
            A_gray = self.transform_gray(B_img)
            mode = 1
        
        B = self.transform_no(B_img)

        if self.opt.which_direction == 'BtoA':
            input_nc = self.opt.output_nc
            output_nc = self.opt.input_nc
        else:
            input_nc = self.opt.input_nc
            output_nc = self.opt.output_nc

        return {'A': A, 'B': B, 'A_gray' : A_gray,
                'A_paths': A_path, 'B_paths': B_path,'mode': mode }

    def __len__(self):
        return max(self.A_size, self.B_size)

    def name(self):
        return 'AlignedDataset_Rand'

