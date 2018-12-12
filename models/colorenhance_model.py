import numpy as np
import torch
import os
from collections import OrderedDict
from torch.autograd import Variable
import itertools
import util.util as util
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks
import sys
from torch.utils.serialization import load_lua
import torch.nn as nn
import torchvision
import random

class ColorEnhance_Model(BaseModel):
    def name(self):
        return 'ColorEnhance_Model'

    def initialize(self, opt):
        BaseModel.initialize(self, opt)

        nb = opt.batchSize
        size = opt.fineSize 

        self.netG_A = networks.define_G(3, 3,opt.ngf, 'unet_upsample', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netG_B = networks.define_G(3, 3,opt.ngf, 'unet_upsample', opt.norm, not opt.no_dropout, opt.init_type, self.gpu_ids)
        self.netF = networks.define_F(self.gpu_ids, use_bn=False)




        if self.isTrain:
            use_sigmoid = opt.no_lsgan
            self.netD_A = networks.define_D(opt.input_nc, opt.ndf,opt.which_model_netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)
            self.netD_B = networks.define_D(opt.input_nc, opt.ndf,opt.which_model_netD,opt.n_layers_D, opt.norm, use_sigmoid, opt.init_type, self.gpu_ids)

        if not self.isTrain or opt.continue_train:
            which_epoch = opt.which_epoch
          
            self.load_network(self.netG_A, 'G_A', which_epoch)
            self.load_network(self.netG_B, 'G_B', which_epoch)

            if self.isTrain:
                self.load_network(self.netD_B, 'D_A', which_epoch)
                self.load_network(self.netD_B, 'D_B', which_epoch)

        if self.isTrain:
            self.old_lr = opt.lr
            self.fake_A_pool = ImagePool(opt.pool_size)
            self.fake_B_pool = ImagePool(opt.pool_size)
            # define loss functions
            self.criterionGAN = networks.GANLoss(use_lsgan=not opt.no_lsgan, tensor=self.Tensor)
            self.criterionL1 = torch.nn.L1Loss()
            self.criterionMSE = torch.nn.MSELoss()
            # initialize optimizers
            self.optimizer_G = torch.optim.Adam(itertools.chain(
                                                 self.netG_A.parameters(), self.netG_B.parameters(),
                                                 ),lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_A = torch.optim.Adam(self.netD_A.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D_B = torch.optim.Adam(self.netD_B.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))


            self.optimizers = []
            self.schedulers = []
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D_A)
            self.optimizers.append(self.optimizer_D_B)
            for optimizer in self.optimizers:
                self.schedulers.append(networks.get_scheduler(optimizer, opt))

        print('---------- Networks initialized -------------')
        networks.print_network(self.netG_A)
        networks.print_network(self.netG_B)


        if self.isTrain:
            networks.print_network(self.netD_A)
            networks.print_network(self.netD_B)
        print('-----------------------------------------------')

    def set_input(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        input_B = input['B']
        input_A_gray = input['A_gray']
        mode = input['mode']
      
       
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
            input_B = input_B.cuda(self.gpu_ids[0], async=True)
        
        self.input_A = input_A
        self.input_B = input_B
        self.input_A_gray = input_A_gray
        self.image_paths = input['A_paths' if AtoB else 'B_paths']
        self.mode = mode

    def set_input_test(self, input):
        AtoB = self.opt.which_direction == 'AtoB'
        input_A = input['A']
        
        if len(self.gpu_ids) > 0:
            input_A = input_A.cuda(self.gpu_ids[0], async=True)
        
        self.input_A = input_A
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        self.real_A = Variable(self.input_A)
        self.real_B = Variable(self.input_B)
        self.real_A_gray = Variable(self.input_A_gray)
        self.mode_A = Variable(self.mode)
        print ("[ColorEnhance] CycleGAN enhance")


    def test(self):
        with torch.no_grad():
            real_A = Variable(self.input_A)
            real_A = real_A.float()

            #print(real_A) # 1/3/848/480

            fake_B = self.netG_A(real_A)
            #print(fake_B)
            #print(whatever)

            self.real_A = real_A.data
            self.real_B = real_A.data
            self.fake_A = real_A.data
            self.fake_B = fake_B.data
            self.rec_A = real_A.data
            self.rec_B = real_A.data
        

    # get image paths
    def get_image_paths(self):
        return self.image_paths

    def backward_D_basic(self, netD, real, fake):
        # Real
        pred_real = netD(real.float())
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.float().detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        # backward
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        fake_B = self.fake_B_pool.query(self.fake_B)
        loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, fake_B)
        self.loss_D_A = loss_D_A.data[0]

    def backward_D_B(self):
        fake_A = self.fake_A_pool.query(self.fake_A)
        loss_D_B = self.backward_D_basic(self.netD_B, self.real_A_gray, fake_A)
        self.loss_D_B = loss_D_B.data[0]


    def backward_all(self):
        self.real_A = self.real_A.float().cuda() 
        self.real_B = self.real_B.float().cuda()
        self.real_A_gray = self.real_A_gray.float().cuda()

        # Normal cycleGAN
        fake_B =self.netG_A(self.real_A)
        pred_fake = self.netD_A(fake_B)
        loss_G_A = self.criterionGAN(pred_fake, True)

        fake_A = self.netG_B(self.real_B)
        pred_fake = self.netD_B(fake_A)
        loss_G_B = self.criterionGAN(pred_fake, True)

        rec_A = self.netG_B(fake_B)
        rec_B = self.netG_A(fake_A)

        # Gray Added
        loss_cycle_A = self.criterionL1(rec_A,self.real_A_gray) * 0.5
        loss_cycle_B = self.criterionL1(rec_B,self.real_B) * 0.5

        # Idt
        idt_A = self.netG_A(self.real_B)
        loss_idt_A = self.criterionL1(idt_A, self.real_B) * 1.0
 
        idt_B = self.netG_B(self.real_A)
        loss_idt_B = self.criterionL1(idt_B, self.real_A_gray) * 0.5

        # Combine
        loss_G = loss_G_A + loss_G_B + loss_cycle_A + loss_cycle_B + loss_idt_A + loss_idt_B 

        if self.mode_A == 1:
            loss_G = loss_G + self.criterionMSE(fake_B,self.real_B)
            loss_G = loss_G + self.criterionMSE(self.netF(fake_B),self.netF(self.real_B)) * 0.5
            loss_G = loss_G + self.criterionMSE(self.netF(fake_A),self.netF(self.real_A_gray)) * 0.5

        loss_G.backward()

        self.fake_B = fake_B.data
        self.fake_A = fake_A.data
        self.rec_A = rec_A.data
        self.rec_B = rec_B.data
        self.idt_A = idt_A.data
        self.idt_B = idt_B.data

        self.loss_G_A = loss_G_A.data[0]
        self.loss_G_B = loss_G_B.data[0]
        self.loss_cycle_A = loss_cycle_A.data[0]
        self.loss_cycle_B = loss_cycle_B.data[0]
        self.loss_idt_A = loss_idt_A.data[0]
        self.loss_idt_B = loss_idt_B.data[0]
        

    def optimize_parameters(self):
        # forward
        self.forward()

        self.optimizer_G.zero_grad()
        self.backward_all()
        self.optimizer_G.step()


        # D_A
        self.optimizer_D_A.zero_grad()
        self.backward_D_A()
        self.optimizer_D_A.step()

        # D_B        
        self.optimizer_D_B.zero_grad()          # 1-A
        self.backward_D_B()
        self.optimizer_D_B.step()



    def get_current_errors(self):
        ret_errors = OrderedDict([('D_A', self.loss_D_A), ('G_A', self.loss_G_A), ('Cyc_A', self.loss_cycle_A),
                                 ('D_B', self.loss_D_B), ('G_B', self.loss_G_B), ('Cyc_B',  self.loss_cycle_B)])
        if self.opt.identity > 0.0:
            ret_errors['idt_A'] = self.loss_idt_A
            ret_errors['idt_B'] = self.loss_idt_B
        return ret_errors

    def get_current_visuals(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        rec_A = util.tensor2im(self.rec_A)
        real_B = util.tensor2im(self.input_B)
        fake_A = util.tensor2im(self.fake_A)
        rec_B = util.tensor2im(self.rec_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B), ('rec_A', rec_A),
                                   ('real_B', real_B), ('fake_A', fake_A), ('rec_B', rec_B)])
        if self.opt.isTrain and self.opt.identity > 0.0:
            ret_visuals['idt_A'] = util.tensor2im(self.idt_A)
            ret_visuals['idt_B'] = util.tensor2im(self.idt_B)
        return ret_visuals

    def get_current_visuals_test(self):
        real_A = util.tensor2im(self.input_A)
        fake_B = util.tensor2im(self.fake_B)
        ret_visuals = OrderedDict([('real_A', real_A), ('fake_B', fake_B)])
        return ret_visuals

    def save(self, label):
        self.save_network(self.netG_A, 'G_A', label, self.gpu_ids)
        self.save_network(self.netG_B, 'G_B', label, self.gpu_ids)
        self.save_network(self.netD_A, 'D_A', label, self.gpu_ids)
        self.save_network(self.netD_B, 'D_B', label, self.gpu_ids)