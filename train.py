"""
(c) DI Dominik Hirner BSc. 
Institute for graphics and vision (ICG)
University of Technology Graz, Austria
e-mail: dominik.hirner@tugraz.at
"""

import os
import shutil
import sys
import glob
import numpy as np
from numpy import inf
import string
import cv2
import matplotlib.pyplot as plt
import re
import numpy.matlib
import pydot
import math
import time

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch import optim
from torchsummary import summary

import random
import itertools
from sklearn.metrics.pairwise import cosine_similarity
import configparser


config = configparser.ConfigParser()

config.read(sys.argv[1])


#continue from one of our trained weights
transfer_train = config['PARAM']['transfer_train']

#KITTI, MB or ETH
dataset = config['PARAM']['dataset']
#used as prefix for saved weights
model_name = config['PARAM']['model_name']

#folder were npz files for training are stored
#'/media/HDD/TrainingsData/MB/Half-Inf/'
#'/media/HDD/TrainingsData/ETH3D/sets/'
#'/media/HDD/TrainingsData/mc-cnn-kitti/'

input_folder = config['PARAM']['input_folder']

batch_size = int(config['PARAM']['batch_size'])


nr_epochs = int(config['PARAM']['nr_epochs'])

num_conv_feature_maps = int(config['PARAM']['num_conv_feature_maps'])

#this is dependend on hardware limitations!
#every X iteration load new train samples into RAM
load_samples = int(config['PARAM']['load_samples'])

#this shortens an epoch and is simply used so that weights are saved more often!
samples2train = int(config['PARAM']['samples2train'])

#epochs are shortened in order to save more often!
#every X "mini-epoch" save weights!
#used for more fine-granular save options 
save_weights = int(config['PARAM']['save_weights'])

print("Transfer train: " ,transfer_train)
print("Dataset: " ,dataset)
print("Model name: " ,model_name)
print("Input folder: " ,input_folder)
print("Batch-size: " ,batch_size)
print("Number of Epchs: " ,nr_epochs)
print("#Feature-maps per layer: " ,num_conv_feature_maps)
print("Load new samples: " ,load_samples)
print("Samples per Epoch: " ,samples2train)
print("Save weights every epochs: " ,save_weights)



def loadMB():
    
    nr_sets = 10
    
    s_list = []
    
    left_patches_whole = []
    right_pos_patches_whole = []
    right_neg_patches_whole = []
    
        
    all_parts = glob.glob(input_folder + '*')
    
    #dice 25 random parts (perfect,imperfect,light,exp etc.)
    r_parts = random.sample(range(len(all_parts)), nr_sets)
    
    cur_parts = []
    
    for el in range(nr_sets):
        cur_parts.append(all_parts[r_parts[el]])
        
    
    for el in range(nr_sets):
        s_list = []
        
        for item in glob.glob(cur_parts[el]+'/*'):
            s = item.split('/')[-1].split('.')[0].split('_')[-1]
            s_list.append(s)

        s_list = np.unique(s_list)    

        ridx = np.random.randint(0,len(s_list),1)

        left_pos_list = cur_parts[el] +'/left_pos_' + s_list[ridx][0] + '.npz'
        right_pos_list = cur_parts[el] + '/right_pos_' + s_list[ridx][0] + '.npz'
        right_neg_list = cur_parts[el] +'/right_neg_' + s_list[ridx][0]  + '.npz'


        cur_left = np.load(left_pos_list)['arr_0']
        cur_right_pos = np.load(right_pos_list)['arr_0']
        cur_right_neg = np.load(right_neg_list)['arr_0']  
                
        if(el == 0):
            left_patches_whole = cur_left
            right_pos_patches_whole = cur_right_pos
            right_neg_patches_whole = cur_right_neg
        else:
            left_patches_whole = np.concatenate((left_patches_whole, cur_left),axis = 0)
            right_pos_patches_whole = np.concatenate((right_pos_patches_whole, cur_right_pos),axis = 0)
            right_neg_patches_whole = np.concatenate((right_neg_patches_whole, cur_right_neg),axis = 0)
        
        
        

    return left_patches_whole, right_pos_patches_whole, right_neg_patches_whole


def loadETH3D():
    
    s_list = []

    for item in glob.glob(input_folder + '*'):
        s = item.split('/')[-1].split('.')[0].split('_')[-1]
        s_list.append(s)
    
    s_list = np.unique(s_list)    
    
    ridx = np.random.randint(0,len(s_list),1)
    
    left_pos_list = '/media/HDD/TrainingsData/ETH3D/sets/left_pos_' + s_list[ridx][0] + '.npz'
    right_pos_list = '/media/HDD/TrainingsData/ETH3D/sets/right_pos_' + s_list[ridx][0] + '.npz'
    right_neg_list = '/media/HDD/TrainingsData/ETH3D/sets/right_neg_' + s_list[ridx][0]  + '.npz'


    left_patches_whole = np.load(left_pos_list)['arr_0']
    right_pos_patches_whole = np.load(right_pos_list)['arr_0']
    right_neg_patches_whole = np.load(right_neg_list)['arr_0']  

    return left_patches_whole, right_pos_patches_whole, right_neg_patches_whole

def loadKitti():
    
    s_list = []

    for item in glob.glob(input_folder + '*'):
        s = item.split('/')[-1].split('.')[0].split('_')[-1]
        s_list.append(s)
    
    s_list = np.unique(s_list)    
    
    ridx = np.random.randint(0,len(s_list),1)
    
    left_pos_list = '/media/HDD/TrainingsData/mc-cnn-kitti/left_pos_' + s_list[ridx][0] + '.npz'
    right_pos_list = '/media/HDD/TrainingsData/mc-cnn-kitti/right_pos_' + s_list[ridx][0] + '.npz'
    right_neg_list = '/media/HDD/TrainingsData/mc-cnn-kitti/right_neg_' + s_list[ridx][0]  + '.npz'


    left_patches_whole = np.load(left_pos_list)['arr_0']
    right_pos_patches_whole = np.load(right_pos_list)['arr_0']
    right_neg_patches_whole = np.load(right_neg_list)['arr_0']  

    return left_patches_whole, right_pos_patches_whole, right_neg_patches_whole


class SiameseBranch(nn.Module):
    def __init__(self,img_ch=1):
        super(SiameseBranch,self).__init__()
        
        self.Tanh = nn.Tanh()        
        self.Conv1 = nn.Conv2d(img_ch, num_conv_feature_maps, kernel_size = 3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv2 = nn.Conv2d(num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv3 = nn.Conv2d(2*num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        self.Conv4 = nn.Conv2d(3*num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1,bias=True)
        self.Conv5 = nn.Conv2d(4*num_conv_feature_maps, num_conv_feature_maps, kernel_size=3,stride=1,padding = 1,dilation = 1, bias=True)
        
        
    def forward(self,x_in):
        
        x1 = self.Conv1(x_in) 
        x1 = self.Tanh(x1)
                
        x2 = self.Conv2(x1) 
        x2 = self.Tanh(x2)
        
        d2 = torch.cat((x1,x2),dim=1)
        
        x3 = self.Conv3(d2) 
        x3 = self.Tanh(x3)
        
        d3 = torch.cat((x1,x2,x3),dim=1)
        
        x4 = self.Conv4(d3)
        x4 = self.Tanh(x4)
        
        d4 = torch.cat((x1,x2,x3,x4),dim=1)
        
        x5 = self.Conv5(d4)
        x5 = self.Tanh(x5)
        
        return x5
    
    
branch = SiameseBranch()
branch = branch.cuda()


##python3 version!!!!
def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().decode('utf-8').rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data, scale


def filterCostVolMedian(cost_vol):
    
    d,h,w = cost_vol.shape
    cost_vol_filtered = np.zeros((d,h,w))

    for disp in range(d):
        cur_slice = cost_vol[disp,:,:].astype(np.float32)
        cur_slice_filtered = cv2.medianBlur(cur_slice, 5)     
        cost_vol_filtered[disp,:,:] = cur_slice_filtered
        
    return cost_vol_filtered


def createCostVolRL(left_im,right_im,max_disp):

    left_im = np.mean(left_im, axis=2)
    right_im = np.mean(right_im, axis=2)

    a_h, a_w = left_im.shape

    left_im = np.reshape(left_im, [1,1,a_h,a_w])
    right_im = np.reshape(right_im, [1,1,a_h,a_w])

    with torch.no_grad():
        
        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)


        _,f,h,w = left_feat.shape
        cost_vol = np.zeros((max_disp+1,a_h,a_w))

        for disp in range(0,max_disp+1):

            
            if(disp == 0):
                sim_score = cos(right_feat, left_feat)
                sim_score_cpy = sim_score.cpu().data.numpy()
                cost_vol[disp,:,:] = np.squeeze(sim_score_cpy)
                
            else:    
                left_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)
                left_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)
                left_appended = torch.cat([left_feat,left_shift],3)

                _,f,h_ap,w_ap = left_appended.shape
                left_shifted[:,:,:,:] = left_appended[:,:,:,disp:w_ap]

                sim_score = cos(right_feat, left_shifted)
                sim_score_cpy = sim_score.cpu().data.numpy()
                cost_vol[disp,:,:] = np.squeeze(sim_score_cpy)
                
                
                
    del left_im
    del right_im
    del left_imT
    del right_imT
    del left_feat
    del right_feat
    del left_shifted
    del left_shift
    del left_appended
    torch.cuda.empty_cache()                
        
    return cost_vol


def writePFM(file, image, scale=1):
    file = open(file, 'wb')

    color = None

    if image.dtype.name != 'float32':
        raise Exception('Image dtype must be float32.')

    image = np.flipud(image)

    if len(image.shape) == 3 and image.shape[2] == 3:  # color image
        color = True
    elif len(image.shape) == 2 or len(image.shape) == 3 and image.shape[2] == 1:  # greyscale
        color = False
    else:
        raise Exception('Image must have H x W x 3, H x W x 1 or H x W dimensions.')

    file.write('PF\n'.encode() if color else 'Pf\n'.encode())
    file.write('%d %d\n'.encode() % (image.shape[1], image.shape[0]))

    endian = image.dtype.byteorder

    if endian == '<' or endian == '=' and sys.byteorder == 'little':
        scale = -scale

    file.write('%f\n'.encode() % scale)

    image.tofile(file)
    
    
def LR_Check(first_output, second_output):    
    
    rows ,cols = first_output.shape
    mask = np.ones((rows,cols))

    
    for col in range(0,cols):
        for row in range(0,rows):

            disp_val_first = first_output[row,col]
            
            #HACK FOR NOW! 0 Disp should not be possible!
#            if((disp_val_first == 0) or (disp_val_first == 1) or (disp_val_first == 2) or (disp_val_first == 3) or (disp_val_first == 4) or (disp_val_first == 5)):
#                mask[row,col] = 0
#                first_output[row,col] = np.nan
                
#                continue
                
            
            if ((col- int(disp_val_first)) >= cols):
                mask[row,col] = 0
                first_output[row,col] = np.nan
                continue


            if ((col-int(disp_val_first)) < 0.0):
                mask[row,col] = 0
                first_output[row,col] = np.nan  
                continue

            disp_val_second = second_output[row, col - int(disp_val_first)]      

            if(np.abs(int(disp_val_first) - int(disp_val_second)) > 1.1):
                mask[row,col] = 0
                first_output[row,col] = np.nan  
                continue    
    
    return first_output    
    
    
def TestImage(fn_left, fn_right, max_disp, im_to_save, gt = None, filtered = True, lr_check = True):
    
    four_pe = 0.0
    two_pe = 0.0
    pf_pe = 0.0
    
    left = cv2.imread(fn_left)
    right = cv2.imread(fn_right)

    
    disp_map = []
    
    if(filtered):
        
        cost_vol = createCostVol(left,right,max_disp)
        cost_vol_median = filterCostVolMedian(cost_vol)
        cost_vol_median2 = filterCostVolMedian(cost_vol_median)

        cost_vol_filtered = filterCostVolBilat(cost_vol_median2)
        disp_map = np.argmax(cost_vol_filtered, axis=0)                        
        
        if(lr_check):
            cost_vol_RL = createCostVolRL(left,right,max_disp)
            cost_vol_RL_f = filterCostVolBilat(cost_vol_RL)
            disp_map_RL = np.argmax(cost_vol_RL_f, axis=0)     
                        
            final_disp = LR_Check(disp_map, disp_map_RL)

            writePFM(im_to_save, final_disp.astype(np.float32), scale=1)
            
        writePFM(im_to_save, disp_map.astype(np.float32), scale=1)

    else:
        
        cost_vol = createCostVol(left,right,max_disp)
        disp_map = np.argmax(cost_vol, axis=0)  
        
        
        if(lr_check):
            cost_vol_RL = createCostVolRL(left,right,130)
            disp_map_RL = np.argmax(cost_vol_RL, axis=0)     
            final_disp = LR_Check(disp_map, disp_map_RL)
            writePFM(im_to_save, final_disp.astype(np.float32), scale=1)
            
        writePFM(im_to_save, disp_map.astype(np.float32), scale=1) 
        
    
    if(gt is not None):
        gt_im,_ = readPFM(gt)        
        
        h,w = gt_im.shape        
        
        nr_px = h * w
        nr_px = float(nr_px)
        
        abs_error_im = np.abs(disp_map - gt_im)        
        
        four_pe = (float(np.count_nonzero(abs_error_im >= 4.0) ) / nr_px) * 100.0        
        two_pe = (float(np.count_nonzero(abs_error_im >= 2.0) ) / nr_px) * 100.0        
        pf_pe = (float(np.count_nonzero(abs_error_im >= 0.5) ) / nr_px) * 100.0
        
        
    return four_pe, two_pe, pf_pe


def TestMB(mb_folder, output_folder, plot=False, filtered=True):
    #Adiron
    if(filtered):
        
        #Adiron
        adir_four_pe, adir_two_pe, adir_pf_pe = TestImage(mb_folder + '/Adirondack-imperfect/im0.png.H.png', mb_folder + '/Adirondack-imperfect/im1.png.H.png',145, output_folder + '/adiron_filtered.pfm', mb_folder + '/Adirondack-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                
        
        #Backpack
        backpack_four_pe, backpack_two_pe, backpack_pf_pe = TestImage(mb_folder + '/Backpack-imperfect/im0.png.H.png', mb_folder + '/Backpack-imperfect/im1.png.H.png',130, output_folder + '/backpack_filtered.pfm', mb_folder + '/Backpack-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #Bicycle
        bicycle_four_pe, bicycle_two_pe, bicycle_pf_pe = TestImage(mb_folder + '/Bicycle1-imperfect/im0.png.H.png', mb_folder + '/Bicycle1-imperfect/im1.png.H.png',90, output_folder + '/bicycle_filtered.pfm', mb_folder + '/Bicycle1-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #Cable
        cable_four_pe, cable_two_pe, cable_pf_pe = TestImage(mb_folder + '/Cable-imperfect/im0.png.H.png', mb_folder + '/Cable-imperfect/im1.png.H.png',230, output_folder + '/cable_filtered.pfm', mb_folder + '/Cable-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #clasroom1
        class_four_pe, class_two_pe, class_pf_pe = TestImage(mb_folder + '/Classroom1-imperfect/im0.png.H.png', mb_folder + '/Classroom1-imperfect/im1.png.H.png',130, output_folder + '/classroom_filtered.pfm', mb_folder + '/Classroom1-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #couch
        couch_four_pe, couch_two_pe, couch_pf_pe = TestImage(mb_folder + '/Couch-imperfect/im0.png.H.png', mb_folder + '/Couch-imperfect/im1.png.H.png',315, output_folder + '/couch_filtered.pfm', mb_folder + '/Couch-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #flowers
        flowers_four_pe, flowers_two_pe, flowers_pf_pe = TestImage(mb_folder + '/Flowers-imperfect/im0.png.H.png', mb_folder + '/Flowers-imperfect/im1.png.H.png',320, output_folder + '/flowers_filtered.pfm', mb_folder + '/Flowers-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #jade
        jade_four_pe, jade_two_pe, jade_pf_pe = TestImage(mb_folder + '/Jadeplant-imperfect/im0.png.H.png', mb_folder + '/Jadeplant-imperfect/im1.png.H.png',320, output_folder + '/jade_filtered.pfm', mb_folder + '/Jadeplant-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #mask
        mask_four_pe, mask_two_pe, mask_pf_pe = TestImage(mb_folder + '/Mask-imperfect/im0.png.H.png', mb_folder + '/Mask-imperfect/im1.png.H.png',240, output_folder + '/mask_filtered.pfm', mb_folder + '/Mask-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #motorcycle
        motor_four_pe, motor_two_pe, motor_pf_pe = TestImage(mb_folder + '/Motorcycle-imperfect/im0.png.H.png', mb_folder + '/Motorcycle-imperfect/im1.png.H.png',140, output_folder + '/motorcycle_filtered.pfm', mb_folder + '/Motorcycle-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #piano
        piano_four_pe, piano_two_pe, piano_pf_pe = TestImage(mb_folder + '/Piano-imperfect/im0.png.H.png', mb_folder + '/Piano-imperfect/im1.png.H.png',130, output_folder + '/piano_filtered.pfm', mb_folder + '/Piano-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #pipes
        pipes_four_pe, pipes_two_pe, pipes_pf_pe = TestImage(mb_folder + '/Pipes-imperfect/im0.png.H.png', mb_folder + '/Pipes-imperfect/im1.png.H.png',150, output_folder + '/pipes_filtered.pfm', mb_folder + '/Pipes-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #playroom
        playroom_four_pe, playroom_two_pe, playroom_pf_pe = TestImage(mb_folder + '/Playroom-imperfect/im0.png.H.png', mb_folder + '/Playroom-imperfect/im1.png.H.png',165, output_folder + '/playroom_filtered.pfm', mb_folder + '/Playroom-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #playtable
        playtable_four_pe, playtable_two_pe, playtable_pf_pe = TestImage(mb_folder + '/Playtable-imperfect/im0.png.H.png', mb_folder + '/Playtable-imperfect/im1.png.H.png',145, output_folder + '/playtable_filtered.pfm', mb_folder + '/Playtable-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #recycle
        recycle_four_pe, recycle_two_pe, recycle_pf_pe = TestImage(mb_folder + '/Recycle-imperfect/im0.png.H.png', mb_folder + '/Recycle-imperfect/im1.png.H.png',130, output_folder + '/recycle_filtered.pfm', mb_folder + '/Recycle-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #shelves
        shelves_four_pe, shelves_two_pe, shelves_pf_pe = TestImage(mb_folder + '/Shelves-imperfect/im0.png.H.png', mb_folder + '/Shelves-imperfect/im1.png.H.png',120, output_folder + '/shelves_filtered.pfm', mb_folder + '/Shelves-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #shopvac
        shopvac_four_pe, shopvac_two_pe, shopvac_pf_pe = TestImage(mb_folder + '/Shopvac-imperfect/im0.png.H.png', mb_folder + '/Shopvac-imperfect/im1.png.H.png',555, output_folder + '/shopvac_filtered.pfm', mb_folder + '/Shopvac-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #sticks
        sticks_four_pe, sticks_two_pe, sticks_pf_pe = TestImage(mb_folder + '/Sticks-imperfect/im0.png.H.png', mb_folder + '/Sticks-imperfect/im1.png.H.png',155, output_folder + '/sticks_filtered.pfm', mb_folder + '/Sticks-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #storage
        storage_four_pe, storage_two_pe, storage_pf_pe = TestImage(mb_folder + '/Storage-imperfect/im0.png.H.png', mb_folder + '/Storage-imperfect/im1.png.H.png',330, output_folder + '/storage_filtered.pfm', mb_folder + '/Storage-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #swords1
        swords1_four_pe, swords1_two_pe, swords1_pf_pe = TestImage(mb_folder + '/Sword1-imperfect/im0.png.H.png', mb_folder + '/Sword1-imperfect/im1.png.H.png',130, output_folder + '/swords1_filtered.pfm', mb_folder + '/Sword1-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #swords2
        swords2_four_pe, swords2_two_pe, swords2_pf_pe = TestImage(mb_folder + '/Sword2-imperfect/im0.png.H.png', mb_folder + '/Sword2-imperfect/im1.png.H.png',185, output_folder + '/swords2_filtered.pfm', mb_folder + '/Sword2-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #umbrella
        umbrella_four_pe, umbrella_two_pe, umbrella_pf_pe = TestImage(mb_folder + '/Umbrella-imperfect/im0.png.H.png', mb_folder + '/Umbrella-imperfect/im1.png.H.png',125, output_folder + '/umbrella_filtered.pfm', mb_folder + '/Umbrella-imperfect/disp0.pfm.H.pfm', lr_check = False)
        torch.cuda.empty_cache()                

        #vintage
        vintage_four_pe, vintage_two_pe, vintage_pf_pe = TestImage(mb_folder + '/Vintage-imperfect/im0.png.H.png', mb_folder + '/Vintage-imperfect/im1.png.H.png',380, output_folder + '/vintage_filtered.pfm', mb_folder + '/Vintage-imperfect/disp0.pfm.H.pfm', lr_check = False)   
        torch.cuda.empty_cache()                
        
    else:   
        
        #Adiron
        adir_four_pe, adir_two_pe, adir_pf_pe = TestImage(mb_folder + '/Adirondack-imperfect/im0.png.H.png', mb_folder + '/Adirondack-imperfect/im1.png.H.png',145, output_folder + '/adiron.pfm', mb_folder + '/Adirondack-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #Backpack
        backpack_four_pe, backpack_two_pe, backpack_pf_pe = TestImage(mb_folder + '/Backpack-imperfect/im0.png.H.png', mb_folder + '/Backpack-imperfect/im1.png.H.png',130, output_folder + '/backpack.pfm', mb_folder + '/Backpack-imperfect/disp0.pfm.H.pfm',False, lr_check = False)
        torch.cuda.empty_cache()                

        #Bicycle
        bicycle_four_pe, bicycle_two_pe, bicycle_pf_pe = TestImage(mb_folder + '/Bicycle1-imperfect/im0.png.H.png', mb_folder + '/Bicycle1-imperfect/im1.png.H.png',90, output_folder + '/bicycle.pfm', mb_folder + '/Bicycle1-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #Cable
        cable_four_pe, cable_two_pe, cable_pf_pe = TestImage(mb_folder + '/Cable-imperfect/im0.png.H.png', mb_folder + '/Cable-imperfect/im1.png.H.png',230, output_folder + '/cable.pfm', mb_folder + '/Cable-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #clasroom1
        class_four_pe, class_two_pe, class_pf_pe = TestImage(mb_folder + '/Classroom1-imperfect/im0.png.H.png', mb_folder + '/Classroom1-imperfect/im1.png.H.png',130, output_folder + '/classroom.pfm', mb_folder + '/Classroom1-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #couch
        couch_four_pe, couch_two_pe, couch_pf_pe = TestImage(mb_folder + '/Couch-imperfect/im0.png.H.png', mb_folder + '/Couch-imperfect/im1.png.H.png',315, output_folder + '/couch.pfm', mb_folder + '/Couch-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #flowers
        flowers_four_pe, flowers_two_pe, flowers_pf_pe = TestImage(mb_folder + '/Flowers-imperfect/im0.png.H.png', mb_folder + '/Flowers-imperfect/im1.png.H.png',320, output_folder + '/flowers.pfm', mb_folder + '/Flowers-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #jade
        jade_four_pe, jade_two_pe, jade_pf_pe = TestImage(mb_folder + '/Jadeplant-imperfect/im0.png.H.png', mb_folder + '/Jadeplant-imperfect/im1.png.H.png',320, output_folder + '/jade.pfm', mb_folder + '/Jadeplant-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #mask
        mask_four_pe, mask_two_pe, mask_pf_pe = TestImage(mb_folder + '/Mask-imperfect/im0.png.H.png', mb_folder + '/Mask-imperfect/im1.png.H.png',240, output_folder + '/mask.pfm', mb_folder + '/Mask-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #motorcycle
        motor_four_pe, motor_two_pe, motor_pf_pe = TestImage(mb_folder + '/Motorcycle-imperfect/im0.png.H.png', mb_folder + '/Motorcycle-imperfect/im1.png.H.png',140, output_folder + '/motorcycle.pfm', mb_folder + '/Motorcycle-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #piano
        piano_four_pe, piano_two_pe, piano_pf_pe = TestImage(mb_folder + '/Piano-imperfect/im0.png.H.png', mb_folder + '/Piano-imperfect/im1.png.H.png',130, output_folder + '/piano.pfm', mb_folder + '/Piano-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #pipes
        pipes_four_pe, pipes_two_pe, pipes_pf_pe = TestImage(mb_folder + '/Pipes-imperfect/im0.png.H.png', mb_folder + '/Pipes-imperfect/im1.png.H.png',150, output_folder + '/pipes.pfm', mb_folder + '/Pipes-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #playroom
        playroom_four_pe, playroom_two_pe, playroom_pf_pe = TestImage(mb_folder + '/Playroom-imperfect/im0.png.H.png', mb_folder + '/Playroom-imperfect/im1.png.H.png',165, output_folder + '/playroom.pfm', mb_folder + '/Playroom-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #playtable
        playtable_four_pe, playtable_two_pe, playtable_pf_pe = TestImage(mb_folder + '/Playtable-imperfect/im0.png.H.png', mb_folder + '/Playtable-imperfect/im1.png.H.png',145, output_folder + '/playtable.pfm', mb_folder + '/Playtable-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #recycle
        recycle_four_pe, recycle_two_pe, recycle_pf_pe = TestImage(mb_folder + '/Recycle-imperfect/im0.png.H.png', mb_folder + '/Recycle-imperfect/im1.png.H.png',130, output_folder + '/recycle.pfm', mb_folder + '/Recycle-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #shelves
        shelves_four_pe, shelves_two_pe, shelves_pf_pe = TestImage(mb_folder + '/Shelves-imperfect/im0.png.H.png', mb_folder + '/Shelves-imperfect/im1.png.H.png',120, output_folder + '/shelves.pfm', mb_folder + '/Shelves-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #shopvac
        shopvac_four_pe, shopvac_two_pe, shopvac_pf_pe = TestImage(mb_folder + '/Shopvac-imperfect/im0.png.H.png', mb_folder + '/Shopvac-imperfect/im1.png.H.png',555, output_folder + '/shopvac.pfm', mb_folder + '/Shopvac-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #sticks
        sticks_four_pe, sticks_two_pe, sticks_pf_pe = TestImage(mb_folder + '/Sticks-imperfect/im0.png.H.png', mb_folder + '/Sticks-imperfect/im1.png.H.png',155, output_folder + '/sticks.pfm', mb_folder + '/Sticks-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #storage
        storage_four_pe, storage_two_pe, storage_pf_pe = TestImage(mb_folder + '/Storage-imperfect/im0.png.H.png', mb_folder + '/Storage-imperfect/im1.png.H.png',330, output_folder + '/storage.pfm', mb_folder + '/Storage-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #swords1
        swords1_four_pe, swords1_two_pe, swords1_pf_pe = TestImage(mb_folder + '/Sword1-imperfect/im0.png.H.png', mb_folder + '/Sword1-imperfect/im1.png.H.png',130, output_folder + '/swords1.pfm', mb_folder + '/Sword1-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #swords2
        swords2_four_pe, swords2_two_pe, swords2_pf_pe = TestImage(mb_folder + '/Sword2-imperfect/im0.png.H.png', mb_folder + '/Sword2-imperfect/im1.png.H.png',185, output_folder + '/swords2.pfm', mb_folder + '/Sword2-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #umbrella
        umbrella_four_pe, umbrella_two_pe, umbrella_pf_pe = TestImage(mb_folder + '/Umbrella-imperfect/im0.png.H.png', mb_folder + '/Umbrella-imperfect/im1.png.H.png',125, output_folder + '/umbrella.pfm', mb_folder + '/Umbrella-imperfect/disp0.pfm.H.pfm', False, lr_check = False)
        torch.cuda.empty_cache()                

        #vintage
        vintage_four_pe, vintage_two_pe, vintage_pf_pe = TestImage(mb_folder + '/Vintage-imperfect/im0.png.H.png', mb_folder + '/Vintage-imperfect/im1.png.H.png',380, output_folder + '/vintage.pfm', mb_folder + '/Vintage-imperfect/disp0.pfm.H.pfm', False, lr_check = False)   
        torch.cuda.empty_cache()                
   
    
    if(plot):
        print('Adirondack:')
        print("4-PE:  {}".format(adir_four_pe))
        print("2-PE: {}".format(adir_two_pe))
        print("0.5-PE: {}".format(adir_pf_pe))
 
        print('Backpack:')
        print("4-PE:  {}".format(backpack_four_pe))
        print("2-PE: {}".format(backpack_two_pe))
        print("0.5-PE: {}".format(backpack_pf_pe))
         
        print('Bicycle:')
        print("4-PE:  {}".format(bicycle_four_pe))
        print("2-PE: {}".format(bicycle_two_pe))
        print("0.5-PE: {}".format(bicycle_pf_pe))
 
        print('Cable:')
        print("4-PE:  {}".format(cable_four_pe))
        print("2-PE: {}".format(cable_two_pe))
        print("0.5-PE: {}".format(cable_pf_pe))
   
        print('Classroom:')
        print("4-PE:  {}".format(class_four_pe))
        print("2-PE: {}".format(class_two_pe))
        print("0.5-PE: {}".format(class_pf_pe))
    
        print('Couch:')
        print("4-PE:  {}".format(couch_four_pe))
        print("2-PE: {}".format(couch_two_pe))
        print("0.5-PE: {}".format(couch_pf_pe))
     
        print('Flowers:')
        print("4-PE:  {}".format(flowers_four_pe))
        print("2-PE: {}".format(flowers_two_pe))
        print("0.5-PE: {}".format(flowers_pf_pe))
 
        print('Jade:')
        print("4-PE:  {}".format(jade_four_pe))
        print("2-PE: {}".format(jade_two_pe))
        print("0.5-PE: {}".format(jade_pf_pe))
  
        print('Mask:')
        print("4-PE:  {}".format(mask_four_pe))
        print("2-PE: {}".format(mask_two_pe))
        print("0.5-PE: {}".format(mask_pf_pe))
  
        print('Motorcycle:')
        print("4-PE:  {}".format(motor_four_pe))
        print("2-PE: {}".format(motor_two_pe))
        print("0.5-PE: {}".format(motor_pf_pe))
   
        print('Piano:')
        print("4-PE:  {}".format(piano_four_pe))
        print("2-PE: {}".format(piano_two_pe))
        print("0.5-PE: {}".format(piano_pf_pe))
    
        print('Pipes:')
        print("4-PE:  {}".format(pipes_four_pe))
        print("2-PE: {}".format(pipes_two_pe))
        print("0.5-PE: {}".format(pipes_pf_pe))
     
        print('Playroom:')
        print("4-PE:  {}".format(playroom_four_pe))
        print("2-PE: {}".format(playroom_two_pe))
        print("0.5-PE: {}".format(playroom_pf_pe))
     
        print('Playtable:')
        print("4-PE:  {}".format(playtable_four_pe))
        print("2-PE: {}".format(playtable_two_pe))
        print("0.5-PE: {}".format(playtable_pf_pe))
      
        print('Recycle:')
        print("4-PE:  {}".format(recycle_four_pe))
        print("2-PE: {}".format(recycle_two_pe))
        print("0.5-PE: {}".format(recycle_pf_pe))
       
        print('Shelves:')
        print("4-PE:  {}".format(shelves_four_pe))
        print("2-PE: {}".format(shelves_two_pe))
        print("0.5-PE: {}".format(shelves_pf_pe))
        
        print('Shopvac:')
        print("4-PE:  {}".format(shopvac_four_pe))
        print("2-PE: {}".format(shopvac_two_pe))
        print("0.5-PE: {}".format(shopvac_pf_pe))
         
        print('Sticks:')
        print("4-PE:  {}".format(sticks_four_pe))
        print("2-PE: {}".format(sticks_two_pe))
        print("0.5-PE: {}".format(sticks_pf_pe))
          
        print('Storage:')
        print("4-PE:  {}".format(storage_four_pe))
        print("2-PE: {}".format(storage_two_pe))
        print("0.5-PE: {}".format(storage_pf_pe))
           
        print('Swords1:')
        print("4-PE:  {}".format(swords1_four_pe))
        print("2-PE: {}".format(swords1_two_pe))
        print("0.5-PE: {}".format(swords1_pf_pe))
           
        print('Swords2:')
        print("4-PE:  {}".format(swords2_four_pe))
        print("2-PE: {}".format(swords2_two_pe))
        print("0.5-PE: {}".format(swords2_pf_pe))
          
        print('Umbrella:')
        print("4-PE:  {}".format(umbrella_four_pe))
        print("2-PE: {}".format(umbrella_two_pe))
        print("0.5-PE: {}".format(umbrella_pf_pe))
           
        print('Vintage:')
        print("4-PE:  {}".format(vintage_four_pe))
        print("2-PE: {}".format(vintage_two_pe))
        print("0.5-PE: {}".format(vintage_pf_pe))

        
    avg_two_pe = (adir_two_pe + backpack_two_pe + bicycle_two_pe + cable_two_pe + class_two_pe + couch_two_pe + flowers_two_pe + jade_two_pe + mask_two_pe + motor_two_pe + piano_two_pe + pipes_two_pe + playroom_two_pe + playtable_two_pe + recycle_two_pe + shelves_two_pe + shopvac_two_pe + sticks_two_pe + storage_two_pe + swords1_two_pe + swords2_two_pe + umbrella_two_pe + vintage_two_pe) / 23.0 
    
    return avg_two_pe


def TestMBHP(mb_folder, output_folder, plot=False, filtered=True):
    #Adiron
    if(filtered):
        
        #Adiron
        adir_four_pe, adir_two_pe, adir_pf_pe = TestImage(mb_folder + '/Adirondack/im0.png', mb_folder + '/Adirondack/im1.png',145, output_folder + '/adiron_filtered.pfm', mb_folder + '/Adirondack/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #ArtL
        artl_four_pe, artl_two_pe, artl_pf_pe = TestImage(mb_folder + '/ArtL/im0.png', mb_folder + '/ArtL/im1.png',128, output_folder + '/artl_filtered.pfm', mb_folder + '/ArtL/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Jade
        jade_four_pe, jade_two_pe, jade_pf_pe = TestImage(mb_folder + '/Jadeplant/im0.png', mb_folder + '/Jadeplant/im1.png',320, output_folder + '/jade_filtered.pfm', mb_folder + '/Jadeplant/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #motorcycle
        motor_four_pe, motor_two_pe, motor_pf_pe = TestImage(mb_folder + '/Motorcycle/im0.png', mb_folder + '/Motorcycle/im1.png',140, output_folder + '/motor_filtered.pfm', mb_folder + '/Motorcycle/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #motorcycleE
        motorE_four_pe, motorE_two_pe, motorE_pf_pe = TestImage(mb_folder + '/MotorcycleE/im0.png', mb_folder + '/MotorcycleE/im1.png',140, output_folder + '/motorE_filtered.pfm', mb_folder + '/MotorcycleE/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Piano
        piano_four_pe, piano_two_pe, piano_pf_pe = TestImage(mb_folder + '/Piano/im0.png', mb_folder + '/Piano/im1.png',130, output_folder + '/piano_filtered.pfm', mb_folder + '/Piano/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #PianoL
        pianoL_four_pe, pianoL_two_pe, pianoL_pf_pe = TestImage(mb_folder + '/PianoL/im0.png', mb_folder + '/PianoL/im1.png',130, output_folder + '/pianoL_filtered.pfm', mb_folder + '/PianoL/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Pipes
        pipes_four_pe, pipes_two_pe, pipes_pf_pe = TestImage(mb_folder + '/Pipes/im0.png', mb_folder + '/Pipes/im1.png',150, output_folder + '/pipes_filtered.pfm', mb_folder + '/Pipes/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Playroom
        playroom_four_pe, playroom_two_pe, playroom_pf_pe = TestImage(mb_folder + '/Playroom/im0.png', mb_folder + '/Playroom/im1.png',165, output_folder + '/playroom_filtered.pfm', mb_folder + '/Playroom/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Playtable
        playtable_four_pe, playtable_two_pe, playtable_pf_pe = TestImage(mb_folder + '/Playtable/im0.png', mb_folder + '/Playtable/im1.png',145, output_folder + '/playtable_filtered.pfm', mb_folder + '/Playtable/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #PlaytableP
#        playtableP_four_pe, playtableP_two_pe, playtableP_pf_pe = TestImage(mb_folder + '/PlaytableP/im0.png', mb_folder + '/PlaytableP/im1.png',145, output_folder + '/playtableP_filtered.pfm', mb_folder + '/PlaytableP/disp0.pfm',filtered = True, lr_check = False)
         #Recycle
        recycle_four_pe, recycle_two_pe, recycle_pf_pe = TestImage(mb_folder + '/Recycle/im0.png', mb_folder + '/Recycle/im1.png',130, output_folder + '/recycle_filtered.pfm', mb_folder + '/Recycle/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Shelves
        shelves_four_pe, shelves_two_pe, shelves_pf_pe = TestImage(mb_folder + '/Shelves/im0.png', mb_folder + '/Shelves/im1.png',120, output_folder + '/shelves_filtered.pfm', mb_folder + '/Shelves/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Teddy
        teddy_four_pe, teddy_two_pe, teddy_pf_pe = TestImage(mb_folder + '/Teddy/im0.png', mb_folder + '/Teddy/im1.png',128, output_folder + '/teddy_filtered.pfm', mb_folder + '/Teddy/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                
        #Vintage
        vintage_four_pe, vintage_two_pe, vintage_pf_pe = TestImage(mb_folder + '/Vintage/im0.png', mb_folder + '/Vintage/im1.png',380, output_folder + '/vintage_filtered.pfm', mb_folder + '/Vintage/disp0.pfm',filtered = True, lr_check = False)
        torch.cuda.empty_cache()                   
    
    
    else:   
        
        #Adiron
        adir_four_pe, adir_two_pe, adir_pf_pe = TestImage(mb_folder + '/Adirondack/im0.png', mb_folder + '/Adirondack/im1.png',145, output_folder + '/adiron.pfm', mb_folder + '/Adirondack/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()                
        #ArtL
        artl_four_pe, artl_two_pe, artl_pf_pe = TestImage(mb_folder + '/ArtL/im0.png', mb_folder + '/ArtL/im1.png',128, output_folder + '/artl.pfm', mb_folder + '/ArtL/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()                
        #Jade
        jade_four_pe, jade_two_pe, jade_pf_pe = TestImage(mb_folder + '/Jadeplant/im0.png', mb_folder + '/Jadeplant/im1.png',320, output_folder + '/jade.pfm', mb_folder + '/Jadeplant/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()
        #motorcycle
        motor_four_pe, motor_two_pe, motor_pf_pe = TestImage(mb_folder + '/Motorcycle/im0.png', mb_folder + '/Motorcycle/im1.png',140, output_folder + '/motor.pfm', mb_folder + '/Motorcycle/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()                
        #motorcycleE
        motorE_four_pe, motorE_two_pe, motorE_pf_pe = TestImage(mb_folder + '/MotorcycleE/im0.png', mb_folder + '/MotorcycleE/im1.png',140, output_folder + '/motorE.pfm', mb_folder + '/MotorcycleE/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()                
        #Piano
        piano_four_pe, piano_two_pe, piano_pf_pe = TestImage(mb_folder + '/Piano/im0.png', mb_folder + '/Piano/im1.png',130, output_folder + '/piano.pfm', mb_folder + '/Piano/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()                
        #PianoL
        pianoL_four_pe, pianoL_two_pe, pianoL_pf_pe = TestImage(mb_folder + '/PianoL/im0.png', mb_folder + '/PianoL/im1.png',130, output_folder + '/pianoL.pfm', mb_folder + '/PianoL/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()
        #Pipes
        pipes_four_pe, pipes_two_pe, pipes_pf_pe = TestImage(mb_folder + '/Pipes/im0.png', mb_folder + '/Pipes/im1.png',150, output_folder + '/pipes.pfm', mb_folder + '/Pipes/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()                
        #Playroom
        playroom_four_pe, playroom_two_pe, playroom_pf_pe = TestImage(mb_folder + '/Playroom/im0.png', mb_folder + '/Playroom/im1.png',165, output_folder + '/playroom.pfm', mb_folder + '/Playroom/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()
        #Playtable
        playtable_four_pe, playtable_two_pe, playtable_pf_pe = TestImage(mb_folder + '/Playtable/im0.png', mb_folder + '/Playtable/im1.png',145, output_folder + '/playtable.pfm', mb_folder + '/Playtable/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()
        #PlaytableP
#        playtableP_four_pe, playtableP_two_pe, playtableP_pf_pe = TestImage(mb_folder + '/PlaytableP/im0.png', mb_folder + '/PlaytableP/im1.png',145, output_folder + '/playtableP.pfm', mb_folder + '/PlaytableP/disp0.pfm',filtered = False, lr_check = False)
         #Recycle
        recycle_four_pe, recycle_two_pe, recycle_pf_pe = TestImage(mb_folder + '/Recycle/im0.png', mb_folder + '/Recycle/im1.png',130, output_folder + '/recycle.pfm', mb_folder + '/Recycle/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()
        #Shelves
        shelves_four_pe, shelves_two_pe, shelves_pf_pe = TestImage(mb_folder + '/Shelves/im0.png', mb_folder + '/Shelves/im1.png',120, output_folder + '/shelves.pfm', mb_folder + '/Shelves/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()
        #Teddy
        teddy_four_pe, teddy_two_pe, teddy_pf_pe = TestImage(mb_folder + '/Teddy/im0.png', mb_folder + '/Teddy/im1.png',128, output_folder + '/teddy.pfm', mb_folder + '/Teddy/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()
        #Vintage
        vintage_four_pe, vintage_two_pe, vintage_pf_pe = TestImage(mb_folder + '/Vintage/im0.png', mb_folder + '/Vintage/im1.png',380, output_folder + '/vintage.pfm', mb_folder + '/Vintage/disp0.pfm',filtered = False, lr_check = False)
        torch.cuda.empty_cache()    
    
    if(plot):
        print('Adirondack:')
        print("4-PE:  {}".format(adir_four_pe))
        print("2-PE: {}".format(adir_two_pe))
        print("0.5-PE: {}".format(adir_pf_pe))
 
        print('ArtL:')
        print("4-PE:  {}".format(artl_four_pe))
        print("2-PE: {}".format(artl_two_pe))
        print("0.5-PE: {}".format(artl_pf_pe))

        print('Jadeplant:')
        print("4-PE:  {}".format(jade_four_pe))
        print("2-PE: {}".format(jade_two_pe))
        print("0.5-PE: {}".format(jade_pf_pe))
         
        print('Motorcycle:')
        print("4-PE:  {}".format(motor_four_pe))
        print("2-PE: {}".format(motor_two_pe))
        print("0.5-PE: {}".format(motor_pf_pe))
          
        print('MotorcycleE:')
        print("4-PE:  {}".format(motorE_four_pe))
        print("2-PE: {}".format(motorE_two_pe))
        print("0.5-PE: {}".format(motorE_pf_pe))
        
        print('Piano:')
        print("4-PE:  {}".format(piano_four_pe))
        print("2-PE: {}".format(piano_two_pe))
        print("0.5-PE: {}".format(piano_pf_pe))
         
        print('PianoL:')
        print("4-PE:  {}".format(pianoL_four_pe))
        print("2-PE: {}".format(pianoL_two_pe))
        print("0.5-PE: {}".format(pianoL_pf_pe))
          
        print('Pipes:')
        print("4-PE:  {}".format(pipes_four_pe))
        print("2-PE: {}".format(pipes_two_pe))
        print("0.5-PE: {}".format(pipes_pf_pe))
          
        print('Playroom:')
        print("4-PE:  {}".format(playroom_four_pe))
        print("2-PE: {}".format(playroom_two_pe))
        print("0.5-PE: {}".format(playroom_pf_pe))
         
        print('Playtable:')
        print("4-PE:  {}".format(playtable_four_pe))
        print("2-PE: {}".format(playtable_two_pe))
        print("0.5-PE: {}".format(playtable_pf_pe))
         
#        print('PlaytableP:')
#        print("4-PE:  {}".format(playtableP_four_pe))
#        print("2-PE: {}".format(playtableP_two_pe))
#        print("0.5-PE: {}".format(playtableP_pf_pe))
        
        print('Recycle:')
        print("4-PE:  {}".format(recycle_four_pe))
        print("2-PE: {}".format(recycle_two_pe))
        print("0.5-PE: {}".format(recycle_pf_pe))

        print('Shelves:')
        print("4-PE:  {}".format(shelves_four_pe))
        print("2-PE: {}".format(shelves_two_pe))
        print("0.5-PE: {}".format(shelves_pf_pe))
 
        print('Teddy:')
        print("4-PE:  {}".format(teddy_four_pe))
        print("2-PE: {}".format(teddy_two_pe))
        print("0.5-PE: {}".format(teddy_pf_pe))
        
        print('Vintage:')
        print("4-PE:  {}".format(vintage_four_pe))
        print("2-PE: {}".format(vintage_two_pe))
        print("0.5-PE: {}".format(vintage_pf_pe))
      
        
    avg_two_pe = (adir_two_pe + artl_two_pe + jade_two_pe + motor_two_pe + motorE_two_pe + piano_two_pe + pianoL_two_pe + pipes_two_pe + playroom_two_pe + recycle_two_pe + shelves_two_pe + teddy_two_pe + vintage_two_pe) / 14.0
    
    return avg_two_pe


Tensor = torch.cuda.FloatTensor
cos = torch.nn.CosineSimilarity()


def createCostVol(left_im,right_im,max_disp):

    left_im = np.mean(left_im, axis=2)
    right_im = np.mean(right_im, axis=2)
            
    a_h, a_w = left_im.shape

    left_im = np.reshape(left_im, [1,1,a_h,a_w])
    right_im = np.reshape(right_im, [1,1,a_h,a_w])
    
    with torch.no_grad():

        left_imT = Variable(Tensor(left_im))
        right_imT = Variable(Tensor(right_im))

        left_feat = branch(left_imT)
        right_feat = branch(right_imT)
        
        _,f,h,w = left_feat.shape
        cost_vol = np.zeros((max_disp,a_h,a_w))

        #0 => max_disp => one less disp!
        #python3 apperently cannot have 0 here for disp: right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
        for disp in range(0,max_disp):

            if(disp == 0):
                sim_score = cos(left_feat, right_feat)
                sim_score_cpy = sim_score.cpu().data.numpy()            
                cost_vol[disp,:,:] = np.squeeze(sim_score_cpy)
                del sim_score_cpy
            else:
                right_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)                      
                right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
                right_appended = torch.cat([right_shift,right_feat],3)

                _,f,h_ap,w_ap = right_appended.shape
                right_shifted[:,:,:,:] = right_appended[:,:,:,:(w_ap-disp)]
                sim_score = cos(left_feat, right_shifted)
                sim_score_cpy = sim_score.cpu().data.numpy()            
                cost_vol[disp,:,:] = np.squeeze(sim_score_cpy)
                del sim_score_cpy
    
    del left_im
    del right_im
    del left_imT
    del right_imT
    del left_feat
    del right_feat
    del right_shifted
    del right_shift
    del right_appended
    torch.cuda.empty_cache()
    
    return cost_vol


def filterCostVolBilat(cost_vol):
    
    d,h,w = cost_vol.shape
    cost_vol_filtered = np.zeros((d,h,w))

    for disp in range(d):
        cur_slice = cost_vol[disp,:,:].astype(np.float32)
        cur_slice_filtered = cv2.bilateralFilter(cur_slice, 11, 75, 75)     
        cost_vol_filtered[disp,:,:] = cur_slice_filtered
        
    return cost_vol_filtered

def getBatch(cur_batch):
    

    samples, h, w = left_patches.shape
    batch_xl = np.zeros((batch_size,h,w))
    batch_xr_pos = np.zeros((batch_size,h,w))
    batch_xr_neg = np.zeros((batch_size,h,w))
    
    for el in range(batch_size):
        
        cur_xl = left_patches[el + cur_batch * batch_size,:,:] 
        cur_xr_pos = right_pos_patches[el + cur_batch * batch_size,:,:] 
        cur_xr_neg = right_neg_patches[el + cur_batch * batch_size,:,:]
        
        batch_xl[el,:,:] = cur_xl
        batch_xr_pos[el,:,:] = cur_xr_pos
        batch_xr_neg[el,:,:] = cur_xr_neg
        
        
    batch_xl = np.reshape(batch_xl, [batch_size,1,h,w])
    batch_xr_pos = np.reshape(batch_xr_pos, [batch_size,1,h,w])
    batch_xr_neg = np.reshape(batch_xr_neg, [batch_size,1,h,w])
    
    return batch_xl, batch_xr_pos, batch_xr_neg


pytorch_total_params = sum(p.numel() for p in branch.parameters() if p.requires_grad)
print("Nr feat: " ,pytorch_total_params)

def my_hinge_loss(s_p, s_n):
    margin = 0.2
    relu = torch.nn.ReLU()
    relu = relu.cuda()
    loss = relu(-((s_p - s_n) - margin))

    return loss

if(dataset == 'KITTI'):
    left_patches_whole, right_pos_patches_whole, right_neg_patches_whole = loadKitti()
    
if(dataset == 'MB'):
    left_patches_whole, right_pos_patches_whole, right_neg_patches_whole = loadMB()    
    
if(dataset == 'ETH'):
    left_patches_whole, right_pos_patches_whole, right_neg_patches_whole = loadETH3D()
    
    
nr_samples = len(left_patches_whole)


#KITTI, MB or ETH
if(transfer_train):
    if(dataset == 'KITTI'):
        
        print('Load weights from KITTI')
        branch.load_state_dict(torch.load('weights/kitti'))
        
    if(dataset == 'MB'):
        print('Load weights from Middlebury')
        branch.load_state_dict(torch.load('weights/mb'))
        
    if(dataset == 'ETH'):
        print('Load weights from ETH3D')
        branch.load_state_dict(torch.load('weights/eth3d'))


optimizer_G = optim.Adam(branch.parameters(), lr=0.000006)

nr_batches = int(samples2train / batch_size)

early_stopping_count = 0

left_patches = []
right_pos_patches = []
right_neg_patches = []

loss_list = []

for i in range(nr_epochs):
    
    if(i % load_samples == 0):
        if(dataset == 'KITTI'):
            left_patches_whole, right_pos_patches_whole, right_neg_patches_whole = loadKitti()
        if(dataset == 'MB'):
            left_patches_whole, right_pos_patches_whole, right_neg_patches_whole = loadMB()    
        if(dataset == 'ETH'):
            left_patches_whole, right_pos_patches_whole, right_neg_patches_whole = loadETH3D()        
        
        
    if(i % 1 == 0):
        left_patches = []
        right_pos_patches = []
        right_neg_patches = []

        idx = np.random.randint(0,nr_samples,samples2train)        

        left_patches = left_patches_whole[idx,:,:]
        right_pos_patches = right_pos_patches_whole[idx,:,:]
        right_neg_patches = right_neg_patches_whole[idx,:,:]   

        
    batch_loss = 0.0
    for cur_batch in range(nr_batches): 
        
        #reset gradients
        optimizer_G.zero_grad()
        
        batch_xl, batch_xr_pos, batch_xr_neg = getBatch(cur_batch)
        
        bs, c, h, w = batch_xl.shape
        batch_loss = 0.0
        
        for el in range(batch_size):
            
            
            cur_xl = batch_xl[el,:,:,:]
            cur_xr_pos = batch_xr_pos[el,:,:,:]
            cur_xr_neg = batch_xr_neg[el,:,:,:]
            
            
            cur_xl = np.reshape(cur_xl, [1,1,h,w])
            cur_xr_pos = np.reshape(cur_xr_pos, [1,1,h,w])
            cur_xr_neg = np.reshape(cur_xr_neg, [1,1,h,w])
                        
            xl = Variable(Tensor(cur_xl))
            xr_pos = Variable(Tensor(cur_xr_pos))
            xr_neg = Variable(Tensor(cur_xr_neg))        


            left_out = branch(xl)
            right_pos_out = branch(xr_pos)
            right_neg_out = branch(xr_neg)
            
            sp = cos(left_out, right_pos_out)
            sn = cos(left_out, right_neg_out)            
            
            loss = my_hinge_loss(sp, sn)
            batch_loss = batch_loss + loss
            batch_loss = batch_loss.mean()      
        
        batch_loss = batch_loss / batch_size

        batch_loss.backward()
        optimizer_G.step()

    
    epoch_loss = batch_loss/nr_batches        
    if(i % save_weights == 0):
        torch.save(branch.state_dict(), '/media/HDD/MCCNN_Results/Model/' + model_name + '_%04i' %(i)) 
        print("EPOCH: {} loss: {}".format(i,epoch_loss))
        
        
val, idx = min((val, idx) for (idx, val) in enumerate(loss_list))

plt.figure()
plt.plot(loss_list,'k')
plt.plot(loss_list, 'r*')