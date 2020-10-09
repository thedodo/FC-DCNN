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
from termcolor import colored

import torch
import torch.nn as nn
from torchvision import models
from torch.autograd import Variable
from torch import optim
from torchsummary import summary

import random
import itertools
from PIL import Image




import argparse
import sys


left_im = sys.argv[1]
right_im = sys.argv[2]
max_disp = sys.argv[3]
out_fn = sys.argv[4]

num_conv_feature_maps = 64

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
    
    
    
Tensor = torch.cuda.FloatTensor
cos = torch.nn.CosineSimilarity()


def filterCostVolMedian(cost_vol):
    
    d,h,w = cost_vol.shape
    cost_vol_filtered = np.zeros((d,h,w))

    for disp in range(d):
        cur_slice = cost_vol[disp,:,:].astype(np.float32)
        cur_slice_filtered = cv2.medianBlur(cur_slice, 5)     
        cost_vol_filtered[disp,:,:] = cur_slice_filtered
        
    return cost_vol_filtered



def filterCostVolBilat(cost_vol):
    
    d,h,w = cost_vol.shape
    cost_vol_filtered = np.zeros((d,h,w))

    for disp in range(d):
        cur_slice = cost_vol[disp,:,:].astype(np.float32)
        cur_slice_filtered = cv2.bilateralFilter(cur_slice, 11, 75, 75)     
        cost_vol_filtered[disp,:,:] = cur_slice_filtered
        
    return cost_vol_filtered


def createCostVol(left_im,right_im,max_disp):
    
    print('Creating cost-volume....')

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
            else:
                right_shifted = torch.cuda.FloatTensor(1,f,h,w).fill_(0)                      
                right_shift = torch.cuda.FloatTensor(1,f,h,disp).fill_(0)  
                right_appended = torch.cat([right_shift,right_feat],3)

                _,f,h_ap,w_ap = right_appended.shape
                right_shifted[:,:,:,:] = right_appended[:,:,:,:(w_ap-disp)]
                sim_score = cos(left_feat, right_shifted)
                sim_score_cpy = sim_score.cpu().data.numpy()            
                cost_vol[disp,:,:] = np.squeeze(sim_score_cpy)
    print('Done')
    return cost_vol


def createCostVolRL(left_im,right_im,max_disp):

    print('Create cost-volume right-to-left...')
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
    print('Done')
        
    return cost_vol



#deprecated! python version
def LR_Check(first_output, second_output):    
    
    print('Check for inconsistencies...')
    
    rows ,cols = first_output.shape
    mask = np.ones((rows,cols))

    
    for col in range(0,cols):
        for row in range(0,rows):

            disp_val_first = first_output[row,col]
            
            #HACK FOR NOW! 0 Disp should not be possible!
            if((disp_val_first == 0) or (disp_val_first == 1) or (disp_val_first == 2) or (disp_val_first == 3) or (disp_val_first == 4) or (disp_val_first == 5)):
                mask[row,col] = 0
                first_output[row,col] = np.nan
                
                continue
                
            
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
    
    print('Done')
   
    return first_output



def FillIncons(mask, disp):

    
        
    w = mask.shape[1]
    h = mask.shape[0]
    
    
    #if whole line is nan => fill with 0
    
    print('Update inconsistent points...')
   
    #find a way to not iterate over image???
    for height in range(h):
        for width in range(w):
            
            if(all(np.isnan(disp[height,0:w]))):
                fill = 0
                break
            

            if not np.isnan(disp[height,width]):
                continue

            if(mask[height,width] == 0):
                if(np.isnan(disp[height,width])):
                    counter = 0
                    fill = 0.0
                    diff_dir = False
                    while(np.isnan(disp[height,width+counter]) and mask[height,width] == 0):
                        counter = counter -1   
                        if(width + counter <= 0):
                            #check other direction!
                            counter = 0
                            diff_dir = True
                            while(np.isnan(disp[height,width+counter])):
                                counter = counter + 1   
                                if(width + counter >= w):
                                    diff_dir = False
                                    fill = 0.0
                                    break
                            #-----try!
                        fill = disp[height,width+counter] 

                    disp[height,width] = fill


            if(mask[height,width] == 1):
                if(np.isnan(disp[height,width])):                
                    left = 0
                    right = 0
                    above = 0
                    under = 0

                    r_above = 0
                    l_above = 0
                    r_under = 0
                    l_under = 0

                    counter = 0                
                    while(np.isnan(disp[height,width-counter]) and mask[height,width] == 1):
                        counter = counter +1                    
                        if((width-counter) < 0):
                            left = 0
                            break
                        left = disp[height,width-counter]

                    counter = 0                                    
                    while(np.isnan(disp[height,width+counter]) and mask[height,width] == 1):
                        counter = counter +1                       
                        if((width+counter) >= w):
                            right = 0
                            break       
                        right = disp[height,width+counter]

                    counter = 0                                    
                    while(np.isnan(disp[height+counter,width]) and mask[height,width] == 1):
                        counter = counter +1                       
                        if((height+counter) >= h):
                            above = 0
                            break       
                        above = disp[height+counter,width]

                    counter = 0                                    
                    while(np.isnan(disp[height-counter,width]) and mask[height,width] == 1):
                        counter = counter +1                       
                        if((height-counter) < 0):
                            under = 0
                            break                                              
                        under = disp[height-counter,width]

                    counter = 0                                    
                    while(np.isnan(disp[height+counter,width+counter])and mask[height,width] == 1):
                        counter = counter +1
                        if((height+counter) >= h):
                            r_above = 0
                            break
                        if((width+counter) >= w):
                            r_above = 0
                            break                        
                        r_above = disp[height+counter,width+counter]

                    counter = 0                                    
                    while(np.isnan(disp[height-counter,width-counter]) and mask[height,width] == 1):
                        counter = counter +1
                        if((height-counter) < 0):
                            l_under = 0
                            break                    
                        l_under = disp[height-counter,width-counter]

                    counter = 0                                    
                    while(np.isnan(disp[height+counter,width-counter]) and mask[height,width] == 1):
                        counter = counter +1
                        if((height+counter) >= h):
                            l_above = 0
                            break                    
                        l_above = disp[height+counter,width-counter]

                    counter = 0                                    
                    while(np.isnan(disp[height-counter,width+counter]) and mask[height,width] == 1):
                        counter = counter +1
                        if(width+counter >= w):
                            r_under = 0
                            break
                        if((height-counter) < 0):
                            r_under = 0
                            break
                        r_under = disp[height-counter,width+counter]

                    fill = np.median([left,right,above,under,r_above,l_above,r_under,l_under])

                    disp[height,width] = fill
                    
    print('Done')
                    
    return disp



#deprecated! slower than cython implementation
def TestImage(fn_left, fn_right, max_disp, im_to_save, gt = None, filtered = True, lr_check = True, fill = True):
    
    four_pe = 0.0
    two_pe = 0.0
    pf_pe = 0.0
    
    left = cv2.imread(fn_left)
    right = cv2.imread(fn_right)
    disp_map = []
    
    if(filtered):
        
        cost_vol = createCostVol(left,right,max_disp)
        cost_vol_filtered = filterCostVolMedian(cost_vol) 
        cost_vol_filtered = filterCostVolMedian(cost_vol_filtered) 
        cost_vol_filtered = filterCostVolMedian(cost_vol_filtered) 
        
        cost_vol_filtered = filterCostVolBilat(cost_vol_filtered)
        
        disp_map = np.argmax(cost_vol_filtered, axis=0) 
                       
        writePFM(im_to_save + '.pfm', disp_map.astype(np.float32), scale=1)
        
        if(lr_check):
            cost_vol_RL = createCostVolRL(left,right,max_disp)
            cost_vol_RL_f = filterCostVolMedian(cost_vol_RL)
            cost_vol_RL_f = filterCostVolMedian(cost_vol_RL_f)   
            cost_vol_RL_f = filterCostVolMedian(cost_vol_RL_f)               
            cost_vol_RL_f = filterCostVolBilat(cost_vol_RL_f)
            disp_map_RL = np.argmax(cost_vol_RL_f, axis=0)             
            
            disp_map = disp_map.astype(np.float)
            disp_map_RL = disp_map_RL.astype(np.float)
            
            
            final_disp = LR_Check(disp_map, disp_map_RL)
            writePFM(im_to_save + '_s.pfm', final_disp.astype(np.float32), scale=1)
        
        
    else:
        
        cost_vol = createCostVol(left,right,max_disp)
        disp_map = np.argmax(cost_vol, axis=0)  
        
        writePFM(im_to_save + '.pfm', disp_map.astype(np.float32), scale=1) 
        
        if(lr_check):
            
            cost_vol_RL = createCostVolRL(left,right,max_disp)
            disp_map_RL = np.argmax(cost_vol_RL, axis=0) 
            
            disp_map = disp_map.astype(np.float)
            disp_map_RL = disp_map_RL.astype(np.float)
            
            final_disp = LR_Check(disp_map, disp_map_RL)
            writePFM(im_to_save + '_s.pfm', final_disp.astype(np.float32), scale=1)
        
    
    if(gt is not None):
        gt_im,_ = readPFM(gt)        
        
        h,w = gt_im.shape        
        
        nr_px = h * w
        nr_px = float(nr_px)
        
        abs_error_im = np.abs(disp_map - gt_im)        
        
        four_pe = (float(np.count_nonzero(abs_error_im >= 4.0) ) / nr_px) * 100.0        
        two_pe = (float(np.count_nonzero(abs_error_im >= 2.0) ) / nr_px) * 100.0        
        pf_pe = (float(np.count_nonzero(abs_error_im >= 0.5) ) / nr_px) * 100.0
        
        
        
    if(fill):
    
        disp = np.array(disp_map)
        im_disp = Image.fromarray(disp) 
        im_disp = np.dstack((im_disp, im_disp, im_disp)).astype(np.uint8)    

        h,w = disp.shape

        shifted = cv2.pyrMeanShiftFiltering(im_disp, 7, 7)

        gray = cv2.cvtColor(shifted, cv2.COLOR_BGR2GRAY)
        thresh = cv2.threshold(gray, 0, 1,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]

        kernel = np.ones((5,5), np.uint8)

        dilation = cv2.dilate(thresh,kernel,iterations = 3)
        mask = cv2.erode(dilation, kernel, iterations=2)    

        cv2.imwrite(im_to_save + 'bilat_and_med_mask.png',mask * 255)

        disp_filled = FillIncons(mask, disp)
        
        disp_filled = cv2.medianBlur(disp_filled.astype(np.float32), 5) 

        writePFM(im_to_save + '_filled.pfm',disp_filled) 
        
        
        
        
    return four_pe, two_pe, pf_pe


weights = sys.argv[1]
left_im = sys.argv[2]
right_im = sys.argv[3]
max_disp = int(sys.argv[4])
out_fn = sys.argv[5]


branch.load_state_dict(torch.load(weights))


disp = TestImage(left_im, right_im, max_disp, out_fn, filtered = True, lr_check = True, fill = True)

