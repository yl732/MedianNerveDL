import os
from io import BytesIO
import tarfile
import tempfile
from six.moves import urllib

from matplotlib import gridspec
from matplotlib import pyplot as plt
import numpy as np
from PIL import Image
import cv2
import argparse
import glob
import time
import math
import pandas as pd
from collections import defaultdict,OrderedDict
from scipy.spatial import distance_matrix
from fpn_unet import cts_model,backbone,cts_transforms
import torch

import matplotlib
matplotlib.use('Agg')
def cal_circularity(contour):
   area = cv2.contourArea(contour)
   perimeter = cv2.arcLength(contour,True)
   if perimeter ==0:
      return 0
   return 4*math.pi*area/(perimeter**2)
def cal_centroid(contour):
   M=cv2.moments(contour)
   if M['m00']==0:
      print('GGGGGGGGGGGGGGGGGGGGGGGGGGGGGzeroGGGGGGGGGGGGGGGGGGGG')
      return 0,0
   cx = int(M['m10']/M['m00'])
   cy = int(M['m01']/M['m00'])
   return cx,cy

def main():
   parser = argparse.ArgumentParser(description="PyTorch Object Detection shoulder Demo")
   parser.add_argument(
      "--predict_dir",
      default="./predict",
      help="path to predict img",
   )
   parser.add_argument(
      "--model_type",
      default='fpn',
      choices=['fpn','unet','cb_unet'],
      help="model type"   
   )
   parser.add_argument(
      "--backbone",
      default='resnet101',
      choices=['resnet101','resnext101_32x8d'],
      help="backbone type"   
   )
   parser.add_argument(
      "--output_dir",
      default="./output",
      help="path to output files",
   )

   parser.add_argument(
      "--model_path",
      default="cts-fpn-resnet101",
      help="path to ground mask",
   )
   parser.add_argument(
      "opts",
      help="Modify model config options using the command-line",
      default=None,
      nargs=argparse.REMAINDER,
   )

   args = parser.parse_args()
   output_shape_dict = {'resnet101':(2048, 1024, 512, 256, 64),'resnext101_32x8d':(2048, 1024, 512, 256, 64)}

   if args.model_type=='deeplabv3p':
      #MODEL = DeepLabModel(args.model_path)
      pass
   else:
      device = torch.device('cuda')
      encoder = getattr(backbone, args.backbone)()
      MODEL = getattr(cts_model, args.model_type)(encoder=encoder, encoder_outshape=output_shape_dict[args.backbone], classes=2)
      MODEL = MODEL.to(device)
      if args.model_path.split('.')[-1]=='tar':
         checkpoint = torch.load(args.model_path)
         MODEL.load_state_dict(checkpoint['best_model_wts'])
      else:
         checkpoint = torch.load(args.model_path)
         MODEL.load_state_dict(checkpoint)
      MODEL.eval()
      transforms = cts_transforms.Compose([
                     cts_transforms.ToTensor(),
                     cts_transforms.Normalize(
                        mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225])])
   print('model loaded successfully!')
   png_list=sorted(glob.glob(os.path.join(args.predict_dir,'**','*.png'),recursive=True))
   png_dict=defaultdict(list)
   for pngdir in png_list:
      png_dict['_'.join(os.path.basename(pngdir).split('_')[:-1])].append(pngdir)
   
   time_his=[]
   pred_mask=np.zeros((1,720,960),dtype=np.int32)
   png_dict=OrderedDict(png_dict)
   for i, (k,v) in enumerate(png_dict.items()):
      print(k)
      pred_cx_his=[]
      pred_cy_his=[]
      pred_circularity_his=[]
      frame_his=[]

      sub_output_dir=os.path.join(args.output_dir,k) 
      if not os.path.exists(sub_output_dir):
         os.makedirs(sub_output_dir)
      previous_mask=np.zeros((1,720,960),dtype=np.uint8)
      for j,pngpath in enumerate(v):
             
         pil_im = Image.open(pngpath).convert('RGB')
         img = np.array(pil_im)
         landmark=np.where(img[85,120:,0]>100)[0]
         pixel2cm=1/np.floor(np.mean(landmark[1:]-landmark[:-1]))
         pred_color=[255,255,0]#[14,201,255]
         file_name = os.path.basename(pngpath).split('.')[0]
         frame_his.append(int(os.path.basename(pngpath).split('_')[-1].split('.')[0]))

         #predict mask
         b_time=time.time()
         #resized_im, seg_map = MODEL.run(pil_im)
         input_img,_ = transforms(pil_im, None)
         input_img = torch.unsqueeze(input_img, 0)
         input_img = input_img.to(device)
         with torch.set_grad_enabled(False):
            outputs=MODEL(input_img)
         _, seg_map = torch.max(outputs, dim=1)
         seg_map=seg_map.detach().cpu().numpy() 
         a_time=time.time()
         print('seg_map type:{}, {}, {}'.format(type(seg_map), seg_map.shape, seg_map.dtype))
         pred_mask = seg_map.astype(np.uint8)
         if np.sum(pred_mask)==0:
            pred_mask=previous_mask
         previous_mask = pred_mask
         pred_thresh = pred_mask[0,:, :, None]
         pred_contours, _ = cv2.findContours(
            pred_thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
         )
         c_area=[]
         print(len(pred_contours)) 
         for cnt in pred_contours:
            area=cv2.contourArea(cnt)
            c_area.append(area)
         print(len(c_area)) 

         if len(pred_contours)==0:
            #save img
            cv2.imwrite(os.path.join(sub_output_dir,'{}.png'.format(file_name)),img) 
            pred_img = cv2.drawContours(img.copy(), pred_contours, -1, pred_color, 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_pred.png'.format(file_name)),pred_img)

            time_his.append(a_time-b_time)
            #parameters
            pred_cx,pred_cy =0,0#cal_centroid(pred_contours[c_id]) 
            pred_circularity =0#cal_circularity(pred_contours[c_id]) 
            pred_cx_his.append(pred_cx)
            pred_cy_his.append(pred_cy)
            pred_circularity_his.append(pred_circularity)
         else:
            c_id=np.argmax(np.array(c_area))
            #save img
            cv2.imwrite(os.path.join(sub_output_dir,'{}.png'.format(file_name)),img) 
            pred_img = cv2.drawContours(img.copy(), pred_contours[c_id], -1, pred_color, 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_pred.png'.format(file_name)),pred_img) 

            time_his.append(a_time-b_time)
            #parameters
            pred_cx,pred_cy =cal_centroid(pred_contours[c_id]) 
            pred_circularity =cal_circularity(pred_contours[c_id]) 
            pred_cx_his.append(pred_cx)
            pred_cy_his.append(pred_cy)
            pred_circularity_his.append(pred_circularity)
         print('i:{}/{}, {}, j:{}/{}, cost time :{} sec'.format(i,len(png_dict),k,j,len(v),a_time-b_time))
      his_dict=OrderedDict({
         'frame_idx':frame_his,
         'pred_cx':pred_cx_his,
         'pred_cy':pred_cy_his,
         'pred_circularity':pred_circularity_his,
      })
      his_pd=pd.DataFrame.from_dict(his_dict)
      his_pd.to_csv(os.path.join(sub_output_dir,'{}_parameters.csv'.format(k)),index=False)

   with open(os.path.join(args.output_dir,'time_avg.txt'),'w') as timetxt:
      timetxt.write('{}'.format(np.mean(np.array(time_his))))

if __name__=='__main__':
   main()
