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

import tensorflow as tf
import matplotlib
matplotlib.use('Agg')

class DeepLabModel(object):
  """Class to load deeplab model and run inference."""

  INPUT_TENSOR_NAME = 'ImageTensor:0'
  OUTPUT_TENSOR_NAME = 'SemanticPredictions:0'
  INPUT_SIZE = 960
  FROZEN_GRAPH_NAME = 'frozen_inference_graph'

  def __init__(self, model_path):
    """Creates and loads pretrained deeplab model."""
    self.graph = tf.Graph()

    graph_def = None
    # Extract frozen graph from tar archive.
    with tf.gfile.GFile(model_path,'rb') as file_handle:        
        graph_def = tf.GraphDef.FromString(file_handle.read())


    if graph_def is None:
      raise RuntimeError('Cannot find inference graph in tar archive.')

    with self.graph.as_default():
      tf.import_graph_def(graph_def, name='')

    self.sess = tf.Session(graph=self.graph)

  def run(self, image):
    """Runs inference on a single image.

    Args:
      image: A PIL.Image object, raw input image.

    Returns:
      resized_image: RGB image resized from original input image.
      seg_map: Segmentation map of `resized_image`.
    """
    width, height = image.size
    resize_ratio = 1.0 * self.INPUT_SIZE / max(width, height)
    target_size = (int(resize_ratio * width), int(resize_ratio * height))
    resized_image = image.convert('RGB').resize(target_size, Image.ANTIALIAS)
    batch_seg_map = self.sess.run(
        self.OUTPUT_TENSOR_NAME,
        feed_dict={self.INPUT_TENSOR_NAME: [np.asarray(resized_image)]})
    seg_map = batch_seg_map
    return resized_image, seg_map




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
def cal_mad(gt_c,pred_c):
   dsm=distance_matrix(np.array(gt_c).reshape(-1,2),np.array(pred_c).reshape(-1,2))
   gt_min = np.amin(dsm,axis=1)
   pred_min = np.amin(dsm,axis=0)
   mad=(np.mean(gt_min)+np.mean(pred_min))/2
   return mad

def main():

   parser = argparse.ArgumentParser(description="PyTorch Object Detection shoulder Demo")
   parser.add_argument(
     "--predict_dir",
     default="./predict",
     help="path to predict img",
   )
   parser.add_argument(
     "--output_dir",
     default="./output",
     help="path to output files",
   )
   parser.add_argument(
     "--gt_dir",
     default="./ground_truth_mask",
     help="path to ground mask",
   )
   parser.add_argument(
     "--model_path",
     default="./frozen_inference_grapht.pb",
     help="path to ground mask",
   )
   parser.add_argument(
     "opts",
     help="Modify model config options using the command-line",
     default=None,
     nargs=argparse.REMAINDER,
   )

   args = parser.parse_args()


   MODEL = DeepLabModel(args.model_path)
   print('model loaded successfully!')


   # update the config options with the config file
   # load image and then run prediction
   '''
   img_name = glob.glob(os.path.join(args.predict_dir,'*.png'))[0]
   img=cv2.imread(img_name,1)

   predictions = shoulder_demo.run_on_opencv_image(img)
   cv2.imwrite(os.path.join(args.output_dir,'{}_mask.png'.format(os.path.basename(img_name).split('.')[0])),predictions)
   '''

   png_list=sorted(glob.glob(os.path.join(args.predict_dir,'**','*.png'),recursive=True))
   png_dict=defaultdict(list)
   for pngdir in png_list:
      png_dict['_'.join(os.path.basename(pngdir).split('_')[:-1])].append(pngdir)
   
   total_iou_his=[]
   total_centroid_err_his=[]
   total_circularity_err_his=[]
   total_circularity_err_2_his=[]
   total_mad_his=[]
   time_his=[]
   avg_iou_his={}
   std_iou_his={}   
   
   avg_mad_his={}
   std_mad_his={}  
   
   pred_mask=np.zeros((1,720,960),dtype=np.int32)
   png_dict=OrderedDict(png_dict)
   for i, (k,v) in enumerate(png_dict.items()):
      #if k != 'case_008_1155390190':
      #   continue
      print(k)
      gt_cx_his=[]
      gt_cy_his=[]
      gt_circularity_his=[]
      pred_cx_his=[]
      pred_cy_his=[]
      pred_circularity_his=[]
      mad_his=[]
      frame_his=[]
      v_iou_his=[]

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
         #ground truth
         file_name = os.path.basename(pngpath).split('.')[0]
         frame_his.append(int(os.path.basename(pngpath).split('_')[-1].split('.')[0]))
         mask_name = '{}_mask.png'.format(file_name)
         mask_dir = glob.glob(os.path.join(args.gt_dir,'**',mask_name),recursive=True)
         gt_mask = cv2.imread(mask_dir[0],-1)[:,:,None].astype(np.uint8)
         
         #print(gt_mask.shape)
         gt_contours, _ = cv2.findContours(
            gt_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE
         )

         #predict mask
         b_time=time.time()
         resized_im, seg_map = MODEL.run(pil_im)
         a_time=time.time()
         print('seg_map type:{}, {}, {}'.format(type(seg_map), seg_map.shape, seg_map.dtype))      
            
         pred_mask = seg_map.astype(np.uint8)
         if np.sum(pred_mask)==0:
            pred_mask=previous_mask
         
         print(pred_mask.shape, gt_mask.shape) 
         intersection=float(np.sum(pred_mask[0]*gt_mask[:,:,0]))
         union=float(np.sum(pred_mask[0]+gt_mask[:,:,0])-intersection)
         
         iou=intersection/union
         print(iou) 
         total_iou_his.append(iou)
         v_iou_his.append(iou)
         
         
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
            gt_img = cv2.drawContours(img.copy(), gt_contours, -1, [0,0,255], 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_gt.png'.format(file_name)),gt_img) 
            pred_img = cv2.drawContours(img.copy(), pred_contours, -1, pred_color, 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_pred.png'.format(file_name)),pred_img) 
            mix_img = cv2.drawContours(pred_img, gt_contours, -1, [0,0,255], 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_mix.png'.format(file_name)),mix_img) 
            
            time_his.append(a_time-b_time)
            #parameters
            gt_cx,gt_cy =cal_centroid(gt_contours[0]) 
            gt_circularity =cal_circularity(gt_contours[0]) 
            gt_cx_his.append(gt_cx)
            gt_cy_his.append(gt_cy)
            gt_circularity_his.append(gt_circularity)
            
            pred_cx,pred_cy =0,0#cal_centroid(pred_contours[c_id]) 
            pred_circularity =0#cal_circularity(pred_contours[c_id]) 
            pred_cx_his.append(pred_cx)
            pred_cy_his.append(pred_cy)
            pred_circularity_his.append(pred_circularity)
            
            mad=100#mad=cal_mad(gt_contours[0],pred_contours[c_id])
            mad_his.append(mad)
            total_mad_his.append(mad)
            
            total_centroid_err_his.append(math.sqrt((pred_cx-gt_cx)**2+(pred_cy-gt_cy)**2))
            total_circularity_err_his.append(abs(pred_circularity-gt_circularity))
            total_circularity_err_2_his.append((pred_circularity-gt_circularity)**2)
         
         else:
            c_id=np.argmax(np.array(c_area))
            #save img
            cv2.imwrite(os.path.join(sub_output_dir,'{}.png'.format(file_name)),img) 
            gt_img = cv2.drawContours(img.copy(), gt_contours, -1, [0,0,255], 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_gt.png'.format(file_name)),gt_img) 
            real_pred_contours=pred_contours.pop(c_id)
           
            pred_img = cv2.drawContours(img.copy(), pred_contours, -1, [255,128,255] , 2)
            
            pred_img = cv2.drawContours(pred_img, real_pred_contours, -1, pred_color, 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_pred.png'.format(file_name)),pred_img) 
            mix_img = cv2.drawContours(pred_img, gt_contours, -1, [0,0,255], 2)
            cv2.imwrite(os.path.join(sub_output_dir,'{}_mix.png'.format(file_name)),mix_img) 
            
            time_his.append(a_time-b_time)
            #parameters
            gt_cx,gt_cy =cal_centroid(gt_contours[0]) 
            gt_circularity =cal_circularity(gt_contours[0]) 
            gt_cx_his.append(gt_cx)
            gt_cy_his.append(gt_cy)
            gt_circularity_his.append(gt_circularity)
            
            pred_cx,pred_cy =cal_centroid(real_pred_contours) 
            pred_circularity =cal_circularity(real_pred_contours) 
            pred_cx_his.append(pred_cx)
            pred_cy_his.append(pred_cy)
            pred_circularity_his.append(pred_circularity)
            
            mad=cal_mad(gt_contours[0],real_pred_contours)
            mad_his.append(mad)
            total_mad_his.append(mad)
            
            total_centroid_err_his.append(math.sqrt((pred_cx-gt_cx)**2+(pred_cy-gt_cy)**2))
            total_circularity_err_his.append(abs(pred_circularity-gt_circularity))
            total_circularity_err_2_his.append((pred_circularity-gt_circularity)**2)


         print('i:{}/{}, {}, j:{}/{}, cost time :{} sec'.format(i,len(png_dict),k,j,len(v),a_time-b_time))
      his_dict=OrderedDict({
         'frame_idx':frame_his,
         'gt_cx':gt_cx_his,
         'gt_cy':gt_cy_his,
         'gt_circularity':gt_circularity_his,
         'pred_cx':pred_cx_his,
         'pred_cy':pred_cy_his,
         'pred_circularity':pred_circularity_his,
         'mad':mad_his,
         'iou':v_iou_his
      })
      his_pd=pd.DataFrame.from_dict(his_dict)
      his_pd.to_csv(os.path.join(sub_output_dir,'{}_parameters.csv'.format(k)),index=False)
      centroid_err = np.mean(np.sqrt((np.array(pred_cx_his)-np.array(gt_cx_his))**2+(np.array(pred_cy_his)-np.array(gt_cy_his))**2))
      
      circularity_err = np.mean(np.absolute(np.array(pred_circularity_his)-np.array(gt_circularity_his)))
      
      circularity_err_2 = np.mean((np.array(pred_circularity_his)-np.array(gt_circularity_his))**2)
      avgiou=np.mean(np.array(v_iou_his))
      avg_iou_his[k]=avgiou
      std_iou_his[k]=np.std(np.array(v_iou_his))
      avgmad=np.mean(np.array(mad_his))
      avg_mad_his[k]=avgmad
      std_mad_his[k]=np.std(np.array(mad_his))
      
      with open(os.path.join(sub_output_dir,'{}_err.txt'.format(k)),'w') as outtxt:
         outtxt.write('mad,{:.4f}\niou,{:.4f}\ncentroid_err,{:.4f}\ncircularity_err,{:.4f}\ncircularity_err_2,{:.4f}'.format(avgmad,avgiou,centroid_err, circularity_err,circularity_err_2))
      print('iou,{:.4f}\ncentroid_err,{:.4f}\ncircularity_err,{:.4f}\ncircularity_err_2,{:.4f}'.format(avgiou,centroid_err, circularity_err,circularity_err_2))
   
   with open(os.path.join(args.output_dir,'parameters_err.txt'),'w') as ptxt:
      ptxt.write('centroid_err:{},{}\n'.format(np.mean(np.array(total_centroid_err_his)),np.std(np.array(total_centroid_err_his))))
      ptxt.write('circularity_l1_err:{},{}\n'.format(np.mean(np.array(total_circularity_err_his)),np.std(np.array(total_circularity_err_his))))
      ptxt.write('circularity_mse_err:{},{}\n'.format(np.mean(np.array(total_circularity_err_2_his)),np.std(np.array(total_circularity_err_2_his))))
   with open(os.path.join(args.output_dir,'time_avg.txt'),'w') as timetxt:
      timetxt.write('{}'.format(np.mean(np.array(time_his))))
   with open(os.path.join(args.output_dir,'avg_total_iou.txt'),'w') as ioutxt:
      ioutxt.write('{},{}\n'.format(np.mean(np.array(total_iou_his)),np.std(np.array(total_iou_his))))
   with open(os.path.join(args.output_dir,'avg_total_mad.txt'),'w') as madtxt:
      madtxt.write('{},{}\n'.format(np.mean(np.array(total_mad_his)),np.std(np.array(total_mad_his))))
         
   with open(os.path.join(args.output_dir,'total_iou_his.csv'),'w') as iouhis:
      for iou in total_iou_his:
         iouhis.write('{}\n'.format(iou))
   with open(os.path.join(args.output_dir,'avgiou_byvideo.csv'),'w') as avgioutxt:
      iou_list = [(k,avg_iou_his[k],std_iou_his[k]) for k in sorted(avg_iou_his, key=avg_iou_his.get, reverse=True)]
      for k,iou,std in iou_list:
         avgioutxt.write('{},{},{}\n'.format(k,iou,std))
   
   with open(os.path.join(args.output_dir,'avgmad_byvideo.csv'),'w') as avgmadtxt:
      mad_list = [(k,avg_mad_his[k],std_mad_his[k]) for k in sorted(avg_mad_his, key=avg_mad_his.get, reverse=True)]
      for k,mad,std in mad_list:
         avgmadtxt.write('{},{},{}\n'.format(k,mad,std))
    
   print('draw iou his')
   plt.hist(np.array(total_iou_his),np.linspace(0,1,100))
   plt.ylabel('num of frames',fontsize=22)
   plt.xlabel('IoU',fontsize=22)
   plt.title('IoU histogram',fontsize=24)

   plt.savefig(os.path.join(args.output_dir,'iou_histogram.png'))
   plt.clf()
   plt.close()
      

if __name__=='__main__':
   main()
