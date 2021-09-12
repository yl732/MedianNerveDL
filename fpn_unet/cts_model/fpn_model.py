import torch 
import torch.nn as nn
import torch.nn.functional as F
from .ED_model import EncoderDecoder  

class Conv3x3GNReLU(nn.Module):
   def __init__(self, in_channels, out_channels, upidx):
      super().__init__()
      self.block = nn.Sequential(
         nn.Conv2d(in_channels, out_channels, (3, 3),
                     stride=1, padding=1, bias=False),
         nn.GroupNorm(32, out_channels),
         nn.ReLU(inplace=True)
      
      )
      self.upidx=upidx

   def forward(self, x, image_size_list):
      #print(x.size())
      x = self.block(x)
      #print('conv:{}'.format(x.size()))
      if not image_size_list is None:
         x = F.interpolate(x, size=image_size_list[self.upidx][-2:], mode='bilinear', align_corners=True)
      #print('interpolate:{}'.format(x.size()))
      return x

class FPNBlock(nn.Module):
   def __init__(self, skip_channels, pyramid_channels):
      super().__init__()
      self.skip_conv = nn.Conv2d(skip_channels, pyramid_channels, kernel_size=1)
   
   def forward(self, xskip):
      x, skip = xskip

      x = F.interpolate(x, size=skip.size()[-2:], mode='nearest')
      skip = self.skip_conv(skip)

      x = x + skip

      return x

class SegmentationBlock(nn.Module):
   def __init__(self, in_channels, out_channels, up_num):
      super().__init__()

      blocks = [Conv3x3GNReLU(in_channels, out_channels, upidx=0)]
      if up_num >1:
         for upidx in range(1,up_num):
            blocks.append(Conv3x3GNReLU(out_channels, out_channels, upidx=upidx))
      self.blocks = nn.ModuleList(blocks)
   
   def forward(self, x, image_size_list=None):
      for block in self.blocks:
         x = block(x,image_size_list)
      
      return x 

class FPNDecoder(nn.Module):
   def __init__(self, encoder_channels, pyramid_channels=256, segmentation_channels=128, final_channels=1):
      super().__init__()
      self.conv = nn.Conv2d(encoder_channels[0], pyramid_channels, kernel_size=(1,1))
      
      self.h4 = FPNBlock(encoder_channels[1], pyramid_channels)
      self.h3 = FPNBlock(encoder_channels[2], pyramid_channels)
      self.h2 = FPNBlock(encoder_channels[3], pyramid_channels)

      self.up5 = SegmentationBlock(pyramid_channels, segmentation_channels, up_num=3)
      self.up4 = SegmentationBlock(pyramid_channels, segmentation_channels, up_num=2)
      self.up3 = SegmentationBlock(pyramid_channels, segmentation_channels, up_num=1)
      self.up2 = SegmentationBlock(pyramid_channels, segmentation_channels, up_num=0)

      #self.dropout = nn.Dropout2d(p=dropout, inplace=True)
      self.final_conv = nn.Conv2d(segmentation_channels, final_channels, kernel_size=1, padding=0)

      self.initialize()

   def forward(self, x, ori_image_size):
      c5, c4, c3, c2, _ = x
      
      p5 = self.conv(c5)
      #print("p5:{}".format(p5.size()))
      p4 = self.h4([p5, c4])
      #print("p4:{}".format(p4.size()))
      p3 = self.h3([p4, c3])
      #print("p3:{}".format(p3.size()))
      p2 = self.h2([p3, c2])
      #print("p2:{}".format(p2.size()))

      s5 = self.up5(p5,image_size_list=[p4.size(),p3.size(),p2.size()])
      #print("s5:{}".format(s5.size()))
      s4 = self.up4(p4,image_size_list=[p3.size(),p2.size()])
      #print("s4:{}".format(s4.size()))
      s3 = self.up3(p3,image_size_list=[p2.size()])
      #print("s3:{}".format(s3.size()))
      s2 = self.up2(p2)
      #print("s2:{}".format(s2.size()))

      x = s5 + s4 + s3 + s2

      #x = self.dropout(x)
      x = self.final_conv(x)
      #with torch.no_grad(): 
         #print('final one:{:.6f} final zeros:{:}'.format(torch.sum(x[:,1,:,:]),torch.sum(x[:,0,:,:])))#.format(torch.sum(torch.max(x,dim=1)[1])))
      x = F.interpolate(x, size=ori_image_size[-2:], mode='bilinear', align_corners=True)
      print('final x:{}'.format(x.size()))
      return x
   
   def initialize(self):
      for m in self.modules():
         if isinstance(m, nn.Conv2d):
            nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
            if m.bias is not None:
               nn.init.constant_(m.bias, 0)
         elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0) 
class FPN(EncoderDecoder):
   def __init__(
      self,
      encoder,
      encoder_outshape,
      decoder_pyramid_channels=256,
      decoder_segmentation_channels=128,
      classes=1,
      activation='sigmoid',
   ):
      decoder = FPNDecoder(
         encoder_channels=encoder_outshape,
         pyramid_channels=decoder_pyramid_channels,
         segmentation_channels=decoder_segmentation_channels,
         final_channels=classes,
      )
      super().__init__(encoder, decoder, activation)
