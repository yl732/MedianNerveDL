from .ED_model import EncoderDecoder
import torch
import torch.nn as nn
import torch.nn.functional as F

class Conv2dReLU(nn.Module):
   def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True, **batchnorm_params):
      
      super().__init__()

      layers = [
         nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not(use_batchnorm)),
         nn.ReLU(inplace=True)      
      ]
      
      if use_batchnorm:
         layers.insert(1, nn.BatchNorm2d(out_channels, **batchnorm_params))

      self.block = nn.Sequential(*layers)

   def forward(self, x):
      return self.block(x)

class DecoderBlock(nn.Module):
   def __init__(self, in_channels, out_channels, use_batchnorm=True, attention_type=None):
      super().__init__()
      if attention_type is None:
         self.attention1 = nn.Identity()
         self.attention2 = nn.Identity()
      
      self.block = nn.Sequential(
         Conv2dReLU(in_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm),
         Conv2dReLU(out_channels, out_channels, kernel_size=3, padding=1, use_batchnorm=use_batchnorm)
      )
   
   def forward(self, xskip, image_size=None):
      x, skip = xskip
      if not skip is None:
         #print(x.size(),skip.size())
         x = F.interpolate(x, size=skip.size()[-2:], mode='nearest')
      else:
         x = F.interpolate(x, size=image_size, mode='nearest')
      #print(x.size())
      if skip is not None:
         x = torch.cat([x, skip], dim=1)
         x = self.attention1(x)
      
      x = self.block(x)
      x = self.attention2(x)
      return x

class CenterBlock(DecoderBlock):
   def forward(self, x):
      return self.block(x)

class UnetDecoder(nn.Module):
   def __init__(self, encoder_channels, decoder_channels=(256, 128, 64, 32, 16), final_channels=1, use_batchnorm=True, center=False, attention_type=None):
      super().__init__()
      if center:
         channels = encoder_channels[0]
         self.center = CenterBlock(channels, channels, use_batchnorm=use_batchnorm)
      else:
         self.center = None
      
      
      in_channels = [
         encoder_channels[0] + encoder_channels[1],
         encoder_channels[2] + decoder_channels[0],
         encoder_channels[3] + decoder_channels[1],
         encoder_channels[4] + decoder_channels[2],
         0 + decoder_channels[3]
      ]
      
      out_channels = decoder_channels

      self.layer1 = DecoderBlock(in_channels[0], out_channels[0], use_batchnorm=use_batchnorm, attention_type=attention_type)
      self.layer2 = DecoderBlock(in_channels[1], out_channels[1], use_batchnorm=use_batchnorm, attention_type=attention_type)
      self.layer3 = DecoderBlock(in_channels[2], out_channels[2], use_batchnorm=use_batchnorm, attention_type=attention_type)
      self.layer4 = DecoderBlock(in_channels[3], out_channels[3], use_batchnorm=use_batchnorm, attention_type=attention_type)
      self.layer5 = DecoderBlock(in_channels[4], out_channels[4], use_batchnorm=use_batchnorm, attention_type=attention_type)
      
      self.final_conv = nn.Conv2d(out_channels[4], final_channels, kernel_size=(1,1))

      self.initialize()

   def forward(self, x, image_size):
      encoder_head = x[0]
      skips = x[1:]

      if self.center:
         encoder_head = self.center(encoder_head)
      
      x = self.layer1([encoder_head, skips[0]])
      x = self.layer2([x, skips[1]])
      x = self.layer3([x, skips[2]])
      x = self.layer4([x, skips[3]])
      x = self.layer5([x, None], image_size)
      x = self.final_conv(x)

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


class Unet(EncoderDecoder):
   def __init__(
      self,
      encoder,
      encoder_outshape,
      decoder_use_batchnorm=True,
      decoder_channels=(256, 128, 64, 32, 16),
      classes=1,
      activation='sigmoid',
      center=False,
      attention_type=None
   ):
      decoder = UnetDecoder(encoder_outshape, decoder_channels=decoder_channels, final_channels=classes, use_batchnorm=decoder_use_batchnorm, center=center, attention_type=attention_type)

      super().__init__(encoder, decoder, activation)
