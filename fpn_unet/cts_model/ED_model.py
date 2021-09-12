import numpy as np 
import torch
import torch.nn as nn

class EncoderDecoder(nn.Module):
   def __init__(self, encoder, decoder, activation):
      super().__init__()
      self.encoder = encoder
      self.decoder = decoder

      if callable(activation) or activation is None:
         self.activation = activation
      elif activation == 'softmax':
         self.activation == nn.Softmax(dim=1)
      elif activation == 'sigmoid':
         self.activation = nn.Sigmoid()
      else:
         raise ValueError('Activation should be "sigmoid", "softmax", callable, None')
   
   def forward(self, x):
      image_size = x.size()[-2:]
      x = self.encoder(x)
      x = self.decoder(x, image_size)
      return x 
   
   def predict(self, x):
      
      if self.training:
         self.eval()
      
      with torch.no_grad():
         x = self.forward(x)
         if self.activation:
            x=self.activation(x)
      
      return x
