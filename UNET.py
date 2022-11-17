from torch import nn
import torch
import torchvision.transforms.functional as TF

class UNet(nn.Module):
  def __init__(self, in_channels=3, out_channels=1, init_features=32):
    super(UNet, self).__init__()
    # Encoder
    self.encoder1 = self._block(in_channels, init_features)
    self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder2 = self._block(init_features, init_features*2)
    self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder3 = self._block(init_features*2, init_features*4)
    self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
    self.encoder4 = self._block(init_features*4, init_features*8)
    self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

    self.encoder5 = self._block(init_features*8, init_features*16)
    self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

    # Bottleneck
    self.bottleneck = self._block(init_features*16, init_features*32)
    # Decoder
    
    self.upconv5 = nn.ConvTranspose2d(init_features*32, init_features*16, kernel_size=2, stride=2)
    self.decoder5 = self._block(init_features*32, init_features*16)

    self.upconv4 = nn.ConvTranspose2d(init_features*16, init_features*8, kernel_size=2, stride=2)
    self.decoder4 = self._block(init_features*16, init_features*8)
    self.upconv3 = nn.ConvTranspose2d(init_features*8, init_features*4, kernel_size=2, stride=2)
    self.decoder3 = self._block(init_features*8, init_features*4)
    self.upconv2 = nn.ConvTranspose2d(init_features*4, init_features*2, kernel_size=2, stride=2)
    self.decoder2 = self._block(init_features*4, init_features*2)
    self.upconv1 = nn.ConvTranspose2d(init_features*2, init_features*1, kernel_size=2, stride=2)
    self.decoder1 = self._block(init_features*2, init_features*1)
    # Classifier
    self.cls = nn.Conv2d(init_features, out_channels, kernel_size=1)
    self.sm = nn.Softmax(dim=1)
  def forward(self, x):
    # Encoder
    enc1 = self.encoder1(x)
    enc2 = self.encoder2(self.pool1(enc1))
    enc3 = self.encoder3(self.pool2(enc2))
    enc4 = self.encoder4(self.pool3(enc3))
    enc5 = self.encoder5(self.pool4(enc4))
    # Bottleneck
    bottleneck = self.bottleneck(self.pool5(enc5))
    # Decoder

    dec5 = self.upconv5(bottleneck)
    dec5 = torch.cat((dec5, enc5), dim=1)
    dec5 = self.decoder5(dec5)

    dec4 = self.upconv4(dec5)
    dec4 = torch.cat((dec4, enc4), dim=1)
    dec4 = self.decoder4(dec4)
    
    dec3 = self.upconv3(dec4)
    dec3 = torch.cat((dec3, enc3), dim=1)
    dec3 = self.decoder3(dec3)

    dec2 = self.upconv2(dec3)
    dec2 = torch.cat((dec2, enc2), dim=1)
    dec2 = self.decoder2(dec2)
    
    dec1 = self.upconv1(dec2)
    dec1 = torch.cat((dec1, enc1), dim=1)
    dec1 = self.decoder1(dec1)
    # Classifier
    output = self.sm(self.cls(dec1))
    return output

  def _block(self, in_channels, features):
    module = nn.Sequential(
        nn.Conv2d(in_channels=in_channels, out_channels=features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(num_features=features),
        nn.ReLU(inplace=True),
        nn.Conv2d(in_channels=features, out_channels=features, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(num_features=features),
        nn.ReLU(inplace=True)
    )
    return module