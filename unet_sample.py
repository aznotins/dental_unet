from unet import Unet

n = Unet.load('models/unet3-weights-improvement-04-0.68.hdf5', num_classes=7)
# n.test('/junction/dental_unet/files/train/1.bmp', 'res.bmp')
n.test('/junction/dental_unet/files/train/2.bmp', 'res.bmp')