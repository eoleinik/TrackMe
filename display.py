from PIL import Image
from numpy import *
from pylab import *
import os
from scipy import *
import sift,dsift

dsift.process_image_dsift('video/Fhist.jpg','video/FDsift.sift',30,15,True,resize=(200,200))
l,d = sift.read_features_from_file('video/FDsift.sift')

im = Image.open('video/Fhist.jpg')
im = im.resize((200,200))
im = array(im)
gray()
sift.plot_features(im,l,False)
show()