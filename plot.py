from PIL import Image
from numpy import *
from pylab import *
import sift


imname = 'pic/test.jpg'
im1 = array(Image.open(imname).convert('L'))
sift.process_image(imname,'holmes.sift')
l1,d1 = sift.read_features_from_file('holmes.sift')

figure()
gray()
sift.plot_features(im1,l1,circle=True)
show()