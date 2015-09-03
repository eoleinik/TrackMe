from PIL import Image
import tools
from pylab import *
import os
import sift
"""
featlist = [os.path.join('wrong/',f) for f in os.listdir('wrong/') if f.endswith('.jpg')]
for featfile in featlist:
    newname='wrong/F'+featfile[7:]
    Image.open(featfile).save(newname)

# read image to array
im = array(Image.open('pic/test.jpg')
print type(im)

# create a new figure
figure()
# don't use colors
gray()
# show contours with origin upper left corner
contour(im, origin='image')
axis('equal')
axis('off')

print (ord('b')-ord('B'))
print (ord('a'),ord('z'))
"""

im = Image.open('video/V.png').convert('L')
im = array(im)

"""
v = [0,255,0,9000]

figure()
axis(v)
h = hist(im.flatten(),256)
"""
im,cdf = tools.histeq(im)
"""
figure()
axis(v)
hist(im.flatten(),256)
show()"""
image = Image.fromarray(uint8(im))
sift.process_image('video/Fhist.jpg','video/Five.sift')
l,d = sift.read_features_from_file('video/Five.sift')

figure()
gray()
sift.plot_features(im,l,circle = True)
show()