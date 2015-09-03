from PIL import Image
from numpy import *
from pylab import *
import os
import sift
from sift import VLFeat_dir
import tools
import pca
import pickle

prefix = os.path.dirname(__file__)+'/'

#print os.path.join(BASE_DIR, 'pic')


framesname = 'tmp.frame'

def process_image_dsift(imagename,resultname,size=20,steps=10,
                force_orientation=False,resize=None):
    """ Process an image with densely sampled SIFT descriptors
    and save the results in a file. Optional input: size of features,
    steps between locations, forcing computation of descriptor orientation
    (False means all are oriented upward), tuple for resizing the image."""
    im = Image.open(imagename).convert('L')

    if resize!=None:
        im = im.resize(resize)
    m,n = im.size

    if imagename[-3:]!= 'pgm':
        #create a temporary .pgm file
        im.save('tmp.pgm')
        imagename = 'tmp.pgm'

    #create frames and save to a temp file
    scale = size/3.0
    x,y = meshgrid(range(steps,m,steps),range(steps,n,steps))
    xx,yy = x.flatten(), y.flatten()
    frame = array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])])
    savetxt('tmp.frame',frame.T,fmt='%03.3f')

    if force_orientation:
        cmmd = str("sift "+prefix+imagename+" --output="+prefix+resultname+" --read-frames="+prefix+framesname+" --orientations")
    else:
        cmmd = str("sift "+prefix+imagename+" --output="+prefix+resultname+" --read-frames="+prefix+framesname)
    os.system(VLFeat_dir+cmmd)
    print 'processed', imagename, 'to', resultname

def read_gesture_feature_labels(path):
    #create list of all files ending in .dsift
    featlist = [os.path.join(path,f) for f in os.listdir(path) if f.endswith('.dsift')]

    #read the features
    features = []
    for featfile in featlist:
        l,d = sift.read_features_from_file(featfile)
        features.append(d.flatten())
    features = array(features)

    #create labels
    labels = [featfile.split('/')[-1][0] for featfile in featlist]
    return features,array(labels)

def process_data(path,output,a):
    imlist = tools.get_imlist(path)
    #process images with a fixed size(50,50)
    for filename in imlist:
        featfile = output+filename[len(path):-3]+'dsift'
        process_image_dsift(filename,featfile,2*a,a,resize=(100,100))

def print_confusion(res,labels,classnames):
    n = len(classnames)
    #confusion matrix
    class_ind = dict ([(classnames[i],i) for i in range(n)])
    confuse = zeros((n,n))
    for i in range(len(labels)):
        confuse[class_ind[res[i]],class_ind[labels[i]]] += 1
    print 'Confusion matrix for'
    print classnames
    print confuse

def process_video_frame(im,size=20,steps=10,force_orientation=False,resize=None):
    """Process an image with densely sampled SIFT descriptors
    and save the results in a file. Optional input: size of features,
    steps between locations, forcing computation of descriptor orientation
    (False means all are oriented upward), tuple for resizing the image."""
    im = Image.fromarray(uint8(im)).convert('L')

    #box=(340,60,940,660)
    box=(440,160,840,560)
    im = im.crop(box)

    if resize!=None:
        im = im.resize(resize)
    m,n = im.size

    #create a temporary .pgm file
    im.save('tmp.pgm')
    imagename = 'tmp.pgm'
    resultname = 'video/tmp.dsift'

    #create frames and save to a temp file
    scale = size/3.0
    x,y = meshgrid(range(steps,m,steps),range(steps,n,steps))
    xx,yy = x.flatten(), y.flatten()
    frame = array([xx,yy,scale*ones(xx.shape[0]),zeros(xx.shape[0])])
    savetxt('tmp.frame',frame.T,fmt='%03.3f')

    if force_orientation:
        cmmd = str("sift "+prefix+imagename+" --output="+prefix+resultname+" --read-frames="+prefix+framesname+" --orientations")
    else:
        cmmd = str("sift "+prefix+imagename+" --output="+prefix+resultname+" --read-frames="+prefix+framesname)
    os.system(VLFeat_dir+cmmd)
    #print 'processed', imagename, 'to', resultname


    #reading back
    features = []
    l,d = sift.read_features_from_file(resultname)
    features.append(d.flatten())
    features = array(features)

    return features

def train_and_save(a):
    process_data('train/','train_sifts/',a) #process training data
    features,labels = read_gesture_feature_labels('train_sifts/')
    classnames = unique(labels) #sorted lists of unique class names
    V,S,m = pca.pca(features)
    #keep most important dimensions
    dims = 50
    V = V[:dims]
    features = array([dot(V,f-m) for f in features])
    blist = [features[where (labels==c)[0]] for c in classnames]
    with open('features.pkl', 'wb') as f:
        pickle.dump(blist,f)
        pickle.dump(classnames,f)
        pickle.dump(V,f)
        pickle.dump(m,f)