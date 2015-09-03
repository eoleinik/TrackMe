# Libraries
from random import random
import pickle
import cv2
from pylab import *
from PIL import Image

# User-written
import bayes
import dsift


a=8
#dsift.train_and_save(a)

#load features for Bayes
with open('features.pkl', 'rb'  ) as f:
    blist = pickle.load(f)
    classnames = pickle.load(f)
    V = pickle.load(f)
    m = pickle.load(f)
bc = bayes.BayesClassifier()
bc.train(blist,classnames)

# setup video capture
cap = cv2.VideoCapture(0)

i=0
res=['A']


while True:
    i=i%10000000
    ret,im = cap.read()

    #take every n-th frame
    if i%10==0:
        test_features = dsift.process_video_frame(im,2*a,a,resize=(100,100))
        test_features = array([dot(V,f-m) for f in test_features])

        res = bc.classify(test_features)[0]

    #cv2.rectangle(im,(340,60),(940,660),(0,255,0),1)
    cv2.rectangle(im,(440,160),(840,560),(0,255,0),1)
    text=res[0]
    if text=='L':
        cv2.putText(im,'Hail the admin!',(80,690),cv2.FONT_HERSHEY_COMPLEX,3,(0,255,0),5)
    elif text!='N':
        #letter = dsift.screen_letter(res[0])
        cv2.putText(im,text,(80,620),cv2.FONT_HERSHEY_COMPLEX,6,(0,255,0),5)
    cv2.imshow('video test',im)

    i+=1
    key = cv2.waitKey(10)
    if key == 27:
        break
    if 97<=key<=122:
        im = Image.fromarray(uint8(im)) #.convert('L')
        box=(340,60,940,660)
        im = im.crop(box)
        im=array(im)
        cv2.imwrite('train/'+chr(key-32)+'-vid'+str(int(random()*10**8))+'.jpg',im)