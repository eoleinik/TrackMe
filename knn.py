#K-nearest neighbors classifier
from PIL import Image
from numpy import *
from pylab import *

class KnnClassifier(object):
    def __init__(self,labels,samples):
        """initialize with training set (TS)"""
        self.labels = labels
        self.samples = samples

    def classify(self,point,k=3):
        """Classify new point against k nearest in TS, return class"""
        #compute distance to all points in TS
        dist = array([L2dist(point,s) for s in self.samples])

        #sort them
        ndx = dist.argsort()

        #store K nearest in a dictionary
        votes = {}
        for i in range(k):
            label = self.labels[ndx[i]]
            votes.setdefault(label,0)
            votes[label]+=1

        return max(votes)

def L2dist(p1, p2):
    """Euclidian distance measure"""
    return sqrt(sum((p1-p2)**2))