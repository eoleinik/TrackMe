from numpy import *
from scipy import *
import dsift
import knn

a=8
#dsift.process_data('train/','train_sifts/',a) #process training data
#dsift.process_data('test/','test_sifts/',a)   #process test data

features,labels = dsift.read_gesture_feature_labels('train_sifts/')
test_features,test_labels = dsift.read_gesture_feature_labels('test_sifts/')

classnames = unique(labels) #sorted lists of unique class names

#test kNN
k = 3
knn_classifier = knn.KnnClassifier(labels,features)
res = array([knn_classifier.classify(test_features[i],k) for i in range(len(test_labels))])

#accuracy
acc = sum(1.0*(res==test_labels)) / len(test_labels)
print 'Accuracy:', acc

dsift.print_confusion(res,test_labels,classnames)