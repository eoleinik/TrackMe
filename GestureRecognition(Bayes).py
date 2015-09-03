from numpy import *
import dsift
import pca
import bayes

a=8

#dsift.process_data('train/','train_sifts/',a) #process training data
#dsift.process_data('test/','test_sifts/',a)   #process test data

features,labels = dsift.read_gesture_feature_labels('train_sifts/')
test_features,test_labels = dsift.read_gesture_feature_labels('test_sifts/')

classnames = unique(labels) #sorted lists of unique class names

V,S,m = pca.pca(features)

#keep most important dimensions
dims = 50
V = V[:dims]
features = array([dot(V,f-m) for f in features])
test_features = array([dot(V,f-m) for f in test_features])

#test Bayes
bc = bayes.BayesClassifier()
blist = [features[where (labels==c)[0]] for c in classnames]

bc.train(blist,classnames)
res = bc.classify(test_features)[0]

#accuracy
acc = sum(1.0*(res==test_labels)) / len(test_labels)
print 'Accuracy:', acc

dsift.print_confusion(res,test_labels,classnames)