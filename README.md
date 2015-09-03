# TrackMe
Python app, recognising hand gestures from webcam in real time, using [Scale Invariant Feature Transform](https://en.wikipedia.org/wiki/Scale-invariant_feature_transform) algorithm.
![](https://github.com/JakeOleinik/TrackMe/blob/master/screens%20from%20TrackMe/screen.jpg)
Rough description of the algorithm:

0. Classifier training (see below).

1. Extracts frames from webcam video stream.

2. Dense SIFT descriptor is applied to points, uniformly distributed on the image, to frame to extract features.

3. Features are classified using pre-trained [Bayes Classifier](https://en.wikipedia.org/wiki/Naive_Bayes_classifier) to determine hand gesture.

#Dependencies:
[OpenCV](http://opencv.org) (webcam access)

[VLFeat](http://www.vlfeat.org) (SIFT descriptor)

Packages from `requirements.txt`

#Use
To start, run
```
python GestureRecognition(camera).py
```
and place your palm inside of the green square. The letter will appear, indicating hand gesture.

#Examples:
App was trained to distinguish between 8 classes: [Nothing, A, B, C, F, P, V, Admin's handsome face].
![](https://github.com/JakeOleinik/TrackMe/blob/master/screens%20from%20TrackMe/gestures.jpg)

#Training the Classifier
For the app, I implemented and tried 2 classifiers - Bayes and k-Nearest Neighbours. Bayes was more accurate, classifying gestures correctly up to 90% of the time with good lighting conditions.

The classifier was trained using ~100 images I generated for each class (its settings are pickled in <i>features.py</i>). If you would like to re-train it using the classifier, you can place images in the "test" folder in the root of the project and uncomment the lines in the beginning of `GestureRecognition(camera).py`

Happy playing!
