# Face Recognition with Python

This is a project that used python packages, <a href="https://github.com/opencv/opencv">OpenCV </a>and <a href="https://github.com/ageitgey/face_recognition">face_recognition</a>, to calculate proportion of distance between eyebrow and eye

* OpenCV：Use Haar Feature-based Cascade Classifier to classify different parts of face
* face_recognition：Based on CNN model to recognize different parts of face



# Implementation

**Environment**：Mac OSX 10.13.2

**Editor**：Ananconda Jupyter Notebook

**Python Packages**：

* pillow：python image library
* matplotlib：data visualiztion library

* opencv
* face_recogniton

~~~python
pip install Pillow
pip install matplotlib
pip install opencv-python
pip install face_recognition
~~~



# Some Errors That You May Encounter

Before installing face_recognition, you need to install some packages

* cmake (need to add into environment)

  My Solved Case：

  1. Visit cmake official website and download cmake software
  2. Open cmake software then find "Tools", click "How To Install For Command Line Use"
  3. Choose one of the solutions (I choose the second one)

* dlib



#How To Practice My Ideas ?

1. Recognize the coordinate of eyebrow and eye from image
2. Calculate y-axis of eyebrow , top y-axis of eye and bottom y-axis of eye
3. Calculate eyebrow & eye ratio



# Hassles

1. Can't recognize double eyelids