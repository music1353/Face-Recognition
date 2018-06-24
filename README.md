# Face Recognition with Python

This is a project using python packages, [OpenCV ](https://github.com/opencv/opencv)and [face_recognition](https://github.com/ageitgey/face_recognition), to identify double eyelid and calculate the vertical ratio of subunit below eyelid fold peak to eyebrow-eye unit.

- OpenCV：Use Haar Feature-based Cascade Classifier to classify different parts of face
- face_recognition：Based on CNN model to recognize different parts of face

# Implementation

**Environment**：Mac OSX 10.13.2

**Editor**：Ananconda Jupyter Notebook

**Python Packages**：

- pillow：python image library
- matplotlib：data visualiztion library
- opencv
- face_recogniton

```
pip install Pillow
pip install matplotlib
pip install opencv-python
pip install face_recognition
```

# Errors that Might be Encountered

Before installing face_recognition, installing certain packages is needed.

- cmake (need to add into environment)

  My Solved Case：

  1. Visit cmake official website and download cmake software
  2. Open cmake software. Find "Tools", and click "How To Install For Command Line Use"
  3. Choose one of the solutions (I chose the second one)

- dlib

# How To Practice My Ideas ?

1. Recognize the coordinate of eyebrow and eye from images.
2. Calculate y-axis of lower margin of eyebrow , y-axis of peak of upper eyelid margin, and y-axis of lower eyelid margin.
3. In step 2, if there is double eyelid in the figure, double eyelid fold needs to be included while calculating y-axis of peak of upper eyelid margin.
4. Calculate ratio of subunit below eyelid fold peak to eyebrow-eye unit

# Hassles

1. Not able to recognize double eyelids by recent tools

