# <span style="color:cornflowerblue;">Chapter 8 - Using Deep Learning methods-Introduction and applications </span>

## An Introduction and Outline

The developments of deep learning methods is one of the fastest developing areas in computer science to-date. Its influence on society is rapidly expanding and --some would say-- not always for the good of society. Most of the developments that the press and society call artificial intelligence are based on the application of deep learning methods and techniques. Deep learning is grounded in the application of what are called artificial neural networks. A method of computing where the data structures are used to detect and learn patterns in datasets using one of a few types of processing. 

Generally the field of machine learning (of which deep learning is a sub-field) are divided into the following main categories:

* **<font color="#FF5733;">unsupervised learning</font>**: the algorithm learns to cluster observations together based on shared properties and most similar values thereof. The algorithm learns to detect these clusters or groups from the observations only and is not provided with labels for the observations.
* **<font color="#FF5733;">supervised learning</font>**: in this case the algorithm learns from the datasets by means of observations and labels. The labels provide the relationship between input observations and labels. The learned relations are classifications if the labels represent distinct categories of objects or regressions of the "label" is a real-value.
* **<font color="#FF5733;">semi-supervised learning</font>**: in this case the algorithm learns the relations from a limited set of labeled observations and a large number of unlabeled observations. The techniques transfer the information gleaned from the labeled observations onto the unlabeled observations  after which it can use these observations to improve the learned classification or regression problem.
* **<font color="#FF5733;">self-supervised learning</font>**: in this case the algorithm learns the relations from a limited set of unlabeled observations. The class of Transformer models is the main example family that belongs to this category. The way in which the algorithm learns its task is through modification of the input data (in some algorithmically defined manner with stochastic properties) and then it tries to reconstruct the input data. So in a sense it creates the "labels" from the "input source" and thus the moniker of **self**-supervised learning. This is one of the most powerful deep learning methods in the field. In its latest version it applies to almost all domains relevant for business.
* **<font color="#FF5733;">generative learning</font>**: in this case the algorithm learns to create outputs that are similar in nature to the provided input observations.
* **<font color="#FF5733;">generative learning</font>**: in this case the algorithm learns to create outputs that are similar in nature to the provided input observations.
* **<font color="#FF5733;">reinforcement learning</font>**: this is a very different kind of learning algorithm and its primary objective is learn strategies on how to solve a problem. It finds an optimal approach to find a solution by executing an immense amount of trial-and-error attempts and record which approaches worked best to solve the problem. Ultimately the algorithm learns the optimal way to execute a task.

The field of deep-learning applies to each of these categories. The deep learning methods are simply other ways of doing the activity. What sets the deep learning approach apart from the traditional machine learning methods is that the latter use "pre-defined features" its decision processes are based on the a-priori definition of features that guide the solution. In the deep learning situation the main applications provide unstructured information from which the algorithm learns and applies the features that best solve the problem (classification or regression). It is in this field of feature learning in fields like computer vision, natural language processing or dataset interpretations that the new deep learning methods shine and deliver state-of-the-art solutions. 

In this chapter we will begin the discussion of deep learning with an introduction into the underlying data constructs called Artificial Neural Networks (abbrev. ANN). We will discuss their structure and how various architectures of such ANN exhibit best performances for specific problems. We will explore the interesting field of deep learning through a series of Jupyter Notebooks that illustrate the concepts and how we can use deep learning using the Pytorch ecosystem of tools, libraries and datasets.

### Deep Learning for classification

In the field of deep learning the most used application is that of supervised learning for classification purposes. In chapters [A discussion of classification for earth observation](chpt_8_3.md) we will take a deep dive into the various types of classification that we can use to do to build Earth Observation and Monitoring solutions. Based on the type of classification you want to perform there are different methods to apply. The following types of supervised classifications will be addressed in these chapters:

* **image classification**: this concerns the classification of an entire image (or part of an image) or some other raster-based dataset (part) using a neural network. The primary characteristic of this type of problem is that the image/datafield itself has to be classified. An example would be to classify an image as a "dog-image" or a "cat-image". [An annotated sample project for landscape type classification](chpt_8_3.md)
* **object identification**: this concerns the case where we want to detect the occurrence of a specific object in the frame of the image and we need to draw a bounding box around the object. We identify, classify and indicate objects inside the image. So if we take the previous example, we would now draw a box around the actual "dog-inside-the-image". Some algorithms are able to detect and indicate multiple objects inside a single image. This would then draw boxes around each dog and each cat inside the image. [An annotated sample project for land use object detection](chpt_8_4.md)
### Deep Learning for natural language processing

### 

## Chapter Table of Content

* [Discussing the Pytorch deep learning project template](chpt_8_1.md)
* [An annotated sample project for image classification](chpt_8_2.md)
* [An annotated sample project for landscape type classification](chpt_8_3.md)
* [An annotated sample project for land use object detection](chpt_8_4.md)

---
## Chapter Notebooks

deep learning project template  [<i class="fa-solid fa-arrow-circle-right" style="margin-left:10px;color:teal;"></i>](notebooks/chpt_8/000-project-scaffolding)
: The purpose of this notebook is to provide an executable example of a Jupyter Notebook that you can use for all your Pytorch based deep learning projects. As mentioned in chapter 8.1 of the curriculum moist projects will follow the same steps. They are simply part of every machine learning or deep learning project.

simple image classification project  [<i class="fa-solid fa-arrow-circle-right" style="margin-left:10px;color:teal;"></i>](notebooks/chpt_8/001-image-classification)
: The purpose of this notebook is to provide a simple illustrative example of a full image classification project. Its main purpose is to show how the project scaffolding template is used and what the main steps in a project look like. It is more heavily annotated than regular project notebooks wold be.
