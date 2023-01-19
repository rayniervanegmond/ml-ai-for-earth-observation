
<img style="margin-bottom:20px" src="_images/pneom_logo_sidebar.svg" alt="thumbnail" width="300"/>

# <span style="margin-bottom:20px">Welcome to the "Machine Learning and Deep Learning for Earth Observation and Monitoring" Book</span>

<div style="font-family:serif;font-style:italic;font-size:12pt;margin-bottom:30pt;color:darkred">
    Author: Raynier A. van Egmond<br>
    Organization: Pacific Northwest EOM program</span><br>
    Contact: raynierx@gmail.com</span><br>
    OpenSource: https://github.com/rayniervanegmond</span><br><br>
    Version: 1.0.0
</div>

This book website provides a dynamic textbook that will address the broader application of machine learning and deep learning in the field of Earth Observation and Monitoring. The approach in the book is to look at the various scenarios and use-cases for ML/DL in Earth Observation and Monitoring. The book will address the "traditional" machine learning methods such as Random Forests and Principal Component Analysis as well as the "modern" machine learning methods based on artificial neural networks based on deep learning for classification, segmentation and object detection.

Most of the books written in the space of machine learning, deep learning and AI have a generic approach that doesn't address the content from a specific domain. This book aims to fill this void for the earth observation and monitoring domain. In this domain --which encompasses much of the weather forecasting, environmental monitoring and climate change research-- the application of machine learning and deep learning promises great advances in the development of solutions.

The field however has a few peculiarities of its own that make the application of these new methods somewhat cumbersome. The main issues are:

**there are very few predefined datasets and benchmark datasets that can be used for training the various methods**
: A typical application in the generic computer vision application of these methods is based on preexisting training sets and applying a technique called transfer-learning to train new task applications. It has been shown that these computer vision pre-trained models and datasets do not provide adequate generalization of the features found in the satellite imagery often used for the earth observation and monitoring projects.

**the size of the dataset images is enormous and requires specific preprocessing**
: It is not uncommon for the pictures in the different satellite data products to be in the gigabyte range. Single images can consist of up to 13 bands (channels in computer vision images) and the size of the image can be 10_000 by 12_000 for (W x H) in 32bit floats. To simply run a single image like this "as-is" in a standard mini-batch dataloader will simply low up any CPU or GPU memory. This means that the pictures as well as any training data like bounding boxes or segmentation masks need to be converted into "chips" which later need to be seamlessly reassembled into pictures.

These are but two of the many earth observation and monitoring machine learning and deep learning applications domain.

In this book I will address the various applications of machine learning and deep learning in use-cases that we typically encounter in the earth observation and monitoring domain. Some of the obvious use-cases are:

* application of unsupervised clustering methods to aggregate pixels based on multiple bands of satellite images.
* application of random forest discriminators and classifiers to perform segmentation task on the satellite images.
* development of new segmentation masks based on a series of refinements (similar to semi-supervised learning) to increase the development of Land Use - Land Cover solutions for areas in the world for which no segmentation masks exist.



## Motivation

The **<span style="color:teal;">purpose</span>** for this site is to provide very targeted hands on guidance to the application af machine and deep learning to the field of earth observation and monitoring projects.<span style="font-family:serif;font-style:italic;color:darkred;">what does the book present?</span>

My **<span style="color:teal;">objective</span>** with the material is to enable citizen scientists and students in the technical studies to quickly get up to speed on how to use these interesting new methods and lower the entry barriers to become a productive team member.<span style="font-family:serif;font-style:italic;color:darkred;">what are the learning objectives for the book?</span>

The **<span style="color:teal;">vision</span>** I have for the material is empower the target audiences in making use of the vast amounts of information available to the public for free. The idea of democratization of this type of information is the vision I have for the material presented in this book. <span style="font-family:serif;font-style:italic;color:darkred;">what should the reader be able to achieve with the material book?</span>

The main parts in the book are:

* <span style="color:cornflowerblue;">The Preamble to the Book</span> [<i class="fa-solid fa-circle-arrow-right" style="margin-left:10px;color:teal;"></i>](part_0/introduction)
: This part discusses the best way to make use of the book content and its dynamic nature in the form of the Jupyter Notebooks that were used to write it. Then the second chapter in the preamble servers as a "Reader's Guide" to the outline and content of the book. 

* <span style="color:cornflowerblue;">An Introduction to Earth Observation methods and Techniques.</span> [<i class="fa-solid fa-circle-arrow-right" style="margin-left:10px;color:teal;"></i>](part_1/introduction)
: This part provides the introduction to the field of earth observation and monitoring in general: the geospatial data and methods used in the field and the way in which ML/DL can be applied to solve problems and answer questions.

<!-- 
### Contributors
1. <a href="https://github.com/ProjectPythiaCookbooks/cookbook-template/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=ProjectPythiaCookbooks/cookbook-template" />
</a> -->
