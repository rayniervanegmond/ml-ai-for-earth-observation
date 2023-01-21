#!/usr/bin/env python
# coding: utf-8

# # <span style="color:cornflowerblue;">Chapter 8.1 - Illustrating the workings of the Pytorch environment </span>
# 
# In this chapter we will look at how at some illustrations --without the explanation of the underlying Pytorch codings--  to see how we can use deep learning for some of the categories of machine learning we briefly discussed earlier (supervised, unsupervised, generative). The purpose of this first notebook series and explanations is to simply get a feel how we can use the Pytorch framework.
# 
# These illustrations will get you familiar with the general approach of using the deep learning frameworks. The solutions to any deep learning project typically has the same outline. The main steps are:
# 
# <font color="#FF5733;">1. data pre-processing, cleaning and preparation</font>
# : this concerns the processing of the raw data, the observations and cleaning up the input data. In the case of earth observation and monitoring applications we often have a great deal of work to do before we actually start working with the actual deep learning methods and techniques.
# 
# <font color="#FF5733;">2. dataset and dataloader definition</font>
# : this concerns the casting the (vast amount) of input data into data objects that are defined by the Pytorch framework with the purpose of standardizing the feeding process of the defined deep learning models.
# 
# <font color="#FF5733;">3. deep learning model definition</font>
# : this concerns definition of the architecture of the neural network model and deciding whether or not we will use pre-trained models or do all the training ourselves. The things we do in the model definition stage is the specify the behavior of the model and its learning process.
# 
# <font color="#FF5733;">4. model instantiation</font>
# : once we have the definition of the model architecture complete we need to actually instantiate the model and connect it to the data sources from which it will learn and train.
# 
# <font color="#FF5733;">5. model training and evaluation</font>
# : this is the moment where the deep learning neural network will start processing the input training data and based on the progress adapts its internal parameters the get progressively better in doing its task. The process is basically a cycle of learning, evaluating, updating the parameters until progress is sufficient or does not improve in a relevant amount.
# 
# <font color="#FF5733;">6. model deployment and inference</font>
# : the final stage is where the trained model is saved and used for doing its actual operational task of making inferences or creating generative outputs.
# 
# ## Deep Learning Project Scaffolding
# 
# Since the steps outlined above occur in just about every deep learning project we'll use it as the outline for all the (starter) projects we will be writing in this curriculum. In the notebooks below we will use these steps as the main components of the processing we will do. 
# 
# ---
# 
# ## Chapter Notebooks
# 
# deep learning project template  [<i class="fa-solid fa-arrow-circle-right" style="margin-left:10px;color:teal;"></i>](notebooks/chpt_8/000-project-scaffolding)
# : The purpose of this notebook is to provide an executable example of a Jupyter Notebook that you can use for all your Pytorch based deep learning projects. As mentioned in chapter 8.1 of the curriculum moist projects will follow the same steps. They are simply part of every machine learning or deep learning project.
