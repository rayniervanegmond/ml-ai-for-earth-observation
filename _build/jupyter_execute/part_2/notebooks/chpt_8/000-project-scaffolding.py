#!/usr/bin/env python
# coding: utf-8

# <img src="../../../images/logos/pneom_logo.svg" width=150 alt="PNEOM Curriculum Logo"></img>

# 
#  [to chapter introduction <i class="fa-solid fa-circle-arrow-up" style="margin-left:10px;color:teal;"></i>](../../chpt_8)
# 
#  [to chapter notebooks <i class="fa-solid fa-circle-arrow-left" style="margin-left:10px;color:teal;"></i>](../../chpt_8.html#chapter-notebooks)
# 

# # <span style="color:teal;font-size:20pt;">PNEOM Deep Learning Project Scaffolding </span>

# ## <span style="color:salmon;font-size:16pt;">Overview</span>

# 
# The purpose of this notebook is to provide an executable example of a Jupyter Notebook that you can use for all your Pytorch based deep learning projects. As mentioned in chapter 8.1 of the curriculum moist projects will follow the same steps. They are simply part of every machine learning or deep learning project. 
# 
# Before we can start our work in the Pytorch context we need to setup the Pytorch environment and all the required libraries needed to run the project. This concerns installing (locally in the kernel) any libraries that are not part of the standard Anaconda environment (for the installed libraries of that environment see the file: ["Setup and install of Pytorch13 environment"]("...")). 
# 
# In this setup stage we use the `!pip install <package>` or `!conda install <package>` managers to install any missing packages. See the referenced file to learn how to modify (extend) the suite of default installed packages. Once we have our packages installed and imported we can start processing the source information of stag one of the scaffolding.
# 
# The steps are:
# 
# <font color="#FF5733;">1. data pre-processing, cleaning and preparation</font>
# : this stage concerns the work that needs to be done to modify and convert the source data into a form that can be used for the data ingestion stage of the deep learning models. Clearly the step is different for every project and its main purpose is to massage the data into a standard form that can be used by the Pytorch Dataset and DataLoader classes which are the standard entry form for all Pytorch based deep learning projects.
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

# ## <span style="color:salmon;font-size:16pt;">Methodology</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 0: Environment and Libraries Setup</span>

# In this chapter discuss the architecture for the project and its relation to earlier or other peoples work. 
# 
# * What is the architecture (family) you are using for the project?
# * Why do you think this is (or they are) the most appropriate choices for the problem?
#  
# Then another topic to discus is why the project outline was defined as it is; 
# 
# * What is the source information why did you choose it?
# * How do you think the sources and the way they we used will support the project?

# ## <span style="color:salmon;font-size:16pt;">Project Outline</span>

# ### <span style="font-size:13pt;font-weight:bold;">Introduction</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 0: Environment and Libraries Setup</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 1. Data pre-processing, cleaning and preparation</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 2. Dataset and dataloader definition</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 3. Data pre-processing, cleaning and preparation</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 4. deep learning model definition</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 5. model training and evaluation</span>
# 

# 
# ### <span style="font-size:13pt;font-weight:bold;">Step 6. model deployment and inference</span>

# ## <span style="color:salmon;font-size:16pt;">Discussion</span>
# 
# placeholder text

# ## <span style="color:salmon;font-size:16pt;">Bibliography</span>
# 
# placeholder text

# ---

# ## Overview
# If you have an introductory paragraph, lead with it here! Keep it short and tied to your material, then be sure to continue into the required list of topics below,
# 
# 1. This is a numbered list of the specific topics
# 1. These should map approximately to your main sections of content
# 1. Or each second-level, `##`, header in your notebook
# 1. Keep the size and scope of your notebook in check
# 1. And be sure to let the reader know up front the important concepts they'll be leaving with

# ## Prerequisites
# This section was inspired by [this template](https://github.com/alan-turing-institute/the-turing-way/blob/master/book/templates/chapter-template/chapter-landing-page.md) of the wonderful [The Turing Way](https://the-turing-way.netlify.app/welcome.html) Jupyter Book.
# 
# Following your overview, tell your reader what concepts, packages, or other background information they'll **need** before learning your material. Tie this explicitly with links to other pages here in Foundations or to relevant external resources. Remove this body text, then populate the Markdown table, denoted in this cell with `|` vertical brackets, below, and fill out the information following. In this table, lay out prerequisite concepts by explicitly linking to other Foundations material or external resources, or describe generally helpful concepts.
# 
# Label the importance of each concept explicitly as **helpful/necessary**.
# 
# | Concepts | Importance | Notes |
# | --- | --- | --- |
# | [Intro to Cartopy](https://foundations.projectpythia.org/core/cartopy/cartopy.html) | Necessary | |
# | [Understanding of NetCDF](https://foundations.projectpythia.org/core/data-formats/netcdf-cf.html) | Helpful | Familiarity with metadata structure |
# | Project management | Helpful | |
# 
# - **Time to learn**: estimate in minutes. For a rough idea, use 5 mins per subsection, 10 if longer; add these up for a total. Safer to round up and overestimate.
# - **System requirements**:
#     - Populate with any system, version, or non-Python software requirements if necessary
#     - Otherwise use the concepts table above and the Imports section below to describe required packages as necessary
#     - If no extra requirements, remove the **System requirements** point altogether

# ---

# ## Imports
# Begin your body of content with another `---` divider before continuing into this section, then remove this body text and populate the following code cell with all necessary Python imports **up-front**:

# In[1]:


import sys


# ## Your first content section

# This is where you begin your first section of material, loosely tied to your objectives stated up front. Tie together your notebook as a narrative, with interspersed Markdown text, images, and more as necessary,

# In[2]:


# as well as any and all of your code cells
print("Hello world!")


# ### A content subsection
# Divide and conquer your objectives with Markdown subsections, which will populate the helpful navbar in Jupyter Lab and here on the Jupyter Book!

# In[3]:


# some subsection code
new = "helpful information"


# ### Another content subsection
# Keep up the good work! A note, *try to avoid using code comments as narrative*, and instead let them only exist as brief clarifications where necessary.

# ## Your second content section
# Here we can move on to our second objective, and we can demonstrate

# ### Subsection to the second section
# 
# #### a quick demonstration
# 
# ##### of further and further
# 
# ###### header levels

# as well $m = a * t / h$ text! Similarly, you have access to other $\LaTeX$ equation [**functionality**](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Typesetting%20Equations.html) via MathJax (demo below from link),
# 
# \begin{align}
# \dot{x} & = \sigma(y-x) \\
# \dot{y} & = \rho x - y - xz \\
# \dot{z} & = -\beta z + xy
# \end{align}

# Check out [**any number of helpful Markdown resources**](https://www.markdownguide.org/basic-syntax/) for further customizing your notebooks and the [**Jupyter docs**](https://jupyter-notebook.readthedocs.io/en/stable/examples/Notebook/Working%20With%20Markdown%20Cells.html) for Jupyter-specific formatting information. Don't hesitate to ask questions if you have problems getting it to look *just right*.

# ## Last Section
# 
# If you're comfortable, and as we briefly used for our embedded logo up top, you can embed raw html into Jupyter Markdown cells (edit to see):

# <div class="admonition alert alert-info">
#     <p class="admonition-title" style="font-weight:bold">Info</p>
#     Your relevant information here!
# </div>

# Feel free to copy this around and edit or play around with yourself. Some other `admonitions` you can put in:

# <div class="admonition alert alert-success">
#     <p class="admonition-title" style="font-weight:bold">Success</p>
#     We got this done after all!
# </div>

# <div class="admonition alert alert-warning">
#     <p class="admonition-title" style="font-weight:bold">Warning</p>
#     Be careful!
# </div>

# <div class="admonition alert alert-danger">
#     <p class="admonition-title" style="font-weight:bold">Danger</p>
#     Scary stuff be here.
# </div>

# We also suggest checking out Jupyter Book's [brief demonstration](https://jupyterbook.org/content/metadata.html#jupyter-cell-tags) on adding cell tags to your cells in Jupyter Notebook, Lab, or manually. Using these cell tags can allow you to [customize](https://jupyterbook.org/interactive/hiding.html) how your code content is displayed and even [demonstrate errors](https://jupyterbook.org/content/execute.html#dealing-with-code-that-raises-errors) without altogether crashing our loyal army of machines!

# ---

# ## Summary
# Add one final `---` marking the end of your body of content, and then conclude with a brief single paragraph summarizing at a high level the key pieces that were learned and how they tied to your objectives. Look to reiterate what the most important takeaways were.
# 
# ### What's next?
# Let Jupyter book tie this to the next (sequential) piece of content that people could move on to down below and in the sidebar. However, if this page uniquely enables your reader to tackle other nonsequential concepts throughout this book, or even external content, link to it here!

# ## Resources and references
# Finally, be rigorous in your citations and references as necessary. Give credit where credit is due. Also, feel free to link to relevant external material, further reading, documentation, etc. Then you're done! Give yourself a quick review, a high five, and send us a pull request. A few final notes:
#  - `Kernel > Restart Kernel and Run All Cells...` to confirm that your notebook will cleanly run from start to finish
#  - `Kernel > Restart Kernel and Clear All Outputs...` before committing your notebook, our machines will do the heavy lifting
#  - Take credit! Provide author contact information if you'd like; if so, consider adding information here at the bottom of your notebook
#  - Give credit! Attribute appropriate authorship for referenced code, information, images, etc.
#  - Only include what you're legally allowed: **no copyright infringement or plagiarism**
#  
# Thank you for your contribution!
