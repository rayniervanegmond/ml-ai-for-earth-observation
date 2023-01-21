#!/usr/bin/env python
# coding: utf-8

# <img src="../../../images/logos/pneom_logo.svg" width=150 alt="PNEOM Curriculum Logo"></img>

# 
#  [to chapter introduction <i class="fa-solid fa-circle-arrow-up" style="margin-left:10px;color:teal;"></i>](../../chpt_8)
# 
#  [to chapter notebooks <i class="fa-solid fa-circle-arrow-left" style="margin-left:10px;color:teal;"></i>](../../chpt_8.html#chapter-notebooks)
# 

# 
# # <span style="color:teal;font-size:20pt;">Land Use and Land Cover Semantic Segmentation Project</span>

# ## <span style="color:salmon;font-size:16pt;">Overview</span>

# ...

# ## <span style="color:salmon;font-size:16pt;">Methodology</span>

# ### <span style="font-size:13pt;font-weight:bold;">Step 0: Environment and Libraries Setup</span>

# The project uses the pretrained ResNet model that was trained on over 300K images and has learned to recognize 1000 different things in the images. Because we are training to classify the image based on what is displayed in it we end up with the classification of "a dog-image" or a "train-image". We will discuss in a different sample project <span style="color:red;font-weight:bold">\<REPLACE WITH ACTUAL REFERENCE\></span>. 
# 
# The inference is done on some of the hold-out validation images and also on images you can download from the internet yourself.

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

# ## <span style="color:salmon;font-size:16pt;">Bibliography</span>

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
