# A Reader Guide to "Machine Learning and Deep Learning for Earth Observation and Monitoring"

<span style="font-size:20pt">An Introduction and Outline</span>

This book aims to provide a comprehensive curriculum to teach a thorough understanding of the problems the domain of Earth Observation and Monitoring tries to solve and how machine learning, deep learning and AI can support this research domain. To build this understanding the book provides main sections of content that explain:

<font color="#FF5733;">introduction to the field of Earth Observation and Monitoring (abbrev. EOM)</font>
: This part addresses the types of problems EOM tries to solve. It provides a short discussion of the why we would want to have satellite observation data (images and data collection) of the earth's surface and what the benefits of such observation based approaches are. It then proceeds to discuss in detail the types of data (imagery and other datasets) can be used for these solutions. EOM data can be very varied in nature and the strenght of ML/DL/AI lies in the combination of these data sources to create new insights in relationships between data streams.

<font color="#FF5733;">introduction to the types of data processing in the Earth Observation and Monitoring - Machine Learning (abbrev. EOM-ML) field</font>
: This parts both discussing the "types of questions" EOM practitioners seek to answer as well as a more technical discussion of the ML/DL/AI methods we can bring to the table. The objective of the part is to teach the practitioners to recognize and apply the right tools for a specific problem. This part addresses the application of machine learning forms to geospatial data processing.

<font color="#FF5733;">use-case discussions</font>
: The final parts of the book provides a series of worked examples that show the application of the material explained in the earlier parts of the book to various problems in earth observation and monitoring.

<font color="#FF5733;">earth observation and monitoring machine learning project management</font>
: This parts of the book provides a a framework for defining machine learning based or supported projects in the field of Earth Observation and Monitoring such as environmental monitoring using for instance image classification and field research. Too often EOM-ML projects are started that only address the training of new neural network models. The deployment of such projects then fall short with discouraged stakeholders because the deployment and maintenance phases of such projects were not addressed. The aim of this part is to provide a "starter framework" for EOM-ML project definition and management that will result in successful application that actually make a difference in the world.

## Structure of the Book
The structure of this Jupyter Book is geared to support a very hands on approach to working on ICT solutions for the processing of Earth Observation and Monitoring datasets. Towards this end the chapters and sections in the book are all accompanied with Jupyter notebooks that can be executed in the students local computer environment or online in the cloud. The output of the notebooks is included as static text in the content of the eBook so the content can be printed out in a number of different formats (such as static HTML or Pdf) for offline reading.  

### <span style="color:cornflowerblue;font-size:smaller;">Part 1 "introduction to the field of Earth Observation and Monitoring"</span>

The learning objectives of the first part of the book are that the user will understand what specific problems the field of Earth Observation and Monitoring aims to solve and the larger context in which these projects play a role. Earth Observation and Monitoring basically --as the name describes-- perform satellite-based observations of environmental and geophysical processes that play out across the globe. The main answers the field tries to answer relate to determining if there are important changes occurring at various levels of geospatial resolutions. 

To answer if relevant change is occurring we first need to establish a baseline of the relevant environmental parameters and then over time track any changes in the geo-located information and datasets to detect trends and evaluate these trends. These are the two main purposes of the earth **observation** and **monitoring** process.

The first part of the book contains the following chapters:

["Chapter 1 - What is earth observation and why do we do it?"](../part_1/chpt_1.md) <br> 
["Chapter 2 - What are the main areas of application of earth observation and monitoring?"](../part_1/chpt_2.md) <br> 
["Chapter 3 - How can we apply the different ML techniques to EOM problems?"](../part_1/chpt_3.md) <br> 


### <span style="color:cornflowerblue;font-size:smaller;">Part 2 "ML/DL for Geospatial Data Processing"</span>

The learning objectives for this part of the book are that the user will come understand the exceptional nature of satellite based geospatial information processing and which tools are most appropriate to which problems. Since the field of application of machine learning, deep learning and artificial intelligence to the processing of geospatial data is still very new the users will need to gain an understanding of the kinds of problems rather than simply learning to apply cookie-cutter recipes. Often the practitioner will find out that the standard computer vision approaches are simply note practical in this field; either because information the CV field takes for granted is not available or the technical aspects of the data make such CV recipes impractical or downright impossible.  

To survive in such situation and find solutions the practitioner needs to understand the nature of the problem and figure out how a smart combination of tools and techniques might provide solutions to problems and how these solutions might be automated. This makes the field of applying ML/DL in Geospatial information processing both challenging and interesting. The practitioner better learn to like these problems because they are sure to arise during the projects.

There are many different machine learning and deep learning frameworks that practitioners can choose from. Most notably --with respect to the deep learning field-- there are the Tensorflow framework and the Pytorch framework. For this first release of the book I have chosen to build the material on the Pytorch ecosystem of frameworks and libraries. It seems to be the ruling opinion in the field that the Pytorch framework and its associated libraries is the most dynamic framework with most research projects in the field of deep learning and its applications in EOM (in many domains of deep learning) being done with the Pytorch framework. 

For the "standard" machine learning methods we use the tested SciPy and Scikit-Learn packages. These are simply the go-to libraries for all things data science. For the setup of the local computing environment the eBook provides configuration files that will create the correct environment for running all the Jupyter Notebooks in the various chapters. 

This part of the book contains the following chapters:

["Chapter 4 - What types of geospatial data do we deal with?"](../part_2/chpt_4.md) <br> 
["Chapter 5 - What types of machine learning techniques do we need?"](../part_2/chpt_5.md) <br> 
["Chapter 6 - How do we create the data sources we need?"](../part_2/chpt_6.md) <br> 
["Chapter 7 - Using Machine Learning methods-Introduction and applications"](../part_2/chpt_7.md) <br> 
["Chapter 8 - Using Deep Learning methods-Introduction and applications"](../part_2/chpt_8.md) <br> 

### <span style="color:cornflowerblue;font-size:smaller;">Part 3 "Use-cases of ML/DL for Earth Observation and Monitoring"</span>

The learning objectives for this part of the book are that, through the creation of complete sample projects and discussions, the user will get a practical appreciation for what it takes to do an Earth Observation and Monitoring project using machine learning and deep learning from start to finish. Each of the discussed projects will proceed in a question-and-answer format to illustrate the questions the practitioner needs to ask and consider to create a full machine learning pipeline for the project. Note that in these use-cases we do not address the project management aspects that are the subject of Part 4 of the book.


This part of the book contains the following chapters:

["Chapter 9 - "](../part_3/chpt_9.md) <br> 
["Chapter A - "](../part_3/chpt_A.md) <br> 
["Chapter B - "](../part_3/chpt_B.md) <br> 


### <span style="color:cornflowerblue;font-size:smaller;">Part 4 "A Project Management Framework for ML/DL projects in Earth Observation and Monitoring"</span>

The learning objectives for this part of the book are that, through 

This part of the book contains the following chapters:

["Chapter C - "](../part_4/chpt_C.md) <br> 
["Chapter D - "](../part_4/chpt_D.md) <br> 
["Chapter E - "](../part_4/chpt_E.md) <br>


## Running the Notebooks
You can either run the notebook using [Binder](https://mybinder.org/) or on youC local machine. -->The material that reports on research is mostly provided in the for of Jupyter Notebooks. For instance the Jupyter Books on the PNEOM Curriculum and the Salish Sea Basin project will be executable books. The idea behind this is that you will be able to run all content on your local computer or on a cloud-infrastructure. The material will have `Run on Collab` and `Run on Binder` options or you can clone the books' repositories. 


### <span style="color:cornflowerblue;font-size:smaller;">Running on the Binder Hub</span>

The simplest way to interact with a Jupyter Notebook is through [Binder](https://mybinder.org/), which enables the execution of a
[Jupyter Book](https://jupyterbook.org) in the cloud. The details of how this works are not important for now. All you need to know is how to launch the Jupyter book chapter via Binder. Simply navigate your mouse to the top right corner of the book chapter you are viewing and click
on the rocket ship icon, (see figure below), and be sure to select “launch Binder”. After a moment you should be presented with a
notebook that you can interact with. I.e. you’ll be able to execute and even change the example programs. You’ll see that the code cells
have no output at first, until you execute them by pressing {kbd}`Shift`\+{kbd}`Enter`. Complete details on how to interact with
a live Jupyter notebook are described in [Getting Started with Jupyter](https://foundations.projectpythia.org/foundations/getting-started-jupyter.html).



### <span style="color:cornflowerblue;font-size:smaller;">Running on Your Own Machine</span>

If you are interested in running this material locally on your computer, you will need to follow this workflow:

(Replace "\<jupyterbook-example\>" with the title of the cookbook; eg. pneom-curriculum)   

1. Clone the `https://github.com/rayniervanegmond/<jupyterbook-example>` repository:

   ```bash
    git clone https://github.com/rayniervanegmond/<jupyterbook-example>.git
    ```  
2. Move into the `<jupyterbook-example>` directory
    ```bash
    cd <jupyterbook-example>
    ```  
3. Create and activate your conda environment from the `environment.yml` file
    ```bash
    conda env create -f environment.yml
    conda activate <jupyterbook-example>
    ```  
4.  Move into the `notebooks` directory and start up Jupyterlab
    ```bash
    cd notebooks/
    jupyter lab
    ```

---

## Chapter Notebooks

notebook_template  [<i class="fa-solid fa-arrow-circle-right" style="margin-left:10px;color:teal;"></i>](notebooks/notebook-template)
: this is a link to a pneom curriculum notebooks template
