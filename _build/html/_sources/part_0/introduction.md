# A Reader Guide to ...title of the book...

<span style="font-size:20pt">An Introduction and Outline</span>

This book aims to provide a comprehensive curriculum to learn to apply ...

## Structure
The structure of this Jupyter Book is geared to ... <span style="font-family:serif;font-style:italic;color:darkred;">explain the main principle for struturing the book as it is. Eg. theory, practical examples, case-studies </span>

### <span style="color:cornflowerblue">Section 1 "The Discussion of the Theory"</span>

The learning objectives of the first part of the book are that the user will understand... These objectives are addressed by the chapters on ["Chapter 1 - The Theory Behind ..."](ch_1/introduction.md), ["Chapter 2 - The Practical Applications to..."](ch_2/introduction.md)


## Running the Notebooks
<!-- You can either run the notebook using [Binder](https://mybinder.org/) or on your local machine. -->

The material that reports on research is mostly provided in the for of Jupyter Notebooks. For instance the Jupyter Books on the PNEOM Curriculum and the Salish Sea Basin project will be executable books. The idea behind this is that you will be able to run all content on your local computer or on a cloud-infrastructure. The material will have `Run on Collab` and `Run on Binder` options or you can clone the books' repositories. 

### Running on Binder

The simplest way to interact with a Jupyter Notebook is through [Binder](https://mybinder.org/), which enables the execution of a
[Jupyter Book](https://jupyterbook.org) in the cloud. The details of how this works are not important for now. All you need to know is how to launch the Jupyter book chapter via Binder. Simply navigate your mouse to the top right corner of the book chapter you are viewing and click
on the rocket ship icon, (see figure below), and be sure to select “launch Binder”. After a moment you should be presented with a
notebook that you can interact with. I.e. you’ll be able to execute and even change the example programs. You’ll see that the code cells
have no output at first, until you execute them by pressing {kbd}`Shift`\+{kbd}`Enter`. Complete details on how to interact with
a live Jupyter notebook are described in [Getting Started with Jupyter](https://foundations.projectpythia.org/foundations/getting-started-jupyter.html).

### Running on Your Own Machine
If you are interested in running this material locally on your computer, you will need to follow this workflow:

(Replace "\<jupyterbook-example\>" with the title of the cookbook; eg. pneom-curriculum)   

1. Clone the `https://github.com/rayniervanegmond/<jupyterbook-example>` repository:

   ```bash
    git clone https://github.com/rayniervanegmond/<jupyterbook-example>.git
    ```  
1. Move into the `<jupyterbook-example>` directory
    ```bash
    cd <jupyterbook-example>
    ```  
1. Create and activate your conda environment from the `environment.yml` file
    ```bash
    conda env create -f environment.yml
    conda activate <jupyterbook-example>
    ```  
1.  Move into the `notebooks` directory and start up Jupyterlab
    ```bash
    cd notebooks/
    jupyter lab
    ```

---

## Chapter Notebooks

notebook_template  [<i class="fa-solid fa-arrow-circle-right" style="margin-left:10px;color:teal;"></i>](notebooks/notebook-template)
: this is a link to a pneom curriculum notebooks template
