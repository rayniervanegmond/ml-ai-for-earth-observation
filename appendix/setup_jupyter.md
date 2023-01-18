#  The Setup of the Jupyter Environment

<span style="font-size:20pt">An Introduction and Outline</span>
The benefit of working in Jupyter Book format is that the book material can be made dynamic. This means that part of its content can be interacted with and modified by executing code. In the Jupyter Books that are produced by the PNEOM program we only work with Python executable code inside the Jupyter Notebooks and the larger scale Jupyter Book formats.

For this to work we need to install the Python Interpreter and any libraries that the code inside the Jupyter Books might need. The installation of the required Python version and supporting libraries is handled by an `environment.yml` file that always accompanies a Jupyter Book publication.


## Running on Binder

The simplest way to interact with a Jupyter Notebook is through [Binder](https://mybinder.org/), which enables the execution of a
[Jupyter Book](https://jupyterbook.org) in the cloud. The details of how this works are not important for now. All you need to know is how to launch the Jupyter book chapter via Binder. Simply navigate your mouse to the top right corner of the book chapter you are viewing and click
on the rocket ship icon, (see figure below), and be sure to select “launch Binder”. After a moment you should be presented with a
notebook that you can interact with. I.e. you’ll be able to execute and even change the example programs. You’ll see that the code cells
have no output at first, until you execute them by pressing {kbd}`Shift`\+{kbd}`Enter`. Complete details on how to interact with
a live Jupyter notebook are described in [Getting Started with Jupyter](https://foundations.projectpythia.org/foundations/getting-started-jupyter.html).

## Running on Your Own Machine
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
    conda activate cookbook-example<jupyterbook-example>
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
