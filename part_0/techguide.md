---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# How To Use This Template

This book template is written in the Jupyter Book format which is developed by the jupyter.org community. 

To make optimal use of the material in the book template the writer should organize the material in the book in markdown files and notebook files. The **table of content** model will organize these file into a book structure. The table of content can have a maximum of three levels:

```yaml
format: jb-book
root: README
parts:
  # outline of the preamble and reader guide of the book
  - caption: Preamble
    chapters:
      - file: part_0/preamble
      - file: part_0/introduction

  # Introduction to part one of the book
  - caption: Content Part 1
    chapters:
      - file: part_1/introduction
        sections: # collapsable sub-section
          - file: part_1/chpt_1
            sections: # collapsable sub-section
              - file: part_1/chpt_section_t1
              - file: part_1/chpt_section_t2
          - file: part_1/chpt_2
```

the README level
: This level provides the frontpage text that is outside of the table of content. It is the "homepage" of the book and the title is the markdown '#'-level of the README.md file.

the parts level
: This level provides the TOC entry for the top-level content of the book. It is optional and the Jupyter Books can start with the "caption" level entry. Being a yaml-container element there will be a number of 'captions' for each of the parts in the book. Observe that the parts element is a container only that doesn't provide any title or content data.

the caption level
: This level provides the title of one of the parts in the book. The caption doesn't provide any content; it is not a markdown file.

the file level
: This level provides the title and content of the next level in the table of content. Just like the README.md frontpage this is a markdown file. Any markdown file starts with a top-level markdown header (we use single '#' as the header that provides the title) All the level two ('##') headers provide subdivision headers in the page and they are listed in the top-right navigation menu for the page. <br>**<span style="color:red">THIS APPLIES TO ANY MARKDOWN DOCUMENT IRRESPECTIVE OF ITS TOC LEVEL</span>**

the section level
: This level works similar to the caption container element except that the section it doesn't provide a title. The section documents are listed as sub-elements in the table of content.

the file level
: The section/file level combination can be continued many levels deep.

## Admonitions in Text

It is possible to use nice-looking callout admonition renderings for notes, warnings and information texts in the document. This is done using the following construct:

```html
<div class="admonition note" name="html-admonition" style="background: lightgreen; padding: 10px">
    <p class="title">This is the **title**</p>
    This is the *content*
</div>
```
For the different kinds of admonitions we change the class indicator. We have the option of [note, warning, tip]. Inside the admonition we works as if it was a standard div-element. The type is reflected in the icon and collor of the box. Admonitions can be nested.

<div class="admonition warning" name="html-admonition" style="background: lightgreen; padding: 10px">
<p class="title">This is the warning **title**</p>
This is the *content*
</div>


## Appendices to the Book
The appendix section of this book has chapters that guide the user through any setup processes of required software that might be needed. 

The chapter "Setup Jupyter Environment" [<i class="fa-solid fa-circle-arrow-right" style="margin-left:10px;color:teal;"></i>](../appendix/setup_jupyter.md)
: addresses the installation of the Jupyter environment . On completion the user will have an operational Jupyter Lab and Jupyter Book that runs a Python kernel for the executable cells in the notebooks.


## MyST Extension Samples

### The sphinx-proof extension


**The rendering of a proof using the proofs extension.**
````{prf:proof}
We'll omit the full proof.

But we will prove sufficiency of the asserted conditions.

To this end, let $y \in \mathbb R^n$ and let $S$ be a linear subspace of $\mathbb R^n$.

Let $\hat y$ be a vector in $\mathbb R^n$ such that $\hat y \in S$ and $y - \hat y \perp S$.

Let $z$ be any other point in $S$ and use the fact that $S$ is a linear subspace to deduce

```{math}
\| y - z \|^2
= \| (y - \hat y) + (\hat y - z) \|^2
= \| y - \hat y \|^2  + \| \hat y - z  \|^2
```

Hence $\| y - z \| \geq \| y - \hat y \|$, which completes the proof.
````

**The rendering of a theorem using the proofs extension.**
````{prf:theorem} Orthogonal-Projection-Theorem
:label: my-theorem

Given $y \in \mathbb R^n$ and linear subspace $S \subset \mathbb R^n$,
there exists a unique solution to the minimization problem

```{math}
\hat y := \argmin_{z \in S} \|y - z\|
```

The minimizer $\hat y$ is the unique vector in $\mathbb R^n$ that satisfies

* $\hat y \in S$

* $y - \hat y \perp S$


The vector $\hat y$ is called the **orthogonal projection** of $y$ onto $S$.
````

**The rendering of an algorithm using the proofs extension.**

````{prf:algorithm} Ford–Fulkerson
:label: my-algorithm

**Inputs** Given a Network $G=(V,E)$ with flow capacity $c$, a source node $s$, and a sink node $t$

**Output** Compute a flow $f$ from $s$ to $t$ of maximum value

1. $f(u, v) \leftarrow 0$ for all edges $(u,v)$
2. While there is a path $p$ from $s$ to $t$ in $G_{f}$ such that $c_{f}(u,v)>0$
	for all edges $(u,v) \in p$:

	1. Find $c_{f}(p)= \min \{c_{f}(u,v):(u,v)\in p\}$
	2. For each edge $(u,v) \in p$

		1. $f(u,v) \leftarrow f(u,v) + c_{f}(p)$ *(Send flow along the path)*
		2. $f(u,v) \leftarrow f(u,v) - c_{f}(p)$ *(The flow might be "returned" later)*
````

### The bibliography references extension:

Here is my nifty citation in a chapter document {cite}`perez2011`. There are more ways to control the inline citation formats:

* The citation `{cite:p}`perez2011`` results in {cite:p}`perez2011` 
* The citation `{cite:t}`perez2011`` results in {cite:t}`perez2011` 
* The citation `{cite:ps}`perez2011`` results in{cite:ps}`perez2011`
* The citation `{cite:ts}`perez2011`` results in{cite:ts}`perez2011`


```{margin} This is some margin content
With some explanatory text about the main document flow content. Placing it above the text is should go with is the way to layout the margin content and the main text.
```
This is the main text flow of te document to which we want to add some explanatory text as margin content. It is not really clear how these texts find each other. 

Here's some gnarly math using the mst-nb extension and `amsmath`. This doesn't look too good though -- having issues with page width and content scrolling. This wont go away whatever I try...

\begin{equation}
\frac {\partial u}{\partial x} + \frac{\partial v}{\partial y} 
\end{equation}
Maybe with some text in the middle?

\begin{align*}
2x - 5y &=  8 \\
3x + 9y &=  -12
\end{align*}

To separate the blocks put some text in between:

\begin{equation}
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
\end{equation}

Using inline MathJax for the same function doesn't present the scroll bar issue. It uses exactly the same format as the raw-latex. Without the reference label we don't have the scrollbar issue.

$$
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
$$

Showing it with an additional reference label will cause the scroll-bar issue.

$$
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
$$(label_1)

```
$$
  \int_0^\infty \frac{x^3}{e^x-1}\,dx = \frac{\pi^4}{15}
$$(label_1)
```
Separate the material from the badge. We can use the label 'label_1' as a reference in the text like so: Check out equation {eq}`label_1`. Unfortunately this generates the scrollbar problem.

[![Jupyter Book Badge](https://jupyterbook.org/badge.svg)](<YOUR URL HERE>)


## Executable code in blocks


cell 1: a standard executable cell. It requires the inclusion of the following preamble text at the top of the md-file. Simply include it as the first text of the md-file.

```
---
jupytext:
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---
```

```{code-cell} ipython3
note = "Python syntax highlighting"
print(note)
```

cell 2: make the output vertical scrollable

```{code-cell} ipython3
:tags: ["output_scroll"]
for ii in range(3):
  print("This is a test.")
```

cell 3: only show the output and hide the 'input/executable' cell.

```{code-cell} ipython3
:tags: ["hide-input"]
print("This is a test.")
```

```{code-cell} ipython3
:tags: ["hide-cell"]
print("This is a test.")
```

```{code-cell} ipython3
:tags: ["remove-cell"]
print("This is a test.")
```

Note how the next cell --which generates the function with full python imports and all--  is completely hidden in the output but the result (the image) is glued and can be referenced in the text.
```{code-cell} ipython3
:tags: ["remove-cell"]
from myst_nb import glue
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y, 'b-', linewidth=2)

glue("glued_fig", fig, display=False)
```

If we want to enable the viewing of the code then we would set the tag to "hide-input".

```{code-cell} ipython3
:tags: ["hide-input"]
from myst_nb import glue
import matplotlib.pyplot as plt
import numpy as np

x = np.linspace(0, 10, 200)
y = np.sin(x)
fig, ax = plt.subplots()
ax.plot(x, y, 'b-', linewidth=2)

glue("glued_fig", fig, display=False)
```

This is an inline glue example of a figure: {glue:}`glued_fig`.

This is an example of pasting a glued output as a block:
```{glue:} glued_fig
```

### Gluing across notebooks and documents

In the documentation this is called "Store code outputs and insert into content". The glue tool from MyST-NB allows you to add a key to variables in a notebook, then display those variables in your book by referencing the key. Above we have asserted that the glue-tool works.It follows a two-step process:

* Glue a variable to a name. Do this by using the myst_nb.glue function on a variable that you’d like to re-use elsewhere in the book. You’ll give the variable a name that can be referenced later.
* Reference that variable from your page’s content. Then, when you are writing your content, insert the variable into your text by using a {glue:} role.

Pasting glued variables into your page: Once you have glued variables to their names, you can then paste those variables into your text in your book anywhere you like (even on other pages). These variables can be pasted using one of the roles or directives in the glue family. In the page 8.1 we will reference the same figure.


---

## Chapter Notebooks

notebook_template  [<i class="fa-solid fa-arrow-circle-right" style="margin-left:10px;color:teal;"></i>](notebooks/notebook-template)
: this is a link to a pneom curriculum notebooks template
