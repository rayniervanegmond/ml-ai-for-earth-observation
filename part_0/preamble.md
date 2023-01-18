# How To Use This Template

This book template is written in the Jupyter Book format which is developed by the jupyter.org community. 

To make optimal use of the material in the book template the writer should organize the material in the book in markdown files and notebook files. The **table of content** model will arganize these file into a book structure. The table of content can have a maxmimum of three levels:

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

```heml
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

---

## Chapter Notebooks

notebook_template  [<i class="fa-solid fa-arrow-circle-right" style="margin-left:10px;color:teal;"></i>](notebooks/notebook-template)
: this is a link to a pneom curriculum notebooks template
