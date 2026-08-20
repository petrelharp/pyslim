---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.12
    jupytext_version: 1.9.1
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(sec_introduction)=

# Introduction

This is the documentation for pyslim, a Python API
for reading and modifying {ref}`tskit<tskit:sec_introduction>`
[tree sequence](https://tskit.dev/learn/) files
produced by [SLiM](https://messerlab.org/slim/), 
or modifying files produced by other programs (e.g.,
{ref}`msprime<msprime:sec_intro>`,
[fwdpy11](https://fwdpy11.readthedocs.io/en/stable/pages/tsoverview.html),
and [tsinfer](https://tsinfer.readthedocs.io/)) for use in SLiM. 

SLiM can read and write *tree sequences*, which store genetic genealogies
for entire populations. These can be used to efficiently encode both the state of the
population at various times during a simulation *as well as* the complete genomic
ancestry. Furthermore, SLiM can "load" a saved tree sequence
file to recreate the exact state of the population at the time it was saved.
To do this, SLiM stores some additional information (as {ref}`metadata<tskit:sec_metadata>`)
in the basic tree sequence file.

First, you probably want to read the {ref}`sec_overview`
for a description of what (and *who*)
SLiM records in the tree sequence.
Next, the {ref}`sec_tutorial` walks through the common workflows.

(sec_citation)=

# Citation

If you use pyslim, please cite:

> *Bridging forward-in-time and coalescent simulations using pyslim*, by
> Shyamalika Gopalan, Murillo F. Rodrigues, Peter L. Ralph, and Benjamin C. Haller
> [bioRxiv:2025.09.30.679676](https://doi.org/10.1101/2025.09.30.679676). 

