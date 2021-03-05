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

(sec_python_api)=

# Python API

This page provides detailed documentation for the ``pyslim`` Python API.

## Additions to the tree sequence

The {class}`tskit.TreeSequence` class represents a sequence of correlated trees
that describes the genealogical history of a population.
Here, we describe pyslim's additions to this class.


### The SLiM Tree Sequence


:::{eval-rst}
.. autoclass:: pyslim.SlimTreeSequence()
    :members:
:::


## Constants

:::{eval-rst}
.. data:: slim_tree_sequence.py:NUCLEOTIDES == ['A', 'C', 'G', 'T']

   Nucleotide states in nucleotide models are encoded as integers (0, 1, 2, 3);
   this gives the mapping.
:::

These constants are used to encode the "genome type" of a Node.

:::{eval-rst}
.. data:: GENOME_TYPE_AUTOSOME == 0
:::

:::{eval-rst}
.. data:: GENOME_TYPE_X == 1
:::

:::{eval-rst}
.. data:: GENOME_TYPE_Y == 2
:::


These constants are used to encode other information about Individuals.

:::{eval-rst}
.. data:: INDIVIDUAL_TYPE_HERMAPHRODITE == -1
:::

:::{eval-rst}
.. data:: INDIVIDUAL_TYPE_FEMALE == 0
:::

:::{eval-rst}
.. data:: INDIVIDUAL_TYPE_MALE == 1
:::

:::{eval-rst}
.. data:: INDIVIDUAL_FLAG_MIGRATED == 0x01
:::

And, these are used in the `Individual.flags`:

:::{eval-rst}
.. data:: INDIVIDUAL_ALIVE == 2**16

   This flag is used by SLiM to record information in the :class:`tskit.Individual` metadata.
:::

:::{eval-rst}
.. data:: INDIVIDUAL_REMEMBERED == 2**17

   This flag is used by SLiM to record information in the :class:`tskit.Individual` metadata.
:::

:::{eval-rst}
.. data:: INDIVIDUAL_RETAINED == 2**18

   This flag is used by SLiM to record information in the :class:`tskit.Individual` metadata.
:::


## Metadata

SLiM-specific metadata is made visible to the user by ``.metadata`` properties.
For instance:

```bash
ts.node(4).metadata
# {"slim_id" : 3, "is_null" : 0, "genome_type" : 0}
```

shows that the fourth node in the tree sequence was given pedigree ID ``3`` by SLiM,
is *not* a null genome, and has ``genome_type`` zero, which corresponds to an autosome 
(see below).


### Annotation

These two functions will add default SLiM metadata to a tree sequence (or the
underlying tables), which can then be modified and loaded into SLiM.

:::{eval-rst}
.. autofunction:: pyslim.annotate_defaults
:::

:::{eval-rst}
.. autofunction:: pyslim.annotate_defaults_tables
:::


### Provenances

:::{eval-rst}
.. autoclass:: pyslim.ProvenanceMetadata
   :members:
:::

:::{eval-rst}
.. autofunction:: pyslim.get_provenance
:::

