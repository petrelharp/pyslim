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

```{code-cell}
:tags: [remove-cell]
import pyslim, tskit, msprime
from IPython.display import SVG
import numpy as np
import random
random.seed(23)

ts = tskit.load("example_sim.trees")
tables = ts.dump_tables()
```

```{eval-rst}
.. currentmodule:: pyslim
```


(sec_metadata)=

# Metadata

(sec_metdata_overview)=

## Overview

SLiM puts SLiM-specific information into the *metadata* for the tree sequence,
as well as for each population, individual, node and mutation.
Here is a quick reference to what information is available:
see the SLiM manual for the more technical writeup.
A good way to get a generic metadata example is with {func}`.default_slim_metadata`.

**Top-level:**
If `ts` is your tree sequence, then `ts.metadata` is a dict,
and `ts.metadata["SLiM"]` contains information about the simulation:

- `file_version`: the version of the SLiM tree sequence file format
- `tick`: the value of `community.tick` within SLiM when the file was written out
- `cycle`: the value of `sim.cycle` within SLiM when the file was written out
- `model_type`: either `"WF"` or `"nonWF"`
- `nucleotide_based`: whether this is a nucleotide-based simulation
- `separate_sexes`: whether the simulation has separate sexes or not
- `spatial_dimensionality`: for instance, `""` or `"x"` or `"xy"` (etcetera)
- `spatial_periodicity`: whether space wraps around in some directions (same format as dimensionality)
- `stage`: the *stage* of the life cycle at which the file was written out (either `"first"`, `"early"`, or `"late"`)
- `name`: the *name* of this species in SLiM
- `this_chromosome`: contains, for the chromosome in SLiM recorded in this tree sequence
    * `id`: SLiM's ID 
    * `index`: the index of this chromosome in the list of chromosomes
    * `symbol`: the user-assigned symbol
    * `type`: specifies inheritance type, e.g., `"A"` for autosome
- `chromosomes`: (optional) a list of all chromosomes in the simulation
- `traits`: a list of information for each of the traits:
    * `index`: the index of the trait in SLiM
    * `name`: the name in SLiM for the trait
    * `type`: additive, multiplicative, or logistic
    * `baselineOffset`, `baselineAccumulation`: a value added to all traits, and whether the effect of substitutions
      accumulate in that value
    * `directFitnessEffect`: whether the trait has a direct effect on fitness
    * `individualOffsetMean`, `individualOffsetSD`: parameters governing the individual-level offsets
      (i.e., "environment" effects)

**Populations:**
Information about each SLiM-produced population is written to metatadata.
The format uses JSON and is extensible, so other keys may be present
and some keys may be missing (for instance, there are no spatial bounds
in a nonspatial simulation). The metadata may be `None` for populations
that SLiM did not use. The keys that SLiM uses are:

- `slim_id`: the ID of this population in SLiM
- `name`: the name of the population (by default, `p0`, `p1`, etcetera)
- `description`: a string describing the population
- `selfing_fraction`, `female_cloning_fraction`, `male_cloning_fraction`, and `sex_ratio`: only present when applicable (e.g., in WF simulations)
- `bounds_x0`, `bounds_x1`, `bounds_y0`, `bounds_y1`, `bounds_z0`, and `bounds_z1`: the spatial bounds, when applicable
- `migration_records`: A *list* of entries decribing migration between populations in a WF model.

**Individuals:**
Each individual produced by SLiM contains the following metadata:

- `pedigree_id`: the "pedigree ID", unique within the SLiM simulation
- `pedigree_p1`, `pedigree_p2`: the pedigree IDs of the individuals' two
  parents (they may be equal in the case of selfing, or `-1` to indicate no
  parent, in the case of the initial generation or for cloning)
- `age`: the `.age` property within SLiM at the time the file was written out
- `subpopulation`: the subpopulation within SLiM the individual was in at the time the file was written out
- `sex`: the sex of the individual (either {data}`.INDIVIDUAL_TYPE_FEMALE`, {data}`.INDIVIDUAL_TYPE_MALE`, or {data}`.INDIVIDUAL_TYPE_HERMAPHRODITE`)
- `flags`: additional information; currently only recording whether the individual was a "migrant" or not (see the SLiM manual)
- `tag`, `tagF`: the corresponding properties in SLiM: default values returned by pyslim
  are the special values that SLiM uses to mean that the values are unset
- `tagL0`, `tagL0_set`, etcetera: again, the corresponding properties in SLiM;
  the purpose of `tagLX_set` is to record whether the tag has been set in the simulation
- `per_trait`: a list of information about the trait values for this indivdual; these are in the same order
  as the traits listed in top-level metadata;
    * `phenotype`: the trait value
    * `offset`: the individual's offset (i.e., the "environmental effect")

**Nodes:**
Each "node" produced by SLiM (i.e., "genome" within SLiM) has:

- `slim_id`: the unique ID associated with the genome by SLiM
- `is_vacant`: records the genome is a "vacant" genome (in which case it isn't
  really there, so shouldn't have any mutations or relationships in the tree
  sequence!) - see [](sec_overview_vacant_nodes) for more explanation

**Mutations:**
Prior to SLiM 6.0, mutation metadata was associated with the tskit mutation objects.
Now, this is stored in top-level metadata, under ``ts.metadata["SLiM_mutation_list"]``.
Each entry

- `mutation_id`: the numeric ID of mutation in SLiM
- `mutation_type`: the numeric ID of the `MutationType` within SLiM
- `subpopulation`: the numeric ID of the subpopulation the mutation occurred in
- `slim_time`: the value of `community.tick` when the mutation occurred
- `nucleotide`: either `-1` if there is no associated nucleotide, or the numeric code for the nucleotide (see {data}`.NUCLEOTIDES`)
- `per_trait`: a list of information in the same order as the traits in top-level metadata, recording for each:
    * `effect_size`: the effect on the trait of this mutation
    * `dominance`: its dominance coefficient
    * `hemizygous_dominance`: its hemizygous dominance coefficient (see the SLiM manual)
- `padding`: this is simply empty bytes, here for byte-alignment reasons, and is always `None`


(sec_metadata_using_top_level)=

## Using top-level metadata

If you are going to be using information from top-level metadata,
it is good practice to extract the metadata as a separate python object once
and refer to that object, since otherwise you can incur runtime penalties
for decoding and copying the metadata every time you call `ts.metadata`.
This can be substantial, given the amount of mutation information
in top-level metadata.
For instance, to subtract off baseline offsets from individual's trait values,
we might do:
```{code-cell}
md = ts.metadata
traits = md["SLiM"]["traits"]
values = [
    [x['phenotype'] - y["baselineOffset"] for x, y in zip(ind.metadata['per_trait'], traits)]
    for ind in ts.individuals()
]
```
If we instead inserted ``ts.metadata["SLiM"]["traits"]`` directly into the loop,
this would become infeasibly slow.

In some more detail:
each time python evaluates ``ts.metadata`` (e.g., using ``ts.metadata["SLiM"]``)
a new copy of the metadata dict is decoded and returned. Furthermore, a number
of pyslim functions need to look up information from metadata under the hood.
For instance, previously it was acceptable to run
``[pyslim.slim_time(ts, mut.time) for mut in ts.mutations()]``.
However, this could now easily take hours even for moderately-sized simulations.
There are several recommendations for how to mitigate this:

- If you use information from top-level metadata, make a copy of it
  and refer to that copy instead: so, ``ts_metadata = ts.metadata``
  after ``ts = tskit.load(...)`` and then use `ts_metadata`. However,
  be careful that you use the correct metadata object!

- Use a single pyslim function call rather than many. For instance, run:
  ``slim_times = pyslim.slim_time(ts, ts.mutations_time)`` and extract
  slim times from this vector. Similarly, use {func}`.nodes_vacant`
  instead of {func}`.node_is_vacant`.

- Some pyslim methods will accept a pre-extracted metadata dictionary
  as an optional argument. If this is not provided, those methods will
  extract the metadata again. The methods that now take a `ts_metadata` argument are:
  {func}`.individual_ages`,
  {func}`.individual_ages_at`,
  {func}`.individuals_alive_at`, and
  {func}`.slim_time`.


(sec_metadata_tools)=

## Metadata tools

The dictionaries describing the schema for these metadata entries
are available in `pyslim.slim_metadata_schemas`.
Furthermore, these method may be useful in working with metadata:

```{eval-rst}
.. autofunction:: default_slim_metadata

.. autofunction:: slim_tree_sequence_metadata_schema

.. autofunction:: slim_individual_metadata_schema

.. autofunction:: slim_node_metadata_schema

.. autofunction:: set_tree_sequence_metadata

.. autofunction:: set_metadata_schemas
```


## Modifying SLiM metadata
For more on working with metadata,
see {ref}`tskit's metadata documentation <tskit:sec_metadata>`.


### Top-level metadata

The entries of the top-level metadata dict are *read-only*.
So, although you might think that
`tables.metadata["SLiM"]["model_type"] = "nonWF"`
would switch the model type,
this in fact (silently) does nothing. To modify the top-level metadata,
we must (a) work with tables (as tree sequences are immutable), and (b)
extract the metadata dict, modify the dict, and copy it back in.
Instead, you should do
```{code-cell}
md = tables.metadata
md["SLiM"]["model_type"] = "nonWF"
tables.metadata = md
```
Modifying the top-level metadata
could be used to set spatial bounds on an annotated msprime simulation, for instance.
(This is recorded in the population metadata.)


### Modifying SLiM metadata in tables


To modify the metadata that ``pyslim`` has introduced into
the tree sequence produced by a coalescent simulation,
or the metadata in a SLiM-produced tree sequence,
we need to edit the TableCollection that forms the editable data behind the tree sequence.
For instance, to set the ages of the individuals in the tree sequence to random numbers between 1 and 4,
we will extract a copy of the underlying tables, clear it,
and then iterate over the individuals in the tree sequence,
as we go re-inserting them into the tables
after replacing their metadata with a modified version:

```{code-cell}
tables = ts.dump_tables()
tables.individuals.clear()
for ind in ts.individuals():
    md = ind.metadata
    md["age"] = random.choice([1,2,3,4])
    _ = tables.individuals.append(
        ind.replace(metadata=md)
    )

mod_ts = tables.tree_sequence()

# check that it worked:
print("First ten ages:", [mod_ts.individual(i).metadata["age"] for i in range(10)])
for ind in mod_ts.individuals():
    assert ind.metadata['age'] in [1, 2, 3, 4]

# save out the tree sequence
mod_ts.dump("modified_ts.trees")
```

## Technical details

### Metadata entries

SLiM records additional information in the metadata columns of Individual, Node, and Mutation tables,
in a binary format using the python ``struct`` module.
See {ref}`tskit's metadata documentation <tskit:sec_metadata>`
for details on how this works.
Nothing besides this binary information can be stored in the metadata of these tables if the tree sequence is to be used by SLiM,
and so when ``pyslim`` annotates an existing tree sequence, anything in those columns is overwritten.
Population metadata is stored as JSON, however, which is more flexible.
For more detailed documentation on the contents and format of the metadata, see the SLiM manual.

Of particular note is that *nodes* and *populations* may have empty metadata.
SLiM will not use the metadata of nodes that are not associated with alive individuals,
so this can safely be omitted (and makes recapitation easier).
And, populations not used by SLiM will have empty metadata.
All remaining metadata are required (besides edges and sites, whose metadata is not used at all).
