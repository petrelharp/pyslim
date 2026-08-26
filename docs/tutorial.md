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

import warnings
import pyslim, tskit, msprime
from IPython.display import SVG
import numpy as np
import util

np.random.seed(1234)
warnings.simplefilter('ignore', msprime.TimeUnitsMismatchWarning)
```

```{eval-rst}
.. currentmodule:: pyslim
```


# Tutorial

This tutorial covers the most common uses of tree sequences in SLiM/pyslim.

## Recapitation, simplification, and mutation

Perhaps the most common pyslim operations involve [](sec_tutorial_recapitation),
[](sec_tutorial_simplification), and/or [](sec_tutorial_adding_neutral_mutations).
Below we illustrate all three in the context of running a "hybrid" simulation, combining
both forwards and backwards (coalescent) methods. Such hybrid approaches provide a popular
application of pyslim because coalescent algorithms, although more limited in the degree
of biological realism they can attain, can be much faster than the forwards algorithms
implemented in SLiM.
(See more discussion in [Gopalan et al 2025](https://doi.org/10.1101/2025.09.30.679676).)

A typical use-case is to take an existing SLiM simulation and endow
it with a history derived from a coalescent simulation: this is known as *recapitation*.
For instance, suppose we have a SLiM simulation of a population of 100,000 individuals
that we have run for 10,000 generations without neutral mutations. Now, we wish to
extract whole-genome genotype data for only 1,000 individuals. Here's one way to do it:

1. {func}`.recapitate` :
   The simulation has likely not reached equilibrium - it has not
   *coalesced* entirely; recapitation uses coalescent simulation to provide
   a "prior history" for the initial generation of the simulation.

2. {meth}`simplify() <tskit.TreeSequence.simplify>` : For efficiency, subset the tree
   sequence to only the information relevant for those 1,000 individuals
   we wish to sample.

3. {func}`msprime.sim_mutations` : Add neutral mutations to the tree sequence.

These steps are described below. First, to get something to work with,
you can run this simple SLiM script of a single population of sexual organisms,
fluctuating around 1000 individuals, for 1000 generations:

```{literalinclude} example_sim.slim
```

You can run this in the shell,
setting the random seed (with `-s 23`) so you get exactly the same results
as in the code below:
```{code-cell}
:tags: ["hide-output"]
%%bash
slim -s 23 example_sim.slim
```


(sec_tutorial_recapitation)=

### Recapitation

If some individuals in the final generation of simulation, at some point on the genome,
do not have a MRCA within the tree sequence,
we say that the simulation has not coalesced.
This means, effectively, that the final state of the simulation depends on the genotypes
of the initial generation. If the simulation was begun in an empty state,
we are effectively assuming the initial state had no genetic diversity.
Usually, we'd like instead to assume that the simulation began with some
at least roughly reasonable levels of genetic diversity.

```{figure} _static/pedigree_recapitate.png
---
scale: 42%
align: right
name: pedigree_recapitate
---
Recapitation adds the green nodes by coalescent simulation.
(See [the introduction](sec_left_in_tree_sequence)
for a diagram of the previous state.)
```

Although we can initialize a SLiM simulation with the results of a coalescent simulation,
if SLiM don't actually use the genotypes for anything, it
can be much more efficient to run the coalescent simulation *afterwards*,
hence only doing a coalescent simulation
for the portions of the first-generation ancestors that have
not yet coalesced.
(See the SLiM manual for more explanation, or
[Kelleher et al 2018](https://doi.org/10.1371/journal.pcbi.1006581).)
This is depicted in {numref}`figure {number} <pedigree_recapitate>`:
imagine that at some sites, some of the samples
don't share a common ancestor within the SLiMulated portion of history (shown in blue).
Recapitation starts at the *top* of the genealogies,
and runs a coalescent simulation back through time
to fill out the rest of genealogical history relevant to the samples.
The green chromosomes are new ancestral nodes that have been added to the tree sequence.
This is important - if we did not do this,
then effectively we are assuming the initial population would be genetically homogeneous,
and so our simulation would have less genetic variation than it should have
(since the component of variation from the initial population would be omitted).

Doing this is as simple as:

```{code-cell}
orig_ts = tskit.load("example_sim.trees")
rts = pyslim.recapitate(orig_ts,
            recombination_rate=1e-8,
            ancestral_Ne=200, random_seed=5)
```
(Here and below we set the random seed
so the content of this document does not change;
you should *not* usually explicitly set
the random seed in your scripts.)

This will produce a warning:

> TimeUnitsMismatchWarning: The initial_state has time_units=ticks but time is measured
> in generations in msprime. This may lead to significant discrepancies between the
> timescales.

This is simplfy reminding us to think about generation time
when recapitating a nonWF simulation (see [](sec_time_units)).

We can check that this worked as expected, by verifying that after recapitation
all trees have only one root:

```{code-cell}
orig_max_roots = max(t.num_roots for t in orig_ts.trees())
recap_max_roots = max(t.num_roots for t in rts.trees())
print(f"Maximum number of roots before recapitation: {orig_max_roots}\n"
      f"After recapitation: {recap_max_roots}")
```

The {func}`.recapitate` method
is just a thin wrapper around {func}`msprime.sim_ancestry`,
and you need to set up demography explicitly - for instance, in the example above
we've effectively simulated from an ancestral population of ``Ne=200`` diploids.
If you have more than one population,
you must set migration rates or else coalescence will never happen
(see [](sec_recapitate_with_migration) for an example,
and {func}`.recapitate` for more).

What population size (`ancestral_Ne`) should you use for recapitation?
This question is far beyond the scope of this documentation,
but here are a few helpful points:
First, recapitation is (nearly) exactly identical to having initialized
the SLiM simulation with the randomly produced offspring of the endpoint
of a neutral Wright-Fisher simulation that was begun infinitely far in the past.
This Wright-Fisher model is crude: it does not have separate sexes,
for instance, and is in continuous time unless the 
{ref}`DTWF model <msprime:sec_ancestry_models_dtwf>` is used;
however, a large body of mathematical work on coalescent theory
has shown that this is a good
approximation to a wide class of more realistic models.
Second, this is certainly an approximation to reality,
but then again so is everything else,
and if the start of the SLiM simulation was far enough in the past,
in most cases the precise details won't substantially affect the outcomes.
Third, you can always evaluate the effect of your choices
by changing them a bit and seeing if it affects the results.


#### Recapitation with a nonuniform recombination map

Above, we recapitated using a uniform genetic map.
But, msprime - like SLiM - can simulate with recombination drawn from an arbitrary genetic map.
Let's say we've already got a recombination map as specified by SLiM,
as a vector of "positions" and a vector of "rates".
msprime also needs vectors of positions and rates, but the format is slightly different.
To use the SLiM values for msprime, we need to do three things:

1. Add a 0 at the beginning of the positions,
2. add 1 to the last position.

The reason why msprime "positions" must start with 0 (step 1) is that in SLiM,
a position or "end" indicates the end of a recombination block such that its associated
"rate" applies to everything to the left of that end (see ``initializeRecombinationRate``),
while msprime's format indicates the start of that same block.
In msprime, we will pass in a {class}`msprime.RateMap`,
which requires two things:

- ``position``: A list of n+1 positions, starting at 0, and ending in the sequence length over which the RateMap will apply.
- ``rate``: A list of n positive rates that apply between each position.

So, msprime needs a vector of positions that is 1 longer than what you give SLiM,
but one fewer rate values than positions.

The reason for step 2 is that intervals for tskit (which msprime uses)
are "closed on the left and open on the right",
which means that the genomic interval from 0.0 to 100.0 includes 0.0 but does not include 100.0.
If SLiM has a final genomic position of 99, then it could have mutations occurring at position 99.
Such mutations would *not* be legal, on the other hand, if we set the tskit sequence length to 99,
since the position 99 would be outside of the interval from 0 to 99.
Said another way, if SLiM's final position is 99, the total sequence length is 100,
and so we need to set the end of the genome to 100.
The upshot is that we need to use SLiM's last position plus one - i.e.,
the length of the genome - as the rightmost coordinate.

For instance, suppose that we have a recombination map file in the following (tab-separated) format:

```{literalinclude} _static/recomb_rates.tsv
```

This describes recombination rates across a 100Mb genome with higher rates on the ends
(for instance, 3.2 and 2.8 cM/Mb in the first and last 15Mb respectively)
and lower rates in the middle (0.25 cM/Mb between 50Mb and 85Mb).
The first column gives the starting position, in bp,
for the window whose recombination rate is given in the second column;
the final position is given on the last line with a recombination rate of 0
(it is a failing of this relatively common file format that there's no good
way to say where the chromosome ends).
To read this file into SLiM, remove the first and last lines
and use the `initializeRecombinationRateFromFile()` function in SLiM,
which by default scales rates by the `1e-8` factor required to convert from
cM/Mb to crossovers per bp.

Now, here's code to take the same recombination map used in SLiM,
and use it for recapitation in msprime:

```{code-cell}
recomb_map = msprime.RateMap.read_hapmap(
    "_static/recomb_rates.tsv",
    position_col=0, rate_col=1
)
rts = pyslim.recapitate(orig_ts,
                recombination_rate=recomb_map,
                ancestral_Ne=200, random_seed=7)
assert(max([t.num_roots for t in rts.trees()]) == 1)
```

:::{note}
Starting from msprime 1.0, the default model of recombination
in msprime is *discrete* - recombinations only occur at integer
locations - which matches SLiM's model of recombination.
:::


(sec_tutorial_simplification)=

### Simplification

```{figure} _static/pedigree_simplify.png
---
scale: 42%
align: right
name: pedigree_simplify
---
The result of simplifying the tree sequence
in figure {numref}`figure {number} <pedigree_recapitate>`
to only two of the three samples.
```

Probably, your simulations have produced many more fictitious genomes
than you will be lucky enough to have in real life,
so at some point you may want to reduce your dataset to a realistic sample size.
We can get rid of unneeded samples and any extra information from them by using
an operation called *simplification* (this is the same basic approach that SLiM
implements under the hood to keep down memory usage, as described in
[the introduction](sec_left_in_tree_sequence)).

Depicted in {numref}`figure {number} <pedigree_simplify>`
is the result of applying an explicit call to
{meth}`tskit.TreeSequence.simplify` to our example tree sequence.
In the call we asked to keep only 4
genomes (contained in 2 of the individuals in the current generation). This has
substantially simplified the tree sequence, because only information relevant to the
genealogies of the 4 sample nodes has been kept. (Precisely, simplification retains only
nodes of the tree sequence that are branching points of some marginal genealogy -- see
[Kelleher et al 2018](https://doi.org/10.1371/journal.pcbi.1006581) for details.)
While simplification sounds very appealing - it makes things simpler after all -
it is often not necessary in practice, because tree sequences are very compact,
and many operations with them are quite fast.
(It will, however, speed up many operations, so if you plan to do a large number of simulations,
your workflow could benefit from early simplification.)
So, you should probably not make simplification a standard step in your workflow,
only using it if necessary.

It is important that simplification - if it happens at all -
either (a) comes after recapitation, or (b) is done with the
``keep_input_roots=True`` option (see {meth}`tskit.TreeSequence.simplify`).
This is because simplification will almost certainly remove some of the
ancestral genomes in the first generation,
which are necessary for recapitation,
unless it is asked to "keep the input roots".
If we simplify without this option before recapitating,
some of the first-generation blue chromosomes
in {numref}`figure {number} <pedigree_simplify>`
would not be present, so the coalescent simulation would start from a more recent point in time
than it really should.
As an extreme example, suppose our SLiM simulation has a single diploid who has reproduced
by clonal reproduction for 1,000 generations,
so that the final tree sequence is just two vertical lines of descent going back
to the two chromosomes in the initial individual alive 1,000 generations ago.
Recapitation would produce a shared history for these two chromosomes
that would coalesce some time longer ago than 1,000 generations.
However, if we simplified first, then those two branches going back 1,000 generations would be removed,
since they don't convey any information about the shape of the tree;
and so recapitation might produce a common ancestor more recently than 1,000 generations,
which would be inconsistent with the SLiM simulation.

#### How to simplify

After recapitation,
simplification to the history of 100 individuals alive today
can be done with the {meth}`tskit.TreeSequence.simplify` method:

```{code-cell}
import numpy as np
rng = np.random.default_rng(seed=3)
alive_indivs = pyslim.individuals_alive_at(rts, 0)
keep_indivs = rng.choice(alive_indivs, 100, replace=False)
keep_nodes = []
for i in keep_indivs:
  keep_nodes.extend(rts.individual(i).nodes)

sts = rts.simplify(keep_nodes, keep_input_roots=True)

print(f"Before, there were {rts.num_samples} sample nodes (and {rts.num_individuals} individuals)\n"
      f"in the tree sequence, and now there are {sts.num_samples} sample nodes\n"
      f"(and {sts.num_individuals} individuals).")
```

**Note** that you must pass simplify a list of *node IDs*, not individual IDs.
Here, we used the {func}`.individuals_alive_at` method to obtain the list
of individuals alive today.
Also note that there are *still* more than 100 individuals remaining - 15 non-sample individuals
have not been simplified away,
because they have nodes that are required to describe the genealogies of the samples.
(Since this is a non-Wright-Fisher simulation,
parents and children can be both alive at the same time in the final generation.)



(sec_tutorial_adding_neutral_mutations)=

### Adding neutral mutations to a SLiM simulation

```{figure} _static/pedigree_mutate.png
---
scale: 42%
align: right
name: pedigree_mutate
---
The tree sequence, with mutations added.
```

If you have recorded a tree sequence in SLiM, likely you have not included any neutral mutations,
since it is much more efficient to simply add these on afterwards.
To add these (in a completely equivalent way to having included them during the simulation),
you can use the {func}`msprime.sim_mutations` function, which returns a new tree sequence with additional mutations.
These are added to each branch of the tree sequence
at the rate per unit time that you request,
as depicted in {numref}`figure {number} <pedigree_mutate>`.
We'll add these using the {class}`msprime.SLiMv6MutationModel` (note the `v6` in that name!),
so that the file can be read back into SLiM,
but any of the other mutation models in msprime could be used.
This works as follows, starting with the tree sequence
produced above by recapitation:

```{code-cell}
next_id = pyslim.next_slim_mutation_id(sts)
ts = pyslim.add_mutation_metadata(
        msprime.sim_mutations(
           sts,
           rate=1e-8,
           model=msprime.SLiMv6MutationModel(next_id=next_id),
           keep=True,
        ),
        mutation_type=0,
)

print(f"The tree sequence now has {ts.num_mutations} mutations,\n"
      f"and mean pairwise nucleotide diversity is {ts.diversity():0.3e}.")
```


What's going on here? Let's step through the code.

1. The mutation ``rate = 1e-8`` says mutations are added at a rate of {math}`10^{-8}` per bp.

2. We specify ``keep = True``, to keep any existing mutations.
    In this example there aren't any, so this isn't strictly necessary,
    but this is a good default.

3. If there are existing SLiM mutations on the tree sequence we need to
    make sure any newly added mutations have distinct SLiM IDs,
    so we use {func}`.next_slim_mutation_id` to figure out
    what the next available ID is, and pass it in.

4. After ``msprime`` adds the mutations, if we want to load this into SLiM,
    we need to add information about the mutations using {func}`add_mutation_metadata`.

5. We're passing ``mutation_type=0`` to {func}`add_mutation_metadata`.
    This is because SLiM mutations need a "mutation type",
    and it makes the most sense if we add a type that was unused in the simulation.
    In this example we don't have any existing mutation types,
    so we can safely use ``mutation_type=0``, producing mutations of type ``m0`` in SLiM.


(sec_output)=

### Writing out genotypes to VCF

Downstream applications often need input in VCF format,
which we can get with a call to {meth}`tskit.TreeSequence.write_vcf`.
However, if we do that with this tree sequence, we'll get a malformed VCF,
with empty strings in the REF column and a strange comma-separated list of integers
in the ALT column. The reason for this is because we added mutations
using the {class}`msprime.SLiMv6MutationModel`, and has to do with how SLiM stores enough information
in the tree sequence to be able to load it back in.
So, to write out valid VCF with nucleotides for alleles,
we need to (1) if the SLiM simulation was not a nucleotide model, add nucleotides
to the SLiM mutations with {func}`generate_nucleotides`,
and (2) move those nucleotides over into the "ancestral state"
and "derived state" slots of the tree sequence with {func}`convert_alleles`.
If all your mutations in SLiM were nucleotide mutations, you only need to do (2).
And, beware that (2) is an irreversible step: if you write the tree sequence
produced by {func}`convert_alleles` to a file, you can't load that file into SLiM any more.
So, to do this we'll do:

```{code-cell}
nts = pyslim.generate_nucleotides(ts)
nts = pyslim.convert_alleles(nts)
sample_indivs = np.unique([ts.node(n).individual for n in nts.samples()])
with open("example_sim.vcf", "w") as vcffile:
    nts.write_vcf(vcffile, individuals=sample_indivs[:5])
```

Here we've just extracted genotypes for the first five individuals
(see [](sec_extracting_individuals) for more on this),
and have not taken advantage of the options in {meth}`tskit.TreeSequence.write_vcf`
that help make the output more useful.

For instance, if you want to use the SLiM pedigree IDs for the names in the VCF file,
we could do:

```{code-cell}
pedigree_ids = [
    f"ind_{ts.individual(i).metadata['pedigree_id']}" for i in sample_indivs
]
with open("example_sim2.vcf", "w") as vcffile:
    nts.write_vcf(
        vcffile,
        individuals=sample_indivs[:5],
        individual_names=pedigree_ids[:5],
    )
```


(sec_tutorial_mutation_metadata)=

## Mutation metadata

Because of mutation stacking (see the SLiM manual),
each "tskit mutation" can represent a superposition of more than one
"SLiM mutation".
This is recorded by setting the derived state of the tskit mutation
to a comma-separated string of SLiM mutation IDs
(or the empty string, to denote "no mutations").
So, each SLiM mutation can thus appear in more than one tskit mutation,
and so the [metadata](sec_metadata) about these is stored in top-level metadata,
rather than along with the tskit mutations.
[](sec_tutorial_selected_mutations) has a more in-depth example, but here is a quick overview.
To print out the information about each SLiM mutation "carried"
by a given tskit mutation, whose SLiM IDs are stored in the mutation's metadata
under `"derived_states"`, we'd do:
```{code-cell}
:tags: ["remove-output"]
mut_metadata = pyslim.mutation_metadata(ts)

mut = ts.mutation(0)
for x in mut.metadata["derived_states"]:
    print(mut_metadata[x])
```
```{code-cell}
:tags: ["remove-input"]
mut_metadata = pyslim.mutation_metadata(ts)

mut = ts.mutation(0)
for x in mut.metadata["derived_states"]:
    util.pp(mut_metadata[x])
```
See [](sec_tutorial_selected_mutations) for an example where a tskit mutation
carries more than one (stacked) SLiM mutation.


(sec_extracting_individuals)=

## Extracting SLiM individuals

Another important thing to be able to do is to extract
individuals from a simulation,
for analysis or for outputting their genotypes, for instance.
This section demonstrates some basic manipulations of individuals.

### Extracting a random sample of individuals

The first, most common method to extract individuals is simply to get all
those that were alive at a particular time,
using {func}`.individuals_alive_at`. For instance, to get
the list of individual IDs of all those alive at the end of the
simulation (i.e., zero time units ago), we could do:

```{code-cell}
orig_ts = tskit.load("example_sim.trees")
alive_indivs = pyslim.individuals_alive_at(orig_ts, 0)

print(f"There are {len(alive_indivs)} individuals alive in the final generation.")
```

Here, ``alive_indivs`` is a vector of *individual* IDs,
so one way to take a random sample of living individuals
and write their SNPs to a VCF is:

```{code-cell}
rng = np.random.default_rng(seed=1)
keep_indivs = rng.choice(alive_indivs, 100, replace=False)
ts = msprime.sim_mutations(orig_ts, rate=1e-8, random_seed=1)
with open("example_snps.vcf", "w") as vcffile:
    ts.write_vcf(vcffile, individuals=keep_indivs)
```


### The non-equivalence of "individual" and "sample"

A somewhat confusing concept in tskit is that of a 
{ref}`sample node <tskit:sec_data_model_definitions_sample>`.
If you think of the tree sequence as containing information about some
genetic data, then the sample nodes are those that represent our actual data,
while other nodes represent ancestors.
In practice, many tskit methods apply by default to the samples.

However, we may have individuals in the tree sequence whose nodes are not samples,
and sometimes it is important to know this.
For instance, by default {meth}`tskit.TreeSequence.write_vcf`
will only output columns for sample nodes, silently omitting any non-sample nodes
among the individuals provided.
For instance, if the tree sequence has been simplified to retain only information
about a set of focal individuals, these nodes are marked as samples,
but other individuals who are alive at the end of the simulation may
still be in the tree sequence (e.g., parents of a pair of focal individuals)
but their entire genomes may not be present
and their nodes may not be marked as samples.

Here's an example. We'll simplify down to a random sample of the "children"
(those born during the last tick 
(as mentioned earlier, {meth}`tskit.TreeSequence.simplify` takes a list
of nodes as input):
```{code-cell}
children = [ind.id for ind in orig_ts.individuals() if ind.metadata['age'] == 0]
keep_indivs = rng.choice(children, 10, replace=False)
keep_nodes = []
for i in keep_indivs:
    keep_nodes.extend(orig_ts.individual(i).nodes)
sts = rts.simplify(keep_nodes)
ts = msprime.sim_mutations(sts, rate=1e-8, random_seed=1)
```

Individuals are retained by simplify if any of their nodes are retained,
so we would get an alive individual without sample nodes if, for instance,
a parent and two offspring are all alive, and we happen to keep the offspring
but not the parent.
Now, we write out to VCF:

```{code-cell}
alive_indivs = pyslim.individuals_alive_at(ts, 0)
print(f"There are {len(alive_indivs)} individuals alive.")
with open("example_snps.vcf", "w") as vcffile:
    ts.write_vcf(vcffile, individuals=alive_indivs)
```

We know that this tree sequence has only 10 individuals with sample
(those children, chosen above), but there are more alive individuals.
The resulting VCF has only 10 columns (note some numbers are missed in the
column headers), which could be surprising,
because we passed a longer list to the `individuals=` argument to
{meth}`tskit.TreeSequence.write_vcf`:
```{code-cell}
%%bash
head example_snps.vcf
```

The solution here would be to instead pass a list of the alive individuals who
are samples to `write_vcf`, which we can obtain like so:
```{code-cell}
sample_indivs = [
    i for i in pyslim.individuals_alive_at(ts, 0)
    if ts.node(ts.individual(i).nodes[0]).is_sample()
]
len(sample_indivs)
```
Note that we only check one node of an individual to see if it is a sample;
because we included either all or none of the nodes of each individual,
if one node of an individual is a sample node,
the other is guaranteed to be as well.

### Extracting particular individuals

Now let's see how to examine other attributes of individuals,
e.g., which subpopulation they're in.
To get another example with discrete subpopulations,
let's run another SLiM simulation, similar to the above
but with two populations exchanging migrants:

```{literalinclude} migrants.slim
```

Let's run it:
```{code-cell}
:tags: ["hide-output"]
%%bash
slim -s 32 migrants.slim
```

In the tree sequence, "populations" are associated with nodes, not individuals,
and so to find the population a given individual `ind` was born in,
we find the list of IDs of that individual's nodes with `ind.nodes`,
and then obtain the actual node object with `ts.node( )`.
So, to count up how many individuals are in each population,
we need to look
we could do:

```{code-cell}
orig_ts = tskit.load("migrants.trees")
alive_indivs = pyslim.individuals_alive_at(orig_ts, 0)
num_alive = [0 for _ in range(orig_ts.num_populations)]
for i in alive_indivs:
  ind = orig_ts.individual(i)
  ind_population = orig_ts.node(ind.nodes[0]).population
  num_alive[ind_population] += 1

for pop, num in enumerate(num_alive):
  print(f"Number of individuals in population {pop}: {num}")
```

:::{note}
Our SLiM script started numbering populations at 1, while tskit starts counting at 0,
so there is an empty "population 0" in a SLiM-produced tree sequence.
:::


(sec_recapitate_with_migration)=

## Recapitation with migration between more than one population

Following on the last example,
let's recapitate and mutate the tree sequence.
Recall that this recipe had two populations, ``p1`` and ``p2``,
each of size 1000.
Recapitation takes a bit more thought, because if the two populations stay separate,
it will run forever, unable to coalesce.
By default, {func}`.recapitate` *merges* the two populations into a single
one of size ``ancestral_Ne``.
But, if we'd like them to stay separate, we need to inclue migration between them.
Here's how we set up the demography using
{ref}`msprime's tools <msprime:sec_demography>`:
```{code-cell}
demography = msprime.Demography.from_tree_sequence(orig_ts)
for pop in demography.populations:
    # must set their effective population sizes
    pop.initial_size = 1000

demography.add_migration_rate_change(
    time=orig_ts.metadata['SLiM']['tick'],
    rate=0.1, source="p1", dest="p2",
)
demography.add_migration_rate_change(
    time=orig_ts.metadata['SLiM']['tick'],
    rate=0.1, source="p2", dest="p1",
)
rts = pyslim.recapitate(
        orig_ts, demography=demography,
        recombination_rate=1e-8,
        random_seed=4
)
ts = msprime.sim_mutations(
                    rts, rate=1e-8,
                    model=msprime.SLiMv6MutationModel(),
                    random_seed=7
)
```

Again, there are *three* populations because SLiM starts counting at 1;
the first population is unused (no migrants can go to it).
Let's compute genetic diversity within and between each of the two populations
(we compute the mean density of pairwise nucleotide differences,
often denoted {math}`\pi` and {math}`d_{xy}`).
To do this, we need to extract the node IDs from the individuals of the two populations
that are alive at the end of the simulation.
The method {meth}`tskit.TreeSequence.samples` will give us these nodes,
and {meth}`tskit.TreeSequence.diversity` computes genetic diversity for a given list
of node IDs (*not* individual IDs!),
so we can compute as follows:
```{code-cell}
pop_nodes = [ts.samples(population=p, time=0) for p in range(ts.num_populations)]
diversity = ts.diversity(pop_nodes[1:])
divergence = ts.divergence(pop_nodes[1:])

print(f"There are {ts.num_mutations} mutations across {ts.num_trees} distinct\n"
      f"genealogical trees describing relationships among {ts.num_samples}\n"
      f"sampled genomes, with a mean genetic diversity of {diversity[0]:0.3e}\n"
      f"and {diversity[1]:0.3e} within the two populations,\n"
      f"and a mean divergence of {divergence:0.3e} between them.")
```


## Individual metadata

Each ``Population``, ``Node``, and ``Individual``, as well as the tree
sequence as a whole, carries additional information stored by SLiM in its ``metadata``
property. A fuller description of metadata in general is given in [](sec_metadata),
but as a quick introduction, here is the information available
about an individual in the previous example:

```{code-cell}
:tags: ["remove-output"]
ts.individual(0)
```
```{code-cell}
:tags: ["remove-input"]
util.pp(ts.individual(0))
```

Some information is generic to individuals in tree sequences of any format:
``id`` (the ID internal to the tree sequence),
``flags`` (described [below](sec_individual_flags)),
``location`` (the [x,y,z] coordinates of the individual),
``nodes`` (an array of the node IDs that represent the genomes of this individual),
and ``time`` (the time, in units of "time ago" that the individual was born).

Other information, contained in the ``metadata`` field, is specific to tree sequences
produced by SLiM. This is described in more detail in the SLiM manual, but briefly:

- the  ``pedigree_id`` is SLiM's internal ID for the individual,
- ``age`` and ``subpopulation`` are their age and population at the time they
  were recorded, or at the time
  the simulation stopped if they were still alive  (NB: SLiM uses the word
  "subpopulation" for what is simply called a "population" in tree-sequence parlance)
- ``sex`` is their sex (as an integer, one of {data}`.INDIVIDUAL_TYPE_FEMALE`,
  {data}`.INDIVIDUAL_TYPE_MALE`, or {data}`.INDIVIDUAL_TYPE_HERMAPHRODITE`).
- ``flags`` holds additional information about the individual recorded by SLiM
  (currently, only whether the individual has migrated or not:
  see [](sec_constants_and_flags)).
- the ``tag`` entries contain the correspondingly-named "tags" in SLiM,
  and for the logical tags ``tagLX``, the ``tagLX_set`` records whether or not
  that tag was "set" (as opposed to remaining unset).
  The funny values in ``tag`` and ``tagF`` are those special values that SLiM uses to
  record that *those* entries were not set either.
- the ``per_trait`` entry is a list of information, one for each trait in the simulation.
  (There is always at least one trait:
  even models that do not explicitly set up any traits
  have a default trait set up by SLiM automatically.)


We can use this metadata in many ways, for example, to create an age distribution by sex:

```{code-cell}
import numpy as np
max_age = max([ind.metadata["age"] for ind in ts.individuals()])
age_table = np.zeros((max_age + 1, 2), dtype='int')
age_labels = { pyslim.INDIVIDUAL_TYPE_FEMALE: 'females',
               pyslim.INDIVIDUAL_TYPE_MALE: 'males' }
alive_indivs = pyslim.individuals_alive_at(ts, 0)
for i in alive_indivs:
    ind = ts.individual(i)
    age_table[ind.metadata["age"], ind.metadata["sex"]] += 1

print(f"number\t{age_labels[0]}\t{age_labels[1]}")
for age, x in enumerate(age_table):
    print(f"{age}\t{x[0]}\t{x[1]}")
```

We have looked up how to interpret the ``sex`` attribute
by using the values of {data}`.INDIVIDUAL_TYPE_FEMALE` (which is 0)
and {data}`.INDIVIDUAL_TYPE_MALE` (which is 1).
In a simulation without separate sexes,
all individuals would have sex equal to {data}`.INDIVIDUAL_TYPE_HERMAPHRODITE`
(which is -1).

Several fields associated with individuals are also available as numpy arrays,
across all individuals at once:
{attr}`tskit.TreeSequence.individuals_location`,
{attr}`tskit.TreeSequence.individuals_population`,
{attr}`tskit.TreeSequence.individuals_time` (also see
{func}`.individual_ages` and {func}`.individual_ages_at`).
Using these can sometimes be easier than
iterating over individuals as above. For example,
suppose that we want to randomly sample 10 individuals alive and older than 2 time steps
from each of the populations at the end of the simulation,
and simplify the tree sequence to retain only those individuals.
Since `alive_indivs` (produced with {func}`.individuals_alive_at` above)
and the output of {func}`.individual_ages` and `.individuals_population` 
are arrays of length equal to the number of individuals,
this can be done as follows:
```{code-cell}
ages = pyslim.individual_ages(ts)
adults = alive_indivs[ages[alive_indivs] > 2]
pops = [
   [i for i in adults if ts.individual(i).metadata['subpopulation'] == k]
   for k in [1, 2]
]
sample_inds = [np.random.choice(pop, 10, replace=False) for pop in pops]
sample_nodes = []
for samp in sample_inds:
  for i in samp:
     sample_nodes.extend(ts.individual(i).nodes)
sub_ts = ts.simplify(sample_nodes)
```

Note that here we have used the *subpopulation* attribute that SLiM places in metadata
to find out where each individual lives at the end of the simulation.
We might alternatively have used the *population* attribute of Nodes -
but, this would give each individual's *birth* location.

The resulting tree sequence does indeed have fewer individuals and fewer trees:

```{code-cell}
print(f"There are {sub_ts.num_mutations} mutations across {sub_ts.num_trees} distinct\n"
      f"genealogical trees describing relationships among {sub_ts.num_samples} sampled genomes,\n"
      f"with a mean overall genetic diversity of {sub_ts.diversity()}.")
```

(sec_tutorial_vacant_nodes)=

## Vacant nodes

As discussed in [the Overview](sec_overview_vacant_nodes),
if not all individuals have two copies of the chromosome stored in the tree sequence,
then some nodes will be *vacant*,
which means they are merely a placeholder and don't represent actual genetic material.
The presence of these nodes can cause problems.
For instance, running an msprime simulation backwards from 
a tree sequence with vacant sample nodes
(as in {numref}`figure {number} <pedigree_hap>` of the Overview)
would also simulate ancestry of the vacant nodes.
For this reason, 
you can remove the vacant nodes from the sample with {func}`.remove_vacant`.
Similarly, {func}`.recapitate` removes these nodes
from the sample before running msprime,
which makes it so their ancestry will not be simulated.
Similarly, at present {ref}`statistics in tskit<tskit:sec_stats>`
do not account for missing data, so will return incorrect results
if these vacant nodes are not removed from the sample.

To be clear, in the output from {func}`.recapitate` or {func}`.remove_vacant`
the vacant nodes will still be present,
just not marked as samples (i.e., the ``tskit.NODE_IS_SAMPLE``
flag has been removed from their node flags).
Once they are not part of the sample,
they are essentially invisible to most operations.
However, it is helpful to know that they are there.
Why not remove them entirely, e.g., with ``simplify()``?
They are kept because if you wish to read the tree sequence back into SLiM
then you'll need them;
they can put them back in the sample after being removed
with {func}`.restore_vacant`.

To find out which nodes in the tree sequence are vacant nodes,
use {func}`.nodes_vacant`.

## Historical individuals

As we've seen, a basic tree sequence output by SLiM only contains the currently alive
individuals and the ancestral nodes (genomes) required to reconstruct their genetic
relationships. But you might want more than that. For example, there may be individuals
who are not alive any more, but whose complete ancestry you would like to know. Or
perhaps you'd like to know how the final generation relates to particular individuals in
the past. Or it may be that you want to access the spatial location of historical genomes.
The solution is to *remember* an individual during the simulation,
using the SLiM function ``treeSeqRememberIndividuals()``.
Individuals can be Remembered in two ways, as described below.



```{figure} _static/pedigree_remember.png
---
scale: 40%
align: right
name: pedigree_remember
---
Individuals not alive in the last generation may still be present in the tree sequence
if they are either remembered permanently (purple),
or simply retained with ``permanent=F`` (dotted circle).
```



(sec_remembering_individuals)=

### Permanently remembering individuals

By default, a call to ``treeSeqRememberIndividuals()`` will permanently remember one or
more individuals, by marking their nodes as actual samples: the simulated equivalent of
ancient DNA dug out of permafrost, or stored
in an old collecting tube. This means any tree sequence subsequently recorded will always
contain this individual, its nodes (now marked as samples), and its full ancestry. As
with any other sample nodes, any permanently remembered individuals can be removed from
the tree sequence by [](sec_tutorial_simplification). The result of remembering an
individual in the [introductory example](sec_left_in_tree_sequence) is pictured on the right.


(sec_retaining_individuals)=

### Retaining individuals

Alternatively, you may want to avoid treating historical individuals and their genomes as
actual samples, but temporarily *retain* those individuals as long as they are still
relevant to reconstructing the genetic ancestry of the sample nodes. This can save some
computational burden, as not only will nodes and individuals be removed once they are no
longer ancestral, but also the full ancestry of the retained individuals does not need to
be kept. You can retain individuals in this way by using
``treeSeqRememberIndividuals(..., permanent=F)``.

Since a retained individual's nodes are not marked as samples, they are subject to the
[normal removal process](sec_left_in_tree_sequence), and it is possible to end up
with an individual containing only one genome, as shown in the diagram. However, as soon
as *both* nodes of a retained individual have been lost, the individual itself is deleted
too.

Note that by default, nodes are only kept if they mark a coalescent point (MRCA or branch
point) in one or more of the trees in a tree sequence. This can be changed by
initialising tree sequence recording in SLiM using
``treeSeqInitialize(retainCoalescentOnly=F)``. SLiM will then
preserve all retained individuals while they remain in the genealogy, even if their nodes
are not coalescent points in a tree (so-called "unary nodes"). Similarly, if you later
decide to reduce the number of samples via [](sec_tutorial_simplification),
retained individuals will be kept only if they are still MRCAs in the ancestry of the
selected samples. To preserve them even if their nodes are not coalescent points, you
can specify ``ts.simplify(selected_samples, keep_unary_in_individuals=True)``.

:::{todo}
Add SLiM code which includes retaining and remembering, and perhaps some python code
to show them.
:::

(sec_remembering_everyone)=


### Remembering everyone

Although not needed to reconstruct full genomic history, it is perfectly possible to
apply ``treeSeqRememberIndividuals()`` to every individual in every generation of a
simulation (i.e. everyone who has ever lived). If you simply retain everyone
(with `permanent=F`),
it should not increase the memory burden of your simulation much: most
individuals will be removed as the simulation progresses, since they will not contain
coalescent nodes. However, if you use ``treeSeqInitialize(retainCoalescentOnly=F)``,
the number of individuals in the resulting tree sequence is likely to become very large,
and the efficiencies provided by tree sequence recording will be substantially reduced.
Indeed in this case, retaining will be much the same as permanently remembering everyone
who has ever lived. (It will retain all individuals who are genetic ancestors to the
final generation, which is for a typical-length chromosome a substantial fraction of
everyone who ever lived.)
Nevertheless, if you are willing to sacrifice enough computer memory,
either of these is (perhaps surprisingly) possible, even for medium-sized simulations.


(sec_individual_flags)=

### Individual flags

We have seen that an individual can appear in the tree sequence because it was
Remembered or Retained, and/or alive at the end of the simulation.
The ``Individual.flags`` value stores this information.
For example, to count up the different individual types, we could do this:

:::{todo}
Update this code with the simulation above so that we have some remembered and
retained individuals present
:::

```{code-cell}
indiv_types = {"remembered" : 0,
              "retained" : 0,
              "alive" : 0}
for ind in ts.individuals():
  if ind.flags & pyslim.INDIVIDUAL_REMEMBERED:
     indiv_types['remembered'] += 1
  if ind.flags & pyslim.INDIVIDUAL_RETAINED:
     indiv_types['retained'] += 1
  if ind.flags & pyslim.INDIVIDUAL_ALIVE:
     indiv_types['alive'] += 1

for k in indiv_types:
  print(f"Number of individuals that are {k}: {indiv_types[k]}")
```

:::{note}
In previous versions of SLiM/pyslim, the first generation of individuals were
kept in the tree sequence, to allow [](sec_tutorial_recapitation). With the
addition of the ``keep_input_roots=True`` option to the
[](sec_tutorial_simplification) process, this is no longer necessary,
so these are no longer present, unless you specifically Remember them.
:::


(sec_initial_diversity)=

## Generating initial diversity with msprime

Suppose now that we'd like to *start* a SLiM simulation
with the result of a coalescent simulation.
For instance, we might want to do this instead of recapitating
if we wanted to use msprime to generate genetic diversity that
would then be selected on during the SLiM simulation.
To do this, we'll:
1. simulate a tree sequence with msprime,
2. add SLiM information to the nodes and individuals,
3. add SLiM mutations, and
4. write it out to a ``.trees`` file.

First, we'll (1) run a simulation of 1 Mb of genome sampled in 200 diploids
in a population of 1000 diploids,
and (2) use the {func}`.annotate` function to add default SLiM metadata to the result:
```{code-cell}
demog = msprime.Demography()
demog.add_population(initial_size=1000)
ts = msprime.sim_ancestry(
            samples=200,
            demography=demog,
            recombination_rate=1e-8,
            sequence_length=1e6,
            random_seed=5)
ts = pyslim.annotate(ts, model_type="nonWF", tick=1, stage="early")
assert ts.num_individuals == 200
assert ts.num_samples == 400
```
We have set ``tick`` to 1;
this means that as soon as we load the tree sequence into SLiM,
SLiM will set the current time counter to 1.
(If we set ``tick`` to 100, then any script blocks scheduled to happen before 100
would not execute after loading the tree sequence.)

We now have 200 diploids (so, 400 sampled nodes).
Here's individual 199, which has SLiM metadata:
```{code-cell}
:tags: ["remove-output"]
ind = ts.individual(199)
print(ind)
```
```{code-cell}
:tags: ["remove-input"]
util.pp(ind)
```
Looking at the ``metadata`` above, we see the default values are ``age=0``
hermaphrodites (``sex=-1``), for instance.

Now let's add SLiM mutations.
These will be neutral, as {func}`msprime.sim_mutations`
doesn't have the ability to dynamically modify the selection coefficients
stored in the mutation metadata.
To modify the mutations to be under selection,
see [](sec_vignette_coalescent_diversity).
```{code-cell}
ts = pyslim.add_mutation_metadata(
        msprime.sim_mutations(
                ts, rate=1e-8,
                model=msprime.SLiMv6MutationModel(),
                random_seed=9
        ),
        mutation_type=0,
)
```
The resulting mutations are in SLiM format
because we used {class}`msprime.SLiMv6MutationModel`
Now, each `mutation` object in the tree sequence represents
some number of SLiM mutations, whose SLiM IDs are stored in the mutation's `metadata`
(also in the `derived_state`, but it's recommended to use `metadata`).
For instance, here's which SLiM mutation(s) the first mutation
in the tree sequence represents:
```{code-cell}
ds = ts.mutation(0).metadata["derived_states"]
print(f"SLiM IDs: {ds}")
```
To see the information about these, we pull their information out
using {func}`.mutation_metadata`, which
as above in [](sec_tutorial_mutation_metadata) provides a dictionary
indexed by the SLiM IDs:
```{code-cell}
:tags: ["remove-output"]
mut_metadata = pyslim.mutation_metadata(ts)
for sid in ds:
    print(mut_metadata[sid])
```
```{code-cell}
:tags: ["remove-input"]
for sid in ds:
    util.pp(mut_metadata[sid])
```

Finally, we write this out to a file that can be loaded in to SLiM:
```{code-cell}
ts.dump("initialize_nonWF.trees")
```

Here's a minimal SLiM script that reads in the tree sequence file
and runs it for a bit longer.
We load the tree sequence in `early()`, matching the `stage`
passed to {func}`annotate` above.

```{literalinclude} neutral_restart.slim
```

```{code-cell}
:tags: ["hide-output"]
%%bash
slim -s 123 neutral_restart.slim
```

A more in-depth example is provided at [](sec_vignette_coalescent_diversity).
See the SLiM manual for more about this operation.


## Nucleotide-based models

By default, {func}`.annotate` produces standard SLiM mutations, not "nucleotide-based" mutations.
To demonstrate how to further adjust the starting state of the simulation,
we'll further adjust the tree sequence `ts` from the previous section
to add in information about nucleotides.

First, we need to set the ``nucleotide_based`` property in top-level metadata.
To do this, there are two possibly unfamiliar things:
first, we need to modify the underlying {class}`tskit.TableCollection`
(since tree sequences are immutable);
and second, we have to extract the metadata, modify it, and put it back in
(modifying it in-place will silently do nothing):

```{code-cell}
tables = ts.dump_tables()
md = tables.metadata
md['SLiM']['nucleotide_based'] = True
tables.metadata = md
ts = tables.tree_sequence()
```

Next, we need to generate a reference sequence
and nucleotides for each mutation.
This is easy with {func}`.generate_nucleotides`:

```{code-cell}
ts = pyslim.generate_nucleotides(ts)
ts.dump("initialize_nonWF_nuc.trees")
ts.reference_sequence.data[:20]
```

Now, mutations have a ``nucleotide`` property in metadata that is not ``-1``:

```{code-cell}
:tags: ["remove-output"]
mut_metadata = pyslim.mutation_metadata(ts)
m = ts.mutation(0)
md = [mut_metadata[k] for k in m.metadata["derived_states"]]
print(m)
for x in md:
    print(x)
```

```{code-cell}
:tags: ["remove-input"]
util.pp(m)
for x in md:
    util.pp(x)
```

We can see which nucleotide is the derived state produced by each mutation
by indexing the {data}`.NUCLEOTIDES` object:

```{code-cell}
for k in range(3):
    m = ts.mutation(k)
    print(f"Mutation {k}: position {ts.site(m.site).position}, time {m.time}")
    for sid in m.metadata["derived_states"]:
        md = mut_metadata[sid]
        print(f"  nucleotide: {pyslim.NUCLEOTIDES[md['nucleotide']]}")
```

Here's a script minimally modified from the above to be nucleotide-based:

```{literalinclude} neutral_nucleotide_restart.slim
```

```{code-cell}
:tags: ["hide-output"]
%%bash
slim -s 123 neutral_nucleotide_restart.slim
```

Everything runs reasonably.


(sec_tutorial_selected_mutations)=

## Extracting information about selected mutations

Here is a simple SLiM simulation with two types of mutation:
`m1` are deleterious, and `m2` are beneficial.
Let's see how to extract information about these mutations.

```{literalinclude} selection.slim
```
```{code-cell}
:tags: ["hide-output"]
%%bash
slim -s 23 selection.slim
```

First, let's see how many mutations there are:

```{code-cell}
ts = tskit.load("selection.trees")
print(f"Number of sites: {ts.num_sites}\n"
      f"Number of mutations: {ts.num_mutations}")
```

Note that there are more mutations than sites;
that's because some sites have multiple mutations.
The information about the mutation is put in the mutation's metadata.
Here's the first mutation:

```{code-cell}
:tags: ["remove-output"]
mut_metadata = pyslim.mutation_metadata(ts)
m = ts.mutation(0)
md = [mut_metadata[k] for k in m.metadata["derived_states"]]
print(m)
for x in md:
    print(x)
```

```{code-cell}
:tags: ["remove-input"]
util.pp(m)
for x in md:
    util.pp(x)
```

The trait value is _not_ the same thing as fitness, because other things can affect fitness too – fitnessEffect() callbacks and individual/subpop fitnessScaling values, in particular.  So I think this distinction is important to be clear about.

Since we haven't explicitly defined any traits in this simulation,
the only trait is the default multiplicative trait implicitly defined by SLiM,
which has a direct effect on fitness,
and so the `effect_size` listed under `per_trait`
for this mutation is simply its selection coefficient.
Furthermore, `m.site` tells us the ID of the *site* on the genome that the mutation occurred at,
and we can pull up information about that with the `ts.site( )` method:

```{code-cell}
:tags: ["remove-output"]
s = ts.site(m.site)
md = [
    mut_metadata[k] for m in s.mutations
                         for k in m.metadata["derived_states"]
]
print(s)
for x in md:
    print(x)
```

```{code-cell}
:tags: ["remove-input"]
util.pp(s)
for x in md:
    util.pp(x)
```

This mutation occurred at the position along the genome shown in `site.position`,
which previously had no mutations (since `site.ancestral_state` is the empty string, `''`)
and was given the SLiM mutation ID shown in `m.metadata["derived_states"]`.
The metadata (with `x` the mutation ID, `mut_metadata[x]`, a dict) tells us
the mutation's selection coefficient and which population and at what SLiM time it occurred.
This is not a nucleotide model, so the nucleotide entry is `-1`.
Note that `m.time` and the `slim_time` entry in metadata are in this case redundant:
they contain the same information, but the first is in tskit time
(i.e., number of steps before the tree sequence was written out)
and the second is using SLiM's internal "tick" counter.

Also note that each mutation may have associated a *list* of SLiM mutations,
each with their own metadata.
That's because of SLiM's mutation stacking feature.
We know that some sites have more than one mutation,
so to get an example let's pull out one such mutation.

Let's pull out a mutation that was stacked on top of another one:

```{code-cell}
:tags: ["remove-output"]
for m in ts.mutations():
  if m.parent != tskit.NULL:
     break

pm = ts.mutation(m.parent)
md = [mut_metadata[k] for k in m.metadata["derived_states"]]
pmd = [mut_metadata[k] for k in pm.metadata["derived_states"]]

print(m)
for x in md:
    print(x)
print(pm)
for x in pmd:
    print(x)
```

```{code-cell}
:tags: ["remove-input"]
util.pp(m)
for x in md:
    util.pp(x)
util.pp(ts.mutation(m.parent))
for x in pmd:
    util.pp(x)
```

This mutation (which is `ts.mutation(330)` in the tree sequence)
was the result of SLiM adding a new mutation of type `m1` and selection coefficient -0.1547
on top of an existing mutation, of type `m2` and with (whopping) selection coefficient 1.737.
This happened at generation 998 (i.e., at tskit time 1.0 time units ago),
and the older mutation occurred at generation 83 (at tskit time 916 time units ago).
The older mutation has SLiM mutation ID 1994163,
and the newer mutation had SLiM mutation ID 164833,
so the resulting "derived state" is `'1994163,164833'`.

Now that we understand how SLiM mutations are stored in a tree sequence,
let's look at the allele frequencies.
The allele frequency spectrum for *all* mutations can be obtained using the
{meth}`tskit.TreeSequence.allele_frequency_spectrum` method.
(Note this is the frequency spectrum for the *tskit* mutations,
rather than the SLiM mutations, which may differ because of stacking.)
It is shown here for a sample of size 10 to make the output easy to see:

```{code-cell}
samps = np.random.choice(ts.samples(), 10, replace=False)
afs = ts.allele_frequency_spectrum([samps], span_normalise=False, polarised=True)
print(afs.astype('int'))
```

(The `span_normalise=False` argument gives us counts rather than a density per unit length.)
This shows us that there are 3929 alleles that are found among the tree sequence's samples
that are not present in any of our 10 samples, 585 that are present in just one, etcetera.
The surprisingly large number that are near 50% frequency are perhaps positively selected
and on their way to fixation: we can check if that's true next.
You may have noticed that the sum of the allele frequency spectrum is 5029,
which is not obviously related to the number of mutations (5861) *or* the number of sites (5848).
That's because each derived allele that is inherited by some but not all of the samples
in the tree sequence is counted in the polarised allele frequency spectrum:
Fixed mutations, or mutations that were entirely "overwritten" by subsequent mutations,
do not contribute.
Here's how we can check this:

```{code-cell}
afs_total = 0
for v in ts.variants():
    if len(set(v.genotypes)) > 1:
        afs_total += len(set(v.genotypes) - set([0]))
print(afs_total, sum(afs))
```

These are equal, verifying our interpretation.

At time of writing, we don't have a built-in ``allele_frequency`` method,
so we'll use the following snippet:

```{code-cell}
def allele_counts(ts, sample_sets=None):
   if sample_sets is None:
      sample_sets = [ts.samples()]
   def f(x):
      return x
   return ts.sample_count_stat(sample_sets, f, len(sample_sets),
              span_normalise=False, windows='sites',
              polarised=True, mode='site', strict=False)
```

This will return an array of counts, one for each site in the tree sequence,
giving the number of *all* nonancestral alleles at that site found in the sample set
(so, lumping together any of the various derived alleles we were looking at above).
Then, we'll separate out the counts in this array to get the derived frequency spectra
separately for sites with (a) only `m1` mutations, (b) only `m2` mutations,
and (c) both (for completeness, if there are any).
First, we need to know which site has which of these three mutation types (m1, m2, or both):

```{code-cell}
mut_type = np.zeros(ts.num_sites)
for j, s in enumerate(ts.sites()):
  mt = []
  for m in s.mutations:
     for sid in m.metadata["derived_states"]:
        md = mut_metadata[sid]
        mt.append(md["mutation_type"])
  if len(set(mt)) > 1:
     mut_type[j] = 3
  else:
     mut_type[j] = mt[0]
```

Now, we compute the frequency spectrum, and aggregate it
to produce the allele frequency spectrum separately by mutation type.
We'll use the function `np.bincount` to do this efficiently:

```{code-cell}
freqs = allele_counts(ts, [samps])
# convert the n x 1 array of floats to a vector of integers
freqs = freqs.flatten().astype(int)
mut_afs = np.zeros((len(samps)+1, 3), dtype='int64')
for k in range(3):
  mut_afs[:, k] = np.bincount(freqs[mut_type == k+1], minlength=len(samps) + 1)

print(mut_afs)
```

The first column gives the AFS among these 10 samples for the deleterious alleles,
the second for the beneficial mutations;
the third column for the few sites that had both types of mutation.
Interestingly, there are similar numbers of both types of mutation at intermediate frequency:
perhaps because beneficial mutations are sweeping linked deleterious alleles along with them.
Many fewer benefical alleles are at low frequency, however.

Finally, let's pull out information on the allele with the largest selection coefficient.

```{code-cell}
:tags: ["remove-output"]
sel_coeffs = np.array([
        sum(mut_metadata[k]["per_trait"][0]["effect_size"]
            for k in m.metadata["derived_states"])
        for m in ts.mutations()
])
which_max = np.argmax(sel_coeffs)
m = ts.mutation(which_max)
print(f"Max selection coefficient: {sel_coeffs[which_max]} for site {m.site}")
ts.site(m.site)
```

```{code-cell}
:tags: ["remove-input"]
print(f"Max selection coefficient: {sel_coeffs[which_max]} for site {m.site}")
util.pp(ts.site(m.site))
```

This allele had a whopping selection coefficient of 5.69
and appeared fairly late in the simulation.
Let's find its frequency in the full population:

```{code-cell}
full_freqs = allele_counts(ts)
print(f"The allele is found in {full_freqs[m.site][0]} copies\n"
      f"out of {ts.num_nodes} genomes.")
```

The allele is above 50% in the population, so it is probably on its way to fixation.
Using its SLiM ID (which is shown in its derived state, ``305447``),
we could reload the tree sequence into SLiM,
restart the simulation, and use its ID to track its subsequent progression.


## Possibly important technical notes

Also known as "gotchas".

1. If you use msprime to simulate a tree sequence, and then use that to initialize a SLiM simulation,
    you have to specify the same sequence length in both: as in the examples above,
    the ``sequence_length`` argument to {func}`msprime.sim_ancestry` should be equal to the
    `lastPosition` in SLiM
    *plus 1.0* (e.g., if the base positions in SLiM are 0 to 99, then there are 100 bases in all,
    so the sequence length should be 100).

2. Make sure to distinguish *individuals* and *nodes*!
   ``tskit`` "nodes" correspond to SLiM "haplosomes".
   Individuals in SLiM normally each have two nodes, but retained
   individuals may have nodes removed by simplification: see below,
   and some nodes may be [vacant](sec_tutorial_vacant_nodes).

3. As described above, the Individual table contains entries for

   1. the currently alive individuals,
   2. any individuals that have been permanently remembered with
      ``treeSeqRememberIndividuals()``, and
   3. any individuals that have been temporarily retained with
      ``treeSeqRememberIndividuals(permanent=F)``. Importantly, the nodes in these
      individuals are *not* marked as sample nodes, so they can be lost during
      simplification. This means that a retained individual may only have one node (but
      if both nodes are lost due to simplification, the individual is removed too, and
      will not appear in the Individual table).

4. SLiM requires that the two nodes corresponding to the haplosomes of each individual
    are adjacent in the node table, and are sorted by haplosome ID.
    SLiM always writes out tree sequences like this, but it is possible to make
    tree sequences in python that are legal otherwise but don't satisfy this requirement.
