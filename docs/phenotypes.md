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
import util
import plotnine as p9
import pandas as pd

np.random.seed(1234)
```


(sec_phenotypes)=


# Working with traits and phenotypes

An individual's phenotype for an trait in SLiM is the result of that individual's genetic effects,
and both global and individual-specific "offsets".
SLiM records in metadata both phenotype and offset, so we can use these (by subtraction, for an additive trait)
to determine the genetic value.
We can also compute genetic contributions directly, which can be helpful,
for instance, to decompose genetic variance from different parts of the genome.

Let's start off with this simulation, which has:

- two additive traits,
- with Gaussian selection on both,
- started away from the optimum;
- correlated mutational effects,
- and also neutral mutations.

The recipe also saves the state of the population (by remembering everyone)
every 1,000 generations. You may want to skip the details of the SLiM script
on first reading.

```{literalinclude} phenotypes.slim
```


(sec_phenotypes_trait_values)=

## Trait distributions

First, let's load the result of the simulation and look at trait distributions in each of the saved time points.
```{code-cell}
ts = tskit.load("phenotypes.trees")
ts_metadata = ts.metadata
mut_metadata = pyslim.mutation_metadata(ts)
```
Here's information about the traits:
```{code-cell}
ts_metadata['SLiM']['traits']
```

Here's the information we have about a given individual:
```{code-cell}
ts.individual(0)
```
In particular, phenotype and offset values:
```{code-cell}
ts.individual(0).metadata['per_trait']
```

Next, let's find the birth time of each individual, and see which times we have information for
(this is simple because it's a WF simulation):
```{code-cell}
ind_times = ts.nodes_time[ts.individuals_nodes[:,0]]

from collections import Counter
Counter([int(t) for t in ind_times])
```

We can use the helpful `metadata_vector` method of table collections
to quickly pull out the phenotype and offset vectors:
```{code-cell}
ind_phenotypes = np.column_stack([
    ts.tables.individuals.metadata_vector(["per_trait", j, "phenotype"])
    for j in (0, 1)
])
ind_offsets = np.column_stack([
    ts.tables.individuals.metadata_vector(["per_trait", j, "offset"])
    for j in (0, 1)
])
```

To prepare to make the plots
(using [plotnine](https://plotnine.org))
we'll put everything in a data frame:
```{code-cell}
df = pd.DataFrame({
    'pedigree_id' : ts.tables.individuals.metadata_vector("pedigree_id"),
    'time' : ind_times,
    "phenotype1" : ind_phenotypes[:,0],
    "offset1" : ind_offsets[:,0],
    "phenotype2" : ind_phenotypes[:,1],
    "offset2" : ind_offsets[:,1],
})
```
As noted before, we can get the genetic contributions
by subtracting the individual-level offsets:
```{code-cell}
df['genetic_value1'] = df['phenotype1'] - df['offset1']
df['genetic_value2'] = df['phenotype2'] - df['offset2']
```
Now, here's phenotype distributions across these time slices
```{code-cell}
( df >> 
    p9.ggplot(p9.aes(x='phenotype1', color="time"))
    + p9.geom_histogram(bins=40) + p9.facet_grid("time ~")
)
```
Here's how the population moved through 2d phenotype space:
```{code-cell}
(
    df >> p9.ggplot(p9.aes(x="phenotype1", y="phenotype2", color="time")) 
    + p9.geom_point(alpha=0.5)
)
```
That's the phenotypes; it turns out that most of that spread is actually
"offset", i.e., what's usually called "environmental" noise:
```{code-cell}
(
    df >> p9.ggplot(p9.aes(x="genetic_value1", y="genetic_value2", color="time")) 
    + p9.geom_point(alpha=0.5)
)
```

(sec_phenotypes_mutation_effects)=

## Mutation effects

Now let's see how to use mutation information to calculate genetic values directly.
To do this, let's take a simpler example.
This just has two additive traits, and mutations have independent effects on each trait.
There are two mutation types; the first is underdominant and the second is additive
(using the special `NAN` value for $h$ to indicate independent dominance).

```{literalinclude} phenotypes2.slim
```


```{code-cell}
ts = tskit.load("phenotypes2.trees")
ts_metadata = ts.metadata
mut_metadata = pyslim.mutation_metadata(ts)
```

For example, here's the first mutation:
```{code-cell}
mut = ts.mutation(0)
mut
```
We can see which SLiM mutation(s) this mutation represents
by looking at the mutation's "derived state",
and pulling the information out of the mutation metadata:


```{code-cell}
mut_metadata[mut.metadata["derived_states"][0]]
```
So, we can use the `effect_size` and `dominance`
to calculate genetic effects (we won't need `hemizygous dominance`).
Here is a function that calculates the effect of a pair of alleles.
The function uses the `derived_state` property (a string) rather than the
metadata property because that's what's returned by {meth}`tskit.TreeSequence.variants`,
which we'd like to use next.


```{code-cell}
from collections import Counter

def additive_effect(mut_metadata, a, b, num_traits=2):
    # here a and b are *string* derived states
    # note this works for additive traits:
    # multiplicative ones are "* (1 + hs)" instead of "+ 2hs"
    out = np.zeros((num_traits,))
    muts = Counter(a.split(",")) + Counter(b.split(","))
    for m in muts:
        if m != "":
            md = mut_metadata[int(m)]['per_trait']
            if muts[m] == 1:
                for j in range(num_traits):
                    h = md[j]['dominance']
                    if np.isnan(h):
                        h = 1/2
                    out[j] += 2 * md[j]['effect_size'] * h
            if muts[m] == 2:
                for j in range(num_traits):
                    out[j] += 2 * md[j]['effect_size']
    return out

# for instance, a homozygote for that mutation:
additive_effect(mut_metadata, mut.derived_state, mut.derived_state)
```

Using this and {meth}`tskit.TreeSequence.variants`,
we can compute genetic values for an individual:
```{code-cell}
def additive_genetic_value(ts, mut_metadata, ind, num_traits=2):
    out = np.zeros((num_traits,))
    for v in ts.variants(samples=ind.nodes):
        x, y = [v.alleles[g] for g in v.genotypes]
        out += additive_effect(mut_metadata, x, y)
    return out

additive_genetic_value(ts, mut_metadata, ts.individual(0))
```

The final component is offsets.
To match SLiM we need to add in both the individual's offset
(which is stored in their metadata)
and the global ("baseline") offset.
Within SLiM, the `baselineOffset` property is a sum of the value provided by the user
on initialization of the trait, and the accumulated value of any substitutions
(if various options do not modify this behavior; see the SLiM manual).
Since the tree sequence does not distinguish substitutions from other mutations,
SLiM stores these two components separately, as `baselineOffsetFromUser`
and `baselineOffsetFromSubstitutions`. So, we just need to add in `baselineOffsetFromUser`.


```{code-cell}
def additive_offset(ts_metadata, ind):
    out = np.array([x['baselineOffsetFromUser'] for x in ts_metadata['SLiM']['traits']])
    out += [x['offset'] for x in ind.metadata['per_trait']]
    return out

additive_offset(ts.metadata, ts.individual(0))
```

Now we can compute phenotypes, and check they match what SLiM produced.


```{code-cell}
alive = pyslim.individuals_alive_at(ts, 0)
ind = ts.individual(alive[0])
print(f"Ours: {additive_genetic_value(ts, mut_metadata, ind) + additive_offset(ts_metadata, ind)}")
print(f"SLiM: {np.array([x['phenotype'] for x in ind.metadata['per_trait']])}")
```

That looks good! To compare we need to account for floating-point error
(actually the numbers stored by SLiM and computed by us may differ by $10^{-10}$).


```{code-cell}
for ind in ts.individuals():
    our_pheno = additive_genetic_value(ts, mut_metadata, ind) + additive_offset(ts_metadata, ind)
    slim_pheno = np.array([x['phenotype'] for x in ind.metadata['per_trait']])
    assert np.allclose(our_pheno, slim_pheno)
```

(sec_phenotypes_multiplicative_traits)=

## Multiplicative and logistic traits

To see how multiplicative and logistic traits work,
we'll change the first trait to be multiplicative
and the second logistic:

```{literalinclude} phenotypes3.slim
```

This produces
```{code-cell}
ts = tskit.load("phenotypes3.trees")
ts_metadata = ts.metadata
mut_metadata = pyslim.mutation_metadata(ts)
ts
```

Just for the heck of it, we'll load the information into a data frame
row-wise instead of column-wise:
```{code-cell}
df = pd.DataFrame([
        (
         ts.node(ind.nodes[0]).time,
         ind.metadata['per_trait'][0]['phenotype'],
         ind.metadata['per_trait'][0]['offset'],
         ind.metadata['per_trait'][1]['phenotype'],
         ind.metadata['per_trait'][1]['offset'],
        )
        for ind in ts.individuals()
    ],
    columns=['time', "phenotype1", "offset1", "phenotype2", "offset2"],
)
```

Here's the joint distribution of trait values.
The first is positive (since it's multiplicative)
and the second between 0 and 1 (since it's logistic).


```{code-cell}
df >> p9.ggplot(p9.aes(x='phenotype1', y='phenotype2')) + p9.geom_point()
```


A "logistic" trait is just an additive trait that's been put through the logistic transform,
$x \mapsto 1/(1 + \exp(-x))$. So we can use the code above to verify:


```{code-cell}
alive = pyslim.individuals_alive_at(ts, 0)
ind = ts.individual(alive[0])
ind_pheno = additive_genetic_value(ts, mut_metadata, ind) + additive_offset(ts_metadata, ind)
ind_pheno[1] = 1/(1 + np.exp(-ind_pheno[1]))
print(f"Ours: {ind_pheno}")
print(f"SLiM: {np.array([x['phenotype'] for x in ind.metadata['per_trait']])}")
```


For the multiplicative trait, we need some new functions.
Following the pattern above:


```{code-cell}
def multiplicative_effect(mut_metadata, a, b, num_traits=2):
    # here a and b are *string* derived states
    out = np.ones((num_traits,))
    muts = Counter(a.split(",")) + Counter(b.split(","))
    for m in muts:
        if m != "":
            md = mut_metadata[int(m)]['per_trait']
            if muts[m] == 1:
                for j in range(num_traits):
                    h = md[j]['dominance']
                    if np.isnan(h):
                        h = 1/2
                    out[j] *= (1 + md[j]['effect_size'] * h)
            if muts[m] == 2:
                for j in range(num_traits):
                    out[j] *= (1 + md[j]['effect_size'])
    return out

# for instance, a homozygote for the first mutation:
mut = ts.mutation(0)
multiplicative_effect(mut_metadata, mut.derived_state, mut.derived_state)
```

Genetic values:

```{code-cell}
def multiplicative_genetic_value(ts, mut_metadata, ind, num_traits=2):
    out = np.ones((num_traits,))
    for v in ts.variants(samples=ind.nodes):
        x, y = [v.alleles[g] for g in v.genotypes]
        out *= multiplicative_effect(mut_metadata, x, y)
    return out

multiplicative_genetic_value(ts, mut_metadata, ts.individual(0))
```

Offsets:


```{code-cell}
def multiplicative_offset(ts_metadata, ind):
    out = np.array([x['baselineOffsetFromUser'] for x in ts_metadata['SLiM']['traits']])
    out *= [x['offset'] for x in ind.metadata['per_trait']]
    return out

multiplicative_offset(ts.metadata, ts.individual(0))
```

Let's combine these into a function that works for this simulation.
This is not going to be very efficient (for many reasons),
but efficiency is not important here, since we're just verifying that we understand
how the `phenotype` values SLiM computes relate to what's in the tree sequence.


```{code-cell}
def phenotype(ind):
    out = multiplicative_offset(ts_metadata, ind)
    out[1] = additive_offset(ts_metadata, ind)[1]
    add = additive_genetic_value(ts, mut_metadata, ind)
    mult = multiplicative_genetic_value(ts, mut_metadata, ind)
    out[0] *= mult[0]
    out[1] += add[1]
    out[1] = 1 / (1 + np.exp(-out[1]))
    return out

phenotype(ts.individual(0))
```

Putting this together,

```{code-cell}
alive = pyslim.individuals_alive_at(ts, 0)
ind = ts.individual(alive[0])
print(f"Ours: {phenotype(ind)}")
print(f"SLiM: {np.array([x['phenotype'] for x in ind.metadata['per_trait']])}")
```

More elegant code would pull the trait types out of top-level metadata
and use additive or multiplicative effects accordingly, etcetera.
