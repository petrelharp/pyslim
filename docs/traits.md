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


(sec_traits)=


# Working with traits and phenotypes

An individual's phenotype for a trait in SLiM is determined from that individual's genetic effects,
as well as both global and individual-specific "offsets".
SLiM records in metadata both phenotype and offset, so we can use these (by subtraction, for an additive trait)
to determine the genetic value.
We can also compute genetic contributions directly, which can be helpful,
for instance, to decompose genetic variance from different parts of the genome.

Let's start off with the simulation below, which has:

- two additive traits,
- with Gaussian selection on both,
- started away from the optimum;
- correlated mutational effects,
- and also neutral mutations.

The recipe also saves the state of the population (by remembering everyone)
every 5,000 generations. You may want to skip the details of the SLiM script
on first reading: the simulation is more complex to display interesting trait dynamics.
See the chapter on traits in the SLiM manual for more discussion,
in particular the example *"a model with two quantitative traits and pleiotropy"*,
from which this script was modified.

```{literalinclude} phenotypes.slim
```


(sec_traits_trait_values)=

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
The information matches the parameters passed to `initializeTrait`,
along with `substitutionOffsetH`, which we discuss below.
This is a simulation of hermaphrodites, so the baseline offsets we specified in script
(as `START1` and `START2` in the script)
are recorded here as `baselineOffsetH`. If this simulation had separate sexes,
there would be two baseline offsets: `F` and `M`.

Here's the information we have about a given individual:
```{code-cell}
:tags: ["remove-output"]
ts.individual(0)
```
```{code-cell}
:tags: ["remove-input"]
util.pp(ts.individual(0))
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
num_traits = 2
ind_phenotypes = np.column_stack([
    ts.tables.individuals.metadata_vector(["per_trait", j, "phenotype"])
    for j in range(num_traits)
])
ind_offsets = np.column_stack([
    ts.tables.individuals.metadata_vector(["per_trait", j, "offset"])
    for j in range(num_traits)
])
```
For instance, here are the "phenotype" entries for the first five individuals
(one column per trait):
```{code-cell}
ind_phenotypes[:5,:]
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
df.head()
```
Now, here's phenotype distributions across these time slices.
We can see that the population is moving from its initial position at (-5, 5)
towards the optimum at (20, -20):
```{code-cell}
(
    df >> p9.ggplot(p9.aes(x="phenotype1", y="phenotype2", color="time")) 
    + p9.geom_point()
)
```
That's the phenotypes; it turns out that most of that spread is actually
"offset", i.e., what's usually called "environmental" noise.
We can get the genetic contributions by subtracting the individual-level offsets:
```{code-cell}
df['genetic_value1'] = df['phenotype1'] - df['offset1']
df['genetic_value2'] = df['phenotype2'] - df['offset2']
```
Plotting these, we see the populations have very little genetic variation:
```{code-cell}
(
    df >> p9.ggplot(p9.aes(x="genetic_value1", y="genetic_value2", color="time")) 
    + p9.geom_point()
)
```

(sec_traits_mutation_effects)=

## Mutation effects

Now let's see how to use mutation information to calculate genetic values directly.
To do this, let's take a simpler example.
This just has two additive traits, and mutations have independent effects on each trait.
Both traits are neutral, and so are only affected by drift and are not calculated at all
by SLiM until we call `demandPhenotype`.
There are two mutation types; the first is underdominant and the second is additive
(using the special `NAN` value for {math}`h` to indicate independent dominance).

```{literalinclude} phenotypes2.slim
```

First, we load in the resulting tree sequence,
and extract copies of top-level metadata and mutation metadata,
as described in [](sec_tutorial_mutation_metadata):
```{code-cell}
ts = tskit.load("phenotypes2.trees")
ts_metadata = ts.metadata
mut_metadata = pyslim.mutation_metadata(ts)
```

Here's the first mutation:
```{code-cell}
:tags: ["remove-output"]
mut = ts.mutation(0)
mut
```
```{code-cell}
:tags: ["remove-input"]
util.pp(mut)
```
We can see which SLiM mutation(s) this mutation represents
by looking up those SLiM mutation IDs in the mutation's "derived state",
`mut.metadata["derived_states"]`.
(As noted [previously](sec_tutorial_mutation_metadata),
the same information is stored in `mut.derived_state`,
but we recommend pulling this information out of metadata.)
Then, we can find information about those SLiM mutations
in the top-level mutation metadata (here, `mut_metadata`,
obtained using {func}`.mutation_metadata`).
```{code-cell}
mut_metadata[mut.metadata["derived_states"][0]]
```
So, we can use the `effect_size` and `dominance`
to calculate genetic effects (we won't need `hemizygous dominance`,
since no-one is hemizygous in this simulation).
Here is a function that calculates the effect of a pair of alleles.
The function uses the `derived_state` property (a string) rather than the
metadata property because that's what's returned by {meth}`tskit.TreeSequence.variants`,
which we'll be using below.

```{code-cell}
from collections import Counter

def additive_effect(mut_metadata, a, b):
    # Here a and b are *string* derived states.
    num_traits = len(next(iter(mut_metadata.values()))['per_trait'])
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
def additive_genetic_value(ts, mut_metadata, ind):
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
Within SLiM, the `baselineOffsetX` properties
(where `X` is `F`, `M`, or `H`) values are provided by the user
on initialization of the trait, and contribute to all individual's phenotypes.
Within SLiM there is also the `substitutionOffsetX` property to consider,
but since the tree sequence does not distinguish substitutions from other mutations,
here we do nothing with these.
(For more on this, see the SLiM manual and [](sec_traits_technical_details).)

Here is a function that finds the offsets for an individual,
depending on their sex, and combines them for an additive trait:
```{code-cell}
def additive_offset(ts_metadata, ind):
    k = {
        pyslim.INDIVIDUAL_TYPE_HERMAPHRODITE : "baselineOffsetH",
        pyslim.INDIVIDUAL_TYPE_FEMALE : "baselineOffsetF",
        pyslim.INDIVIDUAL_TYPE_MALE : "baselineOffsetM",
    }[ind.metadata["sex"]]
    out = np.array([x[k] for x in ts_metadata['SLiM']['traits']])
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

Happily, they match. Next we can compare for all individuals;
to actually check for equality we need to account for floating-point error,
since the numbers stored by SLiM and computed by us may differ by `1e-10` or so:

```{code-cell}
for ind in ts.individuals():
    our_pheno = additive_genetic_value(ts, mut_metadata, ind) + additive_offset(ts_metadata, ind)
    slim_pheno = np.array([x['phenotype'] for x in ind.metadata['per_trait']])
    assert np.allclose(our_pheno, slim_pheno)
```

(sec_traits_multiplicative_traits)=

## Multiplicative and logistic traits

To see how multiplicative and logistic traits work,
we'll change the first trait to be multiplicative
and the second to be logistic:

```{literalinclude} phenotypes3.slim
```

First, we load the information,
remembering to re-compute `ts_metadata` and `mut_metadata`:
```{code-cell}
ts = tskit.load("phenotypes3.trees")
ts_metadata = ts.metadata
mut_metadata = pyslim.mutation_metadata(ts)
```

Just for the heck of it, we'll load the individual information into a data frame
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
df.head()
```

Here's the joint distribution of trait values.
The first is positive (since it's multiplicative)
and the second between 0 and 1 (since it's logistic).

```{code-cell}
df >> p9.ggplot(p9.aes(x='phenotype1', y='phenotype2')) + p9.geom_point()
```

A "logistic" trait is just an additive trait that's been put through the logistic transform,
{math}`x \mapsto 1/(1 + \exp(-x))`. So we can use the code above to verify:
```{code-cell}
alive = pyslim.individuals_alive_at(ts, 0)
ind = ts.individual(alive[0])
ind_pheno = additive_genetic_value(ts, mut_metadata, ind) + additive_offset(ts_metadata, ind)
ind_pheno[1] = 1/(1 + np.exp(-ind_pheno[1]))
print(f"Ours: {ind_pheno}")
print(f"SLiM: {np.array([x['phenotype'] for x in ind.metadata['per_trait']])}")
```
The second phenotype matches, since that's the logistic trait.
The first does not, as expected, since we've computed the phenotype as if it were additive,
when in fact it's multiplicative.

For the multiplicative trait, we need some new functions.
Following the pattern above,
and remembering that while the effects on an additive trait are
`+(0, h*2*s, 2*s)`, for a multiplicative trait they are
`*(1, 1+h*s, 1+s)`:
```{code-cell}
def multiplicative_effect(mut_metadata, a, b):
    # here a and b are *string* derived states
    num_traits = len(next(iter(mut_metadata.values()))['per_trait'])
    out = np.zeros((num_traits,))
    muts = Counter(a.split(",")) + Counter(b.split(","))
    for m in muts:
        if m != "":
            md = mut_metadata[int(m)]['per_trait']
            for j in range(num_traits):
                s = md[j]["effect_size"]
                if muts[m] == 1:
                    h = md[j]['dominance']
                    if np.isnan(h):
                        # "independent dominance occurs when (1+hs)(1+hs) equals 1+s,
                        # which occurs when h=(sqrt(1+s)−1)/s"
                        h = (np.sqrt(1 + s) - 1) / s if s != 0 else 0
                    out[j] *= (1 + h * s)
                else:
                    assert  muts[m] == 2
                    out[j] *= max(0, 1 + s)
    return out

# for instance, a homozygote for the first mutation:
mut = ts.mutation(0)
multiplicative_effect(mut_metadata, mut.derived_state, mut.derived_state)
```

Genetic values:
```{code-cell}
def multiplicative_genetic_value(ts, mut_metadata, ind):
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
    k = {
        pyslim.INDIVIDUAL_TYPE_HERMAPHRODITE : "baselineOffsetH",
        pyslim.INDIVIDUAL_TYPE_FEMALE : "baselineOffsetF",
        pyslim.INDIVIDUAL_TYPE_MALE : "baselineOffsetM",
    }[ind.metadata["sex"]]
    out = np.array([x[k] for x in ts_metadata['SLiM']['traits']])
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
    mult = multiplicative_genetic_value(ts, mut_metadata, ind)
    add = additive_genetic_value(ts, mut_metadata, ind)
    out[0] *= mult[0]
    out[1] += add[1]
    out[1] = 1 / (1 + np.exp(-out[1]))
    return out
```

Putting this together,
```{code-cell}
alive = pyslim.individuals_alive_at(ts, 0)
ind = ts.individual(alive[0])
print(f"Ours: {phenotype(ind)}")
print(f"SLiM: {np.array([x['phenotype'] for x in ind.metadata['per_trait']])}")
```
Happily, these again agree, up to floating point error.

More elegant code would pull the trait types out of top-level metadata
and use additive or multiplicative effects accordingly, etcetera.
Furthermore, if some chromosomes are not autosomes, we'd need to identify the ploidy
of each individual, and use the hemizygous dominance coefficient when appropriate.

(sec_traits_technical_details)=

## Technical details

You're better off using SLiM to calculate phenotypes
than doing it yourself in python.
There's lots of corner cases, and
phenotypes can even be modified by the SLiM script directly.
But, there's some situations where it's important to know about those corner cases.

First: phenotypes are determined from the various contributions by
either a sum ("additive" or "logistic") or a product ("multiplicative").
Logistic traits are then transformed.
The contributions come from the global `baselineOffsetX` values
(which one depends on the sex of the individual),
from individual offsets, and from cumulative mutation effects.

The contribution of 0, 1, or 2 copies of a SLiM mutation to a diploid
is either 0, 2hs, or 2s (additive) or 1, 1+hs, 1+s (multiplicative),
where h is the dominance coefficient and s is the effect size.
For a hemizygous individual, the contributions are the same (for 0 and 1 copies),
but using the hemizygous dominance coefficient for h.
For haploids, the contributions of 0 or 1 copy is either 0, 2s or 1, 1+s.
For an individual without the chromosome at all, there is no effect (obviously).

Usually, that's all we need to know.
However, there's some more complications that affect the default trait
(i.e., the trait you get - called `simT` - if you don't explicitly declare any traits),
or if you have set up a trait without baseline accumulation
and with mutations that convert to substitutions.
This can also be important to understand if you remove fixed mutations in python,
and then want to read the file back into SLiM:
to keep phenotypes the same, you have to include the effects of any removed mutations
that count towards the trait in the `baselineOffsetX` value.

Which mutations count?
Within SLiM, when a Mutation fixes, it *might* convert into a Substitution.
Whether this occurs is determined by
the mutation type's `convertToSubstitution` property,
which is True by default in WF models and False by default in nonWF models.
Substitutions are no longer used by SLiM for calculating individual traits,
and so to retain the effects of fixed mutations,
sometimes scripts will set `convertToSubstitution=F` if it was not already set.
Another mechanism to retain these effects is the Trait property `substitutionAccumulation`,
which defaults to True.
If a trait has `substitutionAccumulation=T`, then the effects of any Substitutions
are accumulated in the "substitution offset", and thus contribute to every individual's
phenotype. The net effect of this is that the contributions of any mutations that fix
will be included in the phenotype values of all individuals
as long as `convertToSubstitution=F` for their mutation type,
or `substitutionAccumulation=T` for the trait, or both.

However, the tree sequence does not record whether a given mutation was converted to
a Substitution or not, and also does not record anything about mutation types,
including their `convertToSubstitution` property.
(This is because genome structure is set up before .trees files can be loaded.)
So, when we find a fixed mutation in the tree sequence, we don't know *a priori*
if this is a Substitution or not.
If `substitutionAccumulation=T` for the trait
(information that *is* in the top-level metadata),
then it doesn't matter, mutations will contribute either way.
The problematic combination: `substitutionAccumulation=F` and `convertToSubstitution=T` 
is in most cases biologically undesireable, and only enabled for the default trait
for historical reasons.
In any case, the potentially missing information is in the script: are these mutation types
converting to substitutions or not?

Finally, a note on the `substitutionOffsetX` properties (where `X` is `M`, `F`, or `H`).
This is where SLiM stores the effect of any substitutions on the traits of
males, females, and heterozygotes, respectively.
These are distinct because of the differing effects of fixed mutations on individuals
when some individuals have different numbers of copies of a chromosome than other individuals.
