import numpy as np
import tskit

from pyslim import NUCLEOTIDES

from ._version import *  # noqa F403


def load(*args, **kwargs):
    raise RuntimeError("This method has been removed: use tskit.load( ) instead.")


def mutation_metadata(ts, check=True, ts_metadata=None):
    """
    Returns a dictionary whose keys are the numeric SLiM IDs of mutations,
    and whose values are metadata entries for those mutations.
    These SLiM IDs are found in the metadata of tskit mutations:
    for each mutation ``mut``, as ``mut.metadata["derived_states"]``.

    This is a simple extraction function that places the list of metadata entries
    stored in ``ts.metadata["SLiM_mutation_list"]`` in a dictionary
    indexed by SLiM ID. It is recommended to extract this information once
    and use the result in script, because calling this function many times
    (or, even just referring to ``ts.metadata`` many times)
    can slow down scripts considerably.

    SLiM mutation IDs may also be present in ``mut.derived_state`` as a comma-separated
    string, but accessing these from metadata is preferred because the derived
    state may be changed (by {func}`.convert_alleles`).

    :param tskit.TreeSequence ts: The tree sequence.
    :param bool check: Whether to verify that all mutations are described in top-level
        metadata.
    :param dict ts_metadata: Optionally, the top-level metadata for ``ts``. If
        this does not match the actual top-level metadata, incorrect values may result.

    :returns dict: A dictionary of metadata entries, indexed by SLiM ID
        and in sorted order by SLiM ID.
    """
    if ts_metadata is None:
        ts_metadata = ts.metadata
    # Note that dictionaries preserve insertion order
    ml = ts_metadata["SLiM_mutation_list"]
    ml.sort(key=lambda x: x["mutation_id"])
    out = {mut["mutation_id"]: mut for mut in ml}
    if check:
        ids = {j for mut in ts.mutations() for j in mut.metadata["derived_states"]}
        for k in ids:
            if k not in out:
                raise ValueError(
                    "Top-level mutation metadata is missing "
                    f"information for mutation ID {k}: "
                    "do you need to run "
                    "pyslim.add_mutation_metadata(ts)?"
                )
    return out


def mutation_at(ts, node, position, time=None):
    """
    Finds the mutation present in the genome of ``node`` at ``position``,
    returning -1 if there is no such mutation recorded in the tree
    sequence.  Warning: if ``node`` is not actually in the tree sequence
    (e.g., not ancestral to any samples) at ``position``, then this
    function will return -1, possibly erroneously.  If `time` is provided,
    returns the last mutation at ``position`` inherited by ``node`` that
    occurred at or before ``time`` ago.

    :param int node: The index of a node in the tree sequence.
    :param float position: A position along the genome.
    :param int time: The time ago that we want the nucleotide, or None,
        in which case the ``time`` of ``node`` is used.

    :returns: Index of the mutation in question, or -1 if none.
    """
    if position < 0 or position >= ts.sequence_length:
        raise ValueError("Position {} not valid.".format(position))
    if node < 0 or node >= ts.num_nodes:
        raise ValueError("Node {} not valid.".format(node))
    if time is None:
        time = ts.node(node).time
    tree = ts.at(position)
    out = tskit.NULL
    try:
        site = ts.site(position=position)
    except ValueError:
        pass
    else:
        mut_nodes = []
        # look for only mutations that occurred before `time`
        # not strictly necessary if time was None
        for mut in site.mutations:
            if mut.time >= time:
                mut_nodes.append(mut.node)
        n = node
        while n > -1 and n not in mut_nodes:
            n = tree.parent(n)
        if n >= 0:
            # do careful error checking here
            for mut in site.mutations:
                if mut.node == n and mut.time >= time:
                    # BUG: this can fail if a mutation has two children
                    assert out == tskit.NULL or out == mut.parent
                    out = mut.id
    return out


def nucleotide_at(ts, node, position, time=None, mut_metadata=None):
    """
    Finds the nucleotide present in the genome of ``node`` at ``position``.
    Warning: if ``node`` is not actually in the tree sequence (e.g., not
    ancestral to any samples) at ``position``, then this function will
    return the reference sequence nucleotide, possibly erroneously.  If
    `time` is provided, returns the last nucletide produced by a mutation
    at ``position`` inherited by ``node`` that occurred at or before
    ``time`` ago.

    This method uses a dictionary of mutation metadata, computed by
    :meth:`mut_metadata`. This step can be expensive if there are
    many mutations, so this can be pre-computed and passed in as
    ``mutations``. If not provided, it will be computed.

    :param int node: The index of a node in the tree sequence.
    :param float position: A position along the genome.
    :param int time: The time ago that we want the nucleotide, or None,
        in which case the ``time`` of ``node`` is used.
    :param dict mut_metadata: If provided, a dictionary mapping
        mutation ID to metadata, as returned by ``pyslim.mutation_metadata(ts)``.

    :returns: Index of the nucleotide in ``NUCLEOTIDES`` (0=A, 1=C, 2=G, 3=T).
    """
    if not ts.has_reference_sequence():
        raise ValueError("This tree sequence has no reference sequence.")
    if mut_metadata is None:
        mut_metadata = mutation_metadata(ts)
    mut_id = mutation_at(ts, node, position, time)
    if mut_id == tskit.NULL:
        out = NUCLEOTIDES.index(ts.reference_sequence.data[int(position)])
    else:
        mut = ts.mutation(mut_id)
        _, k = max(
            [(mut_metadata[j]["slim_time"], j) for j in mut.metadata["derived_states"]]
        )
        out = mut_metadata[k]["nucleotide"]
    return out
