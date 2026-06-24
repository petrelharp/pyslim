"""
Test cases for tree sequences.
"""

import copy
import json
import random
import sys

import msprime
import numpy as np
import pytest
import tskit

import pyslim
import tests

from .recipe_specs import recipe_eq, restarted_recipe_eq


def run_with_ts_metadata(f, ts_metadata, *args, **kwargs):
    # check for equality in a method with and without passing in
    # the ts_metadata argument
    a = f(*args, **kwargs)
    new_kwargs = copy.deepcopy(kwargs)
    new_kwargs["ts_metadata"] = ts_metadata
    b = f(*args, **new_kwargs)
    assert len(a) == len(b)
    np.testing.assert_equal(a, b)
    return a


def mutations_above(ts, node, pos):
    for s in ts.sites():
        if s.position == pos:
            tree = ts.at(pos)
            site_muts = s.mutations.copy()
            # parent mutations come before children
            site_muts.reverse()
            while node != tskit.NULL:
                for mut in site_muts:
                    if mut.node == node:
                        yield mut.id
                node = tree.parent(node)
    yield -1


def naive_mutation_at(ts, node, pos, time=None):
    if time is None:
        time = ts.node(node).time
    assert ts.node(node).time <= time
    for mut_id in mutations_above(ts, node, pos):
        if mut_id >= 0 and ts.mutation(mut_id).time >= time:
            break
    return mut_id


def verify_mutation_metadata(ts):
    # Verify that all derived states are properly accounted for
    # in mutation metadata.
    mdl = ts.metadata["SLiM_mutation_list"]
    mut_info = {str(mut["mutation_id"]): mut for mut in mdl}
    assert len(mut_info) == len(mdl)
    for mut in ts.mutations():
        for j in mut.derived_state.split(","):
            assert j in mut_info


class TestMutationMetadata(tests.PyslimTestCase):
    @pytest.mark.parametrize("recipe", recipe_eq("multichrom"), indirect=True)
    def test_mutation_IDs_unique(self, recipe):
        ids = set()
        for _, ts in recipe["ts"].items():
            mut_info = pyslim.mutation_metadata(ts)
            new_ids = set(mut_info.keys())
            assert len(ids.intersection(new_ids)) == 0
            ids = ids.union(new_ids)

    def test_mutation_metadata(self, recipe):
        # test that mutation metadata is properly present
        for _, ts in recipe["ts"].items():
            verify_mutation_metadata(ts)

    @pytest.mark.parametrize("recipe", [next(recipe_eq())], indirect=True)
    def test_check(self, recipe):
        for _, ts in recipe["ts"].items():
            assert ts.num_mutations > 5
            t = ts.dump_tables()
            md = t.metadata
            del md["SLiM_mutation_list"][5:]
            t.metadata = md
            ts = t.tree_sequence()
            mut_info = pyslim.mutation_metadata(ts, check=False)
            assert len(mut_info) == 5
            with pytest.raises(ValueError, match="missing information for mutation"):
                _ = pyslim.mutation_metadata(ts)


class TestSlimTime(tests.PyslimTestCase):
    # Tests for slim_time()

    @pytest.mark.parametrize(
        "recipe", recipe_eq(exclude=["long", "old_mutations"]), indirect=True
    )
    def test_slim_time(self, recipe):
        for _, ts in recipe["ts"].items():
            muts = pyslim.mutation_metadata(ts)
            # the mutations in "init_mutated" examples have mutations that are *added*
            # in *early*, and so their times match in that stage.
            stage = "early" if "init_mutated" in recipe else None
            slim_times = pyslim.slim_time(ts, ts.mutations_time, stage=stage)
            for t, mut in zip(slim_times, ts.mutations()):
                mut_time = max(
                    [muts[int(j)]["slim_time"] for j in mut.derived_state.split(",")]
                )
                assert mut_time == t


class TestNextMutationID(tests.PyslimTestCase):
    """
    Tests for the function that returns the largest SLiM mutation ID.
    """

    def test_next_id(self, recipe):
        for _, ts in recipe["ts"].items():
            mt_ids_str = ",".join(
                tskit.unpack_strings(
                    ts.tables.mutations.derived_state,
                    ts.tables.mutations.derived_state_offset,
                )
            )
            mt_ids = [int(i or 0) for i in mt_ids_str.split(",")]
            max_mt_id = max(mt_ids)
            assert max_mt_id + 1 == pyslim.next_slim_mutation_id(ts)

    @pytest.mark.skipif(sys.platform == "win32", reason="Issue #412")
    @pytest.mark.parametrize(
        "recipe",
        recipe_eq("adds_mutations", exclude="multichrom"),  # <-- TODO
        indirect=True,
    )
    def test_reload_slim(self, recipe, helper_functions, tmp_path):
        recapped = {}
        for chrom, ts in recipe["ts"].items():
            rts = self.do_recapitate(
                ts,
                recombination_rate=1e-8,
                ancestral_Ne=100,
                random_seed=875,
                keep_vacant=True,
            )
            next_id = pyslim.next_slim_mutation_id(rts)
            T = max(1, rts.segregating_sites(mode="branch", span_normalise=False))
            mts = pyslim.add_mutation_metadata(
                msprime.sim_mutations(
                    rts,
                    rate=max(6e-4, 10 / T),
                    keep=True,
                    model=msprime.SLiMMutationModel(type=1, next_id=next_id),
                    random_seed=135,
                ),
                mutation_type=1,
            )
            assert mts.num_mutations > rts.num_mutations
            recapped[chrom] = mts
        multichrom = "multichrom" in recipe
        if multichrom:
            slimfile = "restart_nucleotides_WF_chromosomes.slim"
        else:
            chrom_type = ts.metadata["SLiM"]["this_chromosome"]["type"]
            slimfile = "restart_nucleotides_WF.slim"
        rrts = helper_functions.run_slim_restart(
            recapped,
            slimfile,
            tmp_path,
            multichrom,
            WF=False,
            CHROM_TYPE=chrom_type,
        )
        for chrom, ts in rrts.items():
            # nothing should change
            assert chrom in recipe["ts"]
            assert pyslim.next_slim_mutation_id(mts) == pyslim.next_slim_mutation_id(ts)
            assert ts.num_mutations == recapped[chrom].num_mutations
            ots = recipe["ts"][chrom]
            assert ts.metadata["SLiM"] == ots.metadata["SLiM"]
            mut_info = pyslim.mutation_metadata(ts)
            omut_info = pyslim.mutation_metadata(ots)
            assert len(mut_info) == len(ts.metadata["SLiM_mutation_list"])
            assert len(omut_info) == len(ots.metadata["SLiM_mutation_list"])
            # we've added new mutations but originals should all be there
            for k in omut_info:
                assert k in mut_info
                assert omut_info[k] == mut_info[k]

    def test_invalid_derived_state(self):
        ts = msprime.sim_ancestry(
            4,
            sequence_length=10,
            population_size=10,
            random_seed=10,
        )
        mts = msprime.sim_mutations(
            ts,
            model="jc69",
            rate=0.5,
            random_seed=23,
        )
        with pytest.raises(ValueError, match="values coercible to int"):
            pyslim.next_slim_mutation_id(mts)


class TestRecapitate(tests.PyslimTestCase):
    """
    Tests for recapitation.
    """

    def check_recap_consistency(self, ts, recap, with_ancestral_Ne=True):
        tsmd = ts.metadata
        remd = recap.metadata
        assert tsmd["SLiM"]["tick"] == remd["SLiM"]["tick"]
        assert tsmd["SLiM"]["cycle"] == remd["SLiM"]["cycle"]
        assert tsmd["SLiM"]["stage"] == remd["SLiM"]["stage"]
        assert tsmd["SLiM"]["name"] == remd["SLiM"]["name"]
        assert all(tree.num_roots == 1 for tree in recap.trees())
        assert ts.has_reference_sequence() == recap.has_reference_sequence()
        if ts.has_reference_sequence():
            assert ts.reference_sequence.data == recap.reference_sequence.data

        root_times = list(
            set([ts.node(n).time for t in ts.trees(root_threshold=2) for n in t.roots])
        )
        assert len(root_times) == 1

        if with_ancestral_Ne:
            # check that time recorded in provenance is correct
            assert recap.num_provenances == 2
            recap_prov = json.loads(recap.provenance(1).record)
            recap_events = recap_prov["parameters"]["demography"]["events"]
            assert len(recap_events) == 1
            recap_time = recap_events[0]["time"]
            assert np.allclose(recap_time, root_times[0])
            # the oldest nodes in all trees with SLiM provenance should be at that time
            if recap.num_nodes <= 500:  # takes a long time otherwise
                for t in recap.trees():
                    for n in t.nodes():
                        rn = recap.node(n)
                        if rn.metadata is not None:
                            p = t.parent(n)
                            if p == tskit.NULL or recap.node(p).metadata is None:
                                assert rn.time == root_times[0]

        ts_samples = list(ts.samples())
        for u in recap.samples():
            n1 = recap.node(u)
            assert n1.individual >= 0
            i1 = recap.individual(n1.individual)
            remembered = (pyslim.INDIVIDUAL_REMEMBERED & i1.flags) > 0
            retained = (pyslim.INDIVIDUAL_RETAINED & i1.flags) > 0
            alive = (pyslim.INDIVIDUAL_ALIVE & i1.flags) > 0
            assert alive or remembered or retained
            assert u in ts_samples
            n2 = ts.node(u)
            assert n1.time == n2.time
            assert n1.individual == n2.individual
            assert n1.flags == n2.flags
            assert n1.metadata == n2.metadata
            assert n1.population == n2.population
        assert ts.num_populations <= recap.num_populations
        for k in range(ts.num_populations):
            p1 = ts.population(k)
            p2 = recap.population(k)
            assert p1.metadata == p2.metadata
        # find ancestral pop in which recapitation happens
        tables = ts.tables
        # note: this next one assumes the tick hasn't been set to something other than 1
        anc_nodes = np.where(tables.nodes.time > ts.metadata["SLiM"]["tick"])[0]
        if len(anc_nodes) > 0:
            for pop in ts.populations():
                if pop.metadata is not None and pop.metadata["name"] == "ancestral":
                    break
            assert pop.metadata["name"] == "ancestral"
            assert np.all(tables.nodes.population[anc_nodes] == pop.id)

    # Just test on the first recipe
    @pytest.mark.parametrize("recipe", [next(recipe_eq())], indirect=True)
    def test_recapitate_errors(self, recipe):
        ts = list(recipe["ts"].values())[0]
        with pytest.raises(
            ValueError, match="cannot specify both `demography` and `ancestral_Ne`"
        ):
            _ = self.do_recapitate(
                ts,
                recombination_rate=0.0,
                demography=msprime.Demography.from_tree_sequence(ts),
                ancestral_Ne=10,
                random_seed=123,
            )

    def test_root_mismatch_error(self):
        ts = msprime.sim_ancestry(
            4, sequence_length=10, random_seed=12, recombination_rate=0.1
        )
        assert ts.num_trees > 1
        recap_time = 100
        assert ts.node(ts.first().roots[0]).time < recap_time
        ts = pyslim.annotate(ts, model_type="nonWF", tick=recap_time)
        with pytest.raises(ValueError, match="Not all roots"):
            rts = self.do_recapitate(ts, ancestral_Ne=10)

    def test_unique_names(self):
        ts = msprime.sim_ancestry(4, sequence_length=10, random_seed=12, end_time=1.0)
        # adjust seed if not
        assert min([t.num_roots for t in ts.trees()]) > 1
        ts = pyslim.annotate(ts, model_type="nonWF", tick=1)
        t = ts.dump_tables()
        md = t.populations[0].metadata
        md.update({"name": "ancestral"})
        t.populations[0] = t.populations[0].replace(metadata=md)
        md.update({"name": "ancestral_ancestral", "slim_id": 1})
        t.populations.add_row(metadata=md)
        ts = t.tree_sequence()
        rts = self.do_recapitate(ts, ancestral_Ne=10)
        names = [pop.metadata["name"] for pop in rts.populations()]
        assert len(set(names)) == len(names)
        assert names[0] == "ancestral"
        assert names[-2] == "ancestral_ancestral"

    @pytest.mark.parametrize("recipe", recipe_eq(exclude="no_simplify"), indirect=True)
    def test_recapitation(self, recipe):
        for _, ts in recipe["ts"].items():
            recomb_rate = 1.0 / ts.sequence_length
            recap = self.do_recapitate(
                ts, recombination_rate=recomb_rate, ancestral_Ne=10, random_seed=5
            )
            # there should be no new mutations
            assert ts.num_mutations == recap.num_mutations
            assert ts.num_sites == recap.num_sites
            assert list(ts.tables.sites.position) == list(recap.tables.sites.position)
            self.check_recap_consistency(ts, recap)

            if ts.metadata["SLiM"]["tick"] < 200:
                old_root_time = np.max(ts.tables.nodes.time)
                for t in recap.trees():
                    assert t.num_roots == 1
                    assert recap.node(t.root).time >= old_root_time

    @pytest.mark.parametrize(
        "recipe", recipe_eq(exclude=["long", "no_simplify"]), indirect=True
    )
    def test_with_recomb_map(self, recipe):
        for _, ts in recipe["ts"].items():
            recomb_rate = 1.0 / ts.sequence_length
            recombination_map = msprime.RateMap(
                position=[0.0, ts.sequence_length], rate=[recomb_rate]
            )
            recap = self.do_recapitate(
                ts,
                recombination_rate=recombination_map,
                ancestral_Ne=1e-6,
                random_seed=875,
            )
            self.check_recap_consistency(ts, recap)

    @pytest.mark.parametrize("recipe", recipe_eq("multipop"), indirect=True)
    def test_with_demography(self, recipe):
        for _, ts in recipe["ts"].items():
            recomb_rate = 1.0 / ts.sequence_length
            demography = msprime.Demography.from_tree_sequence(ts)
            for pop in demography.populations:
                pop.initial_size = 100.0
            demography.add_population(
                initial_size=10,
                name="ancestral",
                extra_metadata={"slim_id": ts.num_populations},
            )
            demography.add_population_split(
                time=ts.metadata["SLiM"]["tick"]
                + 20.0,  # assumes tick hasn't been changed
                derived=[
                    p.name for p in demography.populations if p.name != "ancestral"
                ],
                ancestral="ancestral",
            )
            recap = self.do_recapitate(
                ts,
                demography=demography,
                recombination_rate=recomb_rate,
                random_seed=333,
            )
            self.check_recap_consistency(ts, recap, with_ancestral_Ne=False)

    @pytest.mark.parametrize(
        "recipe", recipe_eq(exclude=["long", "starts_later"]), indirect=True
    )
    def test_first_gen_nodes(self, recipe):
        # check that all the roots of the trees are present
        # (note this will fail if some populations were started at different
        # times than others or if the tick has been changed)
        for _, ts in recipe["ts"].items():
            tsmd = ts.metadata
            root_time = tsmd["SLiM"]["tick"]
            is_wf = tsmd["SLiM"]["model_type"] == "WF"
            remembered_stage = tsmd["SLiM"]["stage"]
            if (not is_wf) or (remembered_stage != "late"):
                root_time -= 1
            if (not is_wf) and ("begun_first" in recipe):
                root_time += 1
            if (not is_wf) and ("remembered_first" in recipe):
                root_time -= 1
            if is_wf and ("begun_late" in recipe):
                root_time -= 1
            vacant = pyslim.nodes_vacant(ts)
            for t in ts.trees():
                for u in t.roots:
                    assert vacant[u] or ts.node(u).time == root_time


class TestIndividualAges(tests.PyslimTestCase):
    # tests for individuals_alive_at and individual_ages_at

    @pytest.mark.parametrize("recipe", [next(recipe_eq("everyone"))], indirect=True)
    def test_errors(self, recipe):
        for _, ts in recipe["ts"].items():
            for stage in ["abcd", 10, []]:
                with pytest.raises(ValueError):
                    pyslim.individuals_alive_at(ts, 0, stage=stage)
                with pytest.raises(ValueError):
                    pyslim.individuals_alive_at(ts, 0, remembered_stage=stage)
                with pytest.raises(ValueError):
                    pyslim.individual_ages_at(ts, 0, stage=stage)

    @pytest.mark.parametrize(
        "recipe", [next(recipe_eq("pedigree", "WF"))], indirect=True
    )
    def test_mismatched_remembered_stage(self, recipe):
        for _, ts in recipe["ts"].items():
            if "remembered_early" in recipe:
                with pytest.warns(UserWarning):
                    pyslim.individuals_alive_at(ts, 0, remembered_stage="late")
            else:
                with pytest.warns(UserWarning):
                    pyslim.individuals_alive_at(ts, 0, remembered_stage="early")

    @pytest.mark.parametrize(
        "recipe", recipe_eq("multipop", exclude="remembered_early"), indirect=True
    )
    def test_population(self, recipe):
        for _, ts in recipe["ts"].items():
            individual_populations = ts.individuals_population
            ts_metadata = ts.metadata
            all_inds = run_with_ts_metadata(
                pyslim.individuals_alive_at, ts_metadata, ts, 0
            )
            assert len(all_inds) > 0
            for p in range(ts.num_populations):
                sub_inds = pyslim.individuals_alive_at(
                    ts, 0, population=p, ts_metadata=ts_metadata
                )
                assert set(sub_inds) == set(all_inds[individual_populations == p])
                sub_inds = pyslim.individuals_alive_at(
                    ts, 0, population=[p], ts_metadata=ts_metadata
                )
                assert set(sub_inds) == set(all_inds[individual_populations == p])
            sub_inds = pyslim.individuals_alive_at(
                ts, 0, population=np.arange(p), ts_metadata=ts_metadata
            )
            assert set(sub_inds) == set(all_inds[individual_populations != p])

    @pytest.mark.parametrize(
        "recipe", recipe_eq("nonWF", exclude="remembered_early"), indirect=True
    )
    def test_samples_only(self, recipe):
        for _, ts in recipe["ts"].items():
            ts_metadata = ts.metadata
            all_inds = run_with_ts_metadata(
                pyslim.individuals_alive_at, ts_metadata, ts, 0
            )
            assert set(all_inds) == set(
                pyslim.individuals_alive_at(
                    ts, 0, samples_only=False, ts_metadata=ts_metadata
                )
            )
            sub_inds = np.random.choice(
                all_inds, size=min(len(all_inds), 4), replace=False
            )
            flags = np.array(
                [
                    n.flags & (tskit.NODE_IS_SAMPLE * n.individual in sub_inds)
                    for n in ts.nodes()
                ],
                dtype=np.uint32,
            )
            tables = ts.dump_tables()
            tables.nodes.flags = flags
            new_ts = tables.tree_sequence()
            assert set(sub_inds) == set(
                pyslim.individuals_alive_at(new_ts, 0, samples_only=True)
            )

    @pytest.mark.parametrize(
        "recipe", recipe_eq(exclude=("remembered_early", "long")), indirect=True
    )
    def test_after_simplify(self, recipe):
        for _, ts in recipe["ts"].items():
            sts = ts.simplify()
            orig_inds = pyslim.individuals_alive_at(ts, 0)
            simp_inds = pyslim.individuals_alive_at(sts, 0)
            odict = {ts.individual(i).metadata["pedigree_id"]: i for i in orig_inds}
            sdict = {sts.individual(i).metadata["pedigree_id"]: i for i in simp_inds}
            for slim_id in odict:
                i = odict[slim_id]
                ind = ts.individual(i)
                n = ts.node(ind.nodes[0])
                if n.flags & tskit.NODE_IS_SAMPLE:
                    assert slim_id in sdict

    @pytest.mark.parametrize("recipe", recipe_eq("pedigree"), indirect=True)
    def test_ages(self, recipe):
        for _, ts in recipe["ts"].items():
            ts_metadata = ts.metadata
            info = recipe["info"]
            remembered_stage = "late"
            if "remembered_first" in recipe:
                remembered_stage = "first"
            elif "remembered_early" in recipe:
                remembered_stage = "early"
            assert remembered_stage == ts_metadata["SLiM"]["stage"]
            max_time_ago = ts_metadata["SLiM"]["tick"]
            if remembered_stage in ("first", "early"):
                max_time_ago -= 1
            for time in range(0, max_time_ago):
                slim_tick = ts_metadata["SLiM"]["tick"] - time
                check_stages = ("first", "early", "late")
                if time == 0:
                    if remembered_stage == "first":
                        # if we remember in first we don't know who's still there
                        # in later stages of that time step
                        check_stages = ("first",)
                    elif remembered_stage == "early":
                        check_stages = ("first", "early")
                if time == max_time_ago:
                    if remembered_stage == "early":
                        # if we set up the population in early there aren't individuals in first
                        # of the very first time step
                        check_stages = ("early", "late")
                    elif remembered_stage == "late":
                        # similarly for late
                        check_stages = ("late",)
                for stage in check_stages:
                    alive = pyslim.individuals_alive_at(
                        ts,
                        time,
                        stage=stage,
                        remembered_stage=remembered_stage,
                        ts_metadata=ts_metadata,
                    )
                    ages = pyslim.individual_ages_at(
                        ts,
                        time,
                        stage=stage,
                        remembered_stage=remembered_stage,
                        ts_metadata=ts_metadata,
                    )
                    for ind in ts.individuals():
                        ind_time = ts.node(ind.nodes[0]).time
                        # bad things can happen for the very first individuals
                        # depending on when the subpops are created
                        if ("everyone" in recipe or ind_time == 0) and (
                            remembered_stage == "early" or ind_time < max_time_ago
                        ):
                            slim_id = ind.metadata["pedigree_id"]
                            assert slim_id in info
                            slim_alive = (slim_tick, stage) in info[slim_id]["age"]
                            pyslim_alive = ind.id in alive
                            assert slim_alive == pyslim_alive
                            if slim_alive:
                                slim_age = info[slim_id]["age"][(slim_tick, stage)]
                                if ts_metadata["SLiM"]["model_type"] == "WF":
                                    # SLiM records -1 but we return 0 in late and 1 in early
                                    slim_age = 0 + (stage in ("first", "early"))
                                assert ages[ind.id] == slim_age
                            else:
                                assert np.isnan(ages[ind.id])


class TestHasIndividualParents(tests.PyslimTestCase):
    def verify_has_parents(self, ts):
        right_answer = np.repeat(True, ts.num_individuals)
        node_indivs = ts.tables.nodes.individual
        parent_ids = [set() for _ in ts.individuals()]
        node_parent_ids = [set() for _ in ts.nodes()]
        individual_times = ts.individuals_time
        individual_ages = pyslim.individual_ages(ts)
        for t in ts.trees():
            for i in ts.individuals():
                if len(i.nodes) != 2:
                    right_answer[i.id] = False
                for n in i.nodes:
                    pn = t.parent(n)
                    if pn == tskit.NULL:
                        right_answer[i.id] = False
                    else:
                        p = node_indivs[t.parent(n)]
                        if p == tskit.NULL:
                            right_answer[i.id] = False
                        else:
                            ptime = individual_times[p]
                            parent_alive = True
                            if ts.metadata["SLiM"]["model_type"] == "WF":
                                if individual_times[i.id] + 1 != ptime:
                                    parent_alive = False
                            else:
                                pdeath = ptime - individual_ages[p]
                                if individual_times[i.id] + 1 < pdeath:
                                    parent_alive = False
                            if not parent_alive:
                                right_answer[i.id] = False
                            else:
                                parent_ids[i.id].add(p)
                                node_parent_ids[n].add(p)
        for j, p in enumerate(parent_ids):
            if len(p) == 0:
                right_answer[j] = False
        for j, p in enumerate(node_parent_ids):
            if len(p) != 1:
                ind = ts.node(j).individual
                if ind != tskit.NULL:
                    right_answer[ts.node(j).individual] = False
        right_parents = []
        for j, p in enumerate(parent_ids):
            if right_answer[j]:
                for pp in p:
                    right_parents.append([pp, j])
        with pytest.warns(FutureWarning):
            has_parents = pyslim.has_individual_parents(ts)
        right_parents = np.sort(np.array(right_parents), axis=0)
        with pytest.warns(FutureWarning):
            parents = np.sort(pyslim.individual_parents(ts), axis=0)
        assert np.array_equal(right_answer, has_parents)
        assert np.array_equal(right_parents, parents)

    def get_first_gen(self, ts):
        nodes = ts.tables.nodes
        root_time = np.max(nodes.time)
        first_gen = set(nodes.individual[nodes.time == root_time])
        first_gen.discard(tskit.NULL)
        return np.array(list(first_gen), dtype="int")

    @pytest.mark.parametrize("recipe", recipe_eq("everyone"), indirect=True)
    def test_everyone(self, recipe):
        # since everyone is recorded, only the initial individuals should
        # not have parents
        for _, ts in recipe["ts"].items():
            right_answer = np.repeat(True, ts.num_individuals)
            first_gen = self.get_first_gen(ts)
            assert len(first_gen) > 0
            right_answer[first_gen] = False
            with pytest.warns(FutureWarning):
                has_parents = pyslim.has_individual_parents(ts)
            assert np.array_equal(right_answer, has_parents)
            self.verify_has_parents(ts)

    @pytest.mark.parametrize("recipe", recipe_eq("everyone"), indirect=True)
    def test_post_recap(self, recipe):
        # the same should be true after recapitation
        for _, ts in recipe["ts"].items():
            right_answer = np.repeat(True, ts.num_individuals)
            first_gen = self.get_first_gen(ts)
            right_answer[first_gen] = False
            assert ts.num_populations <= 2
            ts = self.do_recapitate(
                ts, recombination_rate=0.01, ancestral_Ne=10, random_seed=11
            )
            with pytest.warns(FutureWarning):
                has_parents = pyslim.has_individual_parents(ts)
            assert np.array_equal(right_answer, has_parents)
            self.verify_has_parents(ts)

    @pytest.mark.parametrize("recipe", recipe_eq("everyone"), indirect=True)
    def test_post_simplify(self, recipe):
        for _, ts in recipe["ts"].items():
            rng = np.random.default_rng(seed=3)
            individual_times = ts.individuals_time
            md_tick = ts.metadata["SLiM"]["tick"]
            keep_indivs = rng.choice(
                # assumes tick hasn't been changed
                np.where(individual_times < md_tick - 1)[0],
                size=30,
                replace=False,
            )
            keep_nodes = []
            for i in keep_indivs:
                keep_nodes.extend(ts.individual(i).nodes)
            ts = ts.simplify(
                samples=keep_nodes, filter_individuals=True, keep_input_roots=True
            )
            assert ts.num_populations <= 2
            ts = self.do_recapitate(ts, recombination_rate=0.01, ancestral_Ne=10)
            with pytest.warns(FutureWarning):
                has_parents = pyslim.has_individual_parents(ts)
            assert sum(has_parents) > 0
            self.verify_has_parents(ts)

    @pytest.mark.parametrize("recipe", recipe_eq("everyone"), indirect=True)
    def test_pedigree_parents_everyone(self, recipe):
        # We can only guarantee to correctly reconstruct parents when everyone is remembered:
        # for instance, A selfs to produce B who selfs to produce C; if A and C are present
        # but B is not, and A is still alive, we will think that A is C's parent.
        # Or, suppose that X is the parent of Y, but we did not remember X at or after
        # the time that Y was born, so that although X is alive at Y's birth, we don't know it,
        # and so the parentage would not be reported by `individual_parents()`.
        for _, ts in recipe["ts"].items():
            info = recipe["info"]
            with pytest.warns(FutureWarning):
                has_parents = pyslim.has_individual_parents(ts)
            with pytest.warns(FutureWarning):
                parents = pyslim.individual_parents(ts)
            slim_map = {}
            for ind in ts.individuals():
                slim_map[ind.metadata["pedigree_id"]] = ind.id
            ts_to_slim = {sid: [] for sid in slim_map}
            for pa, ch in parents:
                assert pa >= 0 and pa < ts.num_individuals
                assert ch >= 0 and pa < ts.num_individuals
                pa_ind = ts.individual(pa).metadata["pedigree_id"]
                ch_ind = ts.individual(ch).metadata["pedigree_id"]
                ts_to_slim[ch_ind].append(pa_ind)
            for hasp, ind in zip(has_parents, ts.individuals()):
                for n in ind.nodes:
                    assert ts.node(n).is_sample()
                assert len(ind.nodes) == 2
                sid = ind.metadata["pedigree_id"]
                ts_p = ts_to_slim[sid]
                assert hasp == (len(ts_p) > 0)
                # parents, as recorded by SLiM, that we know about:
                slim_p = [x for x in info[sid]["parents"] if x in slim_map]
                assert set(slim_p) == set(ts_p)

    # TODO: excluding multichrom because there's a great-grandparent who makes this
    # test fail in recipe_all_the_chromosome_types.slim
    @pytest.mark.parametrize(
        "recipe", recipe_eq("pedigree", exclude="multichrom"), indirect=True
    )
    def test_pedigree_parents(self, recipe):
        # Less strict test for consistency only: see caveats above in test_pedigree_parents_everyone.
        # In particular, we are only guaranteed to have whole genomes for ALIVE
        # or REMEMBERED individuals (not RETAINED), and the same for parental genomes.
        info = recipe["info"]
        for _, ts in recipe["ts"].items():
            with pytest.warns(FutureWarning):
                has_parents = pyslim.has_individual_parents(ts)
            with pytest.warns(FutureWarning):
                parents = pyslim.individual_parents(ts)
            slim_map = {}
            for ind in ts.individuals():
                slim_map[ind.metadata["pedigree_id"]] = ind.id
            ts_to_slim = {sid: [] for sid in slim_map}
            for pa, ch in parents:
                assert pa >= 0 and pa < ts.num_individuals
                assert ch >= 0 and pa < ts.num_individuals
                pa_ind = ts.individual(pa).metadata["pedigree_id"]
                ch_ind = ts.individual(ch).metadata["pedigree_id"]
                ts_to_slim[ch_ind].append(pa_ind)
            for hasp, ind in zip(has_parents, ts.individuals()):
                all_there = (
                    ind.flags & (pyslim.INDIVIDUAL_ALIVE | pyslim.INDIVIDUAL_REMEMBERED)
                    > 0
                )
                if all_there:
                    for n in ind.nodes:
                        assert ts.node(n).is_sample()
                    assert len(ind.nodes) == 2
                sid = ind.metadata["pedigree_id"]
                ts_p = ts_to_slim[sid]
                assert hasp == (len(ts_p) > 0)
                # parents, as recorded by SLiM, that we know about:
                slim_p = [x for x in info[sid]["parents"] if x in slim_map]
                # all pyslim parents should be legit
                # (so set(ts_p) - set(slim_p) should usually be empty)
                # BUT sometimes we can mistake a (great)^n-grandparent for a parent
                gfolks = []
                for a in set(info[sid]["parents"]) - set(ts_p):
                    gfolks.extend(info[a]["parents"])
                # print("=====  ", hasp, ind)
                # print("sid: ", sid, "ts_p: ", ts_p)
                # print("slim_p: ", slim_p)
                # print(gfolks)
                for a in set(ts_p) - set(slim_p):
                    assert a in gfolks


class TestMutationConsistency(tests.PyslimTestCase):
    """
    Test for consistency between what SLiM has written down in top-level metadata
    and what's in the tree sequence
    """

    @pytest.mark.parametrize("recipe", recipe_eq("refseq"), indirect=True)
    def test_reference_sequence_consistency(self, recipe):
        for n, ts in recipe["ts"].items():
            tsmd = ts.metadata
            chrom_id = tsmd["SLiM"]["this_chromosome"]["id"]
            assert ts.has_reference_sequence()
            ref = list(
                tsmd["SLiM"]["user_metadata"]["reference_sequence"][0][f"chr{chrom_id}"][
                    0
                ]
            )
            ts_ref = ts.reference_sequence.data
            subs = [
                x[0]
                for x in tsmd["SLiM"]["user_metadata"]["substitutions"][0].values()
                if x[0]["chromosome_id"][0] == chrom_id
            ]
            subs.sort(key=lambda x: (x["position"][0], x["fixationTick"][0]))
            for s in subs:
                nuc = s["nucleotide"][0]
                if nuc != "N":
                    ref[s["position"][0]] = nuc
            assert ts_ref == "".join(ref)

    @pytest.mark.parametrize("recipe", recipe_eq("record_mutations"), indirect=True)
    def test_mutation_consistency(self, recipe):
        for n, ts in recipe["ts"].items():
            tsmd = ts.metadata
            chrom_id = tsmd["SLiM"]["this_chromosome"]["id"]
            # this is just making these things not lists, mostly
            debug_info = {
                int(k): {x: y[0] for x, y in v[0].items()}
                for k, v in tsmd["SLiM"]["user_metadata"]["mutations"][0].items()
            }
            mut_info = pyslim.mutation_metadata(ts)
            for mut in ts.mutations():
                for k in mut.derived_state.split(","):
                    k = int(k)
                    assert k in debug_info or mut_info[k]["mutation_id"] == 2
                    assert k in mut_info
                    assert debug_info[k]["chromosome_id"] == chrom_id
                    assert debug_info[k]["position"] == ts.site(mut.site).position
                    assert debug_info[k]["mutationType"] == mut_info[k]["mutation_type"]
                    assert debug_info[k]["originTick"] == mut_info[k]["slim_time"]


class TestReferenceSequence(tests.PyslimTestCase):
    """
    Test for operations involving the reference sequence
    """

    @pytest.mark.parametrize("recipe", recipe_eq(exclude="long"), indirect=True)
    def test_reference_sequence(self, recipe):
        for _, ts in recipe["ts"].items():
            if ts.num_mutations > 0:
                mut_md = ts.mutation(0).metadata
                has_nucleotides = "nucleotides" in recipe
                if not has_nucleotides:
                    assert not ts.has_reference_sequence()
                else:
                    assert type(ts.reference_sequence.data) == str
                    assert len(ts.reference_sequence.data) == ts.sequence_length
                    for u in ts.reference_sequence.data:
                        assert u in pyslim.NUCLEOTIDES
                sts = ts.simplify(ts.samples()[:2])
                assert sts.has_reference_sequence() == ts.has_reference_sequence()
                if sts.has_reference_sequence():
                    assert sts.reference_sequence.data == ts.reference_sequence.data

    def test_mutation_at_errors(self, recipe):
        for _, ts in recipe["ts"].items():
            u = ts.samples()[0]
            with pytest.raises(ValueError):
                pyslim.mutation_at(ts, -2, 3)
            with pytest.raises(ValueError):
                pyslim.mutation_at(ts, u, -3)
            with pytest.raises(ValueError):
                pyslim.mutation_at(ts, ts.num_nodes + 2, 3)
            with pytest.raises(ValueError):
                pyslim.mutation_at(ts, u, ts.sequence_length)

    def test_nucleotide_at_errors(self, recipe):
        for _, ts in recipe["ts"].items():
            u = ts.samples()[0]
            if ts.num_mutations > 0:
                mut_md = ts.mutation(0).metadata
                if not ts.has_reference_sequence():
                    with pytest.raises(ValueError, match="has no reference seq"):
                        pyslim.nucleotide_at(ts, u, 3)

    def test_mutation_at(self, recipe):
        rng = random.Random(42)
        for _, ts in recipe["ts"].items():
            L = int(min(50000, ts.sequence_length))
            for _ in range(min(10, ts.num_sites)):
                site = rng.choice(ts.sites())
                pos = site.position
                tree = ts.at(pos)
                for _ in range(10):
                    node = rng.randint(0, ts.num_nodes - 1)
                    parent = tree.parent(node)
                    if parent != tskit.NULL:
                        b = pyslim.mutation_at(ts, parent, pos)
                        c = pyslim.mutation_at(ts, node, pos, ts.node(parent).time)
                        assert b == c
                    rtime = max([ts.node(r).time for r in tree.roots])
                    ut = ts.node(node).time + (rtime - ts.node(node).time) * rng.random()
                    for time in [None, ts.node(node).time, ut]:
                        a = pyslim.mutation_at(ts, node, pos, time=time)
                        b = naive_mutation_at(ts, node, pos, time=time)
                        assert a == b
            for _ in range(min(10, int(L - ts.num_sites))):
                pos = rng.choice(list(set(range(L)) - set(ts.sites_position)))
                tree = ts.at(pos)
                for _ in range(10):
                    node = rng.randint(0, ts.num_nodes - 1)
                    rtime = max([ts.node(r).time for r in tree.roots])
                    ut = ts.node(node).time + (rtime - ts.node(node).time) * rng.random()
                    for time in [None, ts.node(node).time, ut]:
                        assert naive_mutation_at(ts, node, pos, time=time) == -1

    @pytest.mark.parametrize("recipe", recipe_eq("nucleotides"), indirect=True)
    def test_nucleotide_at(self, recipe):
        random.seed(42)
        for _, ts in recipe["ts"].items():
            if ts.num_mutations > 0:
                mut_metadata = pyslim.mutation_metadata(ts)
                mut_md = ts.mutation(0).metadata
                tsmd = ts.metadata
                # check we've got nucleotide mutations
                nucs = np.array([x["nucleotide"] for x in tsmd["SLiM_mutation_list"]])
                assert np.sum(nucs >= 0) > 1
                mut_info = {
                    str(mut["mutation_id"]): mut for mut in tsmd["SLiM_mutation_list"]
                }
                assert ts.has_reference_sequence()
                assert len(ts.reference_sequence.data) == ts.sequence_length
                for _ in range(100):
                    node = random.randint(0, ts.num_nodes - 1)
                    pos = random.randint(0, int(ts.sequence_length) - 1)
                    tree = ts.at(pos)
                    parent = tree.parent(node)
                    a = pyslim.nucleotide_at(ts, node, pos)
                    if parent == tskit.NULL:
                        nuc = ts.reference_sequence.data[int(pos)]
                        assert a == pyslim.NUCLEOTIDES.index(nuc)
                    else:
                        b = pyslim.nucleotide_at(
                            ts, parent, pos, mut_metadata=mut_metadata
                        )
                        c = pyslim.nucleotide_at(
                            ts,
                            node,
                            pos,
                            ts.node(parent).time,
                            mut_metadata=mut_metadata,
                        )
                        assert b == c
                        for k in np.where(node == ts.tables.mutations.node)[0]:
                            mut = ts.mutation(k)
                            if ts.site(mut.site).position == pos:
                                b = mut_info[mut.derived_state.split(",")[0]][
                                    "nucleotide"
                                ]
                        assert a == b

    @pytest.mark.parametrize("recipe", [next(recipe_eq("nucleotides"))], indirect=True)
    def test_nucleotide_at_without_mut_metadata(self, recipe):
        random.seed(23)
        for _, ts in recipe["ts"].items():
            assert ts.num_mutations > 0
            mut_metadata = pyslim.mutation_metadata(ts)
            for _ in range(100):
                node = random.randint(0, ts.num_nodes - 1)
                pos = random.randint(0, int(ts.sequence_length) - 1)
                a = pyslim.nucleotide_at(ts, node, pos)
                b = pyslim.nucleotide_at(ts, node, pos, mut_metadata=mut_metadata)
                assert a == b

    @pytest.mark.parametrize("recipe", recipe_eq("mutation_spectrum"), indirect=True)
    def test_nucleotide_spectrum(self, recipe):
        # this is modified from Recipe 18.13
        # Also note that we are comparing to "truth" in a recipe where
        # truth is recorded during a mutation callback, in which we have
        # access to the parental genome, so if two adjacent mutations
        # occur in the same meiosis then each will not know about the other.
        for _, ts in recipe["ts"].items():
            mut_info = pyslim.mutation_metadata(ts)
            mutation_spectrum = recipe["mutation_info"]
            M = {
                a + b + c + "," + d: 0
                for a in pyslim.NUCLEOTIDES
                for b in pyslim.NUCLEOTIDES
                for c in pyslim.NUCLEOTIDES
                for d in pyslim.NUCLEOTIDES
            }
            nmuts = 0
            for mut in ts.mutations():
                pos = ts.site(mut.site).position
                if pos > 0 and pos < ts.sequence_length - 1:
                    nmuts += 1
                    mut_list = [mut_info[int(k)] for k in mut.derived_state.split(",")]
                    k = np.argmax([u["slim_time"] for u in mut_list])
                    derived_nuc = mut_list[k]["nucleotide"]
                    left_nuc = pyslim.nucleotide_at(
                        ts, mut.node, pos - 1, time=mut.time + 1.0
                    )
                    right_nuc = pyslim.nucleotide_at(
                        ts, mut.node, pos + 1, time=mut.time + 1.0
                    )
                    parent_nuc = pyslim.nucleotide_at(
                        ts, mut.node, pos, time=mut.time + 1.0
                    )
                    context = "".join(
                        [
                            pyslim.NUCLEOTIDES[k]
                            for k in (left_nuc, parent_nuc, right_nuc)
                        ]
                    )
                    key = context + "," + pyslim.NUCLEOTIDES[derived_nuc]
                    M[key] += 1
            assert sum([M[k] for k in M]) == nmuts
            assert sum([mutation_spectrum[k][0] for k in mutation_spectrum]) == nmuts
            for k in M:
                assert len(mutation_spectrum[k]) == 1
                assert M[k] == mutation_spectrum[k][0]


class TestConvertNucleotides(tests.PyslimTestCase):
    """
    Test for operations involving the converting and generating nucleotides
    """

    def last_slim_mutations(self, ts):
        # iterator over mutations, returning for each mutation in ts a tuple
        # (slim id, slim mutation metadata) of the slim mutation that is the
        # *most recent* one of any possibly stacked mutations. Note that it
        # is possible that this is ambiguous.
        mut_info = pyslim.mutation_metadata(ts)
        for mut in ts.mutations():
            slim_muts = {
                k: v
                for k, v in mut_info.items()
                if str(k) in mut.derived_state.split(",")
            }
            if mut.parent == tskit.NULL:
                parent_slim_ids = []
            else:
                parent_mut = ts.mutation(mut.parent)
                parent_slim_ids = parent_mut.derived_state.split(",")
            max_time = max([md["slim_time"] for md in slim_muts.values()])
            any_new = any(
                [
                    k not in parent_slim_ids
                    for k in slim_muts.keys()
                    if slim_muts[k]["slim_time"] == max_time
                ]
            )
            maybe_these = [
                k
                for k in slim_muts.keys()
                if slim_muts[k]["slim_time"] == max_time
                and ((k not in parent_slim_ids) or (not any_new))
            ]
            k = max(maybe_these)
            yield k, slim_muts[k]

    def verify_converted_nucleotides(self, ts, cts):
        assert ts.has_reference_sequence() == cts.has_reference_sequence()
        if ts.has_reference_sequence():
            assert ts.reference_sequence.data == cts.reference_sequence.data
        assert ts.num_sites == cts.num_sites
        for k, (s, ns) in enumerate(zip(ts.sites(), cts.sites())):
            assert s.position == ns.position
            assert s.metadata == ns.metadata
            assert ns.ancestral_state == ts.reference_sequence.data[int(s.position)]
        for m, cm, (_, sm) in zip(
            ts.mutations(), cts.mutations(), self.last_slim_mutations(ts)
        ):
            assert m.site == cm.site
            assert m.node == cm.node
            assert m.parent == cm.parent
            assert m.time == cm.time
            assert m.metadata == cm.metadata
            nuc = sm["nucleotide"]
            assert nuc in [0, 1, 2, 3]
            assert cm.derived_state == pyslim.NUCLEOTIDES[nuc]
        # should not have changed anything else
        tc = ts.dump_tables()
        ntc = cts.dump_tables()
        tc.sites.clear()
        ntc.sites.clear()
        tc.mutations.clear()
        ntc.mutations.clear()
        tc.provenances.clear()
        ntc.provenances.clear()
        assert tc == ntc

    def scramble_mutations(self, ts):
        # scramble order of mutations so that the most recent is not always first,
        # since we don't have a reliable way to get that out of SLiM
        rng = np.random.default_rng(123)
        t = ts.dump_tables()
        t.mutations.clear()
        for m in ts.mutations():
            a = np.array(m.derived_state.split(","))
            ii = rng.permutation(len(a))
            t.mutations.append(m.replace(derived_state=",".join(a[ii])))
        t.compute_mutation_parents()
        return t.tree_sequence()

    def test_convert_alleles_errors(self):
        ts = msprime.sim_ancestry(4, sequence_length=10, population_size=10)
        with pytest.raises(ValueError, match="must have a valid reference sequence"):
            _ = pyslim.convert_alleles(ts)
        ts = pyslim.annotate(ts, model_type="nonWF", tick=1)
        with pytest.raises(ValueError, match="must have a valid reference sequence"):
            _ = pyslim.convert_alleles(ts)
        mts = msprime.sim_mutations(
            ts, model=msprime.SLiMMutationModel(type=1), rate=0.1, random_seed=23
        )
        assert mts.num_mutations > 0
        mts = pyslim.add_mutation_metadata(mts)
        mtt = mts.dump_tables()
        mtt.reference_sequence.data = "A" * int(mts.sequence_length)
        mts = mtt.tree_sequence()
        with pytest.raises(ValueError, match="must be nucleotide mutations"):
            _ = pyslim.convert_alleles(mts)

    @pytest.mark.parametrize(
        "recipe", recipe_eq("nucleotides", exclude="non-nucleotides"), indirect=True
    )
    def test_convert_alleles(self, recipe):
        for _, ts in recipe["ts"].items():
            verify_mutation_metadata(ts)
            cts = pyslim.convert_alleles(ts)
            self.verify_converted_nucleotides(ts, cts)

            # get some weirder situations in there
            t = ts.dump_tables()
            t.mutations.clear()
            for j, mut in enumerate(ts.mutations()):
                if j % 2 == 0:
                    t.mutations.append(mut)
            t.compute_mutation_parents()
            ts = t.tree_sequence()
            cts = pyslim.convert_alleles(ts)
            self.verify_converted_nucleotides(ts, cts)

    @pytest.mark.parametrize(
        "recipe", recipe_eq("nucleotides", exclude="non-nucleotides"), indirect=True
    )
    def test_convert_alleles_scrambled(self, recipe):
        for _, ts in recipe["ts"].items():
            ts = self.scramble_mutations(ts)
            cts = pyslim.convert_alleles(ts)
            self.verify_converted_nucleotides(ts, cts)

    @pytest.mark.parametrize(
        "recipe", recipe_eq("nucleotides", exclude="non-nucleotides"), indirect=True
    )
    def test_keeps_reference_sequence(self, recipe):
        for _, ts in recipe["ts"].items():
            assert ts.has_reference_sequence()
            nts = pyslim.generate_nucleotides(ts, seed=123)
            assert nts.has_reference_sequence()
            assert ts.reference_sequence == nts.reference_sequence

    def test_generate_nucleotides_errors(self):
        ts = msprime.sim_ancestry(
            4, sequence_length=10, population_size=10, random_seed=777
        )
        with pytest.raises(ValueError, match="must have length equal"):
            _ = pyslim.generate_nucleotides(ts, reference_sequence="AAA")
        with pytest.raises(ValueError, match="must have length equal"):
            _ = pyslim.generate_nucleotides(ts, reference_sequence=[1, 2, 3])
        with pytest.raises(ValueError, match="must be a string of"):
            _ = pyslim.generate_nucleotides(
                ts, reference_sequence="X" * int(ts.sequence_length)
            )
        with pytest.raises(ValueError, match="must be a string of"):
            _ = pyslim.generate_nucleotides(
                ts,
                reference_sequence=np.arange(int(ts.sequence_length)),
            )

    def verify_generate_nucleotides(self, ts, check_transitions=False):
        # if check_transitions is True, verify that derived states differ
        # from parental states - which we try to do but is not guaranteed,
        # for instance, if keep=True, there was more than one mutation in
        # single generation, or in other weird situations.
        assert ts.metadata["SLiM"]["nucleotide_based"]
        assert len(ts.reference_sequence.data) == ts.sequence_length
        mut_info = pyslim.mutation_metadata(ts)
        muts = {}
        ts_muts = {
            j: v["nucleotide"] for j, (_, v) in enumerate(self.last_slim_mutations(ts))
        }
        for mut in ts.mutations():
            aa = ts.reference_sequence.data[int(ts.site(mut.site).position)]
            for i in mut.derived_state.split(","):
                md = mut_info[int(i)]
                nuc = md["nucleotide"]
                assert nuc in [0, 1, 2, 3]
                if i in muts:
                    assert muts[i] == nuc
                muts[i] = nuc
            if check_transitions:
                if mut.parent == tskit.NULL:
                    assert pyslim.NUCLEOTIDES[nuc] != aa
                else:
                    mp = ts.mutation(mut.parent)
                    if mp.derived_state != mut.derived_state:
                        assert (ts_muts[mut.parent] != ts_muts[mut.id]) or (
                            len(mut.derived_state.split(","))
                            > 1 + len(mp.derived_state.split(","))
                        )

    @pytest.mark.parametrize("recipe", recipe_eq(exclude="old_mutations"), indirect=True)
    def test_generate_nucleotides(self, recipe):
        for _, ts in recipe["ts"].items():
            nts = pyslim.generate_nucleotides(ts, keep=False, seed=5)
            self.verify_generate_nucleotides(
                nts,
                check_transitions=("adds_mutations" not in recipe),
            )

    def test_generate_nucleotides_refseq(self):
        ts = msprime.sim_ancestry(
            4,
            sequence_length=10,
            population_size=10,
            random_seed=10,
        )
        ts = pyslim.annotate(ts, model_type="nonWF", tick=1)
        mts = pyslim.add_mutation_metadata(
            msprime.sim_mutations(
                ts, model=msprime.SLiMMutationModel(type=1), rate=0.5, random_seed=23
            )
        )
        refseq = "A" * int(mts.sequence_length)
        nts = pyslim.generate_nucleotides(mts, reference_sequence=refseq, seed=6)
        self.verify_generate_nucleotides(nts, check_transitions=True)
        assert nts.reference_sequence.data == refseq

    def test_generate_nucleotides_keep(self):
        ts = msprime.sim_ancestry(4, sequence_length=10, population_size=10)
        ts = pyslim.annotate(ts, model_type="nonWF", tick=1)
        mts1 = pyslim.add_mutation_metadata(
            msprime.sim_mutations(
                ts, model=msprime.SLiMMutationModel(type=1), rate=0.1, random_seed=23
            )
        )
        nts1 = pyslim.generate_nucleotides(mts1, seed=10, keep=False)
        assert nts1.num_mutations > 0
        self.verify_generate_nucleotides(nts1, check_transitions=False)
        mut_info1 = {
            str(mut["mutation_id"]): mut for mut in nts1.metadata["SLiM_mutation_list"]
        }
        mts2 = pyslim.add_mutation_metadata(
            msprime.sim_mutations(
                nts1,
                model=msprime.SLiMMutationModel(
                    type=2,
                    next_id=nts1.num_mutations,
                ),
                rate=0.1,
                random_seed=24,
            )
        )
        # keep defaults to True
        nts2 = pyslim.generate_nucleotides(mts2, seed=12)
        assert nts2.num_mutations > nts1.num_mutations
        mut_info2 = {
            str(mut["mutation_id"]): mut for mut in nts2.metadata["SLiM_mutation_list"]
        }
        muts1 = {}
        for mut in nts1.mutations():
            for i in mut.derived_state.split(","):
                md = mut_info1[i]
                muts1[i] = md["nucleotide"]
        for mut in nts2.mutations():
            for i in mut.derived_state.split(","):
                md = mut_info2[i]
                if md["mutation_type"] == 1:
                    assert i in muts1
                    assert muts1[i] == md["nucleotide"]
                else:
                    assert md["nucleotide"] in [0, 1, 2, 3]
        nts3 = pyslim.generate_nucleotides(mts2, keep=False, seed=15)
        self.verify_generate_nucleotides(nts3, check_transitions=False)

    @pytest.mark.parametrize("recipe", ["recipe_long_nonWF.slim"], indirect=True)
    def test_generate_and_convert(self, recipe, helper_functions, tmp_path):
        for _, ts in recipe["ts"].items():
            nts = pyslim.generate_nucleotides(ts, seed=123)
            cts = pyslim.convert_alleles(nts)
            self.verify_converted_nucleotides(nts, cts)
            helper_functions.run_slim_restart(
                {"default": nts},
                "restart_nucleotides_nonWF.slim",
                tmp_path,
                multichrom=False,
                WF=False,
            )


class TestDeprecations(tests.PyslimTestCase):
    # test on one arbitrary recipe
    @pytest.mark.skip(reason="TODO")
    @pytest.mark.parametrize("recipe", [next(recipe_eq())], indirect=True)
    def test_slim_tree_sequence(self, recipe):
        ts = list(recipe["ts"].values())[0]
        with pytest.warns(FutureWarning):
            _ = pyslim.SlimTreeSequence(ts)


class TestVacancy(tests.PyslimTestCase):
    def vacancy_values(self, node):
        if node.metadata is None:
            out = None
        else:
            b = node.metadata["is_vacant"]
            out = []
            powers = [1, 2, 4, 8, 16, 32, 64, 128]
            for bb in b:
                x = []
                for p in powers:
                    x.append(bb & p > 0)
                out.extend(x)
        return out

    def get_vacant_samples(self, ts):
        k = ts.metadata["SLiM"]["this_chromosome"]["index"]
        is_vacant = np.full(ts.num_nodes, False)
        for j, n in enumerate(ts.nodes()):
            v = self.vacancy_values(n)
            isv = v is not None and v[k]
            is_vacant[j] = isv and n.is_sample()
        return np.where(is_vacant)[0]

    def verify_remove_vacant(self, ts, rts):
        vacant_samples = self.get_vacant_samples(ts)
        vacant = pyslim.nodes_vacant(ts)
        for v, node in zip(vacant, rts.nodes()):
            assert not (v and (node.is_sample() == 1))
            assert (node.id in vacant_samples) == (
                node.flags & pyslim.NODE_IS_VACANT_SAMPLE > 0
            )

    def verify_restore_vacant(self, ts, rrts):
        vacant_samples = self.get_vacant_samples(ts)
        rr_vacant_samples = self.get_vacant_samples(rrts)
        assert np.all(vacant_samples == rr_vacant_samples)
        ts.tables.assert_equals(
            rrts.tables,
            ignore_ts_metadata=True,
            ignore_provenance=True,
        )

    @pytest.mark.parametrize("recipe", [next(recipe_eq())], indirect=True)
    def test_chromosome_index_errors(self, recipe):
        ts = list(recipe["ts"].values())[0]
        # no SLiM > this_chromosome key
        tables = ts.dump_tables()
        md = tables.metadata
        del md["SLiM"]["this_chromosome"]
        tables.metadata_schema = tskit.MetadataSchema(
            {"codec": "json", "type": "object"}
        )
        tables.metadata = md
        with pytest.raises(ValueError, match="in top-level metadata"):
            pyslim.remove_vacant(tables.tree_sequence())
        # no SLiM key
        del md["SLiM"]
        tables.metadata = md
        with pytest.raises(ValueError, match="in top-level metadata"):
            pyslim.remove_vacant(tables.tree_sequence())

    @pytest.mark.parametrize("recipe", [next(recipe_eq("Y"))], indirect=True)
    def test_restore_vacant_bad_nodes(self, recipe):
        ts = pyslim.remove_vacant(list(recipe["ts"].values())[0])
        tables = ts.dump_tables()
        tables.nodes.clear()
        for n in ts.nodes():
            flags = n.flags
            if n.id in ts.samples():
                flags |= pyslim.NODE_IS_VACANT_SAMPLE
            tables.nodes.append(n.replace(flags=flags))
        with pytest.raises(ValueError, match="is not vacant"):
            _ = pyslim.restore_vacant(tables.tree_sequence())
        tables.nodes.clear()
        for n in ts.nodes():
            md = n.metadata
            if n.flags & pyslim.NODE_IS_VACANT_SAMPLE > 0:
                md = None
            tables.nodes.append(n.replace(metadata=md))
        with pytest.raises(ValueError, match="has no metadata"):
            _ = pyslim.restore_vacant(tables.tree_sequence())

    @pytest.mark.parametrize(
        "recipe", [next(recipe_eq("multichrom", "X", "Y"))], indirect=True
    )
    def test_multiple_remove_vacant_warning(self, recipe):
        done = 0
        for ts in recipe["ts"].values():
            rts = pyslim.remove_vacant(ts)
            if pyslim.has_vacant_samples(ts):
                with pytest.warns(UserWarning, match="flags are being overwritten"):
                    _ = pyslim.remove_vacant(rts)
                done += 1
            else:
                ts.tables.assert_equals(rts.tables)
        assert done > 0

    def test_has_vacant_samples(self, recipe):
        for _, ts in recipe["ts"].items():
            chrom_type = ts.metadata["SLiM"]["this_chromosome"]["type"]
            if chrom_type != "A":
                assert pyslim.has_vacant_samples(ts)

    def test_has_vacant_msprime(self):
        ts = msprime.sim_ancestry(
            4,
            sequence_length=10,
            population_size=10,
            random_seed=10,
        )
        with pytest.raises(ValueError, match="top-level metadata"):
            _ = pyslim.has_vacant_samples(ts)
        tables = ts.dump_tables()
        tables.metadata_schema = tskit.MetadataSchema({"codec": "json"})
        tables.metadata = {}
        with pytest.raises(ValueError, match="top-level metadata"):
            _ = pyslim.has_vacant_samples(ts)
        md = pyslim.default_slim_metadata("tree_sequence")
        del md["SLiM"]["this_chromosome"]
        tables.metadata = md
        with pytest.raises(ValueError, match="top-level metadata"):
            _ = pyslim.has_vacant_samples(ts)
        tables.metadata_schema = pyslim.slim_metadata_schemas["tree_sequence"]
        md = pyslim.default_slim_metadata("tree_sequence")
        tables.metadata = md
        tables.nodes.metadata_schema = pyslim.slim_metadata_schemas["node"]
        assert not pyslim.has_vacant_samples(tables.tree_sequence())

    def test_nodes_vacant(self, recipe):
        np.random.seed(123)
        num_chromosomes = len(recipe["ts"])
        for _, ts in recipe["ts"].items():
            test_nodes = np.random.choice(
                np.arange(ts.num_nodes), size=min(1, ts.num_nodes), replace=False
            )
            k = ts.metadata["SLiM"]["this_chromosome"]["index"]
            vacant = pyslim.nodes_vacant(ts)
            for pv, node in zip(vacant, ts.nodes()):
                v = self.vacancy_values(node)
                isv = v is not None and v[k]
                assert pv == isv
                if node.id in test_nodes:
                    # node_is_vacant is kinda slow so don't test all of them
                    with pytest.warns(FutureWarning, match="deprecated"):
                        assert isv == pyslim.node_is_vacant(ts, node)
                for j in range(num_chromosomes, len(v)):
                    assert not v[j]

    def test_remove_restore_vacant(self, recipe):
        for _, ts in recipe["ts"].items():
            rts = pyslim.remove_vacant(ts)
            self.verify_remove_vacant(ts, rts)
            rrts = pyslim.restore_vacant(rts)
            self.verify_restore_vacant(ts, rrts)

    def test_recapitate_keeps_vacant(self, recipe):
        for chrom, ts in recipe["ts"].items():
            before = pyslim.has_vacant_samples(ts)
            recap_ts = self.do_recapitate(
                ts,
                recombination_rate=1e-8,
                ancestral_Ne=100,
                random_seed=875,
                keep_vacant=True,
            )
            assert before == pyslim.has_vacant_samples(recap_ts)
            # now check we can remove and restore
            rts = pyslim.remove_vacant(recap_ts)
            self.verify_remove_vacant(recap_ts, rts)
            rrts = pyslim.restore_vacant(rts)
            self.verify_restore_vacant(recap_ts, rrts)

    @pytest.mark.parametrize("recipe", recipe_eq("multichrom"), indirect=True)
    def test_remove_restore_vacant_tables(self, recipe):
        for _, ts in recipe["ts"].items():
            rtables = ts.dump_tables()
            pyslim.remove_vacant_tables(rtables)
            self.verify_remove_vacant(ts, rtables.tree_sequence())
            pyslim.restore_vacant_tables(rtables)
            self.verify_restore_vacant(ts, rtables.tree_sequence())

    @pytest.mark.parametrize("recipe", [next(recipe_eq("multichrom"))], indirect=True)
    def test_multiple_restore_vacant(self, recipe):
        # restore_vacant( ) should be idempotent
        for _, ts in recipe["ts"].items():
            rts = pyslim.remove_vacant(ts)
            self.verify_remove_vacant(ts, rts)
            r_rts = pyslim.restore_vacant(rts)
            self.verify_restore_vacant(ts, r_rts)
            rr_rts = pyslim.restore_vacant(r_rts)
            self.verify_restore_vacant(ts, rr_rts)


class TestFlags(tests.PyslimTestCase):
    def test_flags_types(self):
        # they should be unsigned so this should be true
        assert ~pyslim.INDIVIDUAL_ALIVE > 0
        assert ~pyslim.INDIVIDUAL_REMEMBERED > 0
        assert ~pyslim.INDIVIDUAL_RETAINED > 0

    @pytest.mark.parametrize("recipe", [next(recipe_eq("multichrom"))], indirect=True)
    def test_alive_not_alive(self, recipe):
        for flag in (
            pyslim.INDIVIDUAL_ALIVE,
            pyslim.INDIVIDUAL_REMEMBERED,
            pyslim.INDIVIDUAL_RETAINED,
        ):
            yes = None
            no = None
            for _, ts in recipe["ts"].items():
                (this_yes,) = np.where(ts.individuals_flags & flag > 0)
                (this_no,) = np.where(ts.individuals_flags & ~flag > 0)
                if yes is None:
                    yes = this_yes
                else:
                    assert np.all(yes == this_yes)
                if no is None:
                    no = this_no
                else:
                    assert np.all(no == this_no)


class TestSetInitialState(tests.PyslimTestCase):
    def verify_reset(self, its, ots, time=0, individuals=None):
        # we're writing out the tree sequences unchanged
        if individuals is None:
            individuals = [
                ind.id
                for ind in its.individuals()
                if ind.flags & pyslim.INDIVIDUAL_ALIVE > 0
            ]
        slim_ids = [its.individual(k).metadata["pedigree_id"] for k in individuals]
        for ind in ots.individuals():
            sid = ind.metadata["pedigree_id"]
            if sid in slim_ids:
                assert ind.flags & pyslim.INDIVIDUAL_ALIVE > 0
            else:
                assert ind.flags & pyslim.INDIVIDUAL_ALIVE == 0
        assert its.metadata["SLiM"]["tick"] == ots.metadata["SLiM"]["tick"] + time
        assert np.allclose(
            its.mutations_time,
            ots.mutations_time + time,
        )
        assert np.allclose(
            its.nodes_time,
            ots.nodes_time + time,
        )
        assert np.allclose(
            its.migrations_time,
            ots.migrations_time + time,
        )

    @pytest.mark.parametrize(
        "restart_name, recipe", restarted_recipe_eq("no_op"), indirect=["recipe"]
    )
    def test_no_change(self, restart_name, recipe, helper_functions, tmp_path):
        in_ts = {}
        for chrom, ts in recipe["ts"].items():
            in_ts[chrom] = pyslim.set_slim_state(ts)
        # put it through SLiM (which just reads in and writes out)
        out_ts = helper_functions.run_slim_restart(
            in_ts,
            restart_name,
            tmp_path,
            "multichrom" in recipe,
            WF="WF" in recipe,
        )
        for chrom, ts in recipe["ts"].items():
            self.verify_reset(ts, out_ts[chrom])
        # again!!
        out_out_ts = helper_functions.run_slim_restart(
            out_ts,
            restart_name,
            tmp_path,
            "multichrom" in recipe,
            WF="WF" in recipe,
        )
        for chrom, ts in recipe["ts"].items():
            self.verify_reset(ts, out_out_ts[chrom])

    @pytest.mark.parametrize(
        "restart_name, recipe", restarted_recipe_eq("no_op"), indirect=["recipe"]
    )
    def test_shift_time(self, restart_name, recipe, helper_functions, tmp_path):
        in_ts = {}
        time = 3
        for chrom, ts in recipe["ts"].items():
            in_ts[chrom] = pyslim.set_slim_state(ts, time=time)
        out_ts = helper_functions.run_slim_restart(
            in_ts,
            restart_name,
            tmp_path,
            "multichrom" in recipe,
            WF="WF" in recipe,
        )
        for chrom, ts in recipe["ts"].items():
            self.verify_reset(ts, out_ts[chrom], time=time)

    @pytest.mark.parametrize(
        "restart_name, recipe",
        restarted_recipe_eq("no_op", "resettable"),
        indirect=["recipe"],
    )
    @pytest.mark.parametrize("num_indivs", [2, 100])
    @pytest.mark.parametrize("time", [0, None])
    def test_set_individuals(
        self, restart_name, recipe, num_indivs, time, helper_functions, tmp_path
    ):
        in_ts = {}
        ts = list(recipe["ts"].values())[0]
        tsmd = ts.metadata
        assert (
            "user_metadata" in tsmd["SLiM"]
            and "reset_tick" in tsmd["SLiM"]["user_metadata"]
        ), "Simulation not set up for this test."
        reset_tick = tsmd["SLiM"]["user_metadata"]["reset_tick"][0]
        if time is None:
            pytimes = run_with_ts_metadata(
                pyslim.slim_time, tsmd, ts, np.arange(tsmd["SLiM"]["tick"] + 1)
            )
            time = np.searchsorted(pytimes, reset_tick)
        individuals = pyslim.individuals_alive_at(ts, time, ts_metadata=tsmd)[
            :num_indivs
        ]
        for chrom, ts in recipe["ts"].items():
            in_ts[chrom] = pyslim.set_slim_state(ts, time=time, individuals=individuals)
        out_ts = helper_functions.run_slim_restart(
            in_ts,
            restart_name,
            tmp_path,
            "multichrom" in recipe,
            WF="WF" in recipe,
        )
        for chrom, ts in recipe["ts"].items():
            self.verify_reset(ts, out_ts[chrom], time=time, individuals=individuals)
