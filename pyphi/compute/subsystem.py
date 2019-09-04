#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# compute/subsystem.py

"""
Functions for computing subsystem-level properties.
"""

import functools
import logging

from .. import Direction, config, connectivity, memory, utils
from ..models import (CauseEffectStructure, Concept, SystemIrreducibilityAnalysis,
                      _null_sia, cmp, fmt)
from ..models.cuts import Part, KPartition
from ..partition import system_cuts
from ..distribution import flatten
from ..utils import time_annotated
from .distance import ces_distance
from .parallel import MapReduce
from ..relations import ces_distance as relational_ces_distance

# Create a logger for this module.
log = logging.getLogger(__name__)


class ComputeFixedCauseEffectStructure(MapReduce):
    """Engine for computing a |CauseEffectStructure|, when mechanisms and purviews
    are already known or specified."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Computing concepts'

    @property
    def subsystem(self):
        return self.context[0]

    def empty_result(self, *args):
        return []

    @staticmethod
    def compute(concept, subsystem):
        """Recompute the repertories and irreducibility of a |Concept| in the
        context of a |Subsystem| (usually one with an applied system parition),
        keeping its mechanism and purviews fixed.
        """
        new_concept = subsystem.concept(concept.mechanism,
                                        cause_purviews=(concept.cause.purview,),
                                        effect_purviews=(concept.effect.purview,))

        # Don't serialize the subsystem.
        # This is replaced on the other side of the queue, and ensures
        # that all concepts in the CES reference the same subsystem.
        new_concept.subsystem = None
        return new_concept

    def process_result(self, new_concept, concepts):
        """Save all concepts with non-zero |small_phi| to the
        |CauseEffectStructure|.
        """
        if not config.CONCEPTS_MUST_HAVE_BOTH_CAUSES_AND_EFFECTS or new_concept.phi > 0:
            # Replace the subsystem
            new_concept.subsystem = self.subsystem
            concepts.append(new_concept)
        return concepts


@time_annotated
def fixed_ces(subsystem, ces, parallel=False):
    """Return a |CauseEffectStructure|, when mechanisms, purviews, and purviews are
    already known or specified.

    Args:
        subsystem (Subsystem): The subsystem for which to determine the
            |CauseEffectStructure|.
        ces (CauseEffectStructure): The |CauseEffectStructure| from which to
            take the mechanisms, purviews, and partitions.

    Keyword Args:
        parallel (bool): Whether to recompute concepts in parallel. If
            ``True``, overrides :data:`config.PARALLEL_CONCEPT_EVALUATION`.

    Returns:
        CauseEffectStructure: A tuple of every |Concept| in the cause-effect
        structure specified by `ces`, but with recomputed repertories and
        irreducibility values. Reducible concepts are not returned.
    """
    engine = ComputeFixedCauseEffectStructure(ces, subsystem)

    return CauseEffectStructure(engine.run(parallel or
                                           config.PARALLEL_CONCEPT_EVALUATION),
                                subsystem=subsystem)


class ComputeCauseEffectStructure(MapReduce):
    """Engine for computing a |CauseEffectStructure|."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Computing concepts'

    @property
    def subsystem(self):
        return self.context[0]

    def empty_result(self, *args):
        return []

    @staticmethod
    def compute(mechanism, subsystem, purviews, cause_purviews,
                effect_purviews):
        """Compute a |Concept| for a mechanism, in this |Subsystem| with the
        provided purviews.
        """
        concept = subsystem.concept(mechanism,
                                    purviews=purviews,
                                    cause_purviews=cause_purviews,
                                    effect_purviews=effect_purviews)
        # Don't serialize the subsystem.
        # This is replaced on the other side of the queue, and ensures
        # that all concepts in the CES reference the same subsystem.
        concept.subsystem = None
        return concept

    def process_result(self, new_concept, concepts):
        """Save all concepts with non-zero |small_phi| to the
        |CauseEffectStructure|.
        """
        if not config.CONCEPTS_MUST_HAVE_BOTH_CAUSES_AND_EFFECTS or new_concept.phi > 0:
            # Replace the subsystem
            new_concept.subsystem = self.subsystem
            concepts.append(new_concept)
        return concepts


@time_annotated
def ces(subsystem, mechanisms=False, purviews=False, cause_purviews=False,
        effect_purviews=False, parallel=False):
    """Return the conceptual structure of this subsystem, optionally restricted
    to concepts with the mechanisms and purviews given in keyword arguments.

    If you don't need the full |CauseEffectStructure|, restricting the possible
    mechanisms and purviews can make this function much faster.

    Args:
        subsystem (Subsystem): The subsystem for which to determine the
            |CauseEffectStructure|.

    Keyword Args:
        mechanisms (tuple[tuple[int]]): Restrict possible mechanisms to those
            in this list.
        purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        cause_purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        effect_purviews (tuple[tuple[int]]): Same as in |Subsystem.concept()|.
        parallel (bool): Whether to compute concepts in parallel. If ``True``,
            overrides :data:`config.PARALLEL_CONCEPT_EVALUATION`.

    Returns:
        CauseEffectStructure: A tuple of every |Concept| in the cause-effect
        structure.
    """
    if mechanisms is False:
        mechanisms = utils.powerset(subsystem.node_indices, nonempty=True)

    engine = ComputeCauseEffectStructure(mechanisms, subsystem, purviews,
                                         cause_purviews, effect_purviews)

    return CauseEffectStructure(engine.run(parallel or
                                           config.PARALLEL_CONCEPT_EVALUATION),
                                subsystem=subsystem)


def conceptual_info(subsystem):
    """Return the conceptual information for a |Subsystem|.

    This is the distance from the subsystem's |CauseEffectStructure| to the
    null concept.
    """
    ci = ces_distance(ces(subsystem),
                      CauseEffectStructure((), subsystem=subsystem))
    return round(ci, config.PRECISION)


def evaluate_cut(uncut_subsystem, cut, unpartitioned_ces):
    """Compute the system irreducibility for a given cut.
    Args:
        uncut_subsystem (Subsystem): The subsystem without the cut applied.
        cut (Cut): The cut to evaluate.
        unpartitioned_ces (CauseEffectStructure): The cause-effect structure of
            the uncut subsystem.
    Returns:
        SystemIrreducibilityAnalysis: The |SystemIrreducibilityAnalysis| for
        that cut.
    """
    import numpy as np
    from functools import reduce

    log.debug('Evaluating %s...', cut)

    cut_subsystem = uncut_subsystem.apply_cut(cut)

    # Compute that partitioned CES
    if config.SYSTEM_PARTITIONS_CANNOT_CHANGE_CONCEPTS_PURVIEW:
        partitioned_ces = fixed_ces(cut_subsystem, unpartitioned_ces)
    elif config.SYSTEM_PARTITIONS_CANNOT_CREATE_NEW_CONCEPTS:
        mechanisms = unpartitioned_ces.mechanisms
        partitioned_ces = ces(cut_subsystem, mechanisms)
    else:
        # Mechanisms can only produce concepts if they were concepts in the
        # original system, or the cut divides the mechanism.
        mechanisms = set(
            unpartitioned_ces.mechanisms +
            list(cut_subsystem.cut_mechanisms))
        partitioned_ces = ces(cut_subsystem, mechanisms)

    log.debug('Finished evaluating %s.', cut)

    # Compute the contribution of concepts to big phi
    c_phi = ces_distance(unpartitioned_ces, partitioned_ces)
    # Compute the contribution of relations to big phi
    if config.INCLUDE_RELATIONS_IN_BIG_PHI:
        r_phi = relational_ces_distance(unpartitioned_ces, partitioned_ces)
    else:
        r_phi = 0

    if config.SPECIFICATION_RATIO == 'MAX_INTRINSIC_INFO':
        # n = len(uncut_subsystem.node_indices)
        n = uncut_subsystem.size
        c_phi = (c_phi ** 2) / (n * 2 ** (n - 1))
        r_phi = (r_phi ** 2) / (n * (2 ** (2 ** (n + 1))) - (2 ** (n + 1)))
    elif (config.SPECIFICATION_RATIO == 'SUM_PROB') or (config.SPECIFICATION_RATIO == 'PROD_PROB'):
        n = len(uncut_subsystem.node_indices)
        ps = [0] * (2 ** n - 1)
        possible_purviews = utils.powerset(uncut_subsystem.node_indices, nonempty=True)
        for i, possible_purview in enumerate(possible_purviews):
            # cause_repertoire = uncut_subsystem.concept(uncut_subsystem.node_indices, purviews=(possible_purview,))
            cause_repertoire = uncut_subsystem.repertoire(Direction.CAUSE, uncut_subsystem.node_indices, possible_purview)
            effect_repertoire = uncut_subsystem.repertoire(Direction.EFFECT, uncut_subsystem.node_indices, possible_purview)
            pc = np.max(cause_repertoire.flatten())
            pe = np.max(effect_repertoire.flatten())
            ps[i] = np.max([pc, pe])

        # if config.SPECIFICATION_RATIO == 'PROD_PROB':
        #     p = 1.
        #     for pt in ps:
        #         p *= pt
        #     c_phi = p * c_phi
        # else:
        #     p = sum(ps)
        #     c_phi = (p/(2 ** n - 1)) * c_phi

        if config.SPECIFICATION_RATIO == 'PROD_PROB':
            func = lambda x, y: x*y
        else:
            func = lambda x, y: x+y

        c_phi = (reduce(func, ps)/reduce(func, [1]*len(ps))) * c_phi

        if r_phi > 0:
            raise ValueError('Relations not supported yet')

    elif (config.SPECIFICATION_RATIO == 'EXTANT_SUM_PROB') or (config.SPECIFICATION_RATIO == 'EXTANT_PROD_PROB'):

        def distance_no_x(x, y):
            return abs(x * np.nan_to_num(np.log2(x / y)))

        def distance(x, y):
            return abs(x * np.nan_to_num(np.log2(x / y)))

        concepts = unpartitioned_ces.concepts
        ps = []
        infos = []

        # ps = [0] * len(concepts)
        # infos = [0] * len(concepts)

        # print(len(concepts))
        # print('------')
        # print(cut)
        # print(uncut_subsystem.nodes)
        for (conceptidx, concept) in enumerate(concepts):
            # get cause repertoire partitioned by the system cut
            partitioned_cause_repertoire = \
                cut_subsystem.cut_system.cause_repertoire(concept.mechanism, concept.cause_purview)
            qsc = flatten(partitioned_cause_repertoire)
            # print(qsc)
            # get state of the cause purview defining the distinction
            pc, qc = flatten(concept.cause_repertoire), flatten(concept.cause.partitioned_repertoire)
            # print(pc,qc)
            maxcidx = np.argmax([distance(x, y) if x > 0 else 0 for (x, y) in zip(pc, qc)])
            # print(maxcidx)
            # compute information in the cause side given system cut
            infoc = distance_no_x(pc[maxcidx], qsc[maxcidx])
            # print(infoc)

            # same for effect
            partitioned_effect_repertoire = \
                cut_subsystem.cut_system.effect_repertoire(concept.mechanism, concept.effect_purview)
            qse = flatten(partitioned_effect_repertoire)
            # print(qse)
            pe, qe = flatten(concept.effect_repertoire), flatten(concept.effect.partitioned_repertoire)
            # print(pe,qe)
            maxeidx = np.argmax([distance(x, y) if x > 0 else 0 for (x, y) in zip(pe, qe)])
            # print(maxeidx)
            infoe = distance_no_x(pe[maxeidx], qse[maxeidx])
            # print(infoe)

            # touch and die
            p = []
            q = []
            if concept.effect.phi < concept.cause.phi:
                if any(pe != qse):
                    infos.append(concept.phi)
                    p = pe
                    q = qe
            else:
                if any(pc != qsc):
                    infos.append(concept.phi)
                    p = pc
                    q = qc

            if len(p):
                ds = [distance(x, y) if x > 0 else 0 for (x, y) in zip(p, q)]
                maxidx = np.argmax(ds)
                if ds[maxidx].round(decimals=config.PRECISION) != concept.phi:
                    print('problem')
                ps.append(p[maxidx])

            # p = [pc[maxcidx], pe[maxeidx]]
            # info = [infoc, infoe]

            #  # compute big phi with the minimum between cause and effect
            # minidx = np.argmin(info)
            # ps.append(p[minidx])
            # infos.append(info[minidx])

            # # compute big phi with the sum in change in cause and effect
            # ps += p
            # infos += info

            # compute big phi with the maximum between cause and effect
            # maxidx = np.argmax(info)
            # ps.append(p[maxidx])
            # infos.append(info[maxidx])





        if config.SPECIFICATION_RATIO == 'EXTANT_PROD_PROB':
            func = lambda x, y: x*y
        elif config.SPECIFICATION_RATIO == 'EXTANT_SUM_PROB':
            func = lambda x, y: x+y
        else:
            raise ValueError('Unknown SPECIFICATION_RATIO: %s' % config.SPECIFICATION_RATIO)

        if not ps:
            c_phi = 0
        else:
            # print(ps)
            # c_phi = (reduce(func, ps)/reduce(func, [1]*len(ps)))
            # print (c_phi)
            # print (infos)
            c_phi = (reduce(func, ps)/reduce(func, [1]*len(ps))) * sum(infos)
            # print (c_phi, '\n')
            # print (c_phi)

        if r_phi > 0:
            raise ValueError('Relations not supported yet')

    elif config.SPECIFICATION_RATIO == 'SUM_PROB_INFO':
        n = len(uncut_subsystem.node_indices)
        p = 0.
        possible_purviews = utils.powerset(uncut_subsystem.node_indices, nonempty=True)
        for possible_purview in possible_purviews:
            desintegration = KPartition(
                *[Part(mechanism=(i,), purview=()) for i in possible_purview],
                *[Part(mechanism=(), purview=(i,)) for i in possible_purview]
            )
            # cause_repertoire = uncut_subsystem.concept(uncut_subsystem.node_indices, purviews=(possible_purview,))
            cause_repertoire = uncut_subsystem.repertoire(Direction.CAUSE, uncut_subsystem.node_indices, possible_purview)
            dphi, partitioned_cause_repertoire = uncut_subsystem.evaluate_partition(
                Direction.CAUSE, uncut_subsystem.node_indices, possible_purview, desintegration, repertoire=cause_repertoire)
            pc = cause_repertoire.flatten()
            q = partitioned_cause_repertoire.flatten()
            pc_idx = np.argmax([abs(x * np.nan_to_num(np.log2(x / y))) if x > 0 else 0 for (x, y) in zip(pc, q)])

            effect_repertoire = uncut_subsystem.repertoire(Direction.EFFECT, uncut_subsystem.node_indices, possible_purview)
            dphi, partitioned_effect_repertoire = uncut_subsystem.evaluate_partition(
                Direction.EFFECT, uncut_subsystem.node_indices, possible_purview, desintegration, repertoire=effect_repertoire)
            pe = effect_repertoire.flatten()
            q = partitioned_effect_repertoire.flatten()
            pe_idx = np.argmax([abs(x * np.nan_to_num(np.log2(x / y))) if x > 0 else 0 for (x, y) in zip(pe, q)])

            pc = pc[pc_idx]
            pe = pe[pe_idx]
            p += np.max([pc, pe])

        c_phi = (p/(2 ** n - 1)) * c_phi

        if r_phi > 0:
            raise ValueError('Relations not supported yet')

    elif config.SPECIFICATION_RATIO != 'NONE':
        raise ValueError('Unknown SPECIFICATION_RATIO: %s' % config.SPECIFICATION_RATIO)

    return SystemIrreducibilityAnalysis(
        phi=c_phi + r_phi,
        ces=unpartitioned_ces,
        partitioned_ces=partitioned_ces,
        subsystem=uncut_subsystem,
        cut_subsystem=cut_subsystem)


class ComputeSystemIrreducibility(MapReduce):
    """Computation engine for system-level irreducibility."""
    # pylint: disable=unused-argument,arguments-differ

    description = 'Evaluating {} cuts'.format(fmt.BIG_PHI)

    def empty_result(self, subsystem, unpartitioned_ces):
        """Begin with a |SIA| with infinite |big_phi|; all actual SIAs will
        have less.
        """
        return _null_sia(subsystem, phi=float('inf'))

    @staticmethod
    def compute(cut, subsystem, unpartitioned_ces):
        """Evaluate a cut."""
        return evaluate_cut(subsystem, cut, unpartitioned_ces)

    def process_result(self, new_sia, min_sia):
        """Check if the new SIA has smaller |big_phi| than the standing
        result.
        """
        if new_sia.phi == 0:
            self.done = True  # Short-circuit
            return new_sia

        elif new_sia < min_sia:
            return new_sia

        return min_sia


def _ces(subsystem):
    """Parallelize the unpartitioned |CauseEffectStructure| if parallelizing
    cuts, since we have free processors because we're not computing any cuts
    yet.
    """
    return ces(subsystem, parallel=config.PARALLEL_CUT_EVALUATION)


@memory.cache(ignore=["subsystem"])
@time_annotated
def _sia(cache_key, subsystem):
    """Return the minimal information partition of a subsystem.

    Args:
        subsystem (Subsystem): The candidate set of nodes.

    Returns:
        SystemIrreducibilityAnalysis: A nested structure containing all the
        data from the intermediate calculations. The top level contains the
        basic irreducibility information for the given subsystem.
    """
    # pylint: disable=unused-argument

    log.info('Calculating big-phi data for %s...', subsystem)

    # Check for degenerate cases
    # =========================================================================
    # Phi is necessarily zero if the subsystem is:
    #   - not strongly connected;
    #   - empty;
    #   - an elementary micro mechanism (i.e. no nontrivial bipartitions).
    # So in those cases we immediately return a null SIA.
    if not subsystem:
        log.info('Subsystem %s is empty; returning null SIA '
                 'immediately.', subsystem)
        return _null_sia(subsystem)

    if not connectivity.is_strong(subsystem.cm, subsystem.node_indices):
        log.info('%s is not strongly connected; returning null SIA '
                 'immediately.', subsystem)
        return _null_sia(subsystem)

    # Handle elementary micro mechanism cases.
    # Single macro element systems have nontrivial bipartitions because their
    #   bipartitions are over their micro elements.
    if len(subsystem.cut_indices) == 1:
        # If the node lacks a self-loop, phi is trivially zero.
        if not subsystem.cm[subsystem.node_indices][subsystem.node_indices]:
            log.info('Single micro nodes %s without selfloops cannot have '
                     'phi; returning null SIA immediately.', subsystem)
            return _null_sia(subsystem)
        # Even if the node has a self-loop, we may still define phi to be zero.
        elif not config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI:
            log.info('Single micro nodes %s with selfloops cannot have '
                     'phi; returning null SIA immediately.', subsystem)
            return _null_sia(subsystem)
    # =========================================================================

    log.debug('Finding unpartitioned CauseEffectStructure...')
    unpartitioned_ces = _ces(subsystem)

    if not unpartitioned_ces:
        log.info('Empty unpartitioned CauseEffectStructure; returning null '
                 'SIA immediately.')
        # Short-circuit if there are no concepts in the unpartitioned CES.
        return _null_sia(subsystem)

    log.debug('Found unpartitioned CauseEffectStructure.')

    cuts = system_cuts(subsystem.cut_indices, subsystem.cut_node_labels)

    # In principle, SYSTEM_PARTITIONS_CANNOT_CHANGE_CONCEPTS_PURVIEW could be false, but
    # as implemented, MICE tie breaking only makes sense when it is true.
    if config.BREAK_MICE_TIES_USING_BIG_PHI and \
       config.SYSTEM_PARTITIONS_CANNOT_CHANGE_CONCEPTS_PURVIEW:
        log.debug('Breaking MICE ties between CESs...')
        engines = [ComputeSystemIrreducibility(cuts, subsystem, ces)
                   for ces in unpartitioned_ces.ties]
    else:
        engines = [ComputeSystemIrreducibility(cuts, subsystem, unpartitioned_ces)]

    result = max(engine.run(config.PARALLEL_CUT_EVALUATION) for engine in engines)

    if config.CLEAR_SUBSYSTEM_CACHES_AFTER_COMPUTING_SIA:
        log.debug('Clearing subsystem caches.')
        subsystem.clear_caches()

    log.info('Finished calculating big-phi data for %s.', subsystem)

    return result


# TODO(maintainance): don't forget to add any new configuration options here if
# they can change big-phi values
def _sia_cache_key(subsystem):
    """The cache key of the subsystem.

    This includes the native hash of the subsystem and all configuration values
    which change the results of ``sia``.
    """
    return (
        hash(subsystem),
        config.SYSTEM_PARTITIONS_CANNOT_CREATE_NEW_CONCEPTS,
        config.CUT_ONE_APPROXIMATION,
        config.DIVERGENCE,
        config.PRECISION,
        config.VALIDATE_SUBSYSTEM_STATES,
        config.SINGLE_MICRO_NODES_WITH_SELFLOOPS_HAVE_PHI,
        config.CONCEPT_PARTITION_TYPE,
    )


# Wrapper to ensure that the cache key is the native hash of the subsystem, so
# joblib doesn't mistakenly recompute things when the subsystem's MICE cache is
# changed. The cache is also keyed on configuration values which affect the
# value of the computation.
@functools.wraps(_sia)
def sia(subsystem):  # pylint: disable=missing-docstring
    if config.CUT_SYSTEM_CAUSES_AND_EFFECTS_INDEPENDENTLY:
        return sia_concept_style(subsystem)

    return _sia(_sia_cache_key(subsystem), subsystem)


def phi(subsystem):
    """Return the |big_phi| value of a subsystem."""
    return sia(subsystem).phi


class ConceptStyleSystem:
    """A functional replacement for ``Subsystem`` implementing concept-style
    system cuts. The subsystem.direction is the direction (CAUSE or EFFECT),
    in which the cut is applied.
    """

    def __init__(self, subsystem, direction, cut=None):
        self.subsystem = subsystem
        self.direction = direction
        self.cut = cut
        self.cut_system = subsystem.apply_cut(cut)

    def apply_cut(self, cut):
        return ConceptStyleSystem(self.subsystem, self.direction, cut)

    def __getattr__(self, name):
        """Pass attribute access through to the basic subsystem."""
        # Unpickling calls `__getattr__` before the object's dict is populated;
        # check that `subsystem` exists to avoid a recursion error.
        # See https://bugs.python.org/issue5370.
        if 'subsystem' in self.__dict__:
            return getattr(self.subsystem, name)
        raise AttributeError(name)

    def __len__(self):
        return len(self.subsystem)

    @property
    def cause_system(self):
        return {
            Direction.CAUSE: self.cut_system,
            Direction.EFFECT: self.subsystem
        }[self.direction]

    @property
    def effect_system(self):
        return {
            Direction.CAUSE: self.subsystem,
            Direction.EFFECT: self.cut_system
        }[self.direction]

    def mic(self, mechanism, purviews=False, partitions=False):
        """Return the mechanism's maximally-irreducible cause (|MIC|),
        using the appropriate system.
        """
        return self.cause_system.find_mice(Direction.CAUSE,
                                           mechanism,
                                           purviews=purviews,
                                           partitions=partitions)

    def mie(self, mechanism, purviews=False, partitions=False):
        """Return the mechanism's maximally-irreducible effect (|MIE|),
        using the appropriate system.
        """
        return self.effect_system.find_mice(Direction.EFFECT,
                                            mechanism,
                                            purviews=purviews,
                                            partitions=partitions)

    def concept(self, mechanism, purviews=False, cause_purviews=False,
                effect_purviews=False):
        """Compute a concept, using the appropriate system for each side of the
        cut.
        """
        cause = self.cause_system.mic(
            mechanism, purviews=(cause_purviews or purviews))

        effect = self.effect_system.mie(
            mechanism, purviews=(effect_purviews or purviews))

        return Concept(mechanism=mechanism, cause=cause, effect=effect,
                       subsystem=self)

    def __str__(self):
        return 'ConceptStyleSystem{}'.format(self.node_indices)


def directional_sia(subsystem, direction, unpartitioned_ces=None):
    """Calculate a concept-style SystemIrreducibilityAnalysisCause or
    SystemIrreducibilityAnalysisEffect.
    """
    if unpartitioned_ces is None:
        unpartitioned_ces = _ces(subsystem)

    c_system = ConceptStyleSystem(subsystem, direction)
    cuts = system_cuts(c_system.cut_indices, subsystem.node_labels)

    # In principle, SYSTEM_PARTITIONS_CANNOT_CHANGE_CONCEPTS_PURVIEW could be false, but
    # as implemented, MICE tie breaking only makes sense when it is true.
    if config.BREAK_MICE_TIES_USING_BIG_PHI and \
       config.SYSTEM_PARTITIONS_CANNOT_CHANGE_CONCEPTS_PURVIEW:
        log.debug('Breaking MICE ties between CESs...')
        engines = [ComputeSystemIrreducibility(cuts, c_system, ces)
                   for ces in unpartitioned_ces.ties]
    else:
        # TODO: verify that short-cutting works correctly?
        engines = [ComputeSystemIrreducibility(cuts, c_system, unpartitioned_ces)]

    return max(engine.run(config.PARALLEL_CUT_EVALUATION) for engine in engines)


# TODO: only return the minimal SIA, instead of both
class SystemIrreducibilityAnalysisConceptStyle(cmp.Orderable):
    """Represents a |SIA| computed using concept-style system cuts."""

    def __init__(self, sia_cause, sia_effect):
        self.sia_cause = sia_cause
        self.sia_effect = sia_effect

    @property
    def min_sia(self):
        return min(self.sia_cause, self.sia_effect, key=lambda m: m.phi)

    def __getattr__(self, name):
        """Pass attribute access through to the minimal SIA."""
        if ('sia_cause' in self.__dict__ and 'sia_effect' in self.__dict__):
            return getattr(self.min_sia, name)
        raise AttributeError(name)

    def __eq__(self, other):
        return cmp.general_eq(self, other, ['phi'])

    unorderable_unless_eq = ['network']

    def order_by(self):
        return [self.phi, len(self.subsystem)]

    def __repr__(self):
        return repr(self.min_sia)

    def __str__(self):
        return str(self.min_sia)


# TODO: cache
def sia_concept_style(subsystem):
    """Compute a concept-style SystemIrreducibilityAnalysis"""
    unpartitioned_ces = _ces(subsystem)

    sia_cause = directional_sia(subsystem, Direction.CAUSE,
                                unpartitioned_ces)
    sia_effect = directional_sia(subsystem, Direction.EFFECT,
                                 unpartitioned_ces)

    return SystemIrreducibilityAnalysisConceptStyle(sia_cause, sia_effect)
