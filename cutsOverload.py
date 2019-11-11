# -*- coding: utf-8 -*-
"""
Flow based cut algorithms
"""
import itertools
import networkx as nx

# Define the default maximum flow function to use in all flow based
# cut algorithms.
from networkx.algorithms.flow import edmonds_karp
from networkx.algorithms.flow import build_residual_network

from networkx.algorithms.connectivity import minimum_st_node_cut


from networkx.algorithms.connectivity import (build_auxiliary_node_connectivity,
                                        build_auxiliary_edge_connectivity)
from networkx.algorithms.components import number_connected_components
from numpy import var
__author__ = '\n'.join(['Jordi Torrents <jtorrents@milnou.net>'])


# modified from networkx.algorithms.cuts.minimum_node_cut
def minimum_node_cut(G, s=None, t=None, flow_func=None, approximate=-1):
    r"""Returns a set of nodes of minimum cardinality that disconnects G.

    If source and target nodes are provided, this function returns the
    set of nodes of minimum cardinality that, if removed, would destroy
    all paths among source and target in G. If not, it returns a set
    of nodes of minimum cardinality that disconnects G.

    Parameters
    ----------
    G : NetworkX graph

    s : node
        Source node. Optional. Default value: None.

    t : node
        Target node. Optional. Default value: None.

    flow_func : function
        A function for computing the maximum flow among a pair of nodes.
        The function has to accept at least three parameters: a Digraph, 
        a source node, and a target node. And return a residual network 
        that follows NetworkX conventions (see :meth:`maximum_flow` for 
        details). If flow_func is None, the default maximum flow function 
        (:meth:`edmonds_karp`) is used. See below for details. The
        choice of the default function may change from version
        to version and should not be relied on. Default value: None.

    Returns
    -------
    cutset : set
        Set of nodes that, if removed, would disconnect G. If source
        and target nodes are provided, the set contains the nodes that
        if removed, would destroy all paths between source and target.

    Examples
    --------
    >>> # Platonic icosahedral graph has node connectivity 5
    >>> G = nx.icosahedral_graph()
    >>> node_cut = nx.minimum_node_cut(G)
    >>> len(node_cut)
    5

    You can use alternative flow algorithms for the underlying maximum
    flow computation. In dense networks the algorithm
    :meth:`shortest_augmenting_path` will usually perform better
    than the default :meth:`edmonds_karp`, which is faster for
    sparse networks with highly skewed degree distributions. Alternative
    flow functions have to be explicitly imported from the flow package.

    >>> from networkx.algorithms.flow import shortest_augmenting_path
    >>> node_cut == nx.minimum_node_cut(G, flow_func=shortest_augmenting_path)
    True

    If you specify a pair of nodes (source and target) as parameters,
    this function returns a local st node cut.

    >>> len(nx.minimum_node_cut(G, 3, 7))
    5

    If you need to perform several local st cuts among different
    pairs of nodes on the same graph, it is recommended that you reuse
    the data structures used in the maximum flow computations. See 
    :meth:`minimum_st_node_cut` for details.

    Notes
    -----
    This is a flow based implementation of minimum node cut. The algorithm
    is based in solving a number of maximum flow computations to determine
    the capacity of the minimum cut on an auxiliary directed network that
    corresponds to the minimum node cut of G. It handles both directed
    and undirected graphs. This implementation is based on algorithm 11 
    in [1]_.

    See also
    --------
    :meth:`minimum_st_node_cut`
    :meth:`minimum_cut`
    :meth:`minimum_edge_cut`
    :meth:`stoer_wagner`
    :meth:`node_connectivity`
    :meth:`edge_connectivity`
    :meth:`maximum_flow`
    :meth:`edmonds_karp`
    :meth:`preflow_push`
    :meth:`shortest_augmenting_path`

    References
    ----------
    .. [1] Abdol-Hossein Esfahanian. Connectivity Algorithms.
        http://www.cse.msu.edu/~cse835/Papers/Graph_connectivity_revised.pdf

    """
    if (s is not None and t is None) or (s is None and t is not None):
        raise nx.NetworkXError('Both source and target must be specified.')

    # Local minimum node cut.
    if s is not None and t is not None:
        if s not in G:
            raise nx.NetworkXError('node %s not in graph' % s)
        if t not in G:
            raise nx.NetworkXError('node %s not in graph' % t)
        return minimum_st_node_cut(G, s, t, flow_func=flow_func)

    # Global minimum node cut.
    # Analog to the algorithm 11 for global node connectivity in [1].
    if G.is_directed():
        if not nx.is_weakly_connected(G):
            raise nx.NetworkXError('Input graph is not connected')
        iter_func = itertools.permutations

        def neighbors(v):
            return itertools.chain.from_iterable([G.predecessors(v),
                                                  G.successors(v)])
    else:
        if not nx.is_connected(G):
            raise nx.NetworkXError('Input graph is not connected')
        iter_func = itertools.combinations
        neighbors = G.neighbors

    # Reuse the auxiliary digraph and the residual network.
    H = build_auxiliary_node_connectivity(G)
    R = build_residual_network(H, 'capacity')
    kwargs = dict(flow_func=flow_func, auxiliary=H, residual=R)

    # Choose a node with minimum degree.
    v = min(G, key=G.degree)
    # Initial node cutset is all neighbors of the node with minimum degree.
    min_cut = set(G[v])
    # Number of components
    min_num_cc = -1
    min_cc_var = float("inf")
    # Compute st node cuts between v and all its non-neighbors nodes in G.
    for w in set(G) - set(neighbors(v)) - set([v]):
        this_cut = minimum_st_node_cut(G, v, w, **kwargs)
        if len(min_cut) > len(this_cut):
            min_cut = this_cut
            G_copy = G.copy()
            G_copy.remove_nodes_from(this_cut)
            min_num_cc = number_connected_components(G_copy)
            min_cc_var = var([len(c) for c in sorted(nx.connected_components(G_copy))])
        if len(min_cut) == len(this_cut):
            G_copy = G.copy()
            G_copy.remove_nodes_from(this_cut)
            this_num_cc = number_connected_components(G_copy)
            this_cc_var = var([len(c) for c in sorted(nx.connected_components(G_copy))])
#            if this_num_cc >= min_num_cc:
            if this_cc_var < min_cc_var:
                min_cut = this_cut
                min_num_cc = this_num_cc
                min_cc_var = this_cc_var
#            elif this_num_cc == min_num_cc and min_cc_var > this_cc_var:
#                min_cut = this_cut
#                min_num_cc = this_num_cc
#                min_cc_var = this_cc_var
        if len(min_cut) <= approximate:
            break
    # Also for non adjacent pairs of neighbors of v.
    for x, y in iter_func(neighbors(v), 2):
        if y in G[x]:
            continue
        this_cut = minimum_st_node_cut(G, x, y, **kwargs)
        if len(min_cut) > len(this_cut):
            min_cut = this_cut
            G_copy = G.copy()
            G_copy.remove_nodes_from(this_cut)
            min_num_cc = number_connected_components(G_copy)
            min_cc_var = var([len(c) for c in sorted(nx.connected_components(G_copy))])
        if len(min_cut) == len(this_cut):
            G_copy = G.copy()
            G_copy.remove_nodes_from(this_cut)
            this_num_cc = number_connected_components(G_copy)
            this_cc_var = var([len(c) for c in sorted(nx.connected_components(G_copy))])
#            if this_num_cc >= min_num_cc:
            if this_cc_var < min_cc_var:
                min_cut = this_cut
                min_num_cc = this_num_cc
                min_cc_var = this_cc_var
#            elif this_num_cc == min_num_cc and min_cc_var > this_cc_var:
#                min_cut = this_cut
#                min_num_cc = this_num_cc
#                min_cc_var = this_cc_var
        if len(min_cut) <= approximate:
            break

    if approximate > -1 and len(min_cut) > approximate:
        info = {}
        info["Approximation threshold"] = approximate
        info["Minimum cut"] = len(min_cut)
        print(info)
    return min_cut
