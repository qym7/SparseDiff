###############################################################################
#
# Adapted from https://github.com/lrjconan/GRAN/ which in turn is adapted from https://github.com/JiaxuanYou/graph-generation
#
###############################################################################

##Navigate to the ./util/orca directory and compile orca.cpp
# g++ -O2 -std=c++11 -o orca orca.cpp
import os
import copy
import random

import dgl
import wandb
import pygsp as pg
import secrets
import torch
import torch.nn as nn
import numpy as np
import networkx as nx
import subprocess as sp
import concurrent.futures

try:
    import graph_tool.all as gt
except ModuleNotFoundError:
    print("Graph tool could not be loaded")
from datetime import datetime
from scipy.linalg import eigvalsh
from scipy.stats import chi2
from string import ascii_uppercase, digits
from torch_geometric.utils import (
    to_dense_adj,
    is_undirected,
    to_networkx,
    remove_self_loops,
)

from sparse_diffusion.utils import SparsePlaceHolder
from sparse_diffusion.analysis.dist_helper import (
    compute_mmd,
    gaussian_emd,
    gaussian,
    emd,
    gaussian_tv,
    disc,
)
from sparse_diffusion.metrics.neural_metrics import (
    FIDEvaluation,
    MMDEvaluation,
    load_feature_extractor,
)


from sparse_diffusion.utils import SparsePlaceHolder

PRINT_TIME = False
__all__ = [
    "degree_stats",
    "clustering_stats",
    "orbit_stats_all",
    "spectral_stats",
    "eval_acc_lobster_graph",
]


def degree_worker(G):
    return np.array(nx.degree_histogram(G))


def degree_stats(graph_ref_list, graph_pred_list, is_parallel=True, compute_emd=False):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_ref_list):
                sample_ref.append(deg_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for deg_hist in executor.map(degree_worker, graph_pred_list_remove_empty):
                sample_pred.append(deg_hist)
    else:
        for i in range(len(graph_ref_list)):
            degree_temp = np.array(nx.degree_histogram(graph_ref_list[i]))
            sample_ref.append(degree_temp)
        for i in range(len(graph_pred_list_remove_empty)):
            degree_temp = np.array(nx.degree_histogram(graph_pred_list_remove_empty[i]))
            sample_pred.append(degree_temp)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)
    return mmd_dist


###############################################################################


def spectral_worker(G, n_eigvals=-1):
    # eigs = nx.laplacian_spectrum(G)
    try:
        eigs = eigvalsh(nx.normalized_laplacian_matrix(G).todense())
    except:
        eigs = np.zeros(G.number_of_nodes())
    if n_eigvals > 0:
        eigs = eigs[1 : n_eigvals + 1]
    eigs = np.clip(eigs, 1e-5, 2)
    spectral_pmf, _ = np.histogram(eigs, bins=200, range=(-1e-5, 2), density=False)
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def get_spectral_pmf(eigs, max_eig):
    spectral_pmf, _ = np.histogram(
        np.clip(eigs, 0, max_eig), bins=200, range=(-1e-5, max_eig), density=False
    )
    spectral_pmf = spectral_pmf / spectral_pmf.sum()
    return spectral_pmf


def eigval_stats(
    eig_ref_list, eig_pred_list, max_eig=20, is_parallel=True, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_ref_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_pmf,
                eig_pred_list,
                [max_eig for i in range(len(eig_ref_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eig_ref_list)):
            spectral_temp = get_spectral_pmf(eig_ref_list[i])
            sample_ref.append(spectral_temp)
        for i in range(len(eig_pred_list)):
            spectral_temp = get_spectral_pmf(eig_pred_list[i])
            sample_pred.append(spectral_temp)

    if compute_emd:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing eig mmd: ", elapsed)
    return mmd_dist


def eigh_worker(G):
    L = nx.normalized_laplacian_matrix(G).todense()
    try:
        eigvals, eigvecs = np.linalg.eigh(L)
    except:
        eigvals = np.zeros(L[0, :].shape)
        eigvecs = np.zeros(L.shape)
    return (eigvals, eigvecs)


def compute_list_eigh(graph_list, is_parallel=False):
    eigval_list = []
    eigvec_list = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for e_U in executor.map(eigh_worker, graph_list):
                eigval_list.append(e_U[0])
                eigvec_list.append(e_U[1])
    else:
        for i in range(len(graph_list)):
            e_U = eigh_worker(graph_list[i])
            eigval_list.append(e_U[0])
            eigvec_list.append(e_U[1])
    return eigval_list, eigvec_list


def get_spectral_filter_worker(eigvec, eigval, filters, bound=1.4):
    ges = filters.evaluate(eigval)
    linop = []
    for ge in ges:
        linop.append(eigvec @ np.diag(ge) @ eigvec.T)
    linop = np.array(linop)
    norm_filt = np.sum(linop**2, axis=2)
    hist_range = [0, bound]
    hist = np.array(
        [np.histogram(x, range=hist_range, bins=100)[0] for x in norm_filt]
    )  # NOTE: change number of bins
    return hist.flatten()


def spectral_filter_stats(
    eigvec_ref_list,
    eigval_ref_list,
    eigvec_pred_list,
    eigval_pred_list,
    is_parallel=False,
    compute_emd=False,
):
    """Compute the distance between the eigvector sets.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    prev = datetime.now()

    class DMG(object):
        """Dummy Normalized Graph"""

        lmax = 2

    n_filters = 12
    filters = pg.filters.Abspline(DMG, n_filters)
    bound = np.max(filters.evaluate(np.arange(0, 2, 0.01)))
    sample_ref = []
    sample_pred = []
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_ref_list,
                eigval_ref_list,
                [filters for i in range(len(eigval_ref_list))],
                [bound for i in range(len(eigval_ref_list))],
            ):
                sample_ref.append(spectral_density)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                get_spectral_filter_worker,
                eigvec_pred_list,
                eigval_pred_list,
                [filters for i in range(len(eigval_ref_list))],
                [bound for i in range(len(eigval_ref_list))],
            ):
                sample_pred.append(spectral_density)
    else:
        for i in range(len(eigval_ref_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_ref_list[i], eigval_ref_list[i], filters, bound
                )
                sample_ref.append(spectral_temp)
            except:
                pass
        for i in range(len(eigval_pred_list)):
            try:
                spectral_temp = get_spectral_filter_worker(
                    eigvec_pred_list[i], eigval_pred_list[i], filters, bound
                )
                sample_pred.append(spectral_temp)
            except:
                pass

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing spectral filter stats: ", elapsed)
    return mmd_dist


def spectral_stats(
    graph_ref_list, graph_pred_list, is_parallel=True, n_eigvals=-1, compute_emd=False
):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Args:
        graph_ref_list, graph_target_list: two lists of networkx graphs to be evaluated
    """
    sample_ref = []
    sample_pred = []
    # in case an empty graph is generated
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        i = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker, graph_ref_list, [n_eigvals for i in graph_ref_list]
            ):
                sample_ref.append(spectral_density)
                # import pdb; pdb.set_trace()
                if i == 0:
                    print("train spectral_density is", spectral_density)
                i += 1

        i = 0
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for spectral_density in executor.map(
                spectral_worker,
                graph_pred_list_remove_empty,
                [n_eigvals for i in graph_ref_list],
            ):
                # import pdb; pdb.set_trace()
                sample_pred.append(spectral_density)
                if i == 0:
                    print("test spectral_density is", spectral_density)
                i += 1
    else:
        i = 0
        for i in range(len(graph_ref_list)):
            spectral_temp = spectral_worker(graph_ref_list[i], n_eigvals)
            sample_ref.append(spectral_temp)
            if i == 0:
                print("train spectral_density is", spectral_temp)
            i += 1

        i = 0
        for i in range(len(graph_pred_list_remove_empty)):
            spectral_temp = spectral_worker(graph_pred_list_remove_empty[i], n_eigvals)
            sample_pred.append(spectral_temp)
            if i == 0:
                print("test spectral_density is", spectral_temp)
            i += 1

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd)
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_emd)
    else:
        mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian_tv)
    # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing degree mmd: ", elapsed)

    return mmd_dist


###############################################################################


def clustering_worker(param):
    G, bins = param
    clustering_coeffs_list = list(nx.clustering(G).values())
    hist, _ = np.histogram(
        clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
    )
    return hist


def clustering_stats(
    graph_ref_list, graph_pred_list, bins=100, is_parallel=True, compute_emd=False
):
    sample_ref = []
    sample_pred = []
    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    prev = datetime.now()
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_ref_list]
            ):
                sample_ref.append(clustering_hist)
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for clustering_hist in executor.map(
                clustering_worker, [(G, bins) for G in graph_pred_list_remove_empty]
            ):
                sample_pred.append(clustering_hist)

    else:
        for i in range(len(graph_ref_list)):
            clustering_coeffs_list = list(nx.clustering(graph_ref_list[i]).values())
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_ref.append(hist)

        for i in range(len(graph_pred_list_remove_empty)):
            clustering_coeffs_list = list(
                nx.clustering(graph_pred_list_remove_empty[i]).values()
            )
            hist, _ = np.histogram(
                clustering_coeffs_list, bins=bins, range=(0.0, 1.0), density=False
            )
            sample_pred.append(hist)

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=emd, sigma=1.0 / 10)
        mmd_dist = compute_mmd(
            sample_ref,
            sample_pred,
            kernel=gaussian_emd,
            sigma=1.0 / 10,
            distance_scaling=bins,
        )
    else:
        mmd_dist = compute_mmd(
            sample_ref, sample_pred, kernel=gaussian_tv, sigma=1.0 / 10
        )

    elapsed = datetime.now() - prev
    if PRINT_TIME:
        print("Time computing clustering mmd: ", elapsed)
    return mmd_dist


# maps motif/orbit name string to its corresponding list of indices from orca output
motif_to_indices = {
    "3path": [1, 2],
    "4cycle": [8],
}
COUNT_START_STR = "orbit counts:"


def edge_list_reindexed(G):
    idx = 0
    id2idx = dict()
    for u in G.nodes():
        id2idx[str(u)] = idx
        idx += 1

    edges = []
    for u, v in G.edges():
        edges.append((id2idx[str(u)], id2idx[str(v)]))
    return edges


def orca(graph):
    # tmp_fname = f'analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = f'../analysis/orca/tmp_{"".join(secrets.choice(ascii_uppercase + digits) for i in range(8))}.txt'
    tmp_fname = os.path.join(os.path.dirname(os.path.realpath(__file__)), tmp_fname)
    # print(tmp_fname, flush=True)
    f = open(tmp_fname, "w")
    f.write(str(graph.number_of_nodes()) + " " + str(graph.number_of_edges()) + "\n")
    for u, v in edge_list_reindexed(graph):
        f.write(str(u) + " " + str(v) + "\n")
    f.close()

    output = sp.check_output(
        [
            str(
                os.path.join(
                    os.path.dirname(os.path.realpath(__file__)), "../analysis/orca/orca"
                )
            ),
            "node",
            "4",
            tmp_fname,
            "std",
        ]
    )
    output = output.decode("utf8").strip()
    idx = output.find(COUNT_START_STR) + len(COUNT_START_STR) + 2
    output = output[idx:]
    node_orbit_counts = np.array(
        [
            list(map(int, node_cnts.strip().split(" ")))
            for node_cnts in output.strip("\n").split("\n")
        ]
    )

    try:
        os.remove(tmp_fname)
    except OSError:
        pass

    return node_orbit_counts


def motif_stats(
    graph_ref_list,
    graph_pred_list,
    motif_type="4cycle",
    ground_truth_match=None,
    bins=100,
    compute_emd=False,
):
    # graph motif counts (int for each graph)
    # normalized by graph size
    total_counts_ref = []
    total_counts_pred = []

    num_matches_ref = []
    num_matches_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]
    indices = motif_to_indices[motif_type]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_ref.append(match_cnt / G.number_of_nodes())

        # hist, _ = np.histogram(
        #        motif_counts, bins=bins, density=False)
        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_ref.append(motif_temp)

    for G in graph_pred_list_remove_empty:
        orbit_counts = orca(G)
        motif_counts = np.sum(orbit_counts[:, indices], axis=1)

        if ground_truth_match is not None:
            match_cnt = 0
            for elem in motif_counts:
                if elem == ground_truth_match:
                    match_cnt += 1
            num_matches_pred.append(match_cnt / G.number_of_nodes())

        motif_temp = np.sum(motif_counts) / G.number_of_nodes()
        total_counts_pred.append(motif_temp)

    total_counts_ref = np.array(total_counts_ref)[:, None]
    total_counts_pred = np.array(total_counts_pred)[:, None]

    if compute_emd:
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, is_hist=False)
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref, total_counts_pred, kernel=gaussian, is_hist=False
        )
    return mmd_dist


def orbit_stats_all(graph_ref_list, graph_pred_list, compute_emd=False):
    total_counts_ref = []
    total_counts_pred = []

    graph_pred_list_remove_empty = [
        G for G in graph_pred_list if not G.number_of_nodes() == 0
    ]

    for G in graph_ref_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_ref.append(orbit_counts_graph)

    for G in graph_pred_list:
        orbit_counts = orca(G)
        orbit_counts_graph = np.sum(orbit_counts, axis=0) / G.number_of_nodes()
        total_counts_pred.append(orbit_counts_graph)

    total_counts_ref = np.array(total_counts_ref)
    total_counts_pred = np.array(total_counts_pred)

    if compute_emd:
        # mmd_dist = compute_mmd(total_counts_ref, total_counts_pred, kernel=emd, sigma=30.0)
        # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian,
            is_hist=False,
            sigma=30.0,
        )
    else:
        mmd_dist = compute_mmd(
            total_counts_ref,
            total_counts_pred,
            kernel=gaussian_tv,
            is_hist=False,
            sigma=30.0,
        )
    return mmd_dist


def eval_acc_lobster_graph(G_list):
    G_list = [copy.deepcopy(gg) for gg in G_list]
    count = 0
    for gg in G_list:
        if is_lobster_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_tree_graph(G_list):
    count = 0
    for gg in G_list:
        if nx.is_tree(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_grid_graph(G_list, grid_start=10, grid_end=20):
    count = 0
    for gg in G_list:
        if is_grid_graph(gg):
            count += 1
    return count / float(len(G_list))


def eval_acc_sbm_graph(
    G_list,
    p_intra=0.3,
    p_inter=0.005,
    strict=True,
    refinement_steps=1000,
    is_parallel=True,
):
    count = 0.0
    if is_parallel:
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for prob in executor.map(
                is_sbm_graph,
                [gg for gg in G_list],
                [p_intra for i in range(len(G_list))],
                [p_inter for i in range(len(G_list))],
                [strict for i in range(len(G_list))],
                [refinement_steps for i in range(len(G_list))],
            ):
                count += prob
    else:
        for gg in G_list:
            count += is_sbm_graph(
                gg,
                p_intra=p_intra,
                p_inter=p_inter,
                strict=strict,
                refinement_steps=refinement_steps,
            )
    return count / float(len(G_list))


def eval_acc_planar_graph(G_list):
    count = 0
    for gg in G_list:
        if is_planar_graph(gg):
            count += 1
    return count / float(len(G_list))


def is_planar_graph(G):
    return nx.is_connected(G) and nx.check_planarity(G)[0]


def is_lobster_graph(G):
    """
    Check a given graph is a lobster graph or not

    Removing leaf nodes twice:

    lobster -> caterpillar -> path

    """
    ### Check if G is a tree
    if nx.is_tree(G):
        G = G.copy()
        ### Check if G is a path after removing leaves twice
        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        leaves = [n for n, d in G.degree() if d == 1]
        G.remove_nodes_from(leaves)

        num_nodes = len(G.nodes())
        num_degree_one = [d for n, d in G.degree() if d == 1]
        num_degree_two = [d for n, d in G.degree() if d == 2]

        if sum(num_degree_one) == 2 and sum(num_degree_two) == 2 * (num_nodes - 2):
            return True
        elif sum(num_degree_one) == 0 and sum(num_degree_two) == 0:
            return True
        else:
            return False
    else:
        return False


def is_grid_graph(G):
    """
    Check if the graph is grid, by comparing with all the real grids with the same node count
    """
    all_grid_file = f"data/all_grids.pt"
    if os.path.isfile(all_grid_file):
        all_grids = torch.load(all_grid_file)
    else:
        all_grids = {}
        for i in range(2, 20):
            for j in range(2, 20):
                G_grid = nx.grid_2d_graph(i, j)
                n_nodes = f"{len(G_grid.nodes())}"
                all_grids[n_nodes] = all_grids.get(n_nodes, []) + [G_grid]
        torch.save(all_grids, all_grid_file)

    n_nodes = f"{len(G.nodes())}"
    if n_nodes in all_grids:
        for G_grid in all_grids[n_nodes]:
            if nx.faster_could_be_isomorphic(G, G_grid):
                if nx.is_isomorphic(G, G_grid):
                    return True
        return False
    else:
        return False


def is_sbm_graph(G, p_intra=0.3, p_inter=0.005, strict=True, refinement_steps=1000):
    """
    Check if how closely given graph matches a SBM with given probabilites by computing mean probability of Wald test statistic for each recovered parameter
    """

    adj = nx.adjacency_matrix(G).toarray()
    idx = adj.nonzero()
    g = gt.Graph()
    g.add_edge_list(np.transpose(idx))
    try:
        state = gt.minimize_blockmodel_dl(g)
    except ValueError:
        if strict:
            return False
        else:
            return 0.0

    # Refine using merge-split MCMC
    for i in range(refinement_steps):
        state.multiflip_mcmc_sweep(beta=np.inf, niter=10)

    b = state.get_blocks()
    b = gt.contiguous_map(state.get_blocks())
    state = state.copy(b=b)
    e = state.get_matrix()
    n_blocks = state.get_nonempty_B()
    node_counts = state.get_nr().get_array()[:n_blocks]
    edge_counts = e.todense()[:n_blocks, :n_blocks]
    if strict:
        if (
            (node_counts > 40).sum() > 0
            or (node_counts < 20).sum() > 0
            or n_blocks > 5
            or n_blocks < 2
        ):
            return False

    max_intra_edges = node_counts * (node_counts - 1)
    est_p_intra = np.diagonal(edge_counts) / (max_intra_edges + 1e-6)

    max_inter_edges = node_counts.reshape((-1, 1)) @ node_counts.reshape((1, -1))
    np.fill_diagonal(edge_counts, 0)
    est_p_inter = edge_counts / (max_inter_edges + 1e-6)

    W_p_intra = (est_p_intra - p_intra) ** 2 / (est_p_intra * (1 - est_p_intra) + 1e-6)
    W_p_inter = (est_p_inter - p_inter) ** 2 / (est_p_inter * (1 - est_p_inter) + 1e-6)

    W = W_p_inter.copy()
    np.fill_diagonal(W, W_p_intra)
    p = 1 - chi2.cdf(abs(W), 1)
    p = p.mean()
    if strict:
        return p > 0.9  # p value < 10 %
    else:
        return p


def eval_fraction_isomorphic(fake_graphs, train_graphs):
    count = 0
    for fake_g in fake_graphs:
        for train_g in train_graphs:
            if nx.faster_could_be_isomorphic(fake_g, train_g):
                if nx.is_isomorphic(fake_g, train_g):
                    count += 1
                    break
    return count / float(len(fake_graphs))


def eval_fraction_unique(fake_graphs, precise=False):
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True
        if not fake_g.number_of_nodes() == 0:
            for fake_old in fake_evaluated:
                if precise:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.is_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
                else:
                    if nx.faster_could_be_isomorphic(fake_g, fake_old):
                        if nx.could_be_isomorphic(fake_g, fake_old):
                            count_non_unique += 1
                            unique = False
                            break
            if unique:
                fake_evaluated.append(fake_g)

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs

    return frac_unique


def eval_fraction_unique_non_isomorphic_valid(
    fake_graphs, train_graphs, validity_func=(lambda x: True)
):
    count_valid = 0
    count_isomorphic = 0
    count_non_unique = 0
    fake_evaluated = []
    for fake_g in fake_graphs:
        unique = True

        for fake_old in fake_evaluated:
            if nx.faster_could_be_isomorphic(fake_g, fake_old):
                if nx.is_isomorphic(fake_g, fake_old):
                    count_non_unique += 1
                    unique = False
                    break
        if unique:
            fake_evaluated.append(fake_g)
            non_isomorphic = True
            for train_g in train_graphs:
                if nx.faster_could_be_isomorphic(fake_g, train_g):
                    if nx.is_isomorphic(fake_g, train_g):
                        count_isomorphic += 1
                        non_isomorphic = False
                        break
            if non_isomorphic:
                if validity_func(fake_g):
                    count_valid += 1

    frac_unique = (float(len(fake_graphs)) - count_non_unique) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs
    frac_unique_non_isomorphic = (
        float(len(fake_graphs)) - count_non_unique - count_isomorphic
    ) / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set
    frac_unique_non_isomorphic_valid = count_valid / float(
        len(fake_graphs)
    )  # Fraction of distinct isomorphism classes in the fake graphs that are not in the training set and are valid
    return frac_unique, frac_unique_non_isomorphic, frac_unique_non_isomorphic_valid


class SpectreSamplingMetrics(nn.Module):
    def __init__(self, dataloaders, test, compute_emd, metrics_list):
        super().__init__()

        self.train_graphs = self.loader_to_nx(dataloaders["train"])
        # self.val_graphs = self.loader_to_nx(dataloaders["val"])
        # split = 'test' if test else 'val'
        split = "test" if test else "test"  # to be deleted
        self.test_graphs = self.loader_to_nx(dataloaders[split])
        print(self.test_graphs[0])
        # import pdb; pdb.set_trace()
        self.num_graphs_test = len(self.test_graphs)
        # self.num_graphs_val = len(self.val_graphs)
        print("num_train_graphs is", len(self.train_graphs))
        print("num_graphs_test is", self.num_graphs_test)
        # print('num_graphs_val is', self.num_graphs_val)
        self.compute_emd = compute_emd
        self.metrics_list = metrics_list

    def loader_to_nx(self, loader):
        networkx_graphs = []
        for i, batch in enumerate(loader):
            data_list = batch.to_data_list()
            for j, data in enumerate(data_list):
                nx_graph1 = to_networkx(
                    data,
                    node_attrs=None,
                    edge_attrs=None,
                    to_undirected=True,
                    remove_self_loops=True,
                )
                networkx_graphs.append(nx_graph1)  # 0.448

        return networkx_graphs

    def neural_metrics(self, generated):
        # set seed
        seed = 0
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Neural metrics
        gin_model = load_feature_extractor(
            device="cpu"
        )  # take a gin-model with predefined params and random weights
        fid_evaluator = FIDEvaluation(model=gin_model)
        rbf_evaluator = MMDEvaluation(
            model=gin_model, kernel="rbf", sigma="range", multiplier="mean"
        )
        # pass the generated graphs and reference graphs to networkx
        generated_max_comp = []
        test_max_comp = []
        for g in generated:
            # largest_cc = max(nx.connected_components(g), key=len)
            # g = g.subgraph(largest_cc)
            g = dgl.DGLGraph(g)
            generated_max_comp.append(g)
        for g in self.test_graphs:
            # largest_cc = max(nx.connected_components(g), key=len)
            # g = g.subgraph(largest_cc)
            g = dgl.DGLGraph(g)
            test_max_comp.append(g)

        (generated_dataset, reference_dataset), _ = fid_evaluator.get_activations(
            generated_max_comp, test_max_comp
        )
        fid, _ = fid_evaluator.evaluate(
            generated_dataset=generated_dataset, reference_dataset=reference_dataset
        )
        rbf, _ = rbf_evaluator.evaluate(
            generated_dataset=generated_dataset, reference_dataset=reference_dataset
        )

        return fid, rbf

    def forward(self, generated_graphs: SparsePlaceHolder, current_epoch, val_counter):
        num_generated_graphs = generated_graphs.batch.max().item() + 1
        print(
            f"Computing sampling metrics between {num_generated_graphs} generated graphs and {len(self.test_graphs)}"
            f" test graphs -- emd computation: {self.compute_emd}"
        )
        networkx_graphs = []
        adjacency_matrices = []

        print("Building networkx graphs...")
        edge_batch = generated_graphs.batch[generated_graphs.edge_index[0]]
        for graph_idx in range(num_generated_graphs):
            edge_index = generated_graphs.edge_index[:, edge_batch == graph_idx]
            edge_index = (
                edge_index - (generated_graphs.batch < graph_idx).sum()
            )  # delete the offset for edge index
            max_num_nodes = (generated_graphs.batch == graph_idx).sum()
            # original edge_attr are all values bigger than 0
            A = to_dense_adj(
                edge_index=edge_index, max_num_nodes=max_num_nodes
            )  # (1, n, n)
            A = A.int().cpu().numpy()[0]
            adjacency_matrices.append(A)

            nx_graph = nx.from_numpy_array(A)
            networkx_graphs.append(nx_graph)

        # networkx_graphs = self.train_graphs # to be deleted

        print("Saving all adjacency matrices")
        np.savez(f"generated_adjs{random.randint(0, 10000)}.npz", *adjacency_matrices)

        to_log = {}

        if "degree" in self.metrics_list:
            print("Computing degree stats..")
            degree = degree_stats(
                self.test_graphs,
                networkx_graphs,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log["degree"] = degree
            if wandb.run is not None:
                wandb.run.summary["degree"] = degree

        if "clustering" in self.metrics_list:
            print("Computing clustering stats...")
            clustering = clustering_stats(
                self.test_graphs,
                networkx_graphs,
                bins=100,
                is_parallel=True,
                compute_emd=self.compute_emd,
            )
            to_log["clustering"] = clustering
            if wandb.run is not None:
                wandb.run.summary["clustering"] = clustering

        if "orbit" in self.metrics_list:
            print("Computing orbit stats...")
            orbit = orbit_stats_all(
                self.test_graphs, networkx_graphs, compute_emd=self.compute_emd
            )
            to_log["orbit"] = orbit
            if wandb.run is not None:
                wandb.run.summary["orbit"] = orbit

        if "spectre" in self.metrics_list:
            print("Computing spectre stats...")
            spectre = spectral_stats(
                self.test_graphs,
                networkx_graphs,
                is_parallel=False,
                # is_parallel=True,
                n_eigvals=-1,
                compute_emd=self.compute_emd,
            )
            to_log["spectre"] = spectre
            if wandb.run is not None:
                wandb.run.summary["spectre"] = spectre

        if "neural" in self.metrics_list:
            print("Computing neural metrics including FID and RBF MMD")
            fid, rbf = self.neural_metrics(networkx_graphs)
            to_log["fid"] = fid["fid"]
            to_log["rbf mmd"] = rbf["mmd_rbf"]

            if wandb.run is not None:
                wandb.run.summary["fid"] = fid["fid"]
                wandb.run.summary["rbf mmd"] = rbf["mmd_rbf"]

        if "motif" in self.metrics_list:
            print("Computing motif stats")
            motif = motif_stats(
                self.test_graphs,
                networkx_graphs,
                motif_type="4cycle",
                ground_truth_match=None,
                bins=100,
                compute_emd=self.compute_emd,
            )
            to_log["motif"] = motif
            if wandb.run is not None:
                wandb.run.summary["motif"] = motif

        if "sbm" in self.metrics_list:
            print("Computing accuracy...")
            acc = eval_acc_sbm_graph(networkx_graphs, refinement_steps=100, strict=True)
            to_log["sbm_acc"] = acc
            if wandb.run is not None:
                wandb.run.summary["sbmacc"] = acc

        if "planar" in self.metrics_list:
            print("Computing planar accuracy...")
            planar_acc = eval_acc_planar_graph(networkx_graphs)
            to_log["planar_acc"] = planar_acc
            if wandb.run is not None:
                wandb.run.summary["planar_acc"] = planar_acc

        return to_log

    def reset(self):
        pass


class Comm20SamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, dataloaders, test):
        super().__init__(
            dataloaders=dataloaders,
            compute_emd=False,
            test=test,
            metrics_list=["degree", "clustering", "orbit", "neural"],
        )


class PlanarSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, dataloaders, test):
        super().__init__(
            dataloaders=dataloaders,
            compute_emd=False,
            test=test,
            metrics_list=[
                "degree",
                "clustering",
                "orbit",
                "spectre",
                "planar",
                "neural",
            ],
        )


class SBMSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, dataloaders, test):
        super().__init__(
            dataloaders=dataloaders,
            compute_emd=False,
            test=test,
            metrics_list=["degree", "clustering", "orbit", "spectre", "sbm", "neural"],
        )


class ProteinSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, dataloaders, test):
        super().__init__(
            dataloaders=dataloaders,
            compute_emd=False,
            test=test,
            metrics_list=["degree", "clustering", "orbit", "spectre", "neural"],
        )


class PointCloudSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, dataloaders, test):
        super().__init__(
            dataloaders=dataloaders,
            compute_emd=False,
            test=test,
            metrics_list=["degree", "clustering", "orbit", "spectre"],
        )


class EgoSamplingMetrics(SpectreSamplingMetrics):
    def __init__(self, dataloaders, test):
        super().__init__(
            dataloaders=dataloaders,
            compute_emd=False,
            test=test,
            metrics_list=["degree", "clustering", "orbit", "spectre", "neural"],
        )


def degree_histogram(generated_graphs, target, compute_emd=False):
    """Compute the distance between the degree distributions of two unordered sets of graphs.
    Generated graphs: List<torch_geometric.data>
    Target: histogram of degrees
    """

    all_hists = []
    for data in generated_graphs:
        assert is_undirected(data.edge_index, data.edge_attr, data.num_nodes)

        _, degrees = torch.unique(data.edge_index[0], return_counts=True)
        max_degree = max(degrees)
        deg_hist = torch.zeros(max_degree + 1)
        deg_val, deg_count = torch.unique(degrees, return_counts=True)
        for val, count in zip(deg_val, deg_count):
            deg_hist[val] = count
        all_hists.append(deg_hist)

        if compute_emd:
            # EMD option uses the same computation as GraphRNN, the alternative is MMD as computed by GRAN
            mmd_dist = compute_mmd(target, all_hists, kernel=gaussian_emd)
        else:
            mmd_dist = compute_mmd(target, all_hists, kernel=gaussian_tv)
        # mmd_dist = compute_mmd(sample_ref, sample_pred, kernel=gaussian)

        return mmd_dist
