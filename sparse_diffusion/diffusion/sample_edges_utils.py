import torch
import torch.nn.functional as F
from torch_geometric.utils import coalesce

import utils


def condensed_to_matrix_index(condensed_index, num_nodes):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    condensed_index: (E)
    num_nodes: (bs)
    """
    b = 1 - (2 * num_nodes)
    i = torch.div(
        (-b - torch.sqrt(b**2 - 8 * condensed_index)), 2, rounding_mode="floor"
    )
    j = condensed_index + torch.div(i * (b + i + 2), 2, rounding_mode="floor") + 1
    return torch.vstack((i.long(), j.long()))


def matrix_to_condensed_index(matrix_index, num_nodes):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    matrix_index: (2, E)
    num_nodes: (bs).
    """
    n = num_nodes
    i = matrix_index[0]
    j = matrix_index[1]
    index = n * (n - 1) / 2 - (n - i) * (n - i - 1) / 2 + j - i - 1
    return index


def matrix_to_condensed_index_batch(matrix_index, num_nodes, edge_batch):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    matrix_index: (2, E)
    num_nodes: (bs).
    """
    n = num_nodes[edge_batch]
    i = matrix_index[0]
    j = matrix_index[1]
    index = n * (n - 1) / 2 - (n - i) * (n - i - 1) / 2 + j - i - 1
    return index


def condensed_to_matrix_index_batch(condensed_index, num_nodes, edge_batch, ptr):
    """From https://stackoverflow.com/questions/5323818/condensed-matrix-function-to-find-pairs.
    condensed_index: (E) example: [0, 1, 0, 2] where [0, 1] are edges for graph0 and [0,2] edges for graph 1
    num_nodes: (bs)
    edge_batch: (E): tells to which graph each edge belongs
    ptr: (bs+1): contains the offset for the number of nodes in each graph.
    """
    bb = -2 * num_nodes[edge_batch] + 1

    # Edge ptr adds an offset of n (n-1) / 2 to each edge index
    ptr_condensed_index = condensed_index
    ii = torch.div(
        (-bb - torch.sqrt(bb**2 - 8 * ptr_condensed_index)), 2, rounding_mode="floor"
    )
    jj = (
        ptr_condensed_index
        + torch.div(ii * (bb + ii + 2), 2, rounding_mode="floor")
        + 1
    )
    return torch.vstack((ii.long(), jj.long())) + ptr[edge_batch]


def get_computational_graph(
    triu_query_edge_index,
    clean_edge_index,
    clean_edge_attr,
    triu=True,
):
    """
    concat and remove repeated edges of query_edge_index and clean_edge_index
    mask the position of query_edge_index
    in case where query_edge_attr is None, return query_edge_attr as 0
    else, return query_edge_attr for all query_edge_index
    (used in apply noise, when we need to sample the query edge attr)
    """
    # get dimension information
    de = clean_edge_attr.shape[-1]
    device = triu_query_edge_index.device

    # create default query edge attr
    default_query_edge_attr = torch.zeros((triu_query_edge_index.shape[1], de)).to(
        device
    )
    default_query_edge_attr[:, 0] = 1

    # if query_edge_attr is None, use default query edge attr
    if triu:
        # make random edges symmetrical
        query_edge_index, default_query_edge_attr = utils.to_undirected(
            triu_query_edge_index, default_query_edge_attr
        )
        _, default_query_edge_attr = utils.to_undirected(
            triu_query_edge_index, default_query_edge_attr
        )
    else:
        query_edge_index, default_query_edge_attr = triu_query_edge_index, default_query_edge_attr

    # get the computational graph: positive edges + random edges
    comp_edge_index = torch.hstack([clean_edge_index, query_edge_index])
    default_comp_edge_attr = torch.argmax(
        torch.vstack([clean_edge_attr, default_query_edge_attr]), -1
    )

    # reduce repeated edges and get the mask
    assert comp_edge_index.dtype == torch.long
    _, min_default_edge_attr = coalesce(
        comp_edge_index, default_comp_edge_attr, reduce="min"
    )

    max_comp_edge_index, max_default_edge_attr = coalesce(
        comp_edge_index, default_comp_edge_attr, reduce="max"
    )
    query_mask = min_default_edge_attr == 0
    comp_edge_attr = F.one_hot(max_default_edge_attr.long(), num_classes=de).float()

    return query_mask, max_comp_edge_index, comp_edge_attr


def check_symmetry(edge_index):
    cond1 = edge_index[0].sort()[0].equal(edge_index[1].sort()[0])
    cond2 = (edge_index[0] < edge_index[1]).sum() == (
        edge_index[1] < edge_index[0]
    ).sum()
    print((edge_index[0] < edge_index[1]).sum(), (edge_index[1] < edge_index[0]).sum())
    return cond1 and cond2


def mask_query_graph_from_comp_graph(
    triu_query_edge_index, edge_index, edge_attr, num_classes
):
    query_edge_index = utils.to_undirected(triu_query_edge_index)
    # import pdb; pdb.set_trace()

    all_edge_index = torch.hstack([edge_index, query_edge_index])
    all_edge_attr = torch.hstack(
        [
            torch.argmax(edge_attr, -1),
            torch.zeros(query_edge_index.shape[1]).to(edge_index.device),
        ]
    )

    assert all_edge_index.dtype == torch.long
    _, min_edge_attr = coalesce(all_edge_index, all_edge_attr, reduce="min")

    max_edge_index, max_edge_attr = coalesce(
        all_edge_index, all_edge_attr, reduce="max"
    )

    return (
        min_edge_attr == 0,
        F.one_hot(max_edge_attr.long(), num_classes=num_classes),
        max_edge_index,
    )


def sample_non_existing_edge_attr(query_edges_dist_batch, num_edges_to_sample):
    device = query_edges_dist_batch.device
    max_edges_to_sample = int(num_edges_to_sample.max())

    if max_edges_to_sample == 0:
        return torch.tensor([]).to(device)

    query_mask = (
        torch.ones((len(num_edges_to_sample), max_edges_to_sample))
        .cumsum(-1)
        .to(device)
    )
    query_mask[
        query_mask > num_edges_to_sample.unsqueeze(-1).repeat(1, max_edges_to_sample)
    ] = 0
    query_mask[query_mask > 0] = 1
    query_edge_attr = (
        torch.multinomial(query_edges_dist_batch, max_edges_to_sample, replacement=True)
        + 1
    )
    query_edge_attr = query_edge_attr.flatten()[query_mask.flatten().bool()]

    return query_edge_attr
