# Python built-in
import math

# Installed packages
import torch
from torch import Tensor
from torch_geometric.utils import to_dense_adj
import torch_geometric.nn.pool as pool

# My files
import utils
from sparse_diffusion.diffusion.sample_edges_utils import (
    matrix_to_condensed_index,
    condensed_to_matrix_index_batch,
    condensed_to_matrix_index,
    matrix_to_condensed_index_batch,
)


def test_sample_query_edges():
    if False:
        condensed_index = torch.tensor([0, 1, 0, 2])
        num_nodes_per_graph = torch.tensor([5, 4])
        edge_batch = torch.tensor([0, 0, 1, 1])
        offset = torch.tensor([0, 5])
        edge_index, batch = condensed_to_matrix_index_batch(
            condensed_index,
            num_nodes=num_nodes_per_graph,
            edge_batch=edge_batch,
            ptr=offset,
        )
        print([edge_index[:, i] for i in range(edge_index.shape[1])])

    if False:
        for i in range(2, 20):
            num_nodes_per_graph = torch.tensor([i], dtype=torch.long)
            edge_proportion = 0.5
            edge_index, batch = sample_query_edges(num_nodes_per_graph, edge_proportion)
            print_output(edge_index, num_nodes_per_graph)

    if False:
        for i in range(2, 20):
            num_nodes_per_graph = torch.tensor([i])
            edge_proportion = 1
            edge_index, batch = sample_query_edges(num_nodes_per_graph, edge_proportion)
            print_output(edge_index, num_nodes_per_graph)

    if False:
        for i in range(2, 20):
            num_nodes_per_graph = torch.tensor([i])
            edge_proportion = 0.0001
            edge_index, batch = sample_query_edges(num_nodes_per_graph, edge_proportion)
            print_output(edge_index, num_nodes_per_graph)

    if False:
        for i in range(2, 20):
            num_nodes_per_graph = torch.tensor([8, 8])
            edge_proportion = 0.5
            edge_index, batch = sample_query_edges(num_nodes_per_graph, edge_proportion)
            print_output(edge_index, num_nodes_per_graph, batch)

    if False:
        for i in range(2, 20):
            num_nodes_per_graph = torch.tensor([4, 10])
            edge_proportion = 0.5
            edge_index, batch = sample_query_edges(num_nodes_per_graph, edge_proportion)
            print_output(edge_index, num_nodes_per_graph, batch)


def test_sample_existing_edges_batch():
    num_nodes_per_graph = torch.tensor([5, 5])
    num_edges_to_sample = torch.tensor([3, 3])
    existing_edge_index = torch.tensor(
        [[0, 0, 2, 3, 5, 5, 7, 8], [1, 3, 4, 4, 6, 8, 9, 9]]
    )
    batch = torch.tensor([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])

    for i in range(20):
        edge_index = sample_non_existing_edges_batched(
            num_edges_to_sample, existing_edge_index, num_nodes_per_graph, batch
        )

        edge_index = torch.hstack((edge_index, edge_index, existing_edge_index))

        print_output(edge_index, num_nodes_per_graph, batch)


def sampled_condensed_indices_uniformly(
    max_condensed_value, num_edges_to_sample, return_mask=False
):
    """Max_condensed value: (bs) long tensor
    num_edges_to_sample: (bs) long tensor
    Return: condensed_index e.g. [0 1 3 0 2]
    """
    assert (0 <= num_edges_to_sample).all(), (
        num_edges_to_sample <= max_condensed_value
    ).all()
    batch_size = max_condensed_value.shape[0]
    device = max_condensed_value.device

    if (
        len(torch.unique(max_condensed_value)) == 1
        and len(torch.unique(num_edges_to_sample)) == 1
    ):
        max_val = max_condensed_value[0]
        to_sample = num_edges_to_sample[0]
        sampled_condensed = torch.multinomial(
            torch.ones(max_val, device=device), num_samples=to_sample, replacement=False
        )
        edge_batch = torch.zeros(
            num_edges_to_sample[0], device=device, dtype=torch.long
        )
        if batch_size == 1:
            if return_mask:
                condensed_mask = torch.arange(num_edges_to_sample[0], device=device)
                return sampled_condensed, edge_batch, condensed_mask

            return sampled_condensed, edge_batch

        # Case of several graphs of the same size
        # Repeat the edge_index for each graph and aggregate them
        sampled_condensed_repeated = (
            sampled_condensed.unsqueeze(0).expand(batch_size, -1).flatten()
        )
        edge_batch = torch.arange(batch_size, device=device).repeat_interleave(
            to_sample
        )

        if return_mask:
            condensed_mask = torch.arange(num_edges_to_sample[0], device=device)
            condensed_mask = (
                condensed_mask.unsqueeze(0).expand(batch_size, -1).flatten()
            )
            return sampled_condensed_repeated, edge_batch, condensed_mask

        return sampled_condensed_repeated, edge_batch

    # Most general case: graphs of varying sizes
    max_size = torch.max(max_condensed_value)
    # import pdb; pdb.set_trace()
    if max_size > 10**7:
        print("[Warning]: sampling random edges might bew slow")

    randperm_full = torch.randperm(max_size, device=device)  # (max_condensed)
    randperm_expanded = randperm_full.unsqueeze(0).expand(
        batch_size, -1
    )  # (bs, max_condensed)

    # General goal: keep the indices on the left that are not too big for each graph
    # Mask1 is used to mask the indices that are too large for current graph
    mask1 = randperm_expanded < max_condensed_value.unsqueeze(1)  # (bs, max_condensed)

    # Cumsum(mask1) is the number of valid indices on the left of each index
    # Mask2 will select the right number of indices on the left
    mask2 = torch.cumsum(mask1, dim=1) <= num_edges_to_sample.unsqueeze(
        1
    )  # (bs, max_condensed)
    complete_mask = mask1 * mask2
    condensed_index = randperm_expanded[complete_mask]  # (sum(num_edges_per_graph))
    edge_batch = (
        torch.arange(batch_size, device=device)
        .unsqueeze(1)
        .expand(-1, max_size)[complete_mask]
    )

    if return_mask:
        complete_mask = complete_mask.cumsum(-1)[complete_mask] - 1
        return condensed_index, edge_batch, complete_mask

    return condensed_index, edge_batch


def sample_query_edges(
    num_nodes_per_graph: Tensor, edge_proportion=None, num_edges_to_sample=None
):
    """Sample edge_proportion % of edges in each graph
    num_nodes_per_graph: (bs): tensor of int.
    Return: edge_index, batch
    """
    assert num_nodes_per_graph.dtype == torch.long
    # num_nodes could be 1 in QM9
    assert torch.all(num_nodes_per_graph >= 1), num_nodes_per_graph

    batch_size = len(num_nodes_per_graph)
    device = num_nodes_per_graph.device

    n = num_nodes_per_graph
    max_condensed_value = (n * (n - 1) / 2).long()
    if num_edges_to_sample is None and edge_proportion is not None:
        assert 0 < edge_proportion <= 1, edge_proportion
        num_edges_to_sample = torch.ceil(edge_proportion * max_condensed_value).long()
    elif num_edges_to_sample is not None:
        assert num_edges_to_sample.dtype == torch.long
    else:
        raise ValueError(
            "Either edge_proportion or num_edges_to_sample should be provided"
        )

    condensed_index, edge_batch = sampled_condensed_indices_uniformly(
        max_condensed_value, num_edges_to_sample
    )

    if batch_size == 1:
        edge_index = condensed_to_matrix_index(condensed_index, num_nodes=n[0])
        return edge_index, torch.zeros(n, dtype=torch.long, device=device)

    if len(torch.unique(num_nodes_per_graph)) == 1:
        # Case of several graphs of the same size
        # Add the offset to the edge_index
        offset = torch.cumsum(num_nodes_per_graph, dim=0)[:-1]  # (bs - 1)
        offset = torch.cat(
            (torch.zeros(1, device=device, dtype=torch.long), offset)
        )  # (bs)

        edge_index = condensed_to_matrix_index_batch(
            condensed_index,
            num_nodes=num_nodes_per_graph,
            edge_batch=edge_batch,
            ptr=offset,
        )
        return edge_index, torch.arange(batch_size, device=device).repeat_interleave(n)

    # Most general case: graphs of varying sizes
    # condensed_index = randperm_expanded[complete_mask]                                       # (sum(num_edges_per_graph))
    offset = torch.cumsum(num_nodes_per_graph, dim=0)[:-1]  # (bs - 1)
    offset = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.long), offset)
    )  # (bs)
    edge_index = condensed_to_matrix_index_batch(
        condensed_index,
        num_nodes=num_nodes_per_graph,
        edge_batch=edge_batch,
        ptr=offset,
    )
    # Get the batch information
    batch = torch.arange(batch_size, device=device).repeat_interleave(
        num_nodes_per_graph
    )
    return edge_index, batch


def sample_non_existing_edges_batched(
    num_edges_to_sample, existing_edge_index, num_nodes, batch
):
    """Sample non-existing edges from a complete graph.
    num_edges_to_sample: (bs) long
    existing_edge_index: (2, E)
    num_nodes: (bs) long
    batch: (N) long
    existing_edge_index only contains edges that exist in the top part of triangle matrix
    """
    device = existing_edge_index.device
    unit_graph_mask = num_nodes == 1
    unit_graph_mask_offset = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.bool), unit_graph_mask[:-1])
    )

    # Compute the number of existing and non-existing edges.
    num_edges_total = (num_nodes * (num_nodes - 1) / 2).long()
    # Count existing edges using global pooling. In case a graph has no edge, global_add_pool
    # May return something of the wrong length. To avoid this, add a 0 for each graph
    # TODO: check if it can be simplified using the size argument of global add pool
    # full_edge_count = torch.hstack((torch.ones(existing_edge_index.shape[1], device=device),
    #                                 torch.zeros(batch.max()+1, device=device)))   # (ne+bs)
    # full_edge_batch = torch.hstack((batch[existing_edge_index[0]],
    #                                 torch.arange(batch.max()+1, device=device)))  # (ne+bs)
    # num_edges_existing = pool.global_add_pool(x=full_edge_count, batch=full_edge_batch).long()
    num_edges_existing = pool.global_add_pool(
        x=torch.ones(existing_edge_index.shape[1], device=device),
        batch=batch[existing_edge_index[0]],
        size=len(num_edges_to_sample),
    ).long()
    num_non_existing_edges = num_edges_total - num_edges_existing
    assert (num_edges_to_sample <= num_non_existing_edges).all(), (
        num_edges_to_sample,
        num_non_existing_edges,
    )

    # Sample non-existing edge indices without considering existing edges.
    # print("Num edges non existing", num_non_existing_edges)
    # multinomial and not randint because we want to sample without replacement
    sampled_indices, sampled_edge_batch = sampled_condensed_indices_uniformly(
        max_condensed_value=num_non_existing_edges,
        num_edges_to_sample=num_edges_to_sample,
    )

    # Compute the offset (bs, ) for each graph, where offset -> nbr of nodes, sq_offset -> nbr of edges
    # Go from a matrix problem to a 1d problem, it is easier
    existing_edge_batch = batch[existing_edge_index[0]]
    num_edges_total = (num_nodes * (num_nodes - 1) / 2).long()
    sq_offset = torch.cumsum(num_edges_total, dim=0)[:-1]  # (bs - 1)
    # Prepend a 0
    sq_offset = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.long), sq_offset)
    )  # (bs)

    offset = torch.cumsum(num_nodes, dim=0)[
        :-1
    ]  # (bs - 1)                                            # (bs - 1)
    offset = torch.cat(
        (torch.zeros(1, device=device, dtype=torch.long), offset)
    )  # (bs)
    # existing_indices (E, ) is of form [0 1 2 3 4 0 2 3 4]
    rescaled_edge_index = (
        existing_edge_index - offset[existing_edge_batch]
    )  # of form [0 1 2 3 4 0 2 3 4]
    existing_indices = matrix_to_condensed_index_batch(
        rescaled_edge_index, num_nodes=num_nodes, edge_batch=existing_edge_batch
    )

    # Add offset to the sampled indices
    # Example of sampled condensed: [0 3 1 0 2]
    epsilon = 0.1
    sampled_indices_offset = sq_offset[sampled_edge_batch]  # (E_sample, )
    # print("sampled indices", sampled_indices)
    # print("sampled edge batch", sampled_edge_batch)
    samp_ind_w_offset = sampled_indices + sampled_indices_offset
    samp_ind_w_offset = torch.sort(samp_ind_w_offset)[
        0
    ]  # E.g. [0 1 3 6 8], where [0 1 3] belong to a graph of 4 nodes, [6 8] to a graph of 3 nodes
    # print("Sampled indices with offset", samp_ind_w_offset)
    # add small value to create an order later in the sort
    samp_ind_w_offset = samp_ind_w_offset + epsilon

    # Add virtual edges to the existing edges to mark the beginning of each graph, for batch processing
    # After adding epsilon, sqrt_ptr is smaller than all edges of the next graph, and bigger than all edges of the current graph
    # * when there exists graphs with size 1, there might be identical values in sq_offset, also in virtual nodes
    existing_ind_w_offset = existing_indices + sq_offset[existing_edge_batch]
    virtual_nodes = (
        sq_offset - epsilon
    )  # Introduce virtual nodes that will be used later to split graphs
    # add different offset for graphs of size 1 to separate them and their following graphs
    virtual_nodes[unit_graph_mask] = virtual_nodes[unit_graph_mask] - 0.1
    existing_ind_w_offset = torch.cat((existing_ind_w_offset, virtual_nodes))
    existing_ind_w_offset, existing_condensed_offset_argsort = torch.sort(
        existing_ind_w_offset
    )
    # print("Existing condensed indices with offset", existing_ind_w_offset)
    virtual_existing_mask = torch.cat(
        (
            torch.zeros(len(existing_indices), dtype=torch.long, device=device),
            torch.ones(len(sq_offset), dtype=torch.long, device=device),
        )
    )
    virtual_existing_mask = virtual_existing_mask[
        existing_condensed_offset_argsort
    ]  # [1 0 0 0 1 0 0]
    # print('Virtual nodes mask', virtual_existing_mask)

    # Compute the mask of free edges
    # When there exists graphs with size 1, free spots might be negative, which means that
    # existing condensed indices have same neighbor value
    free_spots = (
        torch.diff(existing_ind_w_offset, prepend=torch.tensor([-1]).to(device)) - 1
    )  # [-0.1, 0, 2, 9, 9.9, 18, 25]
    free_spots = torch.ceil(free_spots).long()  # [0,    0, 1, 6, 0,   8,  6]
    # print("Free spots", free_spots)
    # Map these values to index
    cumsum = torch.cumsum(free_spots, dim=0).long()  # [1 2 3 4 5 6 7]
    cumsum_batch = (
        torch.cumsum(virtual_existing_mask, dim=0).long() - 1
    )  # [1 1 1 1 2 2 2] - 1
    # delete the offset of free spots to cumsum
    cumsum_offset = cumsum[virtual_existing_mask.bool()][cumsum_batch]
    # print("Cumsum offset", cumsum_offset)
    # print("Cumsum before removing offset", cumsum)
    cumsum = cumsum - cumsum_offset  # [0 2 5 0 2 5]
    # add the offset of edge number to cumsum
    cumsum = cumsum + sq_offset[cumsum_batch]  # [0 2 5 6 8 11]
    # print("Cumsum", cumsum)
    # Cumsum now contains the number of free spots at the left  -- it is computed separetely for each graph
    # An offset is added on the result

    # Add virtual edges to the sampled edges to mark the end of each graph
    num_sampled_edges = len(sampled_indices)
    num_virtual_nodes = len(sq_offset)
    num_free_spots_indices = len(cumsum)

    # Group the different vectors together: the existing edges, the virtual nodes and the free spots
    grouped = torch.cat((samp_ind_w_offset, virtual_nodes, cumsum))
    # print("grouped", grouped)
    sorted, argsort = torch.sort(grouped)
    # print("sorted", sorted)
    # Create the masks corresponding to these 3 types of objects
    num_total = num_sampled_edges + num_virtual_nodes + num_free_spots_indices
    # mask is created for virtual nodes, in order to reduce the offset for cumsum
    virtual_sampled_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
    virtual_sampled_mask[
        num_sampled_edges : num_sampled_edges + num_virtual_nodes
    ] = True
    virtual_sampled_mask = virtual_sampled_mask[argsort]

    free_spots_ind_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
    free_spots_ind_mask[-num_free_spots_indices:] = True
    free_spots_ind_mask = free_spots_ind_mask[argsort]

    sampled_ind_mask = torch.zeros(num_total, dtype=torch.bool, device=device)
    sampled_ind_mask[:num_sampled_edges] = True
    sampled_ind_mask = sampled_ind_mask[argsort]

    # to_shift tells by how much to shift sampled and virtual edges
    to_shift = torch.cumsum(free_spots_ind_mask, dim=0)  # - sampled_edge_batch
    # print("to_shift", to_shift)
    new_indices = sorted + to_shift
    # remove epsilon added to sampled edges
    new_indices = new_indices[sampled_ind_mask] - epsilon
    # remove cumsum_offset to unify the indices of different graphs from cumsum_mask
    # 1 is added to compensate the fact that cumsum is computed with virtual nodes
    cumsum_offset = to_shift[virtual_sampled_mask.bool()][sampled_edge_batch] + 1
    cumsum_offset[unit_graph_mask_offset[sampled_edge_batch]] = (
        cumsum_offset[unit_graph_mask_offset[sampled_edge_batch]] + 1
    )
    # print("Cumsum offset", cumsum_offset)
    # remove sq_offset contained by sorted
    new_indices = new_indices - cumsum_offset - sq_offset[sampled_edge_batch]
    # print("New indices long", new_indices)
    new_indices = new_indices.round()
    # print('Existing edge indices', existing_indices)
    # Convert to matrix index.
    new_edge_index = condensed_to_matrix_index_batch(
        condensed_index=new_indices,
        num_nodes=num_nodes,
        edge_batch=sampled_edge_batch,
        ptr=offset,
    )

    # # debugging
    # # check if there are repeated edges
    # print('smallest graph size is {}'.format(num_nodes.min()))
    # existing_ind_w_offset = existing_indices + sq_offset[existing_edge_batch]
    # samp_ind_w_offset = new_indices + sq_offset[sampled_edge_batch]
    # repeated = existing_ind_w_offset.round().unsqueeze(1) == samp_ind_w_offset.round().unsqueeze(0)
    # repeated_ind = torch.where(repeated)
    # if repeated.sum()>0:
    #     print('repeated edges')
    #     import pdb; pdb.set_trace()
    #     cur_shift = to_shift[sampled_ind_mask][1188] - cumsum_offset[1188]

    return new_edge_index


def print_output(edge_index: Tensor, num_nodes_per_graph: Tensor, batch=None):
    """edge_index: (2, num_edges)
    num_nodes_per_graph: (bs)
    """
    # Print the adjacency matrix of each graph in the batch
    # Print the proportion of edges of each graph in the batch
    adjs = to_dense_adj(edge_index, batch)
    for i in range(adjs.shape[0]):
        print("Printing matrix for graph {}".format(i))
        n = num_nodes_per_graph[i]
        adj = adjs[i]
        adj = adj[:n, :][:, :n]
        print_adj_matrix(adj)
        print(f"Proportion of edges {100 * torch.sum(adj>0) / (0.5 * n * (n - 1))} %")
        print()

        if adj.max() > 2:
            raise ("there exist repeated edge in the graph!")


def print_adj_matrix(adj: Tensor):
    """adj: (n, n)"""
    n = adj.shape[0]
    print("Adj:")
    for i in range(n):
        for j in range(n):
            print(int(adj[i, j]), end=" ")
        print()


if __name__ == "__main__":
    torch.manual_seed(0)

    if False:
        test_sample_existing_edges_single_graph()

    if False:
        test_sample_existing_edges_batch()

    if False:
        # single graph generation
        for i in range(3, 20):
            sample_edge_proportion = 0.4
            exist_edge_proportion = 0.4
            num_nodes_per_graph = torch.tensor([i], dtype=torch.long)
            num_edges = (num_nodes_per_graph * (num_nodes_per_graph - 1)) / 2
            condensed_index = torch.randperm(int(num_edges))
            condensed_index = condensed_index[: int(num_edges * exist_edge_proportion)]
            existing_edge_index = condensed_to_matrix_index(condensed_index, i)
            num_edges_to_sample = torch.ceil(sample_edge_proportion * num_edges).long()
            batch = (torch.ones(i) * 0).long()
            edge_index = sample_non_existing_edges_batched(
                num_edges_to_sample, existing_edge_index, num_nodes_per_graph, batch
            )

            edge_index = torch.hstack((edge_index, edge_index, existing_edge_index))

            print_output(edge_index, num_nodes_per_graph, batch)

    if False:
        # generation for 2 graphs
        for i in range(3, 10):
            sample_edge_proportion = 0.4
            exist_edge_proportion = 0.1
            num_nodes_per_graph = torch.tensor([i, i + 5], dtype=torch.long)
            batch = torch.cat([torch.ones(i) * 0, torch.ones(i + 5) * 1]).long()
            num_edges = (num_nodes_per_graph * (num_nodes_per_graph - 1)) / 2
            condensed_index0 = torch.randperm(int(num_edges[0]))
            condensed_index0 = condensed_index0[
                : int(num_edges[0] * exist_edge_proportion)
            ]
            existing_edge_index0 = condensed_to_matrix_index(condensed_index0, i)
            condensed_index1 = torch.randperm(int(num_edges[1]))
            condensed_index1 = condensed_index1[
                : int(num_edges[1] * exist_edge_proportion)
            ]
            existing_edge_index1 = (
                condensed_to_matrix_index(condensed_index1, i + 5) + i
            )
            existing_edge_index = torch.hstack(
                (existing_edge_index0, existing_edge_index1)
            ).long()
            num_edges_to_sample = torch.ceil(sample_edge_proportion * num_edges).long()

            # import pdb; pdb.set_trace()
            edge_index = sample_non_existing_edges_batched(
                num_edges_to_sample, existing_edge_index, num_nodes_per_graph, batch
            )

            edge_index = torch.hstack((edge_index, edge_index, existing_edge_index))

            print_output(edge_index, num_nodes_per_graph, batch)

    if False:
        # generation for 3 graphs
        for i in range(3, 10):
            sample_edge_proportion = 0.4
            exist_edge_proportion = 0.2
            num_nodes_per_graph = torch.tensor([i, i + 5, i + 5], dtype=torch.long)
            batch = torch.cat(
                [torch.ones(i) * 0, torch.ones(i + 5) * 1, torch.ones(i + 5) * 2]
            ).long()
            num_edges = (num_nodes_per_graph * (num_nodes_per_graph - 1)) / 2
            condensed_index0 = torch.randperm(int(num_edges[0]))
            condensed_index0 = condensed_index0[
                : int(num_edges[0] * exist_edge_proportion)
            ]
            existing_edge_index0 = condensed_to_matrix_index(condensed_index0, i)
            condensed_index1 = torch.randperm(int(num_edges[1]))
            condensed_index1 = condensed_index1[
                : int(num_edges[1] * exist_edge_proportion)
            ]
            existing_edge_index1 = (
                condensed_to_matrix_index(condensed_index1, i + 5) + i
            )
            existing_edge_index = torch.hstack(
                (
                    existing_edge_index0,
                    existing_edge_index1,
                    existing_edge_index1 + i + 5,
                )
            ).long()
            num_edges_to_sample = torch.ceil(sample_edge_proportion * num_edges).long()

            # import pdb; pdb.set_trace()
            edge_index = sample_non_existing_edges_batched(
                num_edges_to_sample, existing_edge_index, num_nodes_per_graph, batch
            )

            edge_index = torch.hstack((edge_index, edge_index, existing_edge_index))

            print_output(edge_index, num_nodes_per_graph, batch)

    if False:
        # generation with differen proportion
        for i in range(3, 10):
            sample_edge_proportion = torch.tensor([0.2, 0.3, 0.4])
            exist_edge_proportion = torch.tensor([0.2, 0.4, 0.6])
            num_nodes_per_graph = torch.tensor([i, i + 5, i + 5], dtype=torch.long)
            batch = torch.cat(
                [torch.ones(i) * 0, torch.ones(i + 5) * 1, torch.ones(i + 5) * 2]
            ).long()
            num_edges = (num_nodes_per_graph * (num_nodes_per_graph - 1)) / 2
            condensed_index0 = torch.randperm(int(num_edges[0]))
            condensed_index0 = condensed_index0[
                : int((num_edges * exist_edge_proportion)[0])
            ]
            existing_edge_index0 = condensed_to_matrix_index(condensed_index0, i)
            condensed_index1 = torch.randperm(int(num_edges[1]))
            condensed_index1 = condensed_index1[
                : int((num_edges * exist_edge_proportion)[1])
            ]
            existing_edge_index1 = (
                condensed_to_matrix_index(condensed_index1, i + 5) + i
            )
            condensed_index2 = torch.randperm(int(num_edges[2]))
            condensed_index2 = condensed_index2[
                : int((num_edges * exist_edge_proportion)[2])
            ]
            existing_edge_index2 = (
                condensed_to_matrix_index(condensed_index2, i + 5) + i + i + 5
            )
            existing_edge_index = torch.hstack(
                (existing_edge_index0, existing_edge_index1, existing_edge_index2)
            ).long()
            num_edges_to_sample = torch.ceil(sample_edge_proportion * num_edges).long()

            # import pdb; pdb.set_trace()
            edge_index = sample_non_existing_edges_batched(
                num_edges_to_sample, existing_edge_index, num_nodes_per_graph, batch
            )

            edge_index = torch.hstack((edge_index, edge_index, existing_edge_index))

            print_output(edge_index, num_nodes_per_graph, batch)

    if False:
        # oversample: generation with large existing edges (should not exist in practice)
        pass

    if False:
        # generation with very small proportion
        for i in range(3, 10):
            sample_edge_proportion = 0.01
            exist_edge_proportion = 0.3
            num_nodes_per_graph = torch.tensor([i, i + 5], dtype=torch.long)
            batch = torch.cat([torch.ones(i) * 0, torch.ones(i + 5) * 1]).long()
            num_edges = (num_nodes_per_graph * (num_nodes_per_graph - 1)) / 2
            condensed_index0 = torch.randperm(int(num_edges[0]))
            condensed_index0 = condensed_index0[
                : int(num_edges[0] * exist_edge_proportion)
            ]
            existing_edge_index0 = condensed_to_matrix_index(condensed_index0, i)
            condensed_index1 = torch.randperm(int(num_edges[1]))
            condensed_index1 = condensed_index1[
                : int(num_edges[1] * exist_edge_proportion)
            ]
            existing_edge_index1 = (
                condensed_to_matrix_index(condensed_index1, i + 5) + i
            )
            existing_edge_index = torch.hstack(
                (existing_edge_index0, existing_edge_index1)
            ).long()
            num_edges_to_sample = torch.ceil(sample_edge_proportion * num_edges).long()

            # import pdb; pdb.set_trace()
            edge_index = sample_non_existing_edges_batched(
                num_edges_to_sample, existing_edge_index, num_nodes_per_graph, batch
            )

            edge_index = torch.hstack((edge_index, edge_index, existing_edge_index))

            print_output(edge_index, num_nodes_per_graph, batch)

    if False:
        # fill in all gaps
        for i in range(3, 10):
            sample_edge_proportion = 0.7
            exist_edge_proportion = 0.3
            num_nodes_per_graph = torch.tensor([i, i + 5], dtype=torch.long)
            batch = torch.cat([torch.ones(i) * 0, torch.ones(i + 5) * 1]).long()
            num_edges = (num_nodes_per_graph * (num_nodes_per_graph - 1)) / 2
            condensed_index0 = torch.randperm(int(num_edges[0]))
            condensed_index0 = condensed_index0[
                : int(num_edges[0] * exist_edge_proportion)
            ]
            existing_edge_index0 = condensed_to_matrix_index(condensed_index0, i)
            condensed_index1 = torch.randperm(int(num_edges[1]))
            condensed_index1 = condensed_index1[
                : int(num_edges[1] * exist_edge_proportion)
            ]
            existing_edge_index1 = (
                condensed_to_matrix_index(condensed_index1, i + 5) + i
            )
            existing_edge_index = torch.hstack(
                (existing_edge_index0, existing_edge_index1)
            ).long()
            num_edges_to_sample = torch.ceil(sample_edge_proportion * num_edges).long()

            # import pdb; pdb.set_trace()
            edge_index = sample_non_existing_edges_batched(
                num_edges_to_sample, existing_edge_index, num_nodes_per_graph, batch
            )

            edge_index = torch.hstack((edge_index, edge_index, existing_edge_index))

            print_output(edge_index, num_nodes_per_graph, batch)
