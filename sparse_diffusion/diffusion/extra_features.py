import time
import math

import torch
import torch.nn.functional as F
from sparse_diffusion import utils


def batch_trace(X):
    """ Expect a matrix of shape B N N, returns the trace in shape B."""
    diag = torch.diagonal(X, dim1=-2, dim2=-1)
    return diag.sum(dim=-1)


def batch_diagonal(X):
    """ Extracts the diagonal from the last two dims of a tensor. """
    return torch.diagonal(X, dim1=-2, dim2=-1)


class DummyExtraFeatures:
    """This class does not compute anything, just returns empty tensors."""
    def __call__(self, noisy_data):
        X = noisy_data["node_t"]
        if "comp_edge_attr_t" not in noisy_data:
            E = noisy_data["edge_attr_t"]
        else:
            E = noisy_data["comp_edge_attr_t"]
        y = noisy_data["y_t"]
        empty_x = X.new_zeros((*X.shape[:-1], 0))
        empty_e = E.new_zeros((*E.shape[:-1], 0))
        empty_y = y.new_zeros((y.shape[0], 0))
        return utils.SparsePlaceHolder(
            node=empty_x, edge_index=None, edge_attr=empty_e, y=empty_y
        ), 0., 0.


class ExtraFeatures:
    def __init__(self, eigenfeatures: bool, edge_features_type, dataset_info, num_eigenvectors,
                 num_eigenvalues, num_degree, dist_feat, use_positional: bool):
        self.eigenfeatures = eigenfeatures
        self.max_n_nodes = dataset_info.max_n_nodes
        self.edge_features = edge_features_type
        self.use_positional = use_positional
        if use_positional:
            self.positional_encoding = PositionalEncoding(dataset_info.max_n_nodes)
        self.adj_features = AdjacencyFeatures(edge_features_type, num_degree=num_degree, dist_feat=dist_feat)
        if eigenfeatures:
            self.eigenfeatures = EigenFeatures(num_eigenvectors=num_eigenvectors, num_eigenvalues=num_eigenvalues)

    def __call__(self, sparse_noisy_data):
        # make data dense in the beginning to avoid doing this twice for both cycles and eigenvalues
        noisy_data = utils.densify_noisy_data(sparse_noisy_data)
        n = noisy_data["node_mask"].sum(dim=1).unsqueeze(1) / self.max_n_nodes
        start_time = time.time()
        x_feat, y_feat, edge_feat = self.adj_features(noisy_data)  # (bs, n_cycles)
        y_feat = torch.hstack((y_feat, n))
        cycle_time = round(time.time() - start_time, 2)
        eigen_time = 0.

        if self.use_positional:
            node_feat = self.positional_encoding(noisy_data)
            x_feat = torch.cat([x_feat, node_feat], dim=-1)

        if self.eigenfeatures:
            start_time = time.time()
            eval_feat, evec_feat = self.eigenfeatures.compute_features(noisy_data)
            eigen_time = round(time.time() - start_time, 2)
            x_feat = torch.cat((x_feat, evec_feat), dim=-1)
            y_feat = torch.hstack((y_feat, eval_feat))

        return utils.PlaceHolder(X=x_feat, E=edge_feat, y=y_feat), cycle_time, eigen_time


class PositionalEncoding:
    def __init__(self, n_max_dataset, D=30):
        self.n_max = n_max_dataset
        self.d = math.floor(D / 2)

    def __call__(self, dense_noisy_data):
        device = dense_noisy_data['X_t'].device
        n_max_batch = dense_noisy_data['X_t'].shape[1]

        arange_n = torch.arange(n_max_batch, device=device)                                    # n_max
        arange_d = torch.arange(self.d, device=device)                                         # d
        frequencies = math.pi / torch.pow(self.n_max, 2 * arange_d / self.d)    # d

        sines = torch.sin(arange_n.unsqueeze(1) * frequencies.unsqueeze(0))     # N, d
        cosines = torch.cos(arange_n.unsqueeze(1) * frequencies.unsqueeze(0))   # N, d
        encoding = torch.hstack((sines, cosines))                               # N, D
        extra_x = encoding.unsqueeze(0)                                         # 1, N, D
        extra_x = extra_x * dense_noisy_data['node_mask'].unsqueeze(-1)             # B, N, D
        return extra_x

class EigenFeatures:
    """  Some code is taken from : https://github.com/Saro00/DGN/blob/master/models/pytorch/eigen_agg.py. """
    def __init__(self, num_eigenvectors, num_eigenvalues):
        self.num_eigenvectors = num_eigenvectors
        self.num_eigenvalues = num_eigenvalues

    def compute_features(self, noisy_data):
        E_t = noisy_data["E_t"]
        mask = noisy_data["node_mask"]
        A = E_t[..., 1:].sum(dim=-1).float() * mask.unsqueeze(1) * mask.unsqueeze(2)
        L = self.compute_laplacian(A, normalize=False)
        mask_diag = 2 * L.shape[-1] * torch.eye(A.shape[-1]).type_as(L).unsqueeze(0)
        mask_diag = mask_diag * (~mask.unsqueeze(1)) * (~mask.unsqueeze(2))
        L = L * mask.unsqueeze(1) * mask.unsqueeze(2) + mask_diag

        # debug for protein dataset
        if L.isnan().any():
            import pdb; pdb.set_trace()
        eigvals, eigvectors = torch.linalg.eigh(L)
        eigvals = eigvals.type_as(A) / torch.sum(mask, dim=1, keepdim=True)
        eigvectors = eigvectors * mask.unsqueeze(2) * mask.unsqueeze(1)
        # Retrieve eigenvalues features
        n_connected_comp, batch_eigenvalues = self.eigenvalues_features(
            eigenvalues=eigvals, num_eigenvalues=self.num_eigenvalues
        )
        # Retrieve eigenvectors features
        evector_feat = self.eigenvector_features(
            vectors=eigvectors,
            node_mask=noisy_data["node_mask"],
            n_connected=n_connected_comp,
            num_eigenvectors=self.num_eigenvectors,
        )

        evalue_feat = torch.hstack((n_connected_comp, batch_eigenvalues))
        return evalue_feat, evector_feat

    def compute_laplacian(self, adjacency, normalize: bool):
        """
        adjacency : batched adjacency matrix (bs, n, n)
        normalize: can be None, 'sym' or 'rw' for the combinatorial, symmetric normalized or random walk Laplacians
        Return:
            L (n x n ndarray): combinatorial or symmetric normalized Laplacian.
        """
        diag = torch.sum(adjacency, dim=-1)  # (bs, n)
        n = diag.shape[-1]
        D = torch.diag_embed(diag)  # Degree matrix      # (bs, n, n)
        combinatorial = D - adjacency  # (bs, n, n)

        if not normalize:
            return (combinatorial + combinatorial.transpose(1, 2)) / 2

        diag0 = diag.clone()
        diag[diag == 0] = 1e-12

        diag_norm = 1 / torch.sqrt(diag)  # (bs, n)
        D_norm = torch.diag_embed(diag_norm)  # (bs, n, n)
        L = torch.eye(n).unsqueeze(0) - D_norm @ adjacency @ D_norm
        L[diag0 == 0] = 0
        return (L + L.transpose(1, 2)) / 2

    def eigenvalues_features(self, eigenvalues, num_eigenvalues):
        """
        values : eigenvalues -- (bs, n)
        node_mask: (bs, n)
        k: num of non zero eigenvalues to keep
        """
        ev = eigenvalues
        bs, n = ev.shape
        n_connected_components = (ev < 1e-4).sum(dim=-1)

        # if (n_connected_components <= 0).any():
        #     import pdb; pdb.set_trace()
        # assert (n_connected_components > 0).all(), (n_connected_components, ev)

        try:
            to_extend = max(n_connected_components) + num_eigenvalues - n
            if to_extend > 0:
                ev = torch.hstack((ev, 2 * torch.ones(bs, to_extend, device=ev.device)))
            indices = torch.arange(num_eigenvalues, device=ev.device).unsqueeze(0) + n_connected_components.unsqueeze(1)
            first_k_ev = torch.gather(ev, dim=1, index=indices)
        except:
            import pdb; pdb.set_trace()

        return n_connected_components.unsqueeze(-1), first_k_ev

    def eigenvector_features(self, vectors, node_mask, n_connected, num_eigenvectors):
        """
        vectors (bs, n, n) : eigenvectors of Laplacian IN COLUMNS
        returns:
            not_lcc_indicator : indicator vectors of largest connected component (lcc) for each graph  -- (bs, n, 1)
            k_lowest_eigvec : k first eigenvectors for the largest connected component   -- (bs, n, k)
        """
        bs, n = vectors.size(0), vectors.size(1)

        # Create an indicator for the nodes outside the largest connected components
        first_ev = torch.round(vectors[:, :, 0], decimals=3) * node_mask  # bs, n
        # Add random value to the mask to prevent 0 from becoming the mode
        random = torch.randn(bs, n, device=node_mask.device) * (~node_mask)  # bs, n
        first_ev = first_ev + random
        most_common = torch.mode(first_ev, dim=1).values  # values: bs -- indices: bs
        mask = ~(first_ev == most_common.unsqueeze(1))
        not_lcc_indicator = (mask * node_mask).unsqueeze(-1).float()

        # Get the eigenvectors corresponding to the first nonzero eigenvalues
        to_extend = max(n_connected) + num_eigenvectors - n
        if to_extend > 0:
            vectors = torch.cat(
                (vectors, torch.zeros(bs, n, to_extend, device=vectors.device)), dim=2
            )  # bs, n , n + to_extend

        indices = torch.arange(num_eigenvectors, device=vectors.device).long().unsqueeze(0).unsqueeze(0)
        indices = indices + n_connected.unsqueeze(2)  # bs, 1, k
        indices = indices.expand(-1, n, -1)  # bs, n, k
        first_k_ev = torch.gather(vectors, dim=2, index=indices)  # bs, n, k
        first_k_ev = first_k_ev * node_mask.unsqueeze(2)

        return torch.cat((not_lcc_indicator, first_k_ev), dim=-1)


class AdjacencyFeatures:
    """Builds cycle counts for each node in a graph."""
    def __init__(self, edge_features_type, num_degree, max_degree=10, dist_feat=True):
        self.edge_features_type = edge_features_type
        self.max_degree = max_degree
        self.num_degree = num_degree
        self.dist_feat = dist_feat

    def __call__(self, noisy_data):
        adj_matrix = noisy_data["E_t"][..., 1:].int().sum(dim=-1)  # (bs, n, n)
        num_nodes = noisy_data["node_mask"].sum(dim=1)
        self.calculate_kpowers(adj_matrix)

        k3x, k3y = self.k3_cycle()
        k4x, k4y = self.k4_cycle()
        k5x, k5y = self.k5_cycle()
        _, k6y = self.k6_cycle()

        kcyclesx = torch.cat([k3x, k4x, k5x], dim=-1)
        kcyclesy = torch.cat([k3y, k4y, k5y, k6y], dim=-1)

        if self.edge_features_type == "dist":
            edge_feats = self.path_features()
        elif self.edge_features_type == "localngbs":
            edge_feats = self.local_neighbors(num_nodes)
        elif self.edge_features_type == "all":
            dist = self.path_features()
            local_ngbs = self.local_neighbors(num_nodes)
            edge_feats = torch.cat([dist, local_ngbs], dim=-1)
        else:
            edge_feats = torch.zeros((*adj_matrix.shape, 0), device=adj_matrix.device)

        kcyclesx = torch.clamp(kcyclesx, 0, 5) / 5 * noisy_data["node_mask"].unsqueeze(-1)
        y_feat = [torch.clamp(kcyclesy, 0, 5) / 5]
        edge_feats = torch.clamp(edge_feats, 0, 5) / 5

        if self.dist_feat:
        # get degree distribution
            bs, n = noisy_data["node_mask"].shape
            degree = adj_matrix.sum(dim=-1).long()  # (bs, n)
            degree[degree > self.num_degree] = self.num_degree + 1    # bs, n
            one_hot_degree = F.one_hot(degree, num_classes=self.num_degree + 2).float()  # bs, n, num_degree + 2
            one_hot_degree[~noisy_data["node_mask"]] = 0
            degree_dist = one_hot_degree.sum(dim=1)  # bs, num_degree + 2
            s = degree_dist.sum(dim=-1, keepdim=True)
            s[s == 0] = 1
            degree_dist = degree_dist / s
            y_feat.append(degree_dist)

            # get node distribution
            X = noisy_data["X_t"]       # bs, n, dx
            node_dist = X.sum(dim=1)    # bs, dx
            s = node_dist.sum(-1)     # bs
            s[s == 0] = 1
            node_dist = node_dist / s.unsqueeze(-1)     # bs, dx
            y_feat.append(node_dist)

            # get edge distribution
            E = noisy_data["E_t"]
            edge_dist = E.sum(dim=[1, 2])    # bs, de
            s = edge_dist.sum(-1)     # bs
            s[s == 0] = 1
            edge_dist = edge_dist / s.unsqueeze(-1)     # bs, de
            y_feat.append(edge_dist)

        y_feat = torch.cat(y_feat, dim=-1)

        return kcyclesx, y_feat, edge_feats

    def calculate_kpowers(self, adj):
        """ adj: bs, n, n"""
        shape = (self.max_degree, *adj.shape)
        adj = adj.float()
        self.k = torch.zeros(shape, device=adj.device, dtype=torch.float)
        self.d = adj.sum(dim=-1)
        self.k[0] = adj
        for i in range(1, self.max_degree):
            self.k[i] = self.k[i-1] @ adj

        # Warning: index changes by 1 (count from 1 and not 0)
        self.k1, self.k2, self.k3, self.k4, self.k5, self.k6 = [self.k[i] for i in range(6)]

    def k3_cycle(self):
        c3 = batch_diagonal(self.k3)
        return (c3 / 2).unsqueeze(-1).float(), (torch.sum(c3, dim=-1) / 6).unsqueeze(
            -1
        ).float()

    def k4_cycle(self):
        diag_a4 = batch_diagonal(self.k4)
        c4 = (
            diag_a4
            - self.d * (self.d - 1)
            - (self.k1 @ self.d.unsqueeze(-1)).sum(dim=-1)
        )
        return (c4 / 2).unsqueeze(-1).float(), (torch.sum(c4, dim=-1) / 8).unsqueeze(
            -1
        ).float()

    def k5_cycle(self):
        diag_a5 = batch_diagonal(self.k5)
        triangles = batch_diagonal(self.k3)

        c5 = (
            diag_a5
            - 2 * triangles * self.d
            - (self.k1 @ triangles.unsqueeze(-1)).sum(dim=-1)
            + triangles
        )
        return (c5 / 2).unsqueeze(-1).float(), (c5.sum(dim=-1) / 10).unsqueeze(
            -1
        ).float()

    def k6_cycle(self):
        term_1_t = batch_trace(self.k6)
        term_2_t = batch_trace(self.k3 ** 2)
        term3_t = torch.sum(self.k1 * self.k2.pow(2), dim=[-2, -1])
        d_t4 = batch_diagonal(self.k2)
        a_4_t = batch_diagonal(self.k4)
        term_4_t = (d_t4 * a_4_t).sum(dim=-1)
        term_5_t = batch_trace(self.k4)
        term_6_t = batch_trace(self.k3)
        term_7_t = batch_diagonal(self.k2).pow(3).sum(-1)
        term8_t = torch.sum(self.k3, dim=[-2, -1])
        term9_t = batch_diagonal(self.k2).pow(2).sum(-1)
        term10_t = batch_trace(self.k2)

        c6_t = (
            term_1_t
            - 3 * term_2_t
            + 9 * term3_t
            - 6 * term_4_t
            + 6 * term_5_t
            - 4 * term_6_t
            + 4 * term_7_t
            + 3 * term8_t
            - 12 * term9_t
            + 4 * term10_t
        )
        return None, (c6_t / 12).unsqueeze(-1).float()

    def path_features(self):
        path_features = self.k.bool().float()        # max power, bs, n, n
        path_features = path_features.permute(1, 2, 3, 0)    # bs, n, n, max power
        return path_features

    def local_neighbors(self, num_nodes):
        """ Adamic-Adar index for each pair of nodes.
            this function captures the local neighborhood information, commonly used in social network analysis
            [i, j], sum of 1/log(degree(u)), u is a common neighbor of i and j.
        """
        normed_adj = self.k1 / self.k1.sum(-1).unsqueeze(1)        # divide each column by its degree

        normed_adj = torch.sqrt(torch.log(normed_adj).abs())
        normed_adj = torch.nan_to_num(1 / normed_adj, posinf=0)
        normed_adj = torch.matmul(normed_adj, normed_adj.transpose(-2, -1))

        # mask self-loops to 0
        mask = torch.eye(normed_adj.shape[-1]).repeat(normed_adj.shape[0], 1, 1).bool()
        normed_adj[mask] = 0

        # normalization
        normed_adj = (
            normed_adj * num_nodes.log()[:, None, None] / num_nodes[:, None, None]
        )
        return normed_adj.unsqueeze(-1)
