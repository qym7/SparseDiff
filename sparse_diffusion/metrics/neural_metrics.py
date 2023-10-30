import os
import torch
import numpy as np
import time
from scipy import linalg
import sklearn
import dgl
from sklearn.preprocessing import StandardScaler


import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import pairwise_distances

# from dgl.nn.pytorch.conv import GINConv
import dgl.function as fn
from dgl.utils import expand_as_pair
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling


class GINConv(nn.Module):
    r"""
    Description
    -----------
    Graph Isomorphism Network layer from paper `How Powerful are Graph
    Neural Networks? <https://arxiv.org/pdf/1810.00826.pdf>`__.
    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)
    If a weight tensor on each edge is provided, the weighted graph convolution is defined as:
    .. math::
        h_i^{(l+1)} = f_\Theta \left((1 + \epsilon) h_i^{l} +
        \mathrm{aggregate}\left(\left\{e_{ji} h_j^{l}, j\in\mathcal{N}(i)
        \right\}\right)\right)
    where :math:`e_{ji}` is the weight on the edge from node :math:`j` to node :math:`i`.
    Please make sure that `e_{ji}` is broadcastable with `h_j^{l}`.
    Parameters
    ----------
    apply_func : callable activation function/layer or None
        If not None, apply this function to the updated node feature,
        the :math:`f_\Theta` in the formula.
    aggregator_type : str
        Aggregator type to use (``sum``, ``max`` or ``mean``).
    init_eps : float, optional
        Initial :math:`\epsilon` value, default: ``0``.
    learn_eps : bool, optional
        If True, :math:`\epsilon` will be a learnable parameter. Default: ``False``.
    Example
    -------
    >>> import dgl
    >>> import numpy as np
    >>> import torch as th
    >>> from dgl.nn import GINConv
    >>>
    >>> g = dgl.graph(([0,1,2,3,2,5], [1,2,3,4,0,3]))
    >>> feat = th.ones(6, 10)
    >>> lin = th.nn.Linear(10, 10)
    >>> conv = GINConv(lin, 'max')
    >>> res = conv(g, feat)
    >>> res
    tensor([[-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.4821,  0.0207, -0.7665,  0.5721, -0.4682, -0.2134, -0.5236,  1.2855,
            0.8843, -0.8764],
            [-0.1804,  0.0758, -0.5159,  0.3569, -0.1408, -0.1395, -0.2387,  0.7773,
            0.5266, -0.4465]], grad_fn=<AddmmBackward>)
    """
    def __init__(self,
                 apply_func,
                 aggregator_type,
                 init_eps=0,
                 learn_eps=False):
        super(GINConv, self).__init__()
        self.apply_func = apply_func
        self._aggregator_type = aggregator_type
        if aggregator_type == 'sum':
            self._reducer = fn.sum
        elif aggregator_type == 'max':
            self._reducer = fn.max
        elif aggregator_type == 'mean':
            self._reducer = fn.mean
        else:
            raise KeyError('Aggregator type {} not recognized.'.format(aggregator_type))
        # to specify whether eps is trainable or not.
        if learn_eps:
            self.eps = torch.nn.Parameter(torch.FloatTensor([init_eps]))
        else:
            self.register_buffer('eps', torch.FloatTensor([init_eps]))

    def forward(self, graph, feat, edge_weight=None):
        r"""
        Description
        -----------
        Compute Graph Isomorphism Network layer.
        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor or pair of torch.Tensor
            If a torch.Tensor is given, the input feature of shape :math:`(N, D_{in})` where
            :math:`D_{in}` is size of input feature, :math:`N` is the number of nodes.
            If a pair of torch.Tensor is given, the pair must contain two tensors of shape
            :math:`(N_{in}, D_{in})` and :math:`(N_{out}, D_{in})`.
            If ``apply_func`` is not None, :math:`D_{in}` should
            fit the input dimensionality requirement of ``apply_func``.
        edge_weight : torch.Tensor, optional
            Optional tensor on the edge. If given, the convolution will weight
            with regard to the message.
        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, D_{out})` where
            :math:`D_{out}` is the output dimensionality of ``apply_func``.
            If ``apply_func`` is None, :math:`D_{out}` should be the same
            as input dimensionality.
        """
        with graph.local_scope():
            aggregate_fn = self.concat_edge_msg
            # aggregate_fn = fn.copy_src('h', 'm')
            if edge_weight is not None:
                assert edge_weight.shape[0] == graph.number_of_edges()
                graph.edata['_edge_weight'] = edge_weight
                aggregate_fn = fn.u_mul_e('h', '_edge_weight', 'm')

            feat_src, feat_dst = expand_as_pair(feat, graph)
            graph.srcdata['h'] = feat_src
            graph.update_all(aggregate_fn, self._reducer('m', 'neigh'))


            diff = torch.tensor(graph.dstdata['neigh'].shape[1: ]) - torch.tensor(feat_dst.shape[1: ])
            zeros = torch.zeros(feat_dst.shape[0], *diff).to(feat_dst.device)
            feat_dst = torch.cat([feat_dst, zeros], dim=1)
            rst = (1 + self.eps) * feat_dst + graph.dstdata['neigh']
            if self.apply_func is not None:
                rst = self.apply_func(rst)
            return rst

    def concat_edge_msg(self, edges):
        if self.edge_feat_loc not in edges.data:
            return {'m': edges.src['h']}
        else:
            m = torch.cat([edges.src['h'], edges.data[self.edge_feat_loc]], dim=1)
            return {'m': m}


class ApplyNodeFunc(nn.Module):
    """Update the node feature hv with MLP, BN and ReLU."""
    def __init__(self, mlp):
        super(ApplyNodeFunc, self).__init__()
        self.mlp = mlp
        self.bn = nn.BatchNorm1d(self.mlp.output_dim)

    def forward(self, h):
        h = self.mlp(h)
        h = self.bn(h)
        h = F.relu(h)
        return h


class MLP(nn.Module):
    """MLP with linear output"""
    def __init__(self, num_layers, input_dim, hidden_dim, output_dim):
        """MLP layers construction

        Paramters
        ---------
        num_layers: int
            The number of linear layers
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction

        """
        super(MLP, self).__init__()
        self.linear_or_not = True  # default is linear model
        self.num_layers = num_layers
        self.output_dim = output_dim

        if num_layers < 1:
            raise ValueError("number of layers should be positive!")
        elif num_layers == 1:
            # Linear model
            self.linear = nn.Linear(input_dim, output_dim)

        else:
            # Multi-layer model
            self.linear_or_not = False
            self.linears = torch.nn.ModuleList()
            self.batch_norms = torch.nn.ModuleList()

            self.linears.append(nn.Linear(input_dim, hidden_dim))
            for layer in range(num_layers - 2):
                self.linears.append(nn.Linear(hidden_dim, hidden_dim))
            self.linears.append(nn.Linear(hidden_dim, output_dim))

            for layer in range(num_layers - 1):
                self.batch_norms.append(nn.BatchNorm1d((hidden_dim)))

    def forward(self, x):
        if self.linear_or_not:
            # If linear model
            return self.linear(x)
        else:
            # If MLP
            h = x
            for i in range(self.num_layers - 1):
                h = F.relu(self.batch_norms[i](self.linears[i](h)))
            return self.linears[-1](h)


class GIN(nn.Module):
    """GIN model"""
    def __init__(self, num_layers, num_mlp_layers, input_dim, hidden_dim,
                 graph_pooling_type, neighbor_pooling_type, edge_feat_dim=0,
                 final_dropout=0.0, learn_eps=False, output_dim=1, **kwargs):
        """model parameters setting

        Paramters
        ---------
        num_layers: int
            The number of linear layers in the neural network
        num_mlp_layers: int
            The number of linear layers in mlps
        input_dim: int
            The dimensionality of input features
        hidden_dim: int
            The dimensionality of hidden units at ALL layers
        output_dim: int
            The number of classes for prediction
        final_dropout: float
            dropout ratio on the final linear layer
        learn_eps: boolean
            If True, learn epsilon to distinguish center nodes from neighbors
            If False, aggregate neighbors and center nodes altogether.
        neighbor_pooling_type: str
            how to aggregate neighbors (sum, mean, or max)
        graph_pooling_type: str
            how to aggregate entire nodes in a graph (sum, mean or max)
        """

        super().__init__()
        def init_weights_orthogonal(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.orthogonal_(m.weight)
            elif isinstance(m, MLP):
                if hasattr(m, 'linears'):
                    m.linears.apply(init_weights_orthogonal)
                else:
                    m.linear.apply(init_weights_orthogonal)
            elif isinstance(m, nn.ModuleList):
                pass
            else:
                raise Exception()

        self.num_layers = num_layers
        self.learn_eps = learn_eps

        # List of MLPs
        self.ginlayers = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        # self.preprocess_nodes = PreprocessNodeAttrs(
        #     node_attrs=node_preprocess, output_dim=node_preprocess_output_dim)
        # print(input_dim)
        for layer in range(self.num_layers - 1):
            if layer == 0:
                mlp = MLP(num_mlp_layers, input_dim + edge_feat_dim, hidden_dim, hidden_dim)
            else:
                mlp = MLP(num_mlp_layers, hidden_dim + edge_feat_dim, hidden_dim, hidden_dim)
            if kwargs['init'] == 'orthogonal':
                init_weights_orthogonal(mlp)

            self.ginlayers.append(
                GINConv(ApplyNodeFunc(mlp), neighbor_pooling_type, 0, self.learn_eps))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        # Linear function for graph poolings of output of each layer
        # which maps the output of different layers into a prediction score
        self.linears_prediction = torch.nn.ModuleList()

        for layer in range(num_layers):
            if layer == 0:
                self.linears_prediction.append(
                    nn.Linear(input_dim, output_dim))
            else:
                self.linears_prediction.append(
                    nn.Linear(hidden_dim, output_dim))


        if kwargs['init'] == 'orthogonal':
            self.linears_prediction.apply(init_weights_orthogonal)

        self.drop = nn.Dropout(final_dropout)

        if graph_pooling_type == 'sum':
            self.pool = SumPooling()
        elif graph_pooling_type == 'mean':
            self.pool = AvgPooling()
        elif graph_pooling_type == 'max':
            self.pool = MaxPooling()
        else:
            raise NotImplementedError

    def forward(self, g, h):
        # list of hidden representation at each layer (including input)
        hidden_rep = [h]

        # h = self.preprocess_nodes(h)
        for i in range(self.num_layers - 1):
            h = self.ginlayers[i](g, h)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            hidden_rep.append(h)

        score_over_layer = 0

        # perform pooling over all nodes in each graph in every layer
        for i, h in enumerate(hidden_rep):
            pooled_h = self.pool(g, h)
            score_over_layer += self.drop(self.linears_prediction[i](pooled_h))
        return score_over_layer

    def get_graph_embed(self, g, h):
        self.eval()
        with torch.no_grad():
            # return self.forward(g, h).detach().numpy()
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            # perform pooling over all nodes in each graph in every layer
            graph_embed = torch.Tensor([]).to(self.device)
            for i, h in enumerate(hidden_rep):
                pooled_h = self.pool(g, h)
                graph_embed = torch.cat([graph_embed, pooled_h], dim = 1)

            return graph_embed

    def get_graph_embed_no_cat(self, g, h):
        self.eval()
        with torch.no_grad():
            hidden_rep = []
            # h = self.preprocess_nodes(h)
            for i in range(self.num_layers - 1):
                h = self.ginlayers[i](g, h)
                h = self.batch_norms[i](h)
                h = F.relu(h)
                hidden_rep.append(h)

            # perform pooling over all nodes in each graph in every layer
            # graph_embed = torch.Tensor([]).to(self.device)
            # for i, h in enumerate(hidden_rep):
            #     pooled_h = self.pool(g, h)
            #     graph_embed = torch.cat([graph_embed, pooled_h], dim = 1)

            # return graph_embed
            return self.pool(g, hidden_rep[-1]).to(self.device)

    @property
    def edge_feat_loc(self):
        return self.ginlayers[0].edge_feat_loc

    @edge_feat_loc.setter
    def edge_feat_loc(self, loc):
        for layer in self.ginlayers:
            layer.edge_feat_loc = loc


def load_feature_extractor(
    device, num_layers=3, hidden_dim=35, neighbor_pooling_type='sum',
        graph_pooling_type='sum', input_dim=1, edge_feat_dim=0,
        dont_concat=False, num_mlp_layers=2, output_dim=1,
        node_feat_loc='attr', edge_feat_loc='attr', init='orthogonal',
        **kwargs):

    model = GIN(
        num_layers=num_layers, hidden_dim=hidden_dim,
        neighbor_pooling_type=neighbor_pooling_type,
        graph_pooling_type=graph_pooling_type, input_dim=input_dim,
        edge_feat_dim=edge_feat_dim, num_mlp_layers=num_mlp_layers,
        output_dim=output_dim, init=init)

    model.node_feat_loc = node_feat_loc
    model.edge_feat_loc = edge_feat_loc

    use_pretrained = kwargs.get('use_pretrained', False)
    if use_pretrained:
        model_path = kwargs.get('model_path')
        assert model_path is not None, 'Please pass model_path if use_pretrained=True'
        print('loaded', model_path)
        saved_model = torch.load(model_path)
        model.load_state_dict(saved_model['model_state_dict'])

    model.eval()

    if dont_concat:
        model.forward = model.get_graph_embed_no_cat
    else:
        model.forward = model.get_graph_embed

    model.device = device
    return model.to(device)

def time_function(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        results = func(*args, **kwargs)
        end = time.time()
        return results, end - start
    return wrapper

# class OneHotEncoder():
#     def __init__(self, dataset, graphs=None):
#         if graphs is not None:
#             graphs = dgl.batch(graphs)
#             self.max_degree = torch.max(graphs.in_degrees())
#         elif dataset == 'grid':
#             self.max_degree = 4
#         elif dataset == 'zinc':
#             self.max_degree = 4
#         elif dataset == 'lobster':
#             self.max_degree = 18
#         elif dataset == 'community':
#             self.max_degree = 40
#         elif dataset == 'ego':
#             self.max_degree = 99
#         elif dataset == 'proteins':
#             self.max_degree = 15
#         else:
#             raise Exception(f'Unsupported dataset {dataset}, pls pass graphs')
#
#         self.onehot = torch.eye(self.max_degree + 1)
#
#     def encode(self, degrees):
#         onehot = self.onehot.to(degrees.device)
#         degrees = torch.clamp(degrees, max=self.max_degree + 1)
#         return onehot[degrees]


class GINMetric():
    def __init__(self, model):
        self.feat_extractor = model
        self.get_activations = self.get_activations_gin

    @time_function
    def get_activations_gin(self, generated_dataset, reference_dataset):
        return self._get_activations(generated_dataset, reference_dataset)

    def _get_activations(self, generated_dataset, reference_dataset):
        gen_activations = self.__get_activations_single_dataset(generated_dataset)
        ref_activations = self.__get_activations_single_dataset(reference_dataset)

        scaler = StandardScaler()
        scaler.fit(ref_activations)
        ref_activations = scaler.transform(ref_activations)
        gen_activations = scaler.transform(gen_activations)

        return gen_activations, ref_activations

    def __get_activations_single_dataset(self, dataset):
        node_feat_loc = self.feat_extractor.node_feat_loc
        edge_feat_loc = self.feat_extractor.edge_feat_loc

        ndata = [node_feat_loc] if node_feat_loc in dataset[0].ndata\
            else '__ALL__'
        edata = [edge_feat_loc] if edge_feat_loc in dataset[0].edata\
            else '__ALL__'
        graphs = dgl.batch(
            dataset, ndata=ndata, edata=edata).to(self.feat_extractor.device)

        if node_feat_loc not in graphs.ndata:  # Use degree as features
            feats = graphs.in_degrees() + graphs.out_degrees()
            feats = feats.unsqueeze(1).type(torch.float32)
        else:
            feats = graphs.ndata[node_feat_loc]
        feats = feats.to(self.feat_extractor.device)

        graph_embeds = self.feat_extractor(graphs, feats)
        return graph_embeds.cpu().detach().numpy()

    def evaluate(self, *args, **kwargs):
        raise Exception('Must be implemented by child class')


class MMDEvaluation(GINMetric):
    def __init__(self, model, kernel='rbf', sigma='range', multiplier='mean'):
        super().__init__(model)

        if multiplier == 'mean':
            self.__get_sigma_mult_factor = self.__mean_pairwise_distance
        elif multiplier == 'median':
            self.__get_sigma_mult_factor = self.__median_pairwise_distance
        elif multiplier is None:
            self.__get_sigma_mult_factor = lambda *args, **kwargs: 1
        else:
            raise Exception(multiplier)

        if 'rbf' in kernel:
            if sigma == 'range':
                self.base_sigmas = np.array([
                    0.01, 0.1, 0.25, 0.5, 0.75, 1.0, 2.5, 5.0, 7.5, 10.0])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_adaptive_median'
                else:
                    self.name = 'mmd_rbf_adaptive'

            elif sigma == 'one':
                self.base_sigmas = np.array([1])

                if multiplier == 'mean':
                    self.name = 'mmd_rbf_single_mean'
                elif multiplier == 'median':
                    self.name = 'mmd_rbf_single_median'
                else:
                    self.name = 'mmd_rbf_single'

            else:
                raise Exception(sigma)

            self.evaluate = self.calculate_MMD_rbf_quadratic

        elif 'linear' in kernel:
            self.evaluate = self.calculate_MMD_linear_kernel

        else:
            raise Exception()

    def __get_pairwise_distances(self, generated_dataset, reference_dataset):
        return pairwise_distances(
            reference_dataset, generated_dataset,
            metric='euclidean', n_jobs=8) ** 2

    def __mean_pairwise_distance(self, dists_GR):
        return np.sqrt(dists_GR.mean())

    def __median_pairwise_distance(self, dists_GR):
        return np.sqrt(np.median(dists_GR))

    def get_sigmas(self, dists_GR):
        mult_factor = self.__get_sigma_mult_factor(dists_GR)
        return self.base_sigmas * mult_factor

    @time_function
    def calculate_MMD_rbf_quadratic(self, generated_dataset=None, reference_dataset=None):
        # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            (generated_dataset, reference_dataset), _ = self.get_activations(generated_dataset, reference_dataset)

        GG = self.__get_pairwise_distances(generated_dataset, generated_dataset)
        GR = self.__get_pairwise_distances(generated_dataset, reference_dataset)
        RR = self.__get_pairwise_distances(reference_dataset, reference_dataset)

        max_mmd = 0
        sigmas = self.get_sigmas(GR)
        for sigma in sigmas:
            gamma = 1 / (2 * sigma**2)

            K_GR = np.exp(-gamma * GR)
            K_GG = np.exp(-gamma * GG)
            K_RR = np.exp(-gamma * RR)

            mmd = K_GG.mean() + K_RR.mean() - 2 * K_GR.mean()
            max_mmd = mmd if mmd > max_mmd else max_mmd

        return {self.name: max_mmd}

    @time_function
    def calculate_MMD_linear_kernel(self, generated_dataset=None, reference_dataset=None):
        # https://github.com/djsutherland/opt-mmd/blob/master/two_sample/mmd.py
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        G_bar = generated_dataset.mean(axis=0)
        R_bar = reference_dataset.mean(axis=0)
        Z_bar = G_bar - R_bar
        mmd = Z_bar.dot(Z_bar)
        mmd = mmd if mmd >= 0 else 0
        return {'mmd_linear': mmd}


class KIDEvaluation(GINMetric):
    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None):
        import tensorflow as tf
        import tensorflow_gan as tfgan

        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        gen_activations = tf.convert_to_tensor(generated_dataset, dtype=tf.float32)
        ref_activations = tf.convert_to_tensor(reference_dataset, dtype=tf.float32)
        kid = tfgan.eval.kernel_classifier_distance_and_std_from_activations(ref_activations, gen_activations)[0].numpy()
        return {'kid': kid}

class FIDEvaluation(GINMetric):
    # https://github.com/mseitzer/pytorch-fid
    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None):
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        mu_ref, cov_ref = self.__calculate_dataset_stats(reference_dataset)
        mu_generated, cov_generated = self.__calculate_dataset_stats(generated_dataset)
        # print(np.max(mu_generated), np.max(cov_generated), 'mu, cov fid')
        fid = self.compute_FID(mu_ref, mu_generated, cov_ref, cov_generated)
        return {'fid': fid}

    def __calculate_dataset_stats(self, activations):
        # print('activation mean -----------------------------------------', activations.mean())
        mu = np.mean(activations, axis = 0)
        cov = np.cov(activations, rowvar = False)

        return mu, cov

    def compute_FID(self, mu1, mu2, cov1, cov2, eps = 1e-6):
        """Numpy implementation of the Frechet Distance.
        The Frechet distance between two multivariate Gaussians X_1 ~ N(mu_1, C_1)
        and X_2 ~ N(mu_2, C_2) is
                d^2 = ||mu_1 - mu_2||^2 + Tr(C_1 + C_2 - 2*sqrt(C_1*C_2)).
        Stable version by Dougal J. Sutherland.

        Params:
        -- mu1   : Numpy array containing the activations of a layer of the
                   inception net (like returned by the function 'get_predictions')
                   for generated samples.
        -- mu2   : The sample mean over activations, precalculated on an
                   representative data set.
        -- sigma1: The covariance matrix over activations for generated samples.
        -- sigma2: The covariance matrix over activations, precalculated on an
                   representative data set.
        Returns:
        --   : The Frechet Distance.
        """
        assert mu1.shape == mu2.shape, \
            'Training and test mean vectors have different lengths'
        assert cov1.shape == cov2.shape, \
            'Training and test covariances have different dimensions'

        diff = mu1 - mu2
        # Product might be almost singular
        covmean, _ = linalg.sqrtm(cov1.dot(cov2), disp=False)
        # print(np.max(covmean), 'covmean')
        if not np.isfinite(covmean).all():
            msg = ('fid calculation produces singular product; '
                   'adding %s to diagonal of cov estimates') % eps
            print(msg)
            offset = np.eye(cov1.shape[0]) * eps
            covmean = linalg.sqrtm((cov1 + offset).dot(cov2 + offset))

        # Numerical error might give slight imaginary component
        if np.iscomplexobj(covmean):
            if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
                m = np.max(np.abs(covmean.imag))
                #raise ValueError('Imaginary component {}'.format(m))
            covmean = covmean.real

        tr_covmean = np.trace(covmean)
        # print(tr_covmean, 'tr_covmean')

        return (diff.dot(diff) + np.trace(cov1) +
                np.trace(cov2) - 2 * tr_covmean)


class prdcEvaluation(GINMetric):
    # From PRDC github: https://github.com/clovaai/generative-evaluation-prdc/blob/master/prdc/prdc.py#L54
    def __init__(self, *args, use_pr=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.use_pr = use_pr

    @time_function
    def evaluate(self, generated_dataset=None, reference_dataset=None, nearest_k=5):
        """
        Computes precision, recall, density, and coverage given two manifolds.
        Args:
            real_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            fake_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int.
        Returns:
            dict of precision, recall, density, and coverage.
        """
        # start = time.time()
        if not isinstance(generated_dataset, torch.Tensor) and not isinstance(generated_dataset, np.ndarray):
            generated_dataset, reference_dataset, _ = self.get_activations(generated_dataset, reference_dataset)

        real_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(
            reference_dataset, nearest_k)
        distance_real_fake = self.__compute_pairwise_distance(
            reference_dataset, generated_dataset)

        if self.use_pr:
            fake_nearest_neighbour_distances = self.__compute_nearest_neighbour_distances(
                generated_dataset, nearest_k)
            precision = (
                    distance_real_fake <=
                    np.expand_dims(real_nearest_neighbour_distances, axis=1)
            ).any(axis=0).mean()

            recall = (
                    distance_real_fake <=
                    np.expand_dims(fake_nearest_neighbour_distances, axis=0)
            ).any(axis=1).mean()

            f1_pr = 2 / ((1 / (precision + 1e-5)) + (1 / (recall + 1e-5)))
            result = dict(precision=precision, recall=recall, f1_pr=f1_pr)

        else:
            density = (1. / float(nearest_k)) * (
                    distance_real_fake <=
                    np.expand_dims(real_nearest_neighbour_distances, axis=1)
            ).sum(axis=0).mean()

            coverage = (
                    distance_real_fake.min(axis=1) <=
                    real_nearest_neighbour_distances
            ).mean()

            f1_dc = 2 / ((1 / (density + 1e-5)) + (1 / (coverage + 1e-5)))
            result = dict(density=density, coverage=coverage, f1_dc=f1_dc)
        # end = time.time()
        # print('prdc', start - end)
        return result

    def __compute_pairwise_distance(self, data_x, data_y=None):
        """
        Args:
            data_x: numpy.ndarray([N, feature_dim], dtype=np.float32)
            data_y: numpy.ndarray([N, feature_dim], dtype=np.float32)
        Returns:
            numpy.ndarray([N, N], dtype=np.float32) of pairwise distances.
        """
        if data_y is None:
            data_y = data_x
        dists = sklearn.metrics.pairwise_distances(
            data_x, data_y, metric='euclidean', n_jobs=8)
        return dists


    def __get_kth_value(self, unsorted, k, axis=-1):
        """
        Args:
            unsorted: numpy.ndarray of any dimensionality.
            k: int
        Returns:
            kth values along the designated axis.
        """
        indices = np.argpartition(unsorted, k, axis=axis)[..., :k]
        k_smallests = np.take_along_axis(unsorted, indices, axis=axis)
        kth_values = k_smallests.max(axis=axis)
        return kth_values


    def __compute_nearest_neighbour_distances(self, input_features, nearest_k):
        """
        Args:
            input_features: numpy.ndarray([N, feature_dim], dtype=np.float32)
            nearest_k: int
        Returns:
            Distances to kth nearest neighbours.
        """
        distances = self.__compute_pairwise_distance(input_features)
        radii = self.__get_kth_value(distances, k=nearest_k + 1, axis=-1)
        return radii