import datetime
import dgl
import errno
import numpy as np
import os
import pickle
import random
import torch
from torch import nn
from dgl.data.utils import download, get_download_dir, _get_dgl_url
from pprint import pprint
from scipy import sparse
from scipy import io as sio
from dgl.nn.pytorch.utils import Identity
from dgl.nn.pytorch.softmax import edge_softmax
import dgl.function as fn


def set_random_seed(seed=0):
    """Set random seed.
    Parameters
    ----------
    seed : int
        Random seed to use
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


def mkdir_p(path, log=True):
    """Create a directory for the specified path.
    Parameters
    ----------
    path : str
        Path name
    log : bool
        Whether to print result for directory creation
    """
    try:
        os.makedirs(path)
        if log:
            print('Created directory {}'.format(path))
    except OSError as exc:
        if exc.errno == errno.EEXIST and os.path.isdir(path) and log:
            print('Directory {} already exists.'.format(path))
        else:
            raise


def get_date_postfix():
    """Get a date based postfix for directory name.
    Returns
    -------
    post_fix : str
    """
    dt = datetime.datetime.now()
    post_fix = '{}_{:02d}-{:02d}-{:02d}'.format(
        dt.date(), dt.hour, dt.minute, dt.second)

    return post_fix


def setup_log_dir(args, sampling=False):
    """Name and create directory for logging.
    Parameters
    ----------
    args : dict
        Configuration
    Returns
    -------
    log_dir : str
        Path for logging directory
    sampling : bool
        Whether we are using sampling based training
    """
    date_postfix = get_date_postfix()
    log_dir = os.path.join(
        args['log_dir'],
        '{}_{}'.format(args['dataset'], date_postfix))

    if sampling:
        log_dir = log_dir + '_sampling'

    mkdir_p(log_dir)
    return log_dir


def metagraph_graph(g, meta_graph, both_appear=True, weighted=False):
    '''
    :param g: heterogenous graph
    :param meta_graph:
    eg:[[['pa','ap'],['pt','tp']]]
    :return: new heterogenous graph for given meta graph
    '''
    final_adj = 1
    for sub_graph in meta_graph:
        cur_sub_g = 1
        for sub_meta_path in sub_graph:
            metap_adj = 1
            for etype in sub_meta_path:
                metap_adj = metap_adj * g.adj(etype=etype, scipy_fmt='csr', transpose=True)
            if both_appear:
                cur_sub_g = metap_adj.multiply(cur_sub_g)
            elif isinstance(cur_sub_g, int):
                cur_sub_g = metap_adj
            else:
                cur_sub_g = metap_adj + cur_sub_g
        final_adj = final_adj * cur_sub_g
        final_adj.setdiag(0)
        final_adj.eliminate_zeros()
        final_adj = final_adj + 2*sparse.eye(final_adj.shape[0])
        #final_adj[final_adj > 2000] = 2000
        if not weighted:
            final_adj = (final_adj != 0).tocsr()
    srctype = g.to_canonical_etype(meta_graph[0][0][0])[0]
    dsttype = g.to_canonical_etype(meta_graph[-1][-1][-1])[-1]
    assert final_adj.shape[0] == final_adj.shape[1]
    new_g = dgl.graph(final_adj, ntype=srctype)

    for key, value in g.nodes[srctype].data.items():
        new_g.nodes[srctype].data[key] = value
    new_g.edata['edge_weight'] = torch.tensor(final_adj.data, dtype=torch.float32
                                              , device='cpu')
    if srctype != dsttype:
        for key, value in g.nodes[dsttype].data.items():
            new_g.nodes[dsttype].data[key] = value

    return new_g


# The configuration below is from the paper.
default_configure = {
    'lr': 0.005,  # Learning rate
    'num_heads': [8],  # Number of attention heads for node-level attention
    'hidden_units': 8,
    'dropout': 0.6,
    'weight_decay': 0.001,
    'num_epochs': 200,
    'patience': 100,
    'embedding_dim':256
}

sampling_configure = {
    'batch_size': 20
}


def setup(args):
    args.update(default_configure)
    set_random_seed(args['seed'])
    # args['dataset'] = 'ACMRaw' if args['hetero'] else 'ACM'
    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args)
    return args


def setup_for_sampling(args):
    args.update(default_configure)
    args.update(sampling_configure)
    set_random_seed()
    args['device'] = 'cuda: 0' if torch.cuda.is_available() else 'cpu'
    args['log_dir'] = setup_log_dir(args, sampling=True)
    return args


def get_binary_mask(total_size, indices):
    mask = torch.zeros(total_size)
    mask[indices] = 1
    return mask.byte()


def load_acm(remove_self_loop):
    url = 'dataset/ACM3025.pkl'
    data_path = get_download_dir() + '/ACM3025.pkl'
    download(_get_dgl_url(url), path=data_path)

    with open(data_path, 'rb') as f:
        data = pickle.load(f)

    labels, features = torch.from_numpy(data['label'].todense()).long(), \
                       torch.from_numpy(data['feature'].todense()).float()
    num_classes = labels.shape[1]
    labels = labels.nonzero()[:, 1]

    if remove_self_loop:
        num_nodes = data['label'].shape[0]
        data['PAP'] = sparse.csr_matrix(data['PAP'] - np.eye(num_nodes))
        data['PLP'] = sparse.csr_matrix(data['PLP'] - np.eye(num_nodes))

    # Adjacency matrices for meta path based neighbors
    # (Mufei): I verified both of them are binary adjacency matrices with self loops
    author_g = dgl.graph(data['PAP'], ntype='paper', etype='author')
    subject_g = dgl.graph(data['PLP'], ntype='paper', etype='subject')
    gs = [author_g, subject_g]

    train_idx = torch.from_numpy(data['train_idx']).long().squeeze(0)
    val_idx = torch.from_numpy(data['val_idx']).long().squeeze(0)
    test_idx = torch.from_numpy(data['test_idx']).long().squeeze(0)

    num_nodes = author_g.number_of_nodes()
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    print('dataset loaded')
    pprint({
        'dataset': 'ACM',
        'train': train_mask.sum().item() / num_nodes,
        'val': val_mask.sum().item() / num_nodes,
        'test': test_mask.sum().item() / num_nodes
    })

    return gs, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


def load_acm_raw(remove_self_loop):
    assert not remove_self_loop
    url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/ACM.mat'
    download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    p_vs_l = data['PvsL']  # paper-field?
    p_vs_a = data['PvsA']  # paper-author
    p_vs_t = data['PvsT']  # paper-term, bag of words
    p_vs_c = data['PvsC']  # paper-conference, labels come from that
    p_vs_p = data['PvsP']
    # We assign
    # (1) KDD papers as class 0 (data mining),
    # (2) SIGMOD and VLDB papers as class 1 (database),
    # (3) SIGCOMM and MOBICOMM papers as class 2 (communication)
    conf_ids = [0, 1, 9, 10, 13]
    label_ids = [0, 1, 2, 2, 1]

    p_vs_c_filter = p_vs_c[:, conf_ids]
    p_selected = (p_vs_c_filter.sum(1) != 0).A1.nonzero()[0]
    p_vs_l = p_vs_l[p_selected]
    p_vs_a = p_vs_a[p_selected]
    p_vs_t = p_vs_t[p_selected]
    p_vs_c = p_vs_c[p_selected]
    p_vs_p = p_vs_p[p_selected, :][:, p_selected]
    pa = dgl.bipartite(p_vs_a, 'paper', 'pa', 'author')
    ap = dgl.bipartite(p_vs_a.transpose(), 'author', 'ap', 'paper')
    pl = dgl.bipartite(p_vs_l, 'paper', 'pf', 'field')
    lp = dgl.bipartite(p_vs_l.transpose(), 'field', 'fp', 'paper')
    pt = dgl.bipartite(p_vs_t, 'paper', 'pt', "term")
    tp = dgl.bipartite(p_vs_t.transpose(), 'term', 'tp', 'paper')
    # pp = dgl.bipartite(p_vs_p,'paper','pp','paper')
    # ppr = dgl.bipartite(p_vs_p.transpose(),'paper','ppr','paper')
    hg = dgl.hetero_from_relations([pa, ap, pl, lp, pt, tp])

    features = torch.FloatTensor(p_vs_t.toarray())

    pc_p, pc_c = p_vs_c.nonzero()
    labels = np.zeros(len(p_selected), dtype=np.int64)
    for conf_id, label_id in zip(conf_ids, label_ids):
        labels[pc_p[pc_c == conf_id]] = label_id
    labels = torch.LongTensor(labels)

    num_classes = 3

    float_mask = np.zeros(len(pc_p))
    for conf_id in conf_ids:
        pc_c_mask = (pc_c == conf_id)
        float_mask[pc_c_mask] = np.random.permutation(np.linspace(0, 1, pc_c_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('paper')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


def load_dblp_raw(remove_self_loop):
    assert not remove_self_loop
    # url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/LabDBLP.mat'
    fea_path = get_download_dir() + '/DBLP_feat'
    # download(_get_dgl_url(url), path=data_path)

    data = sio.loadmat(data_path)
    selected_aut_id = data['Aut_lab'][:, 0]
    id2index = {v: k for k, v in enumerate(data['AutID'][:, 0])}
    selected_aut_index = np.array([id2index[id] for id in selected_aut_id])
    selected_paper_index = np.where(data['PA'][:, selected_aut_index].sum(1))[0]
    selected_term_index = np.where(data['PT'][selected_paper_index, :].sum(0))[1]

    p_vs_a = data['PA'][selected_paper_index, :][:, selected_aut_index]  # paper-author
    p_vs_t = data['PT'][selected_paper_index, :][:, selected_term_index]  # paper-term, bag of words
    p_vs_c = data['PC'][selected_paper_index, :]  # paper-conference, labels come from that

    pa = dgl.bipartite(p_vs_a, 'paper', 'pa', 'author')
    ap = dgl.bipartite(p_vs_a.transpose(), 'author', 'ap', 'paper')
    pt = dgl.bipartite(p_vs_t, 'paper', 'pt', "term")
    tp = dgl.bipartite(p_vs_t.transpose(), 'term', 'tp', 'paper')
    pc = dgl.bipartite(p_vs_c, 'paper', 'pc', 'conference')
    cp = dgl.bipartite(p_vs_c.transpose(), 'conference', 'cp', 'paper')

    hg = dgl.hetero_from_relations([pa, ap, pt, tp, pc, cp])
    # we assign the terms in the papers which the author published as
    # the author's feature

    # features = torch.FloatTensor(p_vs_a.transpose()*p_vs_t.toarray())
    fea_data = np.loadtxt(fea_path)
    features = torch.FloatTensor(fea_data)
    # pc_p, pc_c = p_vs_c.nonzero()
    labels = data['Aut_lab'][:, 1] - 1

    labels = torch.LongTensor(labels)

    num_classes = 4

    float_mask = np.zeros(len(labels))
    for class_id in range(num_classes):
        pc_a_mask = (data['Aut_lab'][:, 1] == class_id + 1)
        float_mask[pc_a_mask] = np.random.permutation(np.linspace(0, 1, pc_a_mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('author')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


def load_imdb_raw(remove_self_loop):
    assert not remove_self_loop
    # url = 'dataset/ACM.mat'
    data_path = get_download_dir() + '/imdb_3_class.pkl'
    # download(_get_dgl_url(url), path=data_path)
    f = open(data_path, mode="rb")
    data = pickle.load(f)

    m_vs_d = data['md']  # movie-director
    m_vs_a = data['ma']  # movie-actor

    md = dgl.bipartite(m_vs_d, 'movie', 'md', 'director')
    dm = dgl.bipartite(m_vs_d.transpose(), 'director', 'dm', 'movie')
    ma = dgl.bipartite(m_vs_a, 'movie', 'ma', "actor")
    am = dgl.bipartite(m_vs_a.transpose(), 'actor', 'am', 'movie')

    hg = dgl.hetero_from_relations([md, dm, ma, am])
    # we assign the terms in the papers which the author published as
    # the author's feature

    # features = torch.FloatTensor(p_vs_a.transpose()*p_vs_t.toarray())
    fea_data = data['fea'].todense()
    features = torch.FloatTensor(fea_data)
    # pc_p, pc_c = p_vs_c.nonzero()
    labels_arr = np.array(data['labels'])

    labels = torch.LongTensor(labels_arr)

    num_classes = 3

    float_mask = np.zeros(len(labels))
    for class_id in range(num_classes):
        mask = (labels_arr == class_id)
        float_mask[mask] = np.random.permutation(np.linspace(0, 1, mask.sum()))
    train_idx = np.where(float_mask <= 0.2)[0]
    val_idx = np.where((float_mask > 0.2) & (float_mask <= 0.3))[0]
    test_idx = np.where(float_mask > 0.3)[0]

    num_nodes = hg.number_of_nodes('movie')
    train_mask = get_binary_mask(num_nodes, train_idx)
    val_mask = get_binary_mask(num_nodes, val_idx)
    test_mask = get_binary_mask(num_nodes, test_idx)

    return hg, features, labels, num_classes, train_idx, val_idx, test_idx, \
           train_mask, val_mask, test_mask


def load_data(dataset, remove_self_loop=False):
    if dataset == 'ACM':
        return load_acm(remove_self_loop)
    elif dataset == 'ACMRaw':
        return load_acm_raw(remove_self_loop)
    elif dataset == 'DBLP':
        return load_dblp_raw(remove_self_loop)
    elif dataset == 'IMDB':
        return load_imdb_raw(remove_self_loop)
    else:
        return NotImplementedError('Unsupported dataset {}'.format(dataset))


class EarlyStopping(object):
    def __init__(self, patience=10):
        dt = datetime.datetime.now()
        self.filename = 'early_stop_{}_{:02d}-{:02d}-{:02d}.pth'.format(
            dt.date(), dt.hour, dt.minute, dt.second)
        self.patience = patience
        self.counter = 0
        self.best_acc = None
        self.best_loss = None
        self.early_stop = False

    def step(self, loss, acc, model):
        if self.best_loss is None:
            self.best_acc = acc
            self.best_loss = loss
            self.save_checkpoint(model)
        elif (loss > self.best_loss) and (acc < self.best_acc):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            if (loss <= self.best_loss) and (acc >= self.best_acc):
                self.save_checkpoint(model)
            self.best_loss = np.min((loss, self.best_loss))
            self.best_acc = np.max((acc, self.best_acc))
            self.counter = 0
        return self.early_stop

    def save_checkpoint(self, model):
        """Saves model when validation loss decreases."""
        torch.save(model.state_dict(), self.filename)

    def load_checkpoint(self, model):
        """Load the latest checkpoint."""
        model.load_state_dict(torch.load(self.filename))


class GATConv(nn.Module):
    r"""Apply `Graph Attention Network <https://arxiv.org/pdf/1710.10903.pdf>`__
    over an input signal.

    .. math::
        h_i^{(l+1)} = \sum_{j\in \mathcal{N}(i)} \alpha_{i,j} W^{(l)} h_j^{(l)}

    where :math:`\alpha_{ij}` is the attention score bewteen node :math:`i` and
    node :math:`j`:

    .. math::
        \alpha_{ij}^{l} & = \mathrm{softmax_i} (e_{ij}^{l})

        e_{ij}^{l} & = \mathrm{LeakyReLU}\left(\vec{a}^T [W h_{i} \| W h_{j}]\right)

    Parameters
    ----------
    in_feats : int
        Input feature size.
    out_feats : int
        Output feature size.
    num_heads : int
        Number of heads in Multi-Head Attention.
    feat_drop : float, optional
        Dropout rate on feature, defaults: ``0``.
    attn_drop : float, optional
        Dropout rate on attention weight, defaults: ``0``.
    negative_slope : float, optional
        LeakyReLU angle of negative slope.
    residual : bool, optional
        If True, use residual connection.
    activation : callable activation function/layer or None, optional.
        If not None, applies an activation function to the updated node features.
        Default: ``None``.
    """

    def __init__(self,
                 in_feats,
                 out_feats,
                 num_heads,
                 feat_drop=0.,
                 attn_drop=0.,
                 negative_slope=0.2,
                 residual=False,
                 activation=None):
        super(GATConv, self).__init__()
        self._num_heads = num_heads
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.fc = nn.Linear(in_feats, out_feats * num_heads, bias=False)
        self.attn_l = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.attn_r = nn.Parameter(torch.FloatTensor(size=(1, num_heads, out_feats)))
        self.feat_drop = nn.Dropout(feat_drop)
        self.attn_drop = nn.Dropout(attn_drop)
        self.leaky_relu = nn.LeakyReLU(negative_slope)
        if residual:
            if in_feats != out_feats:
                self.res_fc = nn.Linear(in_feats, num_heads * out_feats, bias=False)
            else:
                self.res_fc = Identity()
        else:
            self.register_buffer('res_fc', None)
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        """Reinitialize learnable parameters."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.fc.weight, gain=gain)
        nn.init.xavier_normal_(self.attn_l, gain=gain)
        nn.init.xavier_normal_(self.attn_r, gain=gain)
        if isinstance(self.res_fc, nn.Linear):
            nn.init.xavier_normal_(self.res_fc.weight, gain=gain)

    def forward(self, graph, feat):
        r"""Compute graph attention network layer.

        Parameters
        ----------
        graph : DGLGraph
            The graph.
        feat : torch.Tensor
            The input feature of shape :math:`(N, D_{in})` where :math:`D_{in}`
            is size of input feature, :math:`N` is the number of nodes.

        Returns
        -------
        torch.Tensor
            The output feature of shape :math:`(N, H, D_{out})` where :math:`H`
            is the number of heads, and :math:`D_{out}` is size of output feature.
        """
        graph = graph.local_var()
        h = self.feat_drop(feat)
        feat = self.fc(h).view(-1, self._num_heads, self._out_feats)
        el = (feat * self.attn_l).sum(dim=-1).unsqueeze(-1)
        er = (feat * self.attn_r).sum(dim=-1).unsqueeze(-1)
        graph.ndata.update({'ft': feat, 'el': el, 'er': er})
        # compute edge attention
        graph.apply_edges(fn.u_add_v('el', 'er', 'e'))
        e = self.leaky_relu(graph.edata.pop('e'))
        # multiplcation with edge weights
        e = graph.edata['edge_weight'].reshape((-1, 1, 1)) * e
        # compute softmax
        graph.edata['a'] = self.attn_drop(edge_softmax(graph, e))
        # message passing
        graph.update_all(fn.u_mul_e('ft', 'a', 'm'),
                         fn.sum('m', 'ft'))
        rst = graph.ndata['ft']
        # residual
        if self.res_fc is not None:
            resval = self.res_fc(h).view(h.shape[0], -1, self._out_feats)
            rst = rst + resval
        # activation
        if self.activation:
            rst = self.activation(rst)
        return rst
