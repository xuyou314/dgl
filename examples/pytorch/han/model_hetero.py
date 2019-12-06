"""This model shows an example of using dgl.metapath_reachable_graph on the original heterogeneous
graph.

Because the original HAN implementation only gives the preprocessed homogeneous graph, this model
could not reproduce the result in HAN as they did not provide the preprocessing code, and we
constructed another dataset from ACM with a different set of papers, connections, features and
labels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

import dgl
# from dgl.nn.pytorch import GATConv
from utils import metagraph_graph, GATConv


class SemanticAttention(nn.Module):
    def __init__(self, in_size, hidden_size=128, num_meta=0):
        super(SemanticAttention, self).__init__()

        # self.project = nn.Sequential(
        #     nn.Linear(in_size, hidden_size),
        #     nn.Tanh(),
        #     nn.Linear(hidden_size, 1, bias=False)
        # )
        # self.tranform = nn.Linear(in_size, hidden_size)
        # self.output = nn.Linear(num_meta * 128, 64)  ##
        # self.output2 = nn.Linear(num_meta * 64, 64)  ##metagraph-specific
        self.relu = F.relu
        self.tanh=F.tanh
        # self.output3 = nn.Linear(num_meta * 64, num_meta * 64)
        self.project2 = nn.Linear(num_meta * in_size, hidden_size)
        ##

        # self.project_m =nn.ModuleList([nn.Linear(64, 64) for i in range(num_meta)])
        # self.project_m2 =nn.ModuleList([nn.Linear(64, 64) for i in range(num_meta)])

    def forward(self, z):
        #  w = self.project(z)
        #  beta = torch.softmax(w, dim=1)
        #  print("semantic attention score {}".format(beta.cpu().
        # #                                            detach().numpy().squeeze(-1).mean(0)))
        #  print("semantic attention score {}".format(beta.cpu().
        # #                                           detach().numpy().squeeze(-1).mean(0)))
        # #
        #  return (beta* z).sum(1)
        ##way 2
        # w=self.relu(self.tranform(z))
        # return self.output(torch.reshape(w,[z.shape[0],-1]))
        ##way 3
        # return self.output2(torch.reshape(z,[z.shape[0],-1]))
        ##way 4 with relu
        # return self.relu(self.output2(torch.reshape(z, [z.shape[0], -1])))
        ##way 5
        # hidden_z= self.output3(torch.reshape(z, [z.shape[0], -1]))
        # return torch.reshape(hidden_z,[z.shape[0],5,-1]).sum(1)
        ##way 6
        # hidden_1=self.relu(self.output3(torch.reshape(z, [z.shape[0], -1])))
        # return self.output2(hidden_1)
        ##way 7 ##without parameter penalty
        # return self.relu(self.project2(torch.reshape(z, [z.shape[0], -1])))
        ##way 8 without penalty and relu
        return self.project2(torch.reshape(z, [z.shape[0], -1]))
        ##way 9
        # hs = []
        # for i in range(len(self.project_m)):
        #     h1 = self.project_m[i](z[:, i, :])
        #     hs.append(h1)
        # all_h = torch.cat(hs,dim=1)
        # return self.project2(all_h)
        ##way 10
        # hs = []
        # for i in range(len(self.project_m2)):
        #     h1 = self.project_m2[i](z[:, i, :])
        #     hs.append(h1)
        # all_h = torch.cat(hs, dim=1)
        # return all_h
        # way 11
        # return torch.reshape(z,[z.shape[0],-1])


class HANLayer(nn.Module):
    """
    HAN layer.

    Arguments
    ---------
    meta_paths : list of metapaths, each as a list of edge types
    in_size : input feature dimension
    out_size : output feature dimension
    layer_num_heads : number of attention heads
    dropout : Dropout probability

    Inputs
    ------
    g : DGLHeteroGraph
        The heterogeneous graph
    h : tensor
        Input features

    Outputs
    -------
    tensor
        The output feature
    """

    def __init__(self, meta_graphs, in_size, out_size, layer_num_heads, dropout,
                 embedding_dim,
                 use_both=False, weighted=False):
        super(HANLayer, self).__init__()

        # One GAT layer for each meta path based adjacency matrix
        self.gat_layers = nn.ModuleList()
        self.use_both = use_both
        self.weighted = weighted
        for i in range(len(meta_graphs)):
            if (len(meta_graphs[i][0]) != 1) and use_both:
                self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                               dropout, dropout, activation=F.elu))
            self.gat_layers.append(GATConv(in_size, out_size, layer_num_heads,
                                           dropout, dropout, activation=F.elu))
        self.semantic_attention = SemanticAttention(in_size=out_size * layer_num_heads,
                                                    hidden_size=embedding_dim,
                                                    num_meta=len(self.gat_layers))
        self.meta_graphs = meta_graphs

        self._cached_graph = None
        self._cached_coalesced_graph = []

    def forward(self, g, h):
        semantic_embeddings = []

        if self._cached_graph is None or self._cached_graph is not g:
            self._cached_graph = g
            self._cached_coalesced_graph.clear()
            for meta_graph in self.meta_graphs:
                self._cached_coalesced_graph.append(metagraph_graph(
                    g, meta_graph, weighted=self.weighted))
                if (len(meta_graph[0]) != 1) and self.use_both:
                    self._cached_coalesced_graph.append(metagraph_graph(
                        g, meta_graph, both_appear=False,
                        weighted=self.weighted))

        for i in range(len(self._cached_coalesced_graph)):
            new_g = self._cached_coalesced_graph[i]
            semantic_embeddings.append(self.gat_layers[i](new_g, h).flatten(1))
        semantic_embeddings = torch.stack(semantic_embeddings, dim=1)  # (N, M, D * K)

        return self.semantic_attention(semantic_embeddings)  # (N, D * K)


class HAN(nn.Module):
    def __init__(self, meta_paths, in_size, hidden_size, out_size, num_heads, dropout,
                 embedding_dim,
                 use_both=False, weighted=False):
        super(HAN, self).__init__()
        self.weighted = weighted
        self.layers = nn.ModuleList()
        self.layers.append(HANLayer(meta_paths, in_size, hidden_size, num_heads[0], dropout,
                                    use_both=use_both, weighted=self.weighted,
                                    embedding_dim=embedding_dim))
        for l in range(1, len(num_heads)):
            self.layers.append(HANLayer(meta_paths, hidden_size * num_heads[l - 1],
                                        hidden_size, num_heads[l], dropout,
                                        use_both=use_both, weighted=self.weighted,
                                        embedding_dim=embedding_dim))
        self.predict = nn.Linear(embedding_dim , out_size)

    def forward(self, g, h):
        for gnn in self.layers:
            h = gnn(g, h)

        return self.predict(h), h
