import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
import dgl.function as fn
import numpy as np
import scipy.sparse as ssp
from dgl.data import citation_graph as citegrh
import networkx as nx
##load data
data = citegrh.load_cora()
adj=nx.adjacency_matrix(data.graph)
#reorder
ids_shuffle = np.arange(2708)
np.random.shuffle(ids_shuffle)
adj=adj[ids_shuffle,:][:,ids_shuffle]
data.features=data.features[ids_shuffle]
data.labels=data.labels[ids_shuffle]
##
train_nodes=np.arange(1208)
test_nodes=np.arange(1708,2708)
train_adj= adj[train_nodes, :][:, train_nodes]
test_adj=adj[test_nodes,:][:,test_nodes]
trainG=dgl.DGLGraph(train_adj)
allG=dgl.DGLGraph(adj)
h=torch.tensor(data.features[train_nodes], dtype=torch.float64)
test_h=torch.tensor(data.features[test_nodes],dtype=torch.float64)
all_h=torch.tensor(data.features,dtype=torch.float64)
train_nodes=torch.tensor(train_nodes)
test_nodes=torch.tensor(test_nodes)
y_train=torch.tensor(data.labels[train_nodes])
y_test=torch.tensor(data.labels[test_nodes])
##configuration
lamb=0.5
weight_decay=5e-4
input_size=h.shape[1]
hidden_size=16
output_size=7
##Sampling size for each layer
layer_sizes=[256,256]
batch_size=256
class Sampler(object):
    def __init__(self, graph):
        self.graph = graph

    def __len__(self):
        raise NotImplementedError

    def __iter__(self):
        """Iterator

        The iterator must return a tuple at a time.  The first element must be an array of node IDs from which
        NodeFlow is grown.  Other elements can be arbitrary auxiliary data correspond to the current minibatch.
        """
        raise NotImplementedError


class NodeSampler(Sampler):
    """Minibatch sampler that samples batches of nodes uniformly from the given graph and list of seeds.
    """

    def __init__(self, graph, seeds, batch_size, input_size=None,layer_sample_nodes=None):
        super().__init__(graph)
        self.seeds = seeds
        self.batch_size = batch_size
        self.input_size=input_size
        if input_size:
            self.sample_weight=torch.randn((input_size,2),dtype=torch.float64,requires_grad=True)
            nn.init.xavier_uniform_(self.sample_weight)
            self.layer_sample_node=layer_sample_nodes
    def __len__(self):
        return len(self.seeds) // self.batch_size

    def __iter__(self):
        """Returns
        (1) The seed node IDs, for NodeFlow generation,
        (2) Indices of the seeds in the original seed array, as auxiliary data.
        """
        batches = torch.randperm(len(self.seeds)).split(self.batch_size)
        for i in range(len(self)):
            if len(batches[i]) < self.batch_size:
                break
            if self.input_size:
                yield self.seeds[batches[i]], batches[i],self.sample_weight,self.layer_sample_node
            else:
                yield self.seeds[batches[i]], batches[i]


def create_nodeflow(layer_mappings, block_mappings, block_aux_data, rel_graphs, seed_map):
    hg = dgl.hetero_from_relations(rel_graphs)
    hg.layer_mappings = layer_mappings
    hg.block_mappings = block_mappings
    hg.block_aux_data = block_aux_data
    hg.seed_map = seed_map
    return hg
def normalize_adj(adj):
    """Symmetrically normalize adjacency matrix."""
    adj=ssp.eye(adj.shape[0])+adj
    adj = ssp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = ssp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt)

class Generator(object):
    def __init__(self, sampler=None, num_workers=0):
        self.sampler = sampler
        self.num_workers = num_workers

        if num_workers > 0:
            raise NotImplementedError('multiprocessing')

    def __call__(self, seeds, *auxiliary):
        """
        The __call__ function must take in an array of seeds, and any auxiliary data, and
        return a NodeFlow grown from the seeds and conditioned on the auxiliary data.
        """
        raise NotImplementedError

    def __iter__(self):
        for sampled in self.sampler:
            seeds = sampled[0]
            auxiliary = sampled[1:]
            hg=self(seeds,*auxiliary)
            yield (hg,auxiliary[0])


class IterativeGenerator(Generator):
    """NodeFlow generator.  Only works on homographs.
    """

    def __init__(self, graph,num_blocks, sampler=None, num_workers=0, coalesce=False):
        super().__init__(sampler, num_workers)
        self.graph = graph
        self.num_blocks = num_blocks
        self.coalesce = coalesce

    def __call__(self, seeds, *auxiliary):
        """Default implementation of IterativeGenerator.
        """
        curr_frontier = seeds  # Current frontier to grow neighbors from
        layer_mappings = []  # Mapping from layer node ID to parent node ID
        block_mappings = []  # Mapping from block edge ID to parent edge ID, or -1 if nonexistent
        block_aux_data = []
        rel_graphs = []

        if self.coalesce:
            curr_frontier = torch.LongTensor(np.unique(seeds.numpy()))
            invmap = {x: i for i, x in enumerate(curr_frontier.numpy())}
            seed_map = [invmap[x] for x in seeds.numpy()]
        else:
            seed_map = list(range(len(seeds)))

        layer_mappings.append(curr_frontier.numpy())

        for i in reversed(range(self.num_blocks)):
            neighbor_nodes, neighbor_edges, num_neighbors, aux_result = self.stepback(curr_frontier,i, *auxiliary)
            #self._build_graph(i,curr_frontier,neighbor_nodes,neighbor_edges,num_neighbors, aux_result, rel_graphs,layer_mappings,block_mappings, block_aux_data)
            prev_frontier_srcs = neighbor_nodes
            # The un-coalesced mapping from block edge ID to parent edge ID
            prev_frontier_edges = neighbor_edges.numpy()
            nodes_idx_map = dict({*zip(neighbor_nodes.numpy(), range(len(aux_result)))})
            # Coalesce nodes
            if self.coalesce:
                prev_frontier = np.unique(prev_frontier_srcs.numpy())
                prev_frontier_invmap = {x: j for j, x in enumerate(prev_frontier)}
                block_srcs = np.array([prev_frontier_invmap[s] for s in prev_frontier_srcs.numpy()])
            else:
                prev_frontier = prev_frontier_srcs.numpy()
                block_srcs = np.arange(len(prev_frontier_edges))
            aux_result=aux_result[[nodes_idx_map[i] for i in prev_frontier]]
            block_dsts = np.arange(len(curr_frontier)).repeat(num_neighbors)

            rel_graphs.insert(0, dgl.bipartite(
                (block_srcs, block_dsts),
                'layer%d' % i, 'block%d' % i, 'layer%d' % (i + 1),
                (len(prev_frontier), len(curr_frontier))
            ))
            # rel_graphs.insert(0, dgl.bipartite(
            #     (block_srcs[-len(curr_frontier):], np.arange(len(curr_frontier))),
            #     'layer%d' % i, 'block%d_self' %i, 'layer%d' % (i+ 1),
            #     (len(prev_frontier), len(curr_frontier))
            # ))

            layer_mappings.insert(0, prev_frontier)
            block_mappings.insert(0, prev_frontier_edges)
            block_aux_data.insert(0, aux_result)

            curr_frontier = torch.LongTensor(prev_frontier)

        return create_nodeflow(
            layer_mappings=layer_mappings,
            block_mappings=block_mappings,
            block_aux_data=block_aux_data,
            rel_graphs=rel_graphs,
            seed_map=seed_map)

    def stepback(self, curr_frontier, *auxiliary):
        """Function that takes in the node set in the current layer, and returns the
        neighbors of each node.

        Parameters
        ----------
        curr_frontier : Tensor
        auxiliary : any auxiliary data yielded by the sampler

        Returns
        -------
        neighbor_nodes, incoming_edges, num_neighbors, auxiliary: Tensor, Tensor, Tensor, any
            num_neighbors[i] contains the number of neighbors generated for curr_frontier[i]

            neighbor_nodes[sum(num_neighbors[0:i]):sum(num_neighbors[0:i+1])] contains the actual
            neighbors as node IDs in the original graph for curr_frontier[i].

            incoming_edges[sum(num_neighbors[0:i]):sum(num_neighbors[0:i+1])] contains the actual
            incoming edges as edge IDs in the original graph for curr_frontier[i], or -1 if the
            edge does not exist, or if we don't care about the edge, in the original graph.

            auxiliary could be of any type containing block-specific additional data.
        """
        raise NotImplementedError
class DefaultGenerator(IterativeGenerator):
    def stepback(self, curr_frontier,layer_index, *auxiliary):
        # Relies on that the same dst node of in_edges are contiguous, and the dst nodes
        # are ordered the same as curr_frontier.

        src, _, eid = self.graph.in_edges(curr_frontier, form='all')
        pre_nodes=torch.unique(torch.cat([src,curr_frontier],dim=0))
        curr_padding = curr_frontier.repeat_interleave(len(pre_nodes))
        cand_padding = pre_nodes.repeat(len(curr_frontier))
        has_loops = curr_padding == cand_padding
        has_edges = self.graph.has_edges_between(cand_padding, curr_padding)
        loops_or_edges = (has_edges.bool() + has_loops).int()
        num_neighbors = loops_or_edges.reshape((len(curr_frontier), -1)).sum(1)
        sample_neighbor=cand_padding[loops_or_edges.bool()]

        q_probs=torch.ones(sample_neighbor.shape[0],dtype=torch.float64)
        return sample_neighbor, eid, num_neighbors,q_probs

class AdaptGenerator(IterativeGenerator):
    def __init__(self,graph, num_blocks,node_feature=None ,sampler=None, num_workers=0, coalesce=False):
        """

        :type graph: dgl.DGLGraph
        """
        self.node_feature=node_feature.double()
        self.norm_adj=normalize_adj(train_adj).tocsr()

        super(AdaptGenerator,self).__init__(graph, num_blocks, sampler, num_workers, coalesce)

    def stepback(self, curr_frontier,layer_index, *auxiliary):
        # Relies on that the same dst node of in_edges are contiguous, and the dst nodes
        # are ordered the same as curr_frontier.

        # retrieve x
        # x = self.graph.ndata['x']
        is_sparse=False
        sample_weights=auxiliary[1]
        layer_size=auxiliary[2][layer_index]
        current_layer_feature=self.node_feature[curr_frontier]

        src,des, eid = self.graph.in_edges(curr_frontier, form='all')
        neighbor_nodes=torch.unique(torch.cat((curr_frontier,src),dim=0),sorted=False)
        sparse_adj=self.norm_adj[curr_frontier,:][:,neighbor_nodes]
        if is_sparse:
             sparse_adj=sparse_adj.tocoo()
             tensor_adj=torch.sparse.DoubleTensor(sparse_adj.row,sparse_adj.col,sparse_adj.data)
        else:
             tensor_adj = torch.tensor(sparse_adj.A)
        hu=torch.matmul(self.node_feature[neighbor_nodes],sample_weights[:,0])
        hv=torch.sum(torch.matmul(self.node_feature[curr_frontier],sample_weights[:,1]))
        adj_part=torch.sqrt(torch.sum(torch.pow(tensor_adj,2),dim=0))
        attension_part=F.relu(hv+hu)+1
        gu=F.relu(hu)+1
        probas=adj_part*attension_part*gu
        probas=probas/torch.sum(probas)
        ##sparse_adj
        canidates=neighbor_nodes[probas.multinomial(num_samples=layer_size,replacement=True)]
        ivmap={x: i for i,x in enumerate(neighbor_nodes.numpy())}

        curr_padding = curr_frontier.repeat_interleave(len(canidates))
        cand_padding = canidates.repeat(len(curr_frontier))
        has_loops=curr_padding==cand_padding
        has_edges= self.graph.has_edges_between(cand_padding,curr_padding)
        loops_or_edges=(has_edges.bool()+has_loops).int()
        num_neighbors= loops_or_edges.reshape((len(curr_frontier),-1)).sum(1)
        # for i in range(len(curr_frontier)):
        #     for j in range(len(canidates)):
        #         if curr_frontier[i]==canidates[j] or self.graph.has_edge_between(canidates[j].item(),curr_frontier[i].item()):
        #             num_neighbors[i]+=1
        eids=torch.zeros(torch.sum(num_neighbors),dtype=torch.int64)-1

        sample_neighbor=cand_padding[loops_or_edges.bool()]
        #curid = 0
        # for i in range(len(curr_frontier)):
        #     for j in range(len(canidates)):
        #         if curr_frontier[i]==canidates[j] or self.graph.has_edge_between(canidates[j].item(),curr_frontier[i].item()):
        #             if curr_frontier[i]==canidates[j]:
        #                 eids[curid]=-1
        #             else:
        #                 eids[curid]=self.graph.edge_id(canidates[j],curr_frontier[i])
        #             sample_neighbor[curid]=canidates[j]
        #             curid+=1
        q_prob=probas[[ivmap[i] for i in sample_neighbor.numpy()]]
        #sample_neighbor=sample_neighbor.repeat(len(curr_frontier))
        #num_neighbors = torch.LongTensor(len(sample_neighbor)).repeat(len(curr_frontier))
        has_edge_ids=torch.where(has_edges)[0]
        all_ids=torch.where(loops_or_edges)[0]
        edges_ids_map=torch.where(has_edge_ids[:,None]==all_ids[None,:])[1]
        eids[edges_ids_map]=self.graph.edge_ids(cand_padding,curr_padding)
        return sample_neighbor, eids, num_neighbors,q_prob
class SAGEConv2(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 aggregator_type='mean',
                 feat_drop=0.,
                 bias=True,
                 norm=None,
                 activation=None,
                 G=None):
        super(SAGEConv2, self).__init__()
        self._in_feats = in_feats
        self._out_feats = out_feats
        self.norm = norm
        self.feat_drop = nn.Dropout(feat_drop)
        self.activation = activation
        #self.fc_self = nn.Linear(in_feats, out_feats, bias=bias).double()
        self.fc_neigh = nn.Linear(in_feats, out_feats, bias=bias).double()
        self.reset_parameters()
        self.G=G
    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        #nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)

    def forward(self, graph, hidden_feat, node_feat, layer_id, sample_weights, norm_adj=None, var_loss=None, is_test=False):
        """
        graph: Bipartite.  Has two edge types.  The first one represents the connection to
        the desired nodes from neighbors.  The second one represents the computation
        dependence of the desired nodes themselves.
        :type graph: dgl.DGLHeteroGraph
        """
        # local_var is not implemented for heterograph
        # graph = graph.local_var()
        hidde_feat = self.feat_drop(hidden_feat)



        neighbor_etype_name = 'block%d' % layer_id
        #self_etype_name = 'block%d_self' % layer_id
        src_name = 'layer%d' % layer_id
        dst_name = 'layer%d' % (layer_id + 1)

        graph.nodes[src_name].data['hidden_feat'] = hidden_feat
        graph.nodes[src_name].data['node_feat']=node_feat[graph.layer_mappings[layer_id]]
        if not is_test:
            graph.nodes[src_name].data['norm_deg'] = 1 / torch.sqrt(
                trainG.in_degrees(graph.layer_mappings[layer_id]).double() + 1)
            graph.nodes[dst_name].data['norm_deg'] = 1 / torch.sqrt(
                trainG.in_degrees(graph.layer_mappings[layer_id + 1]).double() + 1)
        else:
            graph.nodes[src_name].data['norm_deg'] = 1 / torch.sqrt(
                allG.in_degrees(graph.layer_mappings[layer_id]).double() + 1)
            graph.nodes[dst_name].data['norm_deg'] = 1 / torch.sqrt(
                allG.in_degrees(graph.layer_mappings[layer_id + 1]).double() + 1)
        graph.nodes[dst_name].data['node_feat']=node_feat[graph.layer_mappings[layer_id+1]]
        graph.nodes[src_name].data['q_probs']=graph.block_aux_data[layer_id]
        def send_func(edges):
            hu=torch.matmul(edges.src['node_feat'],sample_weights[:,0])
            hv=torch.matmul(edges.dst['node_feat'],sample_weights[:,1])
            attensions=edges.src['norm_deg']*edges.dst['norm_deg']*(F.relu(hu+hv)+0.1)/edges.src['q_probs']/len(hu)

            hidden=edges.src['hidden_feat']*torch.reshape(attensions,[-1,1])
            return {"hidden":hidden}
        def recv_func(nodes):
            #print(nodes.mailbox['hidden'].shape)
            msgs=torch.sum(nodes.mailbox['hidden'],dim=1)
            return {'neigh':msgs}
        #def receive_fuc(nodes):
        # aggregate from neighbors
        graph[neighbor_etype_name].update_all(message_func=send_func ,reduce_func=recv_func)
        # copy h to dst nodes from corresponding src nodes, marked by "self etype"
        #graph[self_etype_name].update_all(fn.copy_src('h', 'm'), fn.sum('m', 'h'))

        h_neigh = graph.nodes[dst_name].data['neigh']
        #h_self = graph.nodes[dst_name].data['h']
        rst = self.fc_neigh(h_neigh)

        # activation
        if self.activation is not None:
            rst = self.activation(rst)
        # normalization
        if self.norm is not None:
            rst = self.norm(rst)

        if var_loss and not is_test:
            pre_sup=self.fc_neigh(hidden_feat) #u*h
            support=norm_adj[graph.layer_mappings[layer_id+1],:][:,graph.layer_mappings[layer_id]]##v*u
            hu=torch.matmul(node_feat[graph.layer_mappings[layer_id]],sample_weights[:,0])
            hv=torch.matmul(node_feat[graph.layer_mappings[layer_id+1]],sample_weights[:,1])
            attensions=(F.relu(torch.reshape(hu,[1,-1])+torch.reshape(hv,[-1,1]))+1)/graph.block_aux_data[layer_id]/len(hu)
            adjust_support=torch.tensor(support.A,dtype=torch.float64)*attensions
            support_mean=adjust_support.sum(0)
            mu_v=torch.mean(rst,dim=0) #h
            diff=torch.reshape(support_mean,[-1,1])*pre_sup-torch.reshape(mu_v,[1,-1])
            loss=torch.sum(diff*diff)/len(hu)/len(hv)
            return rst,loss
        return rst


class GraphSAGENet2(nn.Module):
    def __init__(self,sample_weights,node_feature):
        super().__init__()
        self.layers = nn.ModuleList([
            SAGEConv2(input_size, hidden_size, 'mean',bias=False ,activation=F.relu),
            SAGEConv2(hidden_size, output_size, 'mean',bias=False, activation=F.relu),
        ])
        #self.classifier = nn.Linear(80, 1).double()
        self.sample_weights=sample_weights
        self.node_feature=node_feature
        self.norm_adj=normalize_adj(trainG.adjacency_matrix_scipy())

    def forward(self, nf, h,is_test=False):
        for i, layer in enumerate(self.layers):
            if i==len(self.layers)-1 and not is_test:
                h,loss = layer(nf, h, self.node_feature,i,self.sample_weights,norm_adj=self.norm_adj,var_loss=True,is_test=is_test)
            else:
                h = layer(nf, h, self.node_feature,i,self.sample_weights,is_test=is_test)
        #y = self.classifier(h)
        if is_test:
            return h
        return h,loss




# Create a sampler
train_sampler = NodeSampler(graph=trainG, seeds=train_nodes, batch_size=batch_size, input_size=input_size, layer_sample_nodes=layer_sizes)
# Initialize a generator with the created sampler
test_sampler = NodeSampler(graph=trainG, seeds=test_nodes, batch_size=len(test_nodes), input_size=input_size, layer_sample_nodes=layer_sizes)
##Generator for training
nf_generator =AdaptGenerator(graph=trainG, node_feature=all_h, sampler=train_sampler, num_blocks=len(layer_sizes), coalesce=True)
#Generator for testing
test_generator=DefaultGenerator(graph=allG,sampler=test_sampler,num_blocks=len(layer_sizes),coalesce=True)
model2 = GraphSAGENet2(train_sampler.sample_weight,all_h)
params=list(model2.parameters())
params.append(train_sampler.sample_weight)
opt = torch.optim.Adam(params=params)
model2.train()

for epoch in range(500):
    # Equivalently:
    # for seeds, sample_indices in train_sampler:
    #    nf = nf_generator(seeds)
    for nf, sample_indices in nf_generator:
        seed_map = nf.seed_map
        train_y_hat, regloss = model2(nf, h[nf.layer_mappings[0]])
        train_y_hat=train_y_hat[seed_map]
        #print("train",train_y_hat)
        y_train_batch = y_train[sample_indices]
        y_pred=torch.argmax(train_y_hat, dim=1)
        train_acc=torch.sum(torch.eq(y_pred,y_train_batch)).item()/batch_size
        loss = F.cross_entropy(train_y_hat.squeeze(), y_train_batch)
        #print(regloss.item(),loss.item())
        l2_loss=torch.norm(params[0])
        total_loss=regloss*lamb+loss+l2_loss*weight_decay
        opt.zero_grad()
        total_loss.backward()
        opt.step()
        #print(train_sampler.sample_weight)
    for test_nf,test_sample_indices in test_generator:
        seed_map = test_nf.seed_map
        test_y_hat= model2(test_nf, all_h[test_nf.layer_mappings[0]], is_test=True)
        #print("test",test_y_hat)
        test_y_hat=test_y_hat[seed_map]
        y_test_batch=y_test[test_sample_indices]
        y_pred=torch.argmax(test_y_hat, dim=1)
        test_acc=torch.sum(torch.eq(y_pred,y_test_batch)).item()/len(y_pred)
    print("eqoch{} train accuracy {}, regloss {}, loss {} ,test accuracy {}".format(epoch,train_acc, regloss.item()*lamb,loss.item(),test_acc))