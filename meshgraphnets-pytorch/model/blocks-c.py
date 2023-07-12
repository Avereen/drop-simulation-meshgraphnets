import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_scatter import scatter_add
from utils.utils import decompose_graph
from torch_geometric.data import Data
import numpy as np

'''
The given code defines two classes: `EdgeBlock` and `NodeBlock`, both 
subclasses of `torch.nn.Module`. These classes are used for building blocks
 in a graph neural network.

The `EdgeBlock` class takes a custom function (`custom_func`) as an argument 
during initialization. In the `forward` method, the `graph` argument is 
decomposed into node attributes (`node_attr`), edge indices (`edge_index`),
edge attributes (`edge_attr`), and others. The `senders_idx` and 
`receivers_idx` are extracted from `edge_index`, and the corresponding node 
attributes are collected into `senders_attr` and `receivers_attr`, 
respectively. The collected attributes and edge attributes are concatenated 
to form `collected_edges`. This `collected_edges` tensor is then passed 
through the `self.net` function (the custom function provided during 
initialization) to obtain the updated `edge_attr`. Finally, a `Data` object 
is created with the updated `edge_attr` and other attributes, and returned 
as the output.

The `NodeBlock` class is similar to `EdgeBlock`. It also takes a custom 
function (`custom_func`) as an argument during initialization. In the
`forward` method, the `graph` argument is decomposed to obtain the edge 
attributes (`edge_attr`). The received edge attributes are aggregated using 
the `scatter_add` function along the receiver indices (`receivers_idx`). 
The node attributes (`graph.x`) and the aggregated received edges are
collected and concatenated into `collected_nodes`. The `collected_nodes` 
tensor is then passed through the `self.net` function to obtain the updated 
node attributes (`x`). Finally, a `Data` object is created with the updated 
`x`, edge attributes (`edge_attr`), and edge indices (`graph.edge_index`), 
and returned as the output.

In both classes, the `custom_func` argument represents a custom function that 
defines the specific operations to be applied to the collected attributes or 
edges. These functions can be defined externally and passed to the classes 
during initialization, allowing flexibility in designing different neural 
network architectures for graph data.
'''

class EdgeBlock(nn.Module):

    def __init__(self, custom_func=None):
        
        super(EdgeBlock, self).__init__()
        self.net = custom_func


    def forward(self, graph):

        node_attr, edge_index, edge_attr, _ = decompose_graph(graph)
        senders_idx, receivers_idx = edge_index
        edges_to_collect = []

        senders_attr = node_attr[senders_idx]
        receivers_attr = node_attr[receivers_idx]

        edges_to_collect.append(senders_attr)
        edges_to_collect.append(receivers_attr)
        edges_to_collect.append(edge_attr)

        collected_edges = torch.cat(edges_to_collect, dim=1)
        
        edge_attr_ = self.net(collected_edges)   # Update

        return Data(x=node_attr, edge_attr=edge_attr_, edge_index=edge_index)



    
class NodeBlock(nn.Module):

    def __init__(self, custom_func=None):

        super(NodeBlock, self).__init__()

        self.net = custom_func

    def forward(self, graph):
        # Decompose graph
        edge_attr = graph.edge_attr
        nodes_to_collect = []
        
        _, receivers_idx = graph.edge_index
        num_nodes = graph.num_nodes
        agg_received_edges = scatter_add(edge_attr, receivers_idx, dim=0, dim_size=num_nodes)

        nodes_to_collect.append(graph.x)
        nodes_to_collect.append(agg_received_edges)
        collected_nodes = torch.cat(nodes_to_collect, dim=-1)
        x = self.net(collected_nodes)
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)
       
            
            
        