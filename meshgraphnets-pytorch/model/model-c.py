import torch.nn as nn
from .blocks import EdgeBlock, NodeBlock
from utils.utils import decompose_graph, copy_geometric_data
from torch_geometric.data import Data

'''
The given code defines a neural network architecture for a message-passing 
graph neural network. Here's a breakdown of the functions and classes defined 
in the code:

build_mlp: This function constructs a multi-layer perceptron (MLP) module given
the input size, hidden size, and output size. It uses the nn.Sequential module
from PyTorch to define a sequence of linear layers with ReLU activation 
functions. If lay_norm is True, it adds a layer normalization module 
(nn.LayerNorm) after the MLP layers. The function returns the constructed MLP
module.

Encoder: This class defines an encoder module for the graph neural network. 
It takes as input the sizes of the edge features (edge_input_size), node 
features (node_input_size), and hidden size. In the constructor, it creates an 
edge block encoder (eb_encoder) and a node block encoder (nb_encoder), both 
implemented as MLP modules using the build_mlp function. In the forward method, 
it decomposes the input graph into node and edge attributes using the 
decompose_graph function. Then, it applies the encoders to the node and edge 
attributes separately and returns a Data object containing the encoded node 
features (x), edge attributes (edge_attr), and the edge index of the input 
graph.

GnBlock: This class defines a graph neural network block for message passing. 
It takes the hidden size as input. In the constructor, it creates a custom 
function MLP for node attributes (nb_custom_func) and edge attributes 
(eb_custom_func) using the build_mlp function. It also initializes an edge 
block module (eb_module) and a node block module (nb_module) with the 
respective custom functions. In the forward method, it applies the edge block 
module and the node block module to the input graph. It returns a Data object 
with updated node features (x), edge attributes (edge_attr), and edge index.

Decoder: This class defines a decoder module for the graph neural network. It 
takes the hidden size and output size as input. In the constructor, it creates 
a decoding module as an MLP using the build_mlp function. The decoding module 
takes the hidden size as the input size and outputs the specified output size. 
In the forward method, it applies the decoding module to the input graph's node 
features (x) and returns the decoded output.

EncoderProcesserDecoder: This class defines the main architecture of the graph 
neural network. It takes the number of message passing steps 
(message_passing_num), node input size, edge input size, and hidden size as 
input. In the constructor, it initializes an encoder module (encoder), a list 
of message passing blocks (processer_list), and a decoder module (decoder). 
The encoder is created using the Encoder class with the specified edge input 
size, node input size, and hidden size. The message passing blocks are created 
as instances of the GnBlock class, with each block having the specified hidden 
size. The decoder is created using the Decoder class with the specified hidden 
size and output size (defaulted to 2). In the forward method, it performs the 
forward pass of the graph neural network. It applies the encoder to the input 
graph, then iteratively applies each message passing block in the 
processer_list to update the graph features. Finally, it applies the decoder 
to the updated graph features and returns the decoded output.

The code combines these modules and classes to define a graph neural network 
architecture consisting of an encoder, multiple message passing blocks, and a 
decoder. The encoder encodes the input graph, the message passing blocks 
propagate information through the graph, and the decoder decodes the final 
graph features to produce the output.
'''

def build_mlp(in_size, hidden_size, out_size, lay_norm=True):

    module = nn.Sequential(nn.Linear(in_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, hidden_size), nn.ReLU(), nn.Linear(hidden_size, out_size))
    if lay_norm: return nn.Sequential(module,  nn.LayerNorm(normalized_shape=out_size))
    return module


class Encoder(nn.Module):

    def __init__(self,
                edge_input_size=128,
                node_input_size=128,
                hidden_size=128):
        super(Encoder, self).__init__()

        self.eb_encoder = build_mlp(edge_input_size, hidden_size, hidden_size)
        self.nb_encoder = build_mlp(node_input_size, hidden_size, hidden_size)
    
    def forward(self, graph):

        node_attr, _, edge_attr, _ = decompose_graph(graph)
        node_ = self.nb_encoder(node_attr)
        edge_ = self.eb_encoder(edge_attr)
        
        return Data(x=node_, edge_attr=edge_, edge_index=graph.edge_index)



class GnBlock(nn.Module):

    def __init__(self, hidden_size=128):

        super(GnBlock, self).__init__()


        eb_input_dim = 3 * hidden_size
        nb_input_dim = 2 * hidden_size
        nb_custom_func = build_mlp(nb_input_dim, hidden_size, hidden_size)
        eb_custom_func = build_mlp(eb_input_dim, hidden_size, hidden_size)
        
        self.eb_module = EdgeBlock(custom_func=eb_custom_func)
        self.nb_module = NodeBlock(custom_func=nb_custom_func)

    def forward(self, graph):
    
        graph_last = copy_geometric_data(graph)
        graph = self.eb_module(graph)
        graph = self.nb_module(graph)
        edge_attr = graph_last.edge_attr + graph.edge_attr
        x = graph_last.x + graph.x
        return Data(x=x, edge_attr=edge_attr, edge_index=graph.edge_index)



class Decoder(nn.Module):

    def __init__(self, hidden_size=128, output_size=2):
        super(Decoder, self).__init__()
        self.decode_module = build_mlp(hidden_size, hidden_size, output_size, lay_norm=False)

    def forward(self, graph):
        return self.decode_module(graph.x)


class EncoderProcesserDecoder(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, hidden_size=128):

        super(EncoderProcesserDecoder, self).__init__()

        self.encoder = Encoder(edge_input_size=edge_input_size, node_input_size=node_input_size, hidden_size=hidden_size)
        
        processer_list = []
        for _ in range(message_passing_num):
            processer_list.append(GnBlock(hidden_size=hidden_size))
        self.processer_list = nn.ModuleList(processer_list)
        
        self.decoder = Decoder(hidden_size=hidden_size, output_size=2)

    def forward(self, graph):

        graph= self.encoder(graph)
        for model in self.processer_list:
            graph = model(graph)
        decoded = self.decoder(graph)

        return decoded







