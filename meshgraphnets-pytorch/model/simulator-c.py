from .model import EncoderProcesserDecoder
import torch.nn as nn
import torch
from torch_geometric.data import Data
from utils import normalization
import os

'''
The code provided defines a class called `Simulator`, which is a subclass of 
`torch.nn.Module`. This class represents a simulator model and is used for 
training and inference.

Let's break down the different components of the code:

1. Import Statements:
   - `from .model import EncoderProcesserDecoder`: Importing the 
   `EncoderProcesserDecoder` model from the local `.model` module.
   - `import torch.nn as nn`: Importing the `torch.nn` module, which 
   provides various neural network components and functionalities.
   - `import torch`: Importing the `torch` module, which is the main PyTorch 
   library.
   - `from torch_geometric.data import Data`: Importing the `Data` class from 
   the `torch_geometric.data` module, which is used for handling graph data.
   - `from utils import normalization`: Importing the `normalization` module 
   from the local `utils` module.
   - `import os`: Importing the `os` module for handling file paths and 
   directories.

2. Class Definition: `Simulator`
   - The `Simulator` class is defined as a subclass of `nn.Module`.
   - The constructor (`__init__` method) initializes the simulator model.
     - It takes several arguments: `message_passing_num`, `node_input_size`, 
     `edge_input_size`, `device`, and `model_dir`.
     - `message_passing_num` represents the number of message passing 
     iterations in the `EncoderProcesserDecoder` model.
     - `node_input_size` and `edge_input_size` specify the sizes of the input 
     features for nodes and edges in the graph.
     - `device` specifies the device (CPU or GPU) on which the model will be 
     trained and run.
     - `model_dir` is the directory path for saving and loading the model 
     checkpoints.
     - Within the constructor, various components are initialized:
       - `node_input_size` and `edge_input_size` are assigned to instance 
       variables.
       - The `EncoderProcesserDecoder` model is created with the specified 
       parameters and moved to the specified device.
       - Two instances of `normalization.Normalizer` are created for output 
       normalization and node feature normalization.
       - A print statement confirms the initialization of the model.

3. Methods:
   - `update_node_attr(self, frames, types: torch.Tensor)`: This method updates 
   the node attributes of the graph based on the input frames and types.
     - It takes two arguments: `frames` and `types`.
     - `frames` represent the input frames, and `types` represent the node 
     types.
     - The frames are concatenated with a one-hot encoded representation of the 
     node types.
     - The concatenated tensor is then normalized using the `_node_normalizer` 
     instance.
     - The normalized tensor is returned as the updated node attributes.
   - `velocity_to_accelation(self, noised_frames, next_velocity)`: This method 
   calculates the acceleration based on the input frames and next velocity.
     - It takes two arguments: `noised_frames` and `next_velocity`.
     - The acceleration is computed as the difference between the next velocity 
     and the noised frames.
     - The acceleration tensor is returned.
   - `forward(self, graph: Data, velocity_sequence_noise)`: This method 
   performs the forward pass of the simulator model.
     - It takes two arguments: `graph` and `velocity_sequence_noise`.
     - If the model is in training mode, it performs the following steps:
       - Extracts the node type, frames, and target tensors from the input 
       graph.
       - Adds the velocity sequence noise to the frames to create
       `noised_frames`.
       - Calls `update_node_attr` to update the node attributes based on the 
       noised frames and node type.
       - Sets the graph's node attributes to the updated node attributes.
       - Passes the graph through the model to obtain the predicted output.
       - Calculates the target acceleration based on the noised frames and 
       target tensors.
       - Normalizes the target acceleration using the `_output_normalizer` 
       instance.
       - Returns the predicted output and the normalized target acceleration.
     - If the model is not in training mode, it performs similar steps but 
     without the target acceleration calculation and normalization.
       - Instead, it uses the predicted output to update the frames and obtain 
       the predicted velocity.
       - The predicted velocity is returned.
   - `load_checkpoint(self, ckpdir=None)`: This method loads a saved checkpoint 
   for the simulator model.
     - It takes an optional argument `ckpdir` which specifies the directory 
     path of the checkpoint.
     - If no directory path is provided, it uses the default `model_dir` 
     specified during initialization.
     - The checkpoint is loaded using `torch.load` and the model's state dict 
     is updated with the loaded values.
     - Other attributes of the model (such as normalizers) are updated based 
     on the loaded values.
     - A print statement confirms the successful loading of the checkpoint.
   - `save_checkpoint(self, savedir=None)`: This method saves the current state 
   of the simulator model as a checkpoint.
     - It takes an optional argument `savedir` which specifies the directory 
     path to save the checkpoint.
     - If no directory path is provided, it uses the default `model_dir` 
     specified during initialization.
     - The model's state dict, as well as the state of the output and node 
     normalizers, are saved in a dictionary.
     - The dictionary is then saved using `torch.save` to the specified 
     directory path.
     - A print statement confirms the successful saving of the checkpoint.

Overall, this code defines a simulator model that takes graph data as input and 
performs forward computations to predict outputs. It also provides 
functionality for loading and saving checkpoints of the model.
'''

class Simulator(nn.Module):

    def __init__(self, message_passing_num, node_input_size, edge_input_size, device, model_dir='checkpoint/simulator.pth') -> None:
        super(Simulator, self).__init__()

        self.node_input_size =  node_input_size
        self.edge_input_size = edge_input_size
        self.model_dir = model_dir
        self.model = EncoderProcesserDecoder(message_passing_num=message_passing_num, node_input_size=node_input_size, edge_input_size=edge_input_size).to(device)
        self._output_normalizer = normalization.Normalizer(size=2, name='output_normalizer', device=device)
        self._node_normalizer = normalization.Normalizer(size=node_input_size, name='node_normalizer', device=device)
        # self._edge_normalizer = normalization.Normalizer(size=edge_input_size, name='edge_normalizer', device=device)

        print('Simulator model initialized')

    def update_node_attr(self, frames, types:torch.Tensor):
        node_feature = []

        node_feature.append(frames) #velocity
        node_type = torch.squeeze(types.long())
        one_hot = torch.nn.functional.one_hot(node_type, 9)
        node_feature.append(one_hot)
        node_feats = torch.cat(node_feature, dim=1)
        attr = self._node_normalizer(node_feats, self.training)

        return attr

    def velocity_to_accelation(self, noised_frames, next_velocity):

        acc_next = next_velocity - noised_frames
        return acc_next


    def forward(self, graph:Data, velocity_sequence_noise):
        
        if self.training:
            
            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            target = graph.y

            noised_frames = frames + velocity_sequence_noise
            node_attr = self.update_node_attr(noised_frames, node_type)
            graph.x = node_attr
            predicted = self.model(graph)

            target_acceration = self.velocity_to_accelation(noised_frames, target)
            target_acceration_normalized = self._output_normalizer(target_acceration, self.training)

            return predicted, target_acceration_normalized

        else:

            node_type = graph.x[:, 0:1]
            frames = graph.x[:, 1:3]
            node_attr = self.update_node_attr(frames, node_type)
            graph.x = node_attr
            predicted = self.model(graph)

            velocity_update = self._output_normalizer.inverse(predicted)
            predicted_velocity = frames + velocity_update

            return predicted_velocity

    def load_checkpoint(self, ckpdir=None):
        
        if ckpdir is None:
            ckpdir = self.model_dir
        dicts = torch.load(ckpdir)
        self.load_state_dict(dicts['model'])

        keys = list(dicts.keys())
        keys.remove('model')

        for k in keys:
            v = dicts[k]
            for para, value in v.items():
                object = eval('self.'+k)
                setattr(object, para, value)

        print("Simulator model loaded checkpoint %s"%ckpdir)

    def save_checkpoint(self, savedir=None):
        if savedir is None:
            savedir=self.model_dir

        os.makedirs(os.path.dirname(self.model_dir), exist_ok=True)
        
        model = self.state_dict()
        _output_normalizer = self._output_normalizer.get_variable()
        _node_normalizer  = self._node_normalizer.get_variable()
        # _edge_normalizer = self._edge_normalizer.get_variable()

        to_save = {'model':model, '_output_normalizer':_output_normalizer, '_node_normalizer':_node_normalizer}

        torch.save(to_save, savedir)
        print('Simulator model saved at %s'%savedir)