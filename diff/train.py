from dataset import FPC
from model.simulator import Simulator
import torch
from utils.noise import get_velocity_noise
import time
from utils.utils import NodeType
from torch_geometric.loader import DataLoader
import torch_geometric.transforms as T

import argparse
parser = argparse.ArgumentParser(description='Implementation of MeshGraphNets')
parser.add_argument("--gpu",
                    type=int,
                    default=0,
                    help="gpu number: 0 or 1")

parser.add_argument("--model_dir",
                    type=str,
                    default='checkpoint/simulator.pth')

parser.add_argument("--dataset_dir",
                    type=str,
                    default='/mnt/c/Users/alexv/Desktop/pytorch/data/hd5')

parser.add_argument("--test_split", type=str, default='test')

parser.add_argument("--batch_size", type=int, default=1)

parser.add_argument("--print_batch", type=int, default=1)

parser.add_argument("--save_batch", type=int, default=50)

parser.add_argument("--max_epoch", type=int, default=2)

parser.add_argument("--max_workers", type=bool, default=True)

args = parser.parse_args()


noise_std = 2e-2



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'The target device is {device}')
simulator = Simulator(message_passing_num=15, node_input_size=11, edge_input_size=3, device=device)
optimizer = torch.optim.Adam(simulator.parameters(), lr=1e-4)
print('Optimizer initialized')


def train(model:Simulator, dataloader, optimizer):

    for batch_index, graph in enumerate(dataloader):

        graph = transformer(graph)
        graph = graph.to(device)

        node_type = graph.x[:, 0] #"node_type, cur_v, pressure, time"
        velocity_sequence_noise = get_velocity_noise(graph, noise_std=noise_std, device=device)
        predicted_acc, target_acc = model(graph, velocity_sequence_noise)
        mask = torch.logical_or(node_type==NodeType.NORMAL, node_type==NodeType.OUTFLOW)
        
        errors = ((predicted_acc - target_acc)**2)[mask]
        loss = torch.mean(errors)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch_index % args.print_batch == 0:
            print('batch %d [loss %.2e]'%(batch_index, loss.item()))

        if batch_index % args.save_batch == 0:
            model.save_checkpoint()

if __name__ == '__main__':
    "alloting the more resources"
    import resource, sys, os, threading
    #os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:4000"
    #resource.setrlimit(resource.RLIMIT_STACK, (resource.RLIM_INFINITY, resource.RLIM_INFINITY))
    #os.system('ulimit -s unlimited; some_executable')
    # resource.setrlimit(resource.RLIMIT_STACK, (2**29,-1))
    sys.setrecursionlimit(10**6)
    # import threading
    threading.stack_size(2**26)
    # threading.Thread(target=main).start()
    torch.cuda.empty_cache()
    dataset_fpc = FPC(dataset_dir=args.dataset_dir, split=args.test_split, max_epochs=args.max_epoch)
    train_loader = DataLoader(dataset=dataset_fpc, batch_size=args.batch_size, num_workers=2)
    transformer = T.Compose([T.FaceToEdge(), T.Cartesian(norm=False), T.Distance(norm=False)])
    train(simulator, train_loader, optimizer)
