"""
Store global parameters here
"""
import torch
import os
from enum import Enum
import multiprocessing

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

class SamplingAlgorithms(Enum):
  uniform   = 1 # it sucks
  sampler   = 2
  bpr       = 3
  alldata   = 4
  GMF       = 5
  Mixture   = 6
  light_gcn = 7
  light_gcn_mixture =8

sampling_type = SamplingAlgorithms.Mixture

# hyperparameters 
config = {}
config['alpha'] = 100
config['beta']  = 20
config['eta']   = 0.5
config['epsilon'] = 0.001
config['latent_dim_rec'] = 32
config['latent_dim_var'] = 16
config['batch_size'] = 256
config['bpr_batch_size'] = 4096
config['all_batch_size'] = 32768
config['lightGCN_n_layers']=2
config['test_u_batch_size'] = 100
# ======================
TRAIN_epochs = 1000
LOAD = True
PATH = '../checkpoints'
top_k = 5
topks = [5,10,20,40]
comment = f"MF_{sampling_type.name}"
tensorboard = True
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2


# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)





# parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--rlr",
                    type=float,
                    default=0.003,
                    help="recmodel learning rate")
parser.add_argument("--vlr", 
                    type=float,
                    default=0.001,
                    help="varmodel learning rate")
parser.add_argument('--batch',
                    type=int, 
                    default=64)
parser.add_argument('--sample',
                    type=str, 
                    default=sampling_type.name)




logo = r"""
███████╗ █████╗ ███╗   ███╗██████╗ ██╗     ██╗███╗   ██╗ ██████╗ 
██╔════╝██╔══██╗████╗ ████║██╔══██╗██║     ██║████╗  ██║██╔════╝ 
███████╗███████║██╔████╔██║██████╔╝██║     ██║██╔██╗ ██║██║  ███╗
╚════██║██╔══██║██║╚██╔╝██║██╔═══╝ ██║     ██║██║╚██╗██║██║   ██║
███████║██║  ██║██║ ╚═╝ ██║██║     ███████╗██║██║ ╚████║╚██████╔╝
╚══════╝╚═╝  ╚═╝╚═╝     ╚═╝╚═╝     ╚══════╝╚═╝╚═╝  ╚═══╝ ╚═════╝ 
"""
# font: ANSI Shadow
# refer to http://patorjk.com/software/taag/#p=display&f=ANSI%20Shadow&t=Sampling
print(logo)