"""
Store global parameters here
"""
import torch
import os
from enum import Enum

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
# hyperparameters 
config = {}
config['alpha'] = 100
config['beta']  = 20
config['eta']   = 0.5
config['epsilon'] = 0.001
config['latent_dim_rec'] = 64
config['latent_dim_var'] = 64
config['batch_size'] = 200
# ======================
TRAIN_epochs = 1000
LOAD = True
PATH = '../checkpoints'
top_k = 5
comment = "MF_version_add_reg"
tensorboard = True
GPU = torch.cuda.is_available()


class SamplingAlgorithms(Enum):
  uniform = 1
  sampler = 2

sampling_type = SamplingAlgorithms.uniform

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)





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