"""
Store global parameters here
"""
import torch
from enum import Enum

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
comment = "MF_version_add_reg"
tensorboard = True
GPU = torch.cuda.is_available()

class SamplingAlgorithms(Enum):
  uniform = 1
  sampler = 2

sampling_type = SamplingAlgorithms.sampler

# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)



logo = r"""
  _____
 /     \
/  __   \
| |  \__|
\ \
 \ \
 _\ \  
/ |\_\
\ \__||
 \____|
"""
print(logo)