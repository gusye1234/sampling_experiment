import torch
# hyperparameters 
config = {}
config['alpha'] = 100
config['beta']  = 20
config['eta']   = 0.5
config['epsilon'] = 0.001
config['latent_dim_rec'] = 64
config['latent_dim_var'] = 64
config['batch_size'] = 5
# ======================
TRAIN_epochs = 1000
comment = "MF version"
tensorboard = True
GPU = torch.cuda.is_available()


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
 \____/
"""
print(logo)