"""
Store global parameters here
"""
import torch
import os
from enum import Enum
import multiprocessing
from parse import parse_args

args = parse_args()
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

Recmodels = {
  1:"mf",
}

Varmodels = {
  1:'mf',
  2:'mf_itemper',
  3:'lgn',
  4:'lgn_itemper_single',
  5:'lgn_itemper_matrix',
  6:'lgn_itemper_matrix_nohyper'
}

losses = {
  1: 'elbo'
}

samplings = {
  1: 'all_data',
  2: 'all_data_xij',
  3: 'all_data_nobatch',
  4: 'all_data_nobatch_xij',
  5: 'fast_sampling', # not yet,
}


rec_type = Recmodels[1]
var_type = Varmodels[args.vartype]
loss_type = losses[1]
sample_type = samplings[args.sampletype]



# hyperparameters 
config = {}

config['eta']   = 0.5
config['epsilon'] = 0.001
config['keep_prob'] = args.keepprob
config['dropout'] = args.dropout
# ================================
config['latent_dim_rec'] = args.recdim
config['latent_dim_var'] = args.vardim
config['rec_lr'] = args.reclr
config['var_lr'] = args.varlr
config['rec_weight_decay'] = args.recdecay
config['var_weight_decay'] = args.vardecay
config['x_weight_decay'] = args.xdecay
config['w_weight_decay'] = args.wdecay
config['gamma_KL'] = args.gamma_KL
config['lightGCN_n_layers']= args.layer
config['hyper_x'] = args.hyperx
# ===============================
config['batch_size'] = 32768
config['all_batch_size'] = 32768
config['test_u_batch_size'] = 100
config['xij_dim'] = 1
config['num_xij'] = 1
# ======================
TRAIN_epochs = 1401
LOAD = False
PATH = '../checkpoints'
top_k = 5
topks = 5
comment = f"{sample_type}_{rec_type}_{var_type}"
tensorboard = True
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
multi_cores = False
ontest = args.ontest
seed = args.seed


# let pandas shut up
from warnings import simplefilter
simplefilter(action="ignore", category=FutureWarning)




