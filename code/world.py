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

class SamplingAlgorithms(Enum):
  uniform   = 1 # it sucks
  sampler   = 2
  bpr       = 3
  alldata   = 4
  GMF       = 5
  Mixture   = 6
  light_gcn = 7
  light_gcn_mixture =8
  Sample_positive_all = 9
  Alldata_train_ELBO = 10
sampling_type = SamplingAlgorithms.Alldata_train_ELBO

# hyperparameters 
config = {}
config['alpha'] = 100
config['beta']  = 20
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
config['lightGCN_n_layers']= args.layer
# ===============================
config['batch_size'] = 32768
config['bpr_batch_size'] = 4096
config['all_batch_size'] = 32768
config['test_u_batch_size'] = 100
config['xij_dim'] = 8
config['num_xij'] = 1
# ======================
TRAIN_epochs = 1000
LOAD = True
PATH = '../checkpoints'
top_k = 5
topks = [5,10]
comment = f"MF_{sampling_type.name}"
tensorboard = True
GPU = torch.cuda.is_available()
device = torch.device('cuda' if GPU else "cpu")
CORES = multiprocessing.cpu_count() // 2
multi_cores = False

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