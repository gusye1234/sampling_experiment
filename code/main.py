import world
import torch
from torch.utils.data import DataLoader
import model
import utils
import dataloader
import TrainProcedure
import numpy as np
from tensorboardX import SummaryWriter
from pprint import pprint
import os
import time
import register
# loading data...
dataset   = dataloader.LastFM()
lm_loader = DataLoader(dataset, batch_size=world.config['batch_size'], shuffle=True, drop_last=True) 

world.config['num_users'] = dataset.n_users
world.config['num_items'] = dataset.m_items


print('===========config================')
pprint(world.config)
print("cores for test:", world.CORES)
print(world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print(world.rec_type, world.var_type, world.loss_type, world.sample_type)
print('===========end===================')
world.config['dataset'] = dataset


# initialize models

Recmodel = register.Rec_register[world.rec_type](world.config)
Varmodel = register.Var_register[world.var_type](world.config)
train_method = register.sampling_register[world.sample_type]
elbo = utils.ELBO(world.config,
                  rec_model=Recmodel,
                  var_model=Varmodel)

flag = f"{world.sample_type}_{world.rec_type}_{world.var_type}"
if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, f'Rec_{flag}.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, f'Var_{flag}.pth.tar')))
sampler_gamma_save = utils.sample_for_basic_GMF_loss(k=3)

# train
Neg_k = 1
world.config['total_batch'] = int(len(dataset)/world.config['batch_size'])
Recmodel = Recmodel.to(world.device)
if globals().get('Varmodel'):
    Varmodel = Varmodel.to(world.device)

if world.tensorboard:
    w : SummaryWriter = SummaryWriter("/output/"+ "runs/"+time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
    
    
try:
    for i in range(world.TRAIN_epochs):
        start = time.time()
        print(f'===============[EPOCH {i}]=================')
        # training
        output_information = train_method(dataset, Recmodel, Varmodel, elbo, i, w)
        print(output_information)
        
        # save
        #torch.save(Recmodel.state_dict(), os.path.join(world.PATH, f'Rec_{flag}.pth.tar'))
        #torch.save(Varmodel.state_dict(), os.path.join(world.PATH, f'Var_{flag}.pth.tar'))
        # test
        if i%25 == 0 :
            print('[TEST]')
            testDict = dataset.getTestDict()
            TrainProcedure.Test(dataset, Recmodel, Varmodel, world.top_k, i, w)
        print("total time:", time.time() - start)
finally:
    if world.tensorboard:
        w.close()
