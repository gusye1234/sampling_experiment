import world
import torch
from torch.utils.data import DataLoader
import model
import utils
import dataloader
import numpy as np
import TrainProcedure
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pprint

import time

# loading data...
dataset   = dataloader.LastFM()
lm_loader = DataLoader(dataset, batch_size=world.config['batch_size'], shuffle=False, drop_last=True) 

world.config['num_users'] = dataset.n_users
world.config['num_items'] = dataset.m_items

print('===========config================')
pprint(world.config)
print(world.comment)
print("tensorboard:", world.tensorboard)
print('===========end===================')
# initialize models

if world.sampling_type == 'uniform':
    Recmodel = model.RecMF(world.config)
    elbo     = utils.BCE(Recmodel)
else:
    Recmodel = model.RecMF(world.config)
    # Varmodel = model.VarMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    # register ELBO loss
    elbo = utils.ELBO(world.config, 
                    rec_model=Recmodel, 
                    var_model=Varmodel)
    sampler = utils.Sample_MF(k=1, var_model=Varmodel) 
    # k here doesn't matter, will be batch_size on training 


# train
Neg_k = 2
Total_batch = int(len(dataset)/world.config['batch_size'])


if world.tensorboard:
    w : SummaryWriter = SummaryWriter("./output/"+ "runs/"+time.strftime("%m-%d-%H:%M:%S-") + "-" + world.comment)
try:
    for i in range(world.TRAIN_epochs):
        for batch_i, batch_data in tqdm(enumerate(lm_loader)):
            if world.sampling_type == "uniform":
                loss1 = TrainProcedure.uniform_train(batch_data, dataset, Recmodel, elbo, Neg_k)
                if world.tensorboard:
                    w.add_scalar(f'UniformLoss/BCE', loss1, i*Total_batch + batch_i)
            else:
                loss1, loss2, exposed_users, exposed_items = TrainProcedure.sampler_train(dataset, sampler, Recmodel, Varmodel, elbo, world.config['batch_size'])
                if world.tensorboard:
                    w.add_scalar(f'SamplerLoss/stageOne', loss1, i*Total_batch + batch_i)
                    w.add_scalar(f'SamplerLoss/stageTwo', loss2, i*Total_batch + batch_i)
                    w.add_scalar( 'SamplerLoss/exposedUser', exposed_users, i*Total_batch + batch_i)
                    w.add_scalar( 'SamplerLoss/exposedItems', exposed_items, i*Total_batch + batch_i)
            # print((loss1, loss2))
finally:
    if world.tensorboard:
        w.close()
