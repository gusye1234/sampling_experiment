import world
from world import SamplingAlgorithms
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
print(world.sampling_type)
print('===========end===================')
# initialize models

if world.sampling_type == SamplingAlgorithms.uniform:
    Recmodel = model.RecMF(world.config)
    elbo     = utils.BCE(Recmodel)
elif world.sampling_type == SamplingAlgorithms.sampler:
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
world.config['total_batch'] = int(len(dataset)/world.config['batch_size'])


if world.tensorboard:
    w : SummaryWriter = SummaryWriter("./output/"+ "runs/"+time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
try:
    bar = tqdm(range(world.TRAIN_epochs))
    for i in bar:
        # for batch_i, batch_data in tqdm(enumerate(lm_loader)):
        if world.sampling_type == SamplingAlgorithms.uniform:
            bar.set_description('[training]')
            output_information = TrainProcedure.uniform_train(dataset, lm_loader, Recmodel, elbo, Neg_k, i, w)
        elif world.sampling_type == SamplingAlgorithms.sampler:
            epoch_k = dataset.n_users*5
            # epoch_k = dataset.trainDataSize*5
            bar.set_description(f"[Sample {epoch_k}]")
            output_information = TrainProcedure.sampler_train_no_batch(dataset, sampler, Recmodel, Varmodel, elbo, epoch_k, i, w)
        bar.set_description('[SAVE]' + output_information)
        torch.save(Recmodel.state_dict(), f"../checkpoints/Rec-{world.sampling_type.name}.pth.tar")
        if globals().get('Varmodel'):
            torch.save(Varmodel.state_dict(), f"../checkpoints/Var-{world.sampling_type.name}.pth.tar")
        if i%5 == 0:
            # test
            bar.set_description("[TEST]")
            testDict = dataset.getTestDict()
            TrainProcedure.Test(dataset, Recmodel, world.top_k, i, w)
finally:
    if world.tensorboard:
        w.close()
