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
import os
import time

# loading data...
dataset   = dataloader.LastFM()
lm_loader = DataLoader(dataset, batch_size=world.config['batch_size'], shuffle=True, drop_last=True) 

world.config['num_users'] = dataset.n_users
world.config['num_items'] = dataset.m_items

print('===========config================')
pprint(world.config)
print(world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print(world.sampling_type)
print('===========end===================')

# initialize models
if world.sampling_type == SamplingAlgorithms.uniform:
    Recmodel = model.RecMF(world.config)
    elbo     = utils.BCE(Recmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-uniform.pth.tar')))
        Recmodel.train()

elif world.sampling_type == SamplingAlgorithms.sampler:
    Recmodel = model.RecMF(world.config)
    # Varmodel = model.VarMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    # register ELBO loss
    elbo = utils.ELBO(world.config, 
                    rec_model=Recmodel, 
                    var_model=Varmodel)
    sampler = utils.Sample_MF(k=1, var_model=Varmodel) 
    if word.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-sampler.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-sampler.pth.tar')))

elif world.sampling_type == SamplingAlgorithms.bpr:
    Recmodel = model.RecMF(world.config)
    bpr = utils.BPRLoss(Recmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-bpr.pth.tar')))

elif world.sampling_type == SamplingAlgorithms.alldata:
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel, 
                      var_model=Varmodel)

elif world.sampling_type == SamplingAlgorithms.GMF:
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler = utils.sample_for_basic_GMF_loss(k=1.5)

elif world.sampling_type == SamplingAlgorithms.Mixture:
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler_GMF = utils.sample_for_basic_GMF_loss(k=1.5)
    sampler_fast = utils.Sample_MF(k=1, var_model=Varmodel) # k doesn't matter
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-Mixture.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-Mixture.pth.tar')))
elif world.sampling_type == SamplingAlgorithms.light_gcn:
    print('sampling_LGN')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN(world.config, dataset)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler = utils.Sample_MF(k=1, var_model=Varmodel)

elif world.sampling_type == SamplingAlgorithms.light_gcn_mixture:
    print('sampling_LGN_mixture')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN(world.config, dataset)
    elbo = utils.ELBO(world.config,
                      rec_model = Recmodel,var_model=Varmodel)
    sampler1 = utils.sample_for_basic_GMF_loss(k=3)
    sampler2 = utils.Sample_MF(k=1, var_model=Varmodel)

# train
Neg_k = 3
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
            # bar.set_description('[training]')
            output_information = TrainProcedure.uniform_train(dataset, lm_loader, Recmodel, elbo, Neg_k, i, w)
        elif world.sampling_type == SamplingAlgorithms.sampler:
            epoch_k = dataset.n_users*5
            output_information = TrainProcedure.sampler_train_no_batch(dataset, sampler, Recmodel, Varmodel, elbo, epoch_k, i, w)
        elif world.sampling_type == SamplingAlgorithms.bpr:
            output_information = TrainProcedure.BPR_train(dataset, lm_loader, Recmodel, bpr, i, w)
        elif world.sampling_type == SamplingAlgorithms.alldata:
            output_information = TrainProcedure.Alldata_train(dataset, Recmodel, elbo, i, w=w)
        elif world.sampling_type == SamplingAlgorithms.GMF:
            output_information = TrainProcedure.sampler_train_GMF(dataset, sampler, Recmodel, Varmodel, elbo, i, w)
        elif world.sampling_type == SamplingAlgorithms.Mixture:
            output_information = TrainProcedure.sampler_train_Mixture_GMF_nobatch(dataset, sampler_GMF, sampler_fast, Recmodel, Varmodel, elbo, i, w)
        elif world.sampling_type == SamplingAlgorithms.light_gcn:
            epoch_k = dataset.trainDataSize*5
            output_information = TrainProcedure.sampler_train_no_batch_LGN(dataset, sampler, Recmodel, Varmodel, elbo, epoch_k, i, w)
            print('over train and follow bar')
        elif world.sampling_type == SamplingAlgorithms.light_gcn_mixture:
            epoch_k = 166972
            output_information = TrainProcedure.sampler_train_no_batch_LGN_mixture(dataset, sampler1, sampler2, Recmodel, Varmodel, elbo, epoch_k, i, w)
        
        bar.set_description(output_information)
        torch.save(Recmodel.state_dict(), f"../checkpoints/Rec-{world.sampling_type.name}.pth.tar")
        if globals().get('Varmodel'):
            torch.save(Varmodel.state_dict(), f"../checkpoints/Var-{world.sampling_type.name}.pth.tar")
        if i%5 == 0 and i != 0:
            # test
            bar.set_description("[TEST]")
            testDict = dataset.getTestDict()
            TrainProcedure.Test(dataset, Recmodel, world.top_k, i, w)
finally:
    if world.tensorboard:
        w.close()
