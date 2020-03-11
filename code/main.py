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
print("cores for test:", world.CORES)
print(world.comment)
print("tensorboard:", world.tensorboard)
print("LOAD:", world.LOAD)
print("Weight path:", world.PATH)
print(world.sampling_type)
print('===========end===================')

# initialize models
if world.sampling_type == SamplingAlgorithms.Alldata_train_set_gamma_cross_entrophy:
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    elbo = utils.ELBO(world.config,
                    rec_model=Recmodel,
                    var_model=Varmodel)
    #varmodel is not useful
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-uniform.pth.tar')))
        Recmodel.train()

elif world.sampling_type == SamplingAlgorithms.all_data_MF_MF:
    print('all_data_MF_MF train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_reg(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel,
                      var_model=Varmodel)


elif world.sampling_type == SamplingAlgorithms.all_data_LGN_MF:
    print('all_data_LGN_MF train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN(world.config, dataset)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel,
                      var_model=Varmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-all_data_LGN_MF.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-all_data_LGN_MF.pth.tar')))

elif world.sampling_type == SamplingAlgorithms.all_data_MFxij_MF:
    print('all_data_MFxij_MF train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_xij(world.config)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel,
                      var_model=Varmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-all_data_MFxij_MF.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-all_data_MFxij_MF.pth.tar')))

elif world.sampling_type == SamplingAlgorithms.all_data_LGNxij_MF:
    print('all_data_LGNxij_MF train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN_xij(world.config, dataset)
    elbo = utils.ELBO(world.config,
                      rec_model=Recmodel,
                      var_model=Varmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-all_data_LGNxij_MF.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-all_data_LGNxij_MF.pth.tar')))

elif world.sampling_type == SamplingAlgorithms.Sample_all_dataset:
    print('Sample_all_dataset train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN_xij(world.config, dataset)
    elbo = utils.ELBO(world.config, 
                    rec_model=Recmodel, 
                    var_model=Varmodel)
    sampler = utils.Sample_MF(k=1, var_model=Varmodel) 
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-Sample_all_dataset.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-Sample_all_dataset.pth.tar')))

elif world.sampling_type == SamplingAlgorithms.Sample_positive_all:
    print('Sample_positive_all train')
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN_xij(world.config, dataset)
    elbo = utils.ELBO(world.config,
                    rec_model=Recmodel,
                    var_model=Varmodel)
    sampler = utils.Sample_positive_all(dataset, Varmodel)
    if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Rec-Sample_all_dataset.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, 'Var-Sample_all_dataset.pth.tar')))
elif world.sampling_type == SamplingAlgorithms.all_data_LGNxij2_MF:
    print(world.sampling_type.name)
    Recmodel = model.RecMF(world.config)
    Varmodel = model.LightGCN_xij2(world.config, dataset)
    elbo = utils.ELBO(world.config,
                    rec_model=Recmodel,
                    var_model=Varmodel)
elif world.sampling_type == SamplingAlgorithms.all_data_MFxij2_MF:
    print(world.sampling_type.name)
    Recmodel = model.RecMF(world.config)
    Varmodel = model.VarMF_xij2(world.config, dataset)
    elbo = utils.ELBO(world.config,
                    rec_model=Recmodel,
                    var_model=Varmodel)


Recmodel = Recmodel.to(world.device)
if globals().get('Varmodel'):
    Varmodel = Varmodel.to(world.device)
# train
Neg_k = 3
world.config['total_batch'] = int(len(dataset)/world.config['batch_size'])


if world.tensorboard:
    w : SummaryWriter = SummaryWriter("/output/"+ "runs/"+time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
try:
    bar = tqdm(range(world.TRAIN_epochs))
    for i in bar:
        # for batch_i, batch_data in tqdm(enumerate(lm_loader)):
        if world.sampling_type == SamplingAlgorithms.Alldata_train_set_gamma_cross_entrophy:
            # bar.set_description('[training]')
            output_information = TrainProcedure.Alldata_train_set_gamma_cross_entrophy(dataset, Recmodel, elbo, i, w)
        elif world.sampling_type == SamplingAlgorithms.all_data_MF_MF:
            output_information = TrainProcedure.all_data_MF_MF(dataset, Recmodel, Varmodel, elbo, i, w)
        elif world.sampling_type == SamplingAlgorithms.all_data_LGN_MF:
            output_information = TrainProcedure.all_data_LGN_MF(dataset, Recmodel, Varmodel, elbo, i, w=w)

        elif world.sampling_type == SamplingAlgorithms.all_data_MFxij_MF or world.sampling_type == SamplingAlgorithms.all_data_MFxij2_MF:
            output_information = TrainProcedure.all_data_MFxij_MF(dataset, Recmodel, Varmodel, elbo, i, w=w)

        elif world.sampling_type == SamplingAlgorithms.all_data_LGNxij_MF or world.sampling_type == SamplingAlgorithms.all_data_LGNxij2_MF:
            output_information = TrainProcedure.all_data_LGNxij_MF(dataset, Recmodel, Varmodel, elbo, i, w=w)

        elif world.sampling_type == SamplingAlgorithms.Sample_all_dataset:
            epoch_k = dataset.trainDataSize * 4
            output_information = TrainProcedure.sampler_train(dataset, sampler, Recmodel, Varmodel, elbo, epoch_k, i,w)
        elif world.sampling_type == SamplingAlgorithms.Sample_positive_all:
            epoch_k = dataset.trainDataSize * 4
            output_information = TrainProcedure.Sample_positive_all_LGN(dataset, sampler, Recmodel, Varmodel, elbo, epoch_k, i, w)


        bar.set_description(output_information)
        torch.save(Recmodel.state_dict(), f"../checkpoints/Rec-{world.sampling_type.name}.pth.tar")
        if globals().get('Varmodel'):
            torch.save(Varmodel.state_dict(), f"../checkpoints/Var-{world.sampling_type.name}.pth.tar")
        if i%3 == 0 and i != 0:
            # test
            bar.set_description("[TEST]")
            testDict = dataset.getTestDict()
            TrainProcedure.Test(dataset, Recmodel, world.top_k, i, w)
finally:
    if world.tensorboard:
        w.close()
