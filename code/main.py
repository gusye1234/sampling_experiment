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
from sample import SamplePersonal


utils.set_seed(world.seed)

#########################################
# loading data...
# dataset   = dataloader.Ciao()
dataset   = dataloader.LastFM()

world.config['num_users'] = dataset.n_users
world.config['num_items'] = dataset.m_items

#########################################
#########################################
# print out
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
#########################################
#########################################
# initialize models
Recmodel = register.Rec_register[world.rec_type](world.config)
Varmodel = register.Var_register[world.var_type](world.config)
train_method = register.sampling_register[world.sample_type]

if world.sample_type == 'fast_sampling':
    sampler = SamplePersonal(Varmodel, dataset)
else:
    sampler = None
elbo = utils.ELBO(world.config,
                  rec_model=Recmodel,
                  var_model=Varmodel)
#########################################
#########################################
# load
flag = f"{world.sample_type}_{world.rec_type}_{world.var_type}"
if world.LOAD:
        Recmodel.load_state_dict(torch.load(os.path.join(world.PATH, f'Rec_{flag}.pth.tar')))
        Varmodel.load_state_dict(torch.load(os.path.join(world.PATH, f'Var_{flag}.pth.tar')))
########################################$
#########################################
# train


Recmodel = Recmodel.to(world.device)
if globals().get('Varmodel'):
    Varmodel = Varmodel.to(world.device)


if world.tensorboard:
    w : SummaryWriter = SummaryWriter("./outputs/"+ "runs/"+time.strftime("%m-%d-%Hh%Mm%Ss-") + "-" + world.comment)
else:
    w = None
    
    
try:
    for i in range(world.TRAIN_epochs):
        start = time.time()
        print(f'===============[EPOCH {i}]=================')
        # training
        output_information = train_method(dataset, Recmodel, Varmodel, elbo, i, w=w, sampler=sampler)
        print(output_information)
        
        # save
        #torch.save(Recmodel.state_dict(), os.path.join(world.PATH, f'Rec_{flag}.pth.tar'))
        #torch.save(Varmodel.state_dict(), os.path.join(world.PATH, f'Var_{flag}.pth.tar'))
        # test
        if i%100 == 0 :
            print('[TEST]')
            testDict = dataset.getTestDict()
            TrainProcedure.Test(dataset, Recmodel, Varmodel, world.top_k, i, w)
        print("total time:", time.time() - start)

    with torch.no_grad():
        print("save")
        user_emb, item_emb0, item_emb1 = Varmodel.get_user_item_embedding()
        np.savetxt('lgn_xij_user_emb.txt', np.array(user_emb.cpu().detach().numpy()))
        np.savetxt('lgn_xij_item_emb0_.txt', np.array(item_emb0.cpu().detach().numpy()))
        np.savetxt('lgn_xij_item_emb1.txt', np.array(item_emb1.cpu().detach().numpy()))
finally:
    if world.tensorboard:
        w.close()
