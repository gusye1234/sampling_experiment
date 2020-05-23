import world
import torch
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
from sample import SamplePersonal, sampleUniForVar
import math


utils.set_seed(world.seed)

#########################################
# loading data...
dataset   = dataloader.LastFM()

world.config['num_users'] = dataset.n_users
world.config['num_items'] = dataset.m_items
world.config['trainDataSize'] = dataset.trainDataSize*5
world.config['batch_size'] = 2000000

#########################################
#########################################
# print out
print('===========config================')
pprint(world.config)
print("cores for test:", 
    world.CORES)
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

if world.sample_type == 'fast_sampling' or world.sample_type == 'sample_gamuni_xij_no_batch':
    sampler1 = SamplePersonal(Varmodel, dataset)
    sampler2 = sampleUniForVar(dataset)

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


rdecay = world.config['rec_weight_decay']
varlr = world.config['var_lr']
vdecay = world.config['var_weight_decay']
wdecay = world.config['w_weight_decay']
rlr = world.config['rec_lr']


if world.tensorboard:
    w: SummaryWriter = SummaryWriter("../output/" + "runs/" + time.strftime("%m%d-%Hh") + str(rlr) + str(rdecay) + str(varlr) + str(vdecay) + str(wdecay) + "-" + world.comment)
else:
    w = None
    
    
try:
    for i in range(world.TRAIN_epochs):
        start = time.time()
        print(f'===============[EPOCH {i}]=================')
        # training
        output_information = train_method(dataset, Recmodel, Varmodel, elbo, i, w=w, sampler1=sampler1, sampler2=sampler2)
        print(output_information)
        
        # save
        #torch.save(Recmodel.state_dict(), os.path.join(world.PATH, f'Rec_{flag}.pth.tar'))
        #torch.save(Varmodel.state_dict(), os.path.join(world.PATH, f'Var_{flag}.pth.tar'))
        # test
        if i%50 == 0 :
            print('[TEST]')
            TrainProcedure.Test(dataset, Recmodel, Varmodel, world.top_k, i, w)
        print("total time:", time.time() - start)

    with torch.no_grad():
        print('save_gamma_xij_embedding')

        users_embedding, items_embedding0, items_embedding1 = Varmodel.get_user_item_embedding()
        np.savetxt(f'../users_embedding{rlr}{rdecay}{varlr}{vdecay}{wdecay}.txt',
                   users_embedding.cpu().detach().numpy())
        np.savetxt(f'../items_embedding0{rlr}{rdecay}{varlr}{vdecay}{wdecay}.txt',
                   items_embedding0.cpu().detach().numpy())
        np.savetxt(f'../items_embedding1{rlr}{rdecay}{varlr}{vdecay}{wdecay}.txt',
                   items_embedding1.cpu().detach().numpy())
        print('save_ok')



finally:
    if world.tensorboard:
        w.close()
