import world
import torch
from torch.utils.data import DataLoader
import model
import utils
import dataloader
import numpy as np


# loading data...
last_lm   = dataloader.LastFM()
lm_loader = DataLoader(last_lm, batch_size=world.config['batch_size'], shuffle=True, drop_last=True) 

world.config['num_users'] = last_lm.n_users
world.config['num_items'] = last_lm.m_items

# initialize models
Recmodel = model.RecMF(world.config)
Varmodel = model.VarMF(world.config)

# register ELBO loss
elbo = utils.ELBO(world.config, 
                  rec_model=Recmodel, 
                  var_model=Varmodel)


# train
Neg_k = 1
for i in range(world.TRAIN_epochs):
    for batch_i, batch_data in enumerate(lm_loader):
        users = batch_data.numpy() # (batch_size, 1)
        print(users.shape)
        # 1.
        # sample items
        
        S = utils.UniformSample(users, last_lm, k=Neg_k)
        users = np.tile(users, (1, 1+Neg_k)).reshape(-1)
        S     = S.reshape(-1)
        # 2.
        # process xij
        xij   = last_lm.getUserItemFeedback(users, S)
        
        users = torch.Tensor(users).long()
        items = torch.Tensor(S).long()
        xij   = torch.Tensor(xij)
        # 3.
        # optimize loss
        rating = Recmodel(users, items)
        loss1  = elbo.stageOne(rating, xij)
        
        rating = Recmodel(users, items)
        gamma  = Varmodel(users, items)
        print(rating)
        print(gamma)
        loss2  = elbo.stageTwo(rating, gamma, xij)
        
        print((loss1, loss2))
        
        
        
        
        
        

