import world
import torch
from torch.utils.data import DataLoader
import model
import utils
import dataloader
import numpy as np
from tensorboardX import SummaryWriter
from tqdm import tqdm
from pprint import pprint
import time

# loading data...
last_lm   = dataloader.LastFM()
lm_loader = DataLoader(last_lm, batch_size=world.config['batch_size'], shuffle=False, drop_last=True) 

world.config['num_users'] = last_lm.n_users
world.config['num_items'] = last_lm.m_items

print('===========config================')
pprint(world.config)
print(world.comment)
print("tensorboard:", world.tensorboard)
print('===========end===================')
# initialize models
Recmodel = model.RecMF(world.config)
Varmodel = model.VarMF(world.config)

# register ELBO loss
elbo = utils.ELBO(world.config, 
                  rec_model=Recmodel, 
                  var_model=Varmodel)


# train
Neg_k = 1
Total_batch = int(len(last_lm)/world.config['batch_size'])
if world.tensorboard:
    w : SummaryWriter = SummaryWriter("./output/"+ "runs/"+time.strftime("%m-%d-%H:%M:%S-") + "-" + world.comment)
try:
    for i in range(world.TRAIN_epochs):
        for batch_i, batch_data in tqdm(enumerate(lm_loader)):
            users = batch_data.numpy() # (batch_size, 1)
            # 1.
            # sample items
            
            S = utils.UniformSample(users, last_lm, k=Neg_k)
            # print(S)
            
            users = np.tile(users.reshape((-1,1)), (1, 1+Neg_k)).reshape(-1)
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
            # print(rating)
            # print(gamma)
            loss2  = elbo.stageTwo(rating, gamma, xij)
            
            if world.tensorboard:
                w.add_scalar(f'Loss/stageOne', loss1, i*Total_batch + batch_i)
                w.add_scalar(f'Loss/stageTwo', loss2, i*Total_batch + batch_i)
            # print((loss1, loss2))
finally:
    if world.tensorboard:
        w.close()
            


