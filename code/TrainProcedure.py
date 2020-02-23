import world
import numpy as np
import torch
import utils
import dataloader
from time import time
          
def uniform_train(user_batch, dataset, recommend_model, loss_class, Neg_k):
    batch_data = user_batch
    Recmodel = recommend_model
    bce : utils.BCE = loss_class
    
    users = batch_data.numpy() # (batch_size, 1)
    # 1.
    # sample items
    # start = time()
    S = utils.UniformSample(users, dataset, k=Neg_k)
    # end  = time()
    # print("sample time:", end-start)
    # print(S)
    
    users = np.tile(users.reshape((-1,1)), (1, 1+Neg_k)).reshape(-1)
    S     = S.reshape(-1)
    # 2.
    # process xij
    xij   = dataset.getUserItemFeedback(users, S)
    
    users = torch.Tensor(users).long()
    items = torch.Tensor(S).long()
    xij   = torch.Tensor(xij)
    # 3.
    # optimize loss
    # start = time()
    rating = Recmodel(users, items)
    loss1  = bce.stageOne(rating, xij)
    # end   = time()
    # print(f"{world.sampling_type} opt time:", end-start)
    # rating = Recmodel(users, items)
    # gamma  = Varmodel(users, items)
    # print(rating)
    # print(gamma)
    # loss2  = elbo.stageTwo(rating, gamma, xij)

    return loss1



users_set = set()
items_set = set()

def sampler_train(dataset, sampler, recommend_model, var_model_reg, loss_class, batch_k):
    global users_set, items_set
    sampler : utils.Sample_MF
    dataset : dataloader.BasicDataset
    loss_class : utils.ELBO
    # 1.
    # sampling
    # start = time()
    sampler.compute()
    sampler.setK(batch_k)
    users, items = sampler.sample()
    users_set = users_set.union(list(users.numpy()))
    items_set = items_set.union(list(items.numpy()))
    
    # end = time()
    # print("sample time:", end-start)
    # 2.
    # process xij
    xij = dataset.getUserItemFeedback(users.cpu().numpy().astype('int'), items.cpu().numpy().astype('int'))
    xij = torch.Tensor(xij).cuda() if world.GPU else torch.Tensor(xij)
    
    users = users.long()
    items = items.long()
    # 3.
    # optimize loss
    # start = time()
    rating = recommend_model(users, items)
    loss1  = loss_class.stageOne(rating, xij)
    
    rating = recommend_model(users, items)
    gamma  = var_model_reg(users, items)
    
    loss2  = loss_class.stageTwo(rating, gamma, xij)
    # end = time()
    # print(f"{world.sampling_type } opt time", end-start)
    
    return loss1, loss2, len(users_set), len(items_set)