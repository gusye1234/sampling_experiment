import world
import numpy as np
import torch
import utils


          
def uniform_train(user_batch, dataset, recommend_model, var_model, loss_class, Neg_k):
    batch_data = user_batch
    Recmodel = recommend_model
    Varmodel = var_model
    elbo  = loss_class
    
    users = batch_data.numpy() # (batch_size, 1)
    # 1.
    # sample items
    
    S = utils.UniformSample(users, dataset, k=Neg_k)
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
    rating = Recmodel(users, items)
    loss1  = elbo.stageOne(rating, xij)
    
    rating = Recmodel(users, items)
    gamma  = Varmodel(users, items)
    # print(rating)
    # print(gamma)
    loss2  = elbo.stageTwo(rating, gamma, xij)

    return loss1, loss2