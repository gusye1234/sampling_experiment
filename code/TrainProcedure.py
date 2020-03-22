"""
Design training process
"""
import world
import numpy as np
import torch
import utils
import dataloader
from pprint import pprint
from time import time
from tqdm import tqdm
import model
import multiprocessing
          
CORES = world.CORES


          
def all_data(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    flag = f"all_data_[{world.rec_type}-{world.var_type}]"
    print(flag)
    Recmodel = recommend_model
    Varmodel = var_model
    loss_class: utils.ELBO
    lgn = world.var_type.startswith('lgn')
    
    (epoch_users, epoch_items, epoch_xij) = utils.getAllData(dataset)
    epoch_users, epoch_items, epoch_xij = utils.shuffle(epoch_users, epoch_items, epoch_xij)
    datalen = len(epoch_users)
    
    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):
        batch_users = batch_users.to(world.device)
        batch_items = batch_items.to(world.device)
        batch_xij = batch_xij.to(world.device)
        if batch_i == 0:
            print(len(batch_users))
        Recmodel.train()
        Varmodel.eval()
        rating = Recmodel(batch_users, batch_items)
        batch_gamma = Varmodel(batch_users, batch_items)
        loss1 = loss_class.stageOne(rating, batch_xij, batch_gamma)

        Recmodel.eval()
        Varmodel.train()
        rating = Recmodel(batch_users, batch_items)
        if lgn:
            batch_gamma, reg_loss = Varmodel.forwardWithReg(batch_users, batch_items)
            loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij, reg_loss=reg_loss)
        else:
            batch_gamma = Varmodel(batch_users, batch_items)
            loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij)

        if world.tensorboard:
            w.add_scalar(flag+"/stageone", loss1, epoch*round(datalen/world.config['batch_size']) + batch_i)
            w.add_scalar(flag+"/stagetwo", loss2, epoch*round(datalen/world.config['batch_size']) + batch_i)

    print('end')

    return f"[ALL[{datalen}]]"
       
def all_data_xij(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    flag = f"all_data_xij_[{world.rec_type}-{world.var_type}]"
    print(flag)
    Recmodel = recommend_model
    Varmodel = var_model
    loss_class: utils.ELBO
    lgn = world.var_type.startswith('lgn')
    
    (epoch_users, epoch_items, epoch_xij) = utils.getAllData(dataset)
    epoch_users, epoch_items, epoch_xij = utils.shuffle(epoch_users, epoch_items, epoch_xij)
    datalen = len(epoch_users)
    
    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):
        batch_users = batch_users.to(world.device)
        batch_items = batch_items.to(world.device)
        batch_xij = batch_xij.to(world.device)
        if batch_i == 0:
            print(len(batch_users))
        Recmodel.train()
        Varmodel.eval()
        rating = Recmodel(batch_users, batch_items)
        batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss1 = loss_class.stageOne(rating, batch_xij, batch_gamma)

        Recmodel.eval()
        Varmodel.train()
        rating = Recmodel(batch_users, batch_items)
        
        if lgn:
            batch_gamma, reg_loss = Varmodel.forwardWithReg(batch_users, batch_items, batch_xij)
            loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij, reg_loss=reg_loss)
        else:
            batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
            loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij)

        if world.tensorboard:
            w.add_scalar(flag+'/stageone', loss1, epoch*round(datalen/world.config['batch_size']) + batch_i)
            w.add_scalar(flag+'/stagetwo', loss2, epoch*round(datalen/world.config['batch_size']) + batch_i)
    print('end')
    return f"[ALL[{datalen}]]"

def all_data_xij_no_batch(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    flag = f"all_data_xij_nobatch_[{world.rec_type}-{world.var_type}]"
    print(flag)
    Recmodel = recommend_model
    Varmodel = var_model
    loss_class: utils.ELBO
    lgn = world.var_type.startswith('lgn')

    (epoch_users, epoch_items, epoch_xij) = utils.getAllData(dataset)
    epoch_users, epoch_items, epoch_xij = utils.shuffle(epoch_users, epoch_items, epoch_xij)
    datalen = len(epoch_users)

    epoch_users = epoch_users.to(world.device)
    epoch_items = epoch_items.to(world.device)
    epoch_xij = epoch_xij.to(world.device)

    Recmodel.train()
    Varmodel.eval()
    rating = Recmodel(epoch_users, epoch_items)
    epoch_gamma = Varmodel(epoch_users, epoch_items, epoch_xij)
    loss1 = loss_class.stageOne(rating, epoch_xij, epoch_gamma)

    Recmodel.eval()
    Varmodel.train()
    rating = Recmodel(epoch_users, epoch_items)
    if lgn:
        epoch_gamma, reg_loss = Varmodel.forwardWithReg(epoch_users, epoch_items, epoch_xij)
        # batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss2 = loss_class.stageTwo(rating, epoch_gamma, epoch_xij, reg_loss=reg_loss)
    else:
        epoch_gamma = Varmodel.forwardWithReg(epoch_users, epoch_items, epoch_xij)
        loss2 = loss_class.stageTwo(rating, epoch_gamma, epoch_xij)
    
    if epoch % 50 == 0:
        np.savetxt(f'/output/lgn_xij_gamma{epoch}.txt', np.array(epoch_gamma.cpu().detach().numpy()))
        np.savetxt(f'/output/lgn_xij_rating{epoch}.txt', np.array(rating.cpu().detach().numpy()))
        np.savetxt(f'/output/lgn_xij_x{epoch}.txt', np.array(epoch_xij.cpu().detach().numpy()))
    
    if world.tensorboard:
        w.add_scalar(flag + '/stageOne', loss1, epoch)
        w.add_scalar(flag + '/stageTwo', loss2, epoch)
    print('end')
    return f"[ALL[{datalen}]]"
          
def Alldata_train_set_gamma_cross_entrophy(dataset, recommend_model, loss_class, epoch, w=None):
    print('begin Alldata_train_set_gamma_cross_entrophy!')
    Recmodel : model.RecMF = recommend_model
    loss_class : utils.ELBO
    Recmodel.train()
    gamma = torch.ones((dataset.n_users, dataset.m_items))*0.5
    (epoch_users,
     epoch_items,
     epoch_xij,
     epoch_gamma) = utils.getAllData(dataset, gamma)
    epoch_users, epoch_items, epoch_xij = utils.shuffle(epoch_users, epoch_items, epoch_xij)
    datalen = len(epoch_users)
    rating = Recmodel(epoch_users, epoch_items)
    print(epoch_users[:1000], epoch_items[:1000], epoch_xij[:1000], rating[:1000])
    loss1 = loss_class.stageOne(rating, epoch_xij, epoch_gamma)

    # for (batch_i, (batch_users, batch_items, batch_xij, batch_gamma)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij, epoch_gamma)):
    # if epoch == 0:
    # print(len(batch_users))

    # rating = Recmodel(batch_users, batch_items)
    # loss1 = loss_class.stageOne(rating, batch_xij, batch_gamma)

    # if batch_i == 99:
        #print(batch_users[:1000], batch_items[:1000], batch_xij[:1000], rating[:1000], batch_gamma[:1000])

    # if world.tensorboard:
        # w.add_scalar(f'Alldata_train_set_gamma_cross_entrophy/stageOne', loss1, epoch*world.config['total_batch'] + batch_i)
    if world.tensorboard:
        w.add_scalar("Alldata_train_set_gamma_cross_entrophy/stageOne", loss1, epoch)

    print('end Alldata_train_set_gamma_cross_entrophy!')
    return f"[ALL[{datalen}]]"

def sampler_train(dataset, sampler, recommend_model, var_model_reg, loss_class, epoch_k, epoch,w):
    # global users_set, items_set
    sampler : utils.Sample_MF
    dataset : dataloader.BasicDataset
    recommend_model.train()
    var_model_reg.train()
    loss_class : utils.ELBO
    # 1.
    # sampling
    # start = time()
    sampler.compute()
    epoch_users, epoch_items = sampler.sampleForEpoch(epoch_k) # epoch_k may be 5*n

    epoch_users, epoch_items = utils.shuffle(epoch_users, epoch_items)
    epoch_xij = dataset.getUserItemFeedback(epoch_users.cpu().numpy(),
                                            epoch_items.cpu().numpy()).astype('int')
    # print(f"[{epoch}]Positive Label Sparsity",np.sum(epoch_xij)/len(epoch_xij))
    # print(epoch_users[:5], epoch_items[:5], epoch_xij[:5])
    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):
        batch_users = batch_users.to(world.device)
        batch_items = batch_items.to(world.device)
        batch_xij = batch_xij.to(world.device)
        users = batch_users.long()
        # print(users.size())
        items = batch_items.long()
        xij   = torch.Tensor(batch_xij)
        
        rating = recommend_model(users, items)
        loss1  = loss_class.stageOne(rating, xij)

        rating = recommend_model(users, items)
        gamma  = var_model_reg(users, items)

        loss2  = loss_class.stageTwo(rating, gamma, xij)
        # end = time()
        # print(f"{world.sampling_type } opt time", end-start)
        if world.tensorboard:
            w.add_scalar(f'SamplerLoss/stageOne', loss1, epoch*world.config['total_batch'] + batch_i)
            w.add_scalar(f'SamplerLoss/stageTwo', loss2, epoch*world.config['total_batch'] + batch_i)
    return f"Sparsity {np.sum(epoch_xij)/len(epoch_xij):.3f}"

def Test(dataset, Recmodel, Varmodel, top_k, epoch, w=None):
     dataset : utils.BasicDataset
     testDict : dict = dataset.getTestDict()
     Recmodel : model.RecMF
     Varmodel : model.LightGCN
     with torch.no_grad():
         Recmodel.eval()
         Varmodel.eval()
         users = list(testDict.keys())
         users_tensor = torch.Tensor(list(testDict.keys())).to(world.device)
         #print(users_tensor)
         GroundTrue = [testDict[user] for user in users_tensor.cpu().numpy()]
         rating1 = Recmodel.getUsersRating(users_tensor)
         rating2 = Varmodel.allGamma(users_tensor)
         rating = rating1*rating2
         rating = rating.cpu()
         #print(rating1, rating2, rating)


         # exclude positive train data
         allPos = dataset.getUserPosItems(users)
         exclude_index = []
         exclude_items = []
         for range_i, items in enumerate(allPos):
            exclude_index.extend([range_i]*len(items))
            exclude_items.extend(items)
         rating[exclude_index, exclude_items] = 0.
         # assert torch.all(rating >= 0.)
         # assert torch.all(rating <= 1.)
         # end excluding
         _, top_items = torch.topk(rating, top_k)
         top_items = top_items.cpu().numpy()
         metrics = utils.recall_precisionATk(GroundTrue, top_items, top_k)
         metrics['mrr'] = utils.MRRatK(GroundTrue, top_items, top_k)
         metrics['ndcg'] = utils.NDCGatK(GroundTrue, top_items, top_k)
         print(metrics)
         if world.tensorboard:
             w.add_scalar(f'Test/Recall@{top_k}', metrics['recall'], epoch)
             w.add_scalar(f'Test/Precision@{top_k}', metrics['precision'], epoch)
             w.add_scalar(f'Test/MRR@{top_k}', metrics['mrr'], epoch)
             w.add_scalar(f'Test/NDCG@{top_k}', metrics['ndcg'], epoch)
