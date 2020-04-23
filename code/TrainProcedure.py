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


def all_data_xij_no_batch(dataset, recommend_model, var_model, loss_class, epoch, w=None, **args):
    flag = f"all_data_xij_nobatch_[{world.rec_type}-{world.var_type}]"
    print(flag)
    Recmodel = recommend_model
    Varmodel = var_model
    loss_class: utils.ELBO
    lgn = world.var_type.startswith('lgn')

    (epoch_users, epoch_items, epoch_xij) = utils.getAllData(dataset)
    datalen = len(epoch_users)

    epoch_users = epoch_users.to(world.device)
    epoch_items = epoch_items.to(world.device)
    epoch_xij = epoch_xij.to(world.device)

  
    rating = Recmodel(epoch_users, epoch_items)
    epoch_gamma, reg_loss = Varmodel.forwardWithReg(epoch_users, epoch_items, epoch_xij)
    gam1=epoch_gamma.data
    loss1 = loss_class.stageOne(rating, epoch_xij, gam1)

    rating = Recmodel(epoch_users, epoch_items).data
    if lgn:
        print('lgn reg')
        # batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss2 = loss_class.stageTwo_Prior(rating, epoch_gamma, epoch_xij, reg_loss=reg_loss)
    else:
        print('mf reg')
        loss2 = loss_class.stageTwo_Prior(rating, epoch_gamma, epoch_xij, reg_loss=reg_loss)
    
    if epoch % 100 == 0 :
        np.savetxt(f'/output/lgn_xij_gamma{epoch}.txt', np.array(epoch_gamma.cpu().detach().numpy()))
        #np.savetxt(f'/output/lgn_xij_rating{epoch}.txt', np.array(rating.cpu().detach().numpy()))
        np.savetxt(f'/output/lgn_xij_x{epoch}.txt', np.array(epoch_xij.cpu().detach().numpy()))
        user_emb, item_emb0, item_emb1 = Varmodel.get_user_item_embedding()
        np.savetxt(f'/output/lgn_xij_user_emb{epoch}.txt', np.array(user_emb.cpu().detach().numpy()))
        np.savetxt(f'/output/lgn_xij_item_emb0_{epoch}.txt', np.array(item_emb0.cpu().detach().numpy()))
        np.savetxt(f'/output/lgn_xij_item_emb1_{epoch}.txt', np.array(item_emb1.cpu().detach().numpy()))

    
    if world.tensorboard:
        w.add_scalar(flag + '/stageOne', loss1, epoch)
        w.add_scalar(flag + '/stageTwo', loss2, epoch)
    print('end')
    return f"[ALL[{datalen}]]"

def sample_xij_no_batch(dataset, recommend_model, var_model, loss_class, epoch, w=None, **args):
    flag = f"sample_xij_nobatch_[{world.rec_type}-{world.var_type}]"
    print(flag)
    sampler = args['sampler']
    Recmodel = recommend_model
    Varmodel = var_model
    loss_class: utils.ELBO
    lgn = world.var_type.startswith('lgn')

    epoch_k = dataset.trainDataSize*5
    (epoch_users, epoch_items, epoch_xij, G) = sampler.sample(epoch_k)
    datalen = len(epoch_users)

    epoch_users = epoch_users.to(world.device)
    epoch_items = epoch_items.to(world.device)
    epoch_xij = epoch_xij.to(world.device)

    rating = Recmodel(epoch_users, epoch_items)
    epoch_gamma, reg_loss = Varmodel.forwardWithReg(epoch_users, epoch_items, epoch_xij, G, datalen)
    gam1=epoch_gamma.data
    loss1 = loss_class.stageOne(rating, epoch_xij, gam1, gam1)

    rating = Recmodel(epoch_users, epoch_items).data
    if lgn:
        print('lgn reg')
        # batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss2 = loss_class.stageTwoPrior(rating, epoch_gamma, epoch_xij, epoch_gamma, reg_loss=reg_loss)
    else:
        print('mf reg')
        loss2 = loss_class.stageTwoPrior(rating, epoch_gamma, epoch_xij, epoch_gamma, reg_loss=reg_loss)

    

    if world.tensorboard:
        w.add_scalar(flag + '/stageOne', loss1, epoch)
        w.add_scalar(flag + '/stageTwo', loss2, epoch)
    print('end')
    return f"[ALL[{datalen}]]"



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
