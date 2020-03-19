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

def all_data_MF_MF(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    print('begin all_data_MF_MF!')
    Recmodel: model.RecMF = recommend_model
    Varmodel: model.VarMF_reg = var_model
    loss_class: utils.ELBO


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
        batch_gamma = Varmodel(batch_users, batch_items)
        loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij)

        if world.tensorboard:
            w.add_scalar(f'all_data_MF_MF/stageOne', loss1, epoch*round(datalen/world.config['batch_size']) + batch_i)
            w.add_scalar(f'all_data_MF_MF/stageTwo', loss2, epoch*round(datalen/world.config['batch_size']) + batch_i)

    print('end all_data_MF_MF!')

    return f"[ALL[{datalen}]]"

def all_data_LGN_MF(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    print('begin all_data_LGN_MF!')
    Recmodel: model.RecMF = recommend_model
    Varmodel: model.LightGCN = var_model
    loss_class: utils.ELBO


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
        batch_gamma, reg_loss = Varmodel.forwardWithReg(batch_users, batch_items)
        # batch_gamma = Varmodel(batch_users, batch_items)
        loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij, reg_loss=reg_loss)

        if world.tensorboard:
            w.add_scalar(f'all_data_LGN_MF/stageOne', loss1, epoch*round(datalen/world.config['batch_size']) + batch_i)
            w.add_scalar(f'all_data_LGN_MF/stageTwo', loss2, epoch*round(datalen/world.config['batch_size']) + batch_i)

    print('end all_data_LGN_MF!')

    return f"[ALL[{datalen}]]"


def all_data_MFxij_MF_nobatch(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    Recmodel: model.RecMF = recommend_model
    Varmodel: model.VarMF_xij = var_model
    loss_class: utils.ELBO


    (epoch_users, epoch_items, epoch_xij) = utils.getAllData(dataset)
    epoch_users, epoch_items, epoch_xij = utils.shuffle(epoch_users, epoch_items, epoch_xij)
    datalen = len(epoch_users)
    Recmodel.train()
    Varmodel.eval()
    rating = Recmodel(epoch_users, epoch_items )
    gamma = Varmodel(epoch_users, epoch_items, epoch_xij)
    loss1 = loss_class.stageOne(rating, epoch_xij, gamma)
    
    Recmodel.eval()
    Varmodel.train()
    rating = Recmodel(epoch_users, epoch_items )
    gamma = Varmodel(epoch_users, epoch_items, epoch_xij)
    loss2 = loss_class.stageTwo(rating, gamma, epoch_xij)
    if world.tensorboard:
        w.add_scalar(f'all_data_MFxij_MF/stageOne', loss1, epoch)
        w.add_scalar(f'all_data_MFxij_MF/stageTwo', loss2, epoch)
    print("[loss:]", loss1 + loss2)

    

def all_data_MFxij_MF(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    print('begin all_data_MFxij_MF!')
    Recmodel: model.RecMF = recommend_model
    Varmodel: model.VarMF_xij = var_model
    loss_class: utils.ELBO


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
        batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij)

        if world.tensorboard:
            w.add_scalar(f'all_data_MFxij_MF/stageOne', loss1, epoch*round(datalen/world.config['batch_size']) + batch_i)
            w.add_scalar(f'all_data_MFxij_MF/stageTwo', loss2, epoch*round(datalen/world.config['batch_size']) + batch_i)

    print('end all_data_MFxij_MF!')

    return f"[ALL[{datalen}]]"

def all_data_LGNxij_MF(dataset, recommend_model, var_model, loss_class, epoch, w=None):
    print('begin all_data_LGNxij_MF!')
    Recmodel: model.RecMF = recommend_model
    Varmodel: model.LightGCN_xij = var_model
    loss_class: utils.ELBO


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
        batch_gamma, reg_loss = Varmodel.forwardWithReg(batch_users, batch_items, batch_xij)
        # batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss2 = loss_class.stageTwo(rating, batch_gamma, batch_xij, reg_loss=reg_loss)

        if world.tensorboard:
            w.add_scalar(f'all_data_LGNxij_MF/stageOne', loss1, epoch*round(datalen/world.config['batch_size']) + batch_i)
            w.add_scalar(f'all_data_LGNxij_MF/stageTwo', loss2, epoch*round(datalen/world.config['batch_size']) + batch_i)

    print('end all_data_LGNxij_MF!')

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

def Test(dataset, Recmodel, top_k, epoch, w=None):
     dataset : utils.BasicDataset
     testDict : dict = dataset.getTestDict()
     Recmodel : model.RecMF
     with torch.no_grad():
         Recmodel.eval()
         users = list(testDict.keys())
         users_tensor = torch.Tensor(list(testDict.keys())).to(world.device)
         GroundTrue = [testDict[user] for user in users_tensor.cpu().numpy()]
         rating = Recmodel.getUsersRating(users_tensor)
         rating = rating.cpu()
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

"""def test_one_batch(X):
    sorted_items = X[0].numpy()
    groundTrue = X[1]
    r = utils.getLabel(groundTrue, sorted_items)
    print(r)
    pre, recall, ndcg = [], [], []
    for k in world.topks:
        # ret = utils.recall_precisionATk(groundTrue, sorted_items, k)
        ret = utils.RecallPrecision_ATk(groundTrue, r, k)
        pre.append(ret['precision'])
        recall.append(ret['recall'])
        # ndcg.append(utils.NDCGatK(groundTrue, sorted_items, k))
        ndcg.append(utils.NDCGatK_r(r, k))
    return {'recall':np.array(recall), 
            'precision':np.array(pre), 
            'ndcg':np.array(ndcg)}


def Test(dataset, Recmodel, top_k, epoch, w=None):
    u_batch_size = world.config['test_u_batch_size']
    dataset : utils.BasicDataset
    testDict : dict = dataset.getTestDict()
    Recmodel : model.LightGCN
    # eval mode with no dropout
    Recmodel = Recmodel.eval()
    max_K = max(world.topks)
    if world.multi_cores:
        pool = multiprocessing.Pool(CORES)
    results = {'precision': np.zeros(len(world.topks)), 
              'recall': np.zeros(len(world.topks)), 
              'ndcg': np.zeros(len(world.topks))}
    with torch.no_grad():
        users = list(testDict.keys())
        try:
            assert u_batch_size <= len(users)/10
        except AssertionError:
            print(f"test_u_batch_size is too big for this dataset, try a small one {len(users)//10}")
        users_list = []
        rating_list = []
        groundTrue_list = []
        # ratings = []
        total_batch = len(users)//u_batch_size + 1
        for batch_users in utils.minibatch(users, batch_size=u_batch_size):
            allPos = dataset.getUserPosItems(batch_users)
            groundTrue = [testDict[u] for u in batch_users]
            batch_users_gpu = torch.Tensor(batch_users).long()
            batch_users_gpu = batch_users_gpu.to(world.device)

            rating = Recmodel.getUsersRating(batch_users_gpu)
            rating = rating.cpu()
            exclude_index = []
            exclude_items = []
            for range_i, items in enumerate(allPos):
                exclude_index.extend([range_i]*len(items))
                exclude_items.extend(items)
            rating[exclude_index, exclude_items] = -1e5
            _, rating_K = torch.topk(rating, k=max_K)
            del rating      
            users_list.append(batch_users)
            rating_list.append(rating_K) 
            groundTrue_list.append(groundTrue)
        assert total_batch == len(users_list)
        X = zip(rating_list, groundTrue_list)
        if world.multi_cores:
            pre_results = pool.map(test_one_batch, X)
        else:
            pre_results = []
            for data in X:
                result = test_one_batch(data)
                pre_results.append(result)
        for result in pre_results:
            results['recall'] += result['recall'] / total_batch
            results['precision'] += result['precision'] / total_batch
            results['ndcg'] += result['ndcg'] / total_batch
        if world.tensorboard:
            w.add_scalars(f'Test/Recall@{world.topks}', 
                         {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/Precision@{world.topks}',
                         {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
            w.add_scalars(f'Test/NDCG@{world.topks}',
                         {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
        pool.close()
        return results"""




        
    
            
            
         
            

            
