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
          
def uniform_train(dataset, loader,recommend_model, loss_class, Neg_k, epoch, w=None):
    # batch_data = user_batch
    Recmodel = recommend_model
    bce : utils.BCE = loss_class
    for batch_i, batch_data in enumerate(loader):
        
        users = batch_data.numpy() # (batch_size, 1)
        # 1.
        # sample items
        # start = time()
        S = utils.UniformSample(users, dataset, k=Neg_k)
        # end  = time()
        
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
        if world.tensorboard:
            w.add_scalar(f'UniformLoss/BCE', loss1, epoch*world.config['total_batch'] + batch_i)
    return "uniform"


# users_set = set()
# items_set = set()

def sampler_train(dataset, sampler, recommend_model, var_model_reg, loss_class, epoch_k, epoch,w):
    global users_set, items_set
    sampler : utils.Sample_MF
    dataset : dataloader.BasicDataset
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


def Test(testDict, Recmodel, top_k, epoch, w=None):
    testDict : dict
    Recmodel : model.RecMF
    with torch.no_grad():
        Recmodel.eval()
        users = torch.Tensor(list(testDict.keys()))
        GroundTrue = [testDict[user] for user in users.numpy()]
        rating = Recmodel.getUsersRating(users)
        _, top_items = torch.topk(rating, top_k)
        top_items = top_items.cpu().numpy()
        metrics = utils.recall_precisionATk(GroundTrue, top_items, top_k)
        metrics['mrr'] = utils.MRRatK(GroundTrue, top_items, top_k)
        metrics['ndcg'] = utils.NDCGatK(GroundTrue, top_items, top_k)
        # pprint(metrics)
        if world.tensorboard:
            w.add_scalar(f'Test/Recall@{top_k}', metrics['recall'], epoch)
            w.add_scalar(f'Test/Precision@{top_k}', metrics['precision'], epoch)
            w.add_scalar(f'Test/MRR@{top_k}', metrics['mrr'], epoch)
            w.add_scalar(f'Test/NDCG@{top_k}', metrics['ndcg'], epoch)