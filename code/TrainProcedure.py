"""
Design training process
"""
import world
import numpy as np
import torch
import utils
import model

          
#CORES = world.CORES


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
        loss2 = loss_class.stageTwoPrior(rating, epoch_gamma, epoch_xij, reg_loss=reg_loss)
    else:
        print('mf reg')
        loss2 = loss_class.stageTwoPrior(rating, epoch_gamma, epoch_xij, reg_loss=reg_loss)
    
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

    epoch_k = world.config['trainDataSize']
    (epoch_users, epoch_items, epoch_xij) = sampler.sample(epoch_k)
    datalen = len(epoch_users)

    epoch_users = epoch_users.to(world.device)
    epoch_items = epoch_items.to(world.device)
    epoch_xij = epoch_xij.to(world.device)

    rating = Recmodel(epoch_users, epoch_items)
    epoch_gamma, reg_loss = Varmodel.forwardWithReg(epoch_users, epoch_items, epoch_xij)
    gam1=epoch_gamma.data
    loss1 = loss_class.stageOne(rating, epoch_xij, gam1, pij=gam1)

    rating = Recmodel(epoch_users, epoch_items).data
    if lgn:
        print('lgn reg')
        # batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        loss2 = loss_class.stageTwoPrior(rating, epoch_gamma, epoch_xij, pij=gam1, reg_loss=reg_loss)
    else:
        print('mf reg')
        loss2 = loss_class.stageTwoPrior(rating, epoch_gamma, epoch_xij, pij=gam1, reg_loss=reg_loss)

    

    if world.tensorboard:
        w.add_scalar(flag + '/stageOne', loss1, epoch)
        w.add_scalar(flag + '/stageTwo', loss2, epoch)
    print('end')
    return f"[ALL[{datalen}]]"

def all_data_xij_grad_accu(dataset, recommend_model, var_model, loss_class, epoch, w=None, **args):
    flag = f"all_data_xij_[{world.rec_type}-{world.var_type}]"
    print(flag)
    Recmodel = recommend_model
    Varmodel = var_model
    loss_class: utils.ELBO
    lgn = world.var_type.startswith('lgn')
    epoch_loss1 = 0
    epoch_loss2 = 0
    reg_loss = Varmodel.reg_loss()
    (epoch_users, epoch_items, epoch_xij) = utils.getAllData(dataset)
    datalen = len(epoch_users)
    batch_num = (datalen // world.config['batch_size'])
    if (datalen - batch_num * world.config['batch_size']) > 0:
        batch_num += 1

    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):
        STEP = (batch_i != (batch_num - 1))
        batch_users = batch_users.to(world.device)
        batch_items = batch_items.to(world.device)
        batch_xij = batch_xij.to(world.device)

        rating = Recmodel(batch_users, batch_items)
        batch_gamma = Varmodel(batch_users, batch_items, batch_xij)
        gam1 = batch_gamma.data
        loss1 = loss_class.stageOneAcc(rating, batch_xij, gam1, wait=STEP)
        epoch_loss1 += loss1

    for (batch_i, (batch_users, batch_items, batch_xij)) in enumerate(utils.minibatch(epoch_users, epoch_items, epoch_xij)):
        STEP = (batch_i != (batch_num - 1))
        batch_users = batch_users.to(world.device)
        batch_items = batch_items.to(world.device)
        batch_xij = batch_xij.to(world.device)

        rating = Recmodel(batch_users, batch_items).data
        batch_gamma = Varmodel(batch_users, batch_items, batch_xij)


        if lgn:
            loss2 = loss_class.stageTwoAcc(rating, batch_gamma, batch_xij, reg_loss=reg_loss, wait=STEP)
        else:
            loss2 = loss_class.stageTwoAcc(rating, batch_gamma, batch_xij, reg_loss=reg_loss, wait=STEP)
        epoch_loss2 += loss2

    print('loss', epoch_loss1, epoch_loss2)

    if world.tensorboard:
        w.add_scalar(flag + '/stageone', epoch_loss1, epoch)
        w.add_scalar(flag + '/stagetwo', epoch_loss2, epoch)


def sample_gamuni_xij_no_batch(dataset, recommend_model, var_model, loss_class, epoch, w=None, **args):
    flag = f"sample_gamuni_xij_no_batch[{world.rec_type}-{world.var_type}]"
    print(flag)
    sampler1 = args['sampler1']
    sampler2 = args['sampler2']
    Recmodel = recommend_model
    Varmodel = var_model
    loss_class: utils.ELBO
    lgn = world.var_type.startswith('lgn')

    epoch_k = world.config['trainDataSize']
    (epoch_users1, epoch_items1, epoch_xij1) = sampler1.sample(epoch_k)
    datalen = len(epoch_users1)
    epoch_users1 = epoch_users1.to(world.device)
    epoch_items1 = epoch_items1.to(world.device)
    epoch_xij1 = epoch_xij1.to(world.device)
    rating1 = Recmodel(epoch_users1, epoch_items1)
    loss1 = loss_class.stageOneGam(rating1, epoch_xij1)

    (epoch_users2, epoch_items2, epoch_xij2) = sampler2.sample()
    epoch_users2 = epoch_users2.to(world.device)
    epoch_items2 = epoch_items2.to(world.device)
    epoch_xij2 = epoch_xij2.to(world.device)
    rating2 = Recmodel(epoch_users2, epoch_items2).data
    epoch_gamma, reg_loss = Varmodel.forwardWithReg(epoch_users2, epoch_items2, epoch_xij2)

    if lgn:
        print('lgn reg')
        loss2 = loss_class.stageTwoUni(rating2, epoch_gamma, epoch_xij2, reg_loss=reg_loss)
    else:
        print('mf reg')
        loss2 = loss_class.stageTwoUni(rating2, epoch_gamma, epoch_xij2, reg_loss=reg_loss)

    if world.tensorboard:
        w.add_scalar(flag + '/stageOne', loss1, epoch)
        w.add_scalar(flag + '/stageTwo', loss2, epoch)
    print('end')
    return f"[ALL[{datalen}]]"


def Test(dataset, Recmodel, Varmodel, top_k, epoch, w=None):
     dataset : utils.BasicDataset
     Recmodel : model.RecMF
     Varmodel : model.LightGCN_xij_item_personal_matrix
     with torch.no_grad():
         users = dataset.testUniqueUsers
         users_tensor = torch.LongTensor(users).to(world.device)
         GroundTrue = dataset.groundTruth
         rating1 = Recmodel.getUsersRating(users_tensor)
         rating2 = Varmodel.allGamma(users_tensor)
         rating = rating1*rating2
         rating = rating.cpu()

         # exclude positive train data
         rating[dataset.exclude_index, dataset.exclude_items] = -1.0
         _, top_items = torch.topk(rating, top_k)
         top_items = top_items.numpy()
         #_, sortItems = torch.sort(rating, descending=True)
         #top_items = sortItems[:top_k]
         metrics = utils.recall_precisionATk(GroundTrue, top_items, top_k)
         #metrics['ndcg'] = utils.NDCGatALL(GroundTrue, top_items, top_k)
         print(metrics)
         if world.tensorboard:
             w.add_scalar(f'Test/Recall@{top_k}', metrics['recall'], epoch)
             w.add_scalar(f'Test/Precision@{top_k}', metrics['precision'], epoch)
             #w.add_scalar('Test/NDCG', metrics['ndcg'], epoch)
