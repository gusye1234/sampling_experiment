"""
Define models here
"""
import world
import torch
import dataloader
from torch import nn
import numpy as np

class RecMF(nn.Module):
    """
    create embeddings for recommendation
    input user-item pair to get the rating.
    embedding normally initialized N(0,1)
    """
    def __init__(self, config):
        super(RecMF, self).__init__()
        self.num_users  = config['num_users']
        self.num_items  = config['num_items']
        self.latent_dim = config['latent_dim_rec']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_user.weight.data *= 0.1
        self.embedding_item.weight.data *= 0.1
        print('rec_init')
        self.f = nn.Sigmoid()
    
    def getUsersRating(self, users):
        with torch.no_grad():
            users_emb = self.embedding_user(users.long())
            items_emb = self.embedding_item.weight
            rating = self.f(torch.matmul(users_emb, items_emb.t()))
            return rating

    
    def forward(self, users, items):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        inner_pro = torch.mul(users_emb, items_emb)
        rating    = self.f(torch.sum(inner_pro, dim=1))
        return rating
    
class VarMF_reg(nn.Module):
    """
    create embeddings for variational inference
    input user-item pair to get the Probability.
    embedding normally initialized N(0,1)
    Use sigmoid to regularize User embedding
    Use softmax to regularize Item embedding
    """
    def __init__(self, config):
        super(VarMF_reg, self).__init__()
        self.num_users  = config['num_users']
        self.num_items  = config['num_items']
        self.latent_dim = config['latent_dim_var']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        print('VarMF_reg init!!!')

    def get_user_item_embedding(self):
        with torch.no_grad():
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
            users_emb = self.soft(users_emb)
            items_emb = self.sig(items_emb)
    
            return users_emb, items_emb, items_emb


    
    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        with torch.no_grad():
            users_emb = self.embedding_user(users.long())
            users_emb = self.soft(users_emb)
            allItems  = self.embedding_item.weight
            allItems  = self.sig(allItems)
            gamma     = torch.matmul(users_emb, allItems.t())
            # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
            # gamma = self.f(gamma)
            return gamma
    
    def getUsersEmbedding(self, users):
        """
        calculate users embedding for specific sampling algorithm
        hence no need for gradient
        """
        with torch.no_grad():
            users_emb = self.embedding_user(users)
            return self.soft(users_emb)

    def getAllUsersEmbedding(self):
        with torch.no_grad():
            users_emb = self.embedding_user.weight
            users_emb = self.soft(users_emb)
            return users_emb    

        
    # def getGammaForUsers(self, users):
    #     """
    #     calculate gammas of all items for specific users
    #     """
    #     with torch.no_grad():
    #         users_emb = self.embedding_user(users)
    #         users_emb = self.sig(users_emb)
    #         items = self.getAllItemsEmbedding()
    
    def getAllItemsEmbedding(self):
        """
        calculate all items embedding for specific sampling algorithm
        hence no need for gradient
        """
        with torch.no_grad():
            allItems = self.embedding_item.weight
            allItems =  self.sig(allItems)
            # allItems shape (m, d)
            return allItems
    
    def forward(self, users, items):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(VAR)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users)
        users_emb = self.soft(users_emb)
        items_emb = self.embedding_item(items)
        items_emb = self.sig(items_emb)
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma


class VarMF_xij_item_personal(nn.Module):
    def __init__(self, config):
        super(VarMF_xij_item_personal, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_xij = config['num_xij']
        self.latent_dim = config['latent_dim_var']
        self.xij_dim = config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        # self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.weight_decay1 = config['var_weight_decay']
        self.weight_decay2 = config['x_weight_decay']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        print('VarMF_xij_item_personal init!!!')

    def get_user_item_embedding(self):
        with torch.no_grad():
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
            xij_user_emb = self.embedding_user_xij.weight
            xij_item_emb1 = self.embedding_item_xij1.weight
            xij_item_emb0 = self.embedding_item_xij0.weight
            users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            items_emb0 = self.sig(torch.cat([items_emb, xij_item_emb0], dim=1))
            items_emb1 = self.sig(torch.cat([items_emb, xij_item_emb1], dim=1))

            return users_emb, items_emb1, items_emb0

    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        with torch.no_grad():
            users_emb = self.embedding_user(users.long())
            xij_user_emb = self.embedding_user_xij(users.long())
            allItems  = self.embedding_item.weight
            xij_item_emb0 = self.embedding_item_xij0.weight
            users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            allItems  = self.sig(torch.cat([allItems, xij_item_emb0], dim=1))
            gamma     = torch.matmul(users_emb, allItems.t())
        # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
        # gamma = self.f(gamma)
            return gamma

    def forwardWithReg(self,users, items, xij):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user.weight
        itemEmb_ego = self.embedding_item.weight
        xijUser_ego = self.embedding_user_xij.weight
        xijItem_ego1 = self.embedding_item_xij1.weight
        xijItem_ego0 = self.embedding_item_xij0.weight
        reg_loss_emb = (1 / 2) * self.weight_decay1 * (torch.sum(userEmb_ego  ** 2) 
                                                  +torch.sum(itemEmb_ego ** 2))
        reg_loss_xij = (1 / 2) * self.weight_decay1 * (torch.sum(xijUser_ego  ** 2) 
                                                  +torch.sum(xijItem_ego1 ** 2)
                                                  +torch.sum(xijItem_ego0 ** 2))
        reg_loss = reg_loss_emb + reg_loss_xij
        # reg_loss = reg_loss/float(len(users))
        #reg_loss = (1/2)*self.weight_decay*(torch.norm(userEmb_ego, 2).pow(2) + torch.norm(itemEmb_ego, 2).pow(2))
        #reg_loss = reg_loss/float(len(users))
        return gamma, reg_loss

    def forward(self, users, items, xij):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        # xij_emb = self.embedding_xij(xij.long())
        xij_user_emb = self.embedding_user_xij(users.long())
        xij_item_emb1 = self.embedding_item_xij1(items.long())
        xij_item_emb0 = self.embedding_item_xij0(items.long())
        xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))

        # xij_emb = torch.zeros(len(xij), self.xij_dim).to(world.device)
        # xij_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        # xij_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())

        # users_emb = self.sig(torch.cat([users_emb, xij_emb], dim=1))
        print('mfItemPerEmb1', users_emb[35:40], items_emb[35:40])

        users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
        items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))
        print('mfItemPerEmb2', users_emb[35:40], items_emb[35:40])
        inner_pro = torch.mul(users_emb, items_emb)

        rating = torch.sum(inner_pro, dim=1)

        return rating
 

class LightGCN(nn.Module):
    def __init__(self, config):
        super(LightGCN, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()
        print('LightGCN init!!!')

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.latent_dim = self.config['latent_dim_var']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.uniform_(self.embedding_user.weight, a=-1, b=1)
        nn.init.uniform_(self.embedding_item.weight, a=-1, b=1)
        #nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        #nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        #print('use xavier initilizer')

        self.w_user = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.w_item = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.uniform_(self.w_user.weight, a=-1, b=1)
        nn.init.uniform_(self.w_item.weight, a=-10, b=10)

    
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.weight_decay = self.config['var_weight_decay']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)

    def get_user_item_embedding(self):
        with torch.no_grad():
            all_users, all_items = self.computer()
            
            # xij_item_emb1 = self.embedding_item_xij1(items.long())
            # xij_item_emb0 = self.embedding_item_xij0(items.long())
            # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
            users_emb = self.w_user(all_users)
            items_emb = self.w_item(all_items)

            users_emb = self.soft(users_emb)
            items_emb = self.sig(items_emb)
            
            return users_emb, items_emb, items_emb

    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        with torch.no_grad():
            # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
            # gamma = self.f(gamma)
            all_users, all_items = self.computer()

            users_emb = all_users[users.long()]
            users_emb = self.w_user(users_emb)
            users_emb = self.soft(users_emb)

            items_emb = self.w_item(all_items)
            gamma = torch.matmul(users_emb, self.sig(all_items).t())
            return gamma

    def getUsersEmbedding(self, users):
        """
        calculate users embedding for specific sampling algorithm
        hence no need for gradient
        """
        with torch.no_grad():
            all_users, _ = self.computer()
            users_emb = all_users[users]
            #return users_emb
            return self.soft(users_emb)

    def getAllUsersEmbedding(self):
        print("get_all_user")
        with torch.no_grad():
            all_users, _ = self.computer()
            users_emb = self.sig(all_users)
            return users_emb
        
    def getAllItemsEmbedding(self):
        """
        calculate all items embedding for specific sampling algorithm
        hence no need for gradient
        """
        print("get_all_item")
        with torch.no_grad():
            _, all_items = self.computer()
            items_emb = self.soft(all_items)
            # allItems shape (m, d)
            return items_emb
    
    def __dropout(self, keep_prob):
        size = self.Graph.size()
        index = self.Graph.indices().t()
        values = self.Graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index]/keep_prob
        g = torch.sparse.IntTensor(index.t(), values, size)
        return g
        
    
    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        # if self.training:
        # g_droped = self.__dropout(self.keep_prob)
        if self.drop:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            print('no drop!!!')
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def forwardWithReg(self,users, items):
        gamma = self.forward(users, items)
        userEmb_ego = self.embedding_user.weight
        itemEmb_ego = self.embedding_item.weight
        w_user_ego = self.w_user.weight
        w_item_ego = self.w_item.weight
        reg_loss = (1 / 2) * self.weight_decay * (torch.sum(userEmb_ego  ** 2) 
                                                  +torch.sum(itemEmb_ego ** 2)
                                                  +torch.sum(w_user_ego ** 2)
                                                  +torch.sum(w_item_ego ** 2))
        # reg_loss = reg_loss/float(len(users))
        #reg_loss = (1/2)*self.weight_decay*(torch.norm(userEmb_ego, 2).pow(2) + torch.norm(itemEmb_ego, 2).pow(2))
        #reg_loss = reg_loss/float(len(users))
        return gamma, reg_loss
    
    def forward(self, users, items):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        #all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        print('emb1', users_emb[35:39], items_emb[35:39])
        users_emb = self.w_user(users_emb)
        items_emb = self.w_item(items_emb)
        print('linear trans', self.w_user.weight, self.w_item.weight)
        print('emb2', users_emb[35:39], items_emb[35:39])
        users_emb = self.soft(users_emb)
        items_emb = self.sig(items_emb)
        print('emb3', users_emb[35:39], items_emb[35:39])
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
        # return embeddings
        


class LightGCN_xij_item_personal_single(nn.Module):
    def __init__(self, config):
        super(LightGCN_xij_item_personal_single, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()
        print('LightGCN_xij_item_personal_single!!!')

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.hyper_x = self.config['hyper_x']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use xavier initilizer')

        #self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.w_user = torch.nn.Parameter(torch.Tensor(np.random.randn(1)))
        self.w_item = torch.nn.Parameter(torch.Tensor(np.random.randn(1)))

        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.weight_decay1 = self.config['var_weight_decay']
        self.weight_decay2 = self.config['x_weight_decay']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)
        # print("save_txt")
        # np.savetxt('init_weight.txt', np.array(self.embedding_user.weight.detach()))

    def __dropout(self, keep_prob):
        size = self.Graph.size()
        index = self.Graph.indices().t()
        values = self.Graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.IntTensor(index.t(), values, size)
        return g

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        # if self.training:
        # g_droped = self.__dropout(self.keep_prob)
        if self.drop:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        #print('allgamma')
        with torch.no_grad():
            # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
            # gamma = self.f(gamma)
            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]
            #xij_user_emb = self.embedding_user_xij(users.long())
            hyper_x = self.hyper_x
            xij_user_emb = torch.tensor([hyper_x]*len(users)).reshape(-1, 1).to(world.device)
            users_emb = self.w_user * users_emb
            #users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            users_emb = torch.cat([(1-hyper_x)*self.soft(users_emb), xij_user_emb], dim=1)

            xij_item_emb = self.embedding_item_xij0.weight
            all_items = self.w_item * all_items
            all_items = self.sig(torch.cat([all_items, xij_item_emb], dim=1))

            gamma = torch.matmul(users_emb, all_items.t())
            return gamma

    def get_user_item_embedding(self):
        with torch.no_grad():
            all_users, all_items = self.computer()
            
            # xij_item_emb1 = self.embedding_item_xij1(items.long())
            # xij_item_emb0 = self.embedding_item_xij0(items.long())
            # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
            #xij_user_emb = self.embedding_user_xij.weight
            hyper_x = self.hyper_x
            xij_user_emb = torch.tensor([hyper_x]*self.num_users).reshape(-1, 1).to(world.device)

            xij_item_emb0 = self.embedding_item_xij0.weight
            xij_item_emb1 = self.embedding_item_xij1.weight
            
        
            users_emb = self.w_user*all_users
            items_emb = self.w_item*all_items
            users_emb = torch.cat([(1-hyper_x)*self.soft(users_emb), xij_user_emb], dim=1)
            #users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            items_emb0 = self.sig(torch.cat([items_emb, xij_item_emb0], dim=1))
            items_emb1 = self.sig(torch.cat([items_emb, xij_item_emb1], dim=1))
            return users_emb, items_emb0, items_emb1

    def forwardWithReg(self, users, items, xij):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user.weight
        itemEmb_ego = self.embedding_item.weight
        #xij_user_emb = self.embedding_user_xij.weight
        xij_item_emb0 = self.embedding_item_xij0.weight
        xij_item_emb1 = self.embedding_item_xij1.weight
        reg_loss = (1 / 2) * self.weight_decay1 * (torch.sum(userEmb_ego  ** 2) 
                                                  +torch.sum(itemEmb_ego ** 2)
                                                  +torch.sum(self.w_user**2)
                                                  +torch.sum(self.w_item**2)
                                                  +torch.sum(xij_item_emb0 **2)
                                                  +torch.sum(xij_item_emb1 **2))

        #reg_loss_x = (1 / 2) * self.weight_decay2 * (torch.sum(xij_user_emb **2)
                                                    #+torch.sum(xij_item_emb0 **2)
                                                    #+torch.sum(xij_item_emb1 **2))
        
        #print('decay', self.weight_decay1, self.weight_decay2)
        #reg_loss = reg_loss + reg_loss_x
        #reg_loss = reg_loss / float(len(users))
        return gamma, reg_loss

    def forward(self, users, items, xij):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        # xij_item_emb1 = self.embedding_item_xij1(items.long())
        # xij_item_emb0 = self.embedding_item_xij0(items.long())
        # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
        #xij_user_emb = self.embedding_user_xij(users.long())
        hyper_x = self.hyper_x
        xij_user_emb = torch.tensor([hyper_x]*len(users)).reshape(-1, 1).to(world.device)
        
        xij_item_emb = torch.zeros(len(xij), self.xij_dim).to(world.device)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
        
        users_emb = self.w_user*users_emb
        items_emb = self.w_item*items_emb
        users_emb = torch.cat([(1-hyper_x)*self.soft(users_emb), xij_user_emb], dim=1)
        items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))

        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
        


class LightGCN_xij_item_personal_matrix(nn.Module):
    def __init__(self, config):
        super(LightGCN_xij_item_personal_matrix, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()
        print('LightGCN_xij_item_personal_matrix!!!')

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.hyper_x = self.config['hyper_x']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.uniform_(self.embedding_user.weight, a=-1, b=1)
        nn.init.uniform_(self.embedding_item.weight, a=-1, b=1)
        print(self.embedding_user.weight[0:2], self.embedding_item.weight[0:2])
        print('use nn.init.uniform initilizer')

        #self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        nn.init.uniform_(self.embedding_item_xij1.weight, a=-1, b=1)
        nn.init.uniform_(self.embedding_item_xij0.weight, a=-1, b=1)

        self.w_user = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.w_item = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.constant_(self.w_user.weight, 1)
        nn.init.constant_(self.w_item.weight, 1)
        print(self.w_user.weight)
        print(self.w_item.weight)

        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.weight_decay1 = self.config['var_weight_decay']
        self.weight_decay2 = self.config['x_weight_decay']
        #self.weight_decay3 = self.config['w_weight_decay']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)
        
    def get_user_item_embedding(self):
        with torch.no_grad():
            all_users, all_items = self.computer()
            
            # xij_item_emb1 = self.embedding_item_xij1(items.long())
            # xij_item_emb0 = self.embedding_item_xij0(items.long())
            # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
            #xij_user_emb = self.embedding_user_xij.weight
            hyper_x = self.hyper_x
            xij_user_emb = torch.tensor([hyper_x]*self.num_users).reshape(-1, 1).to(world.device)
            
            xij_item_emb0 = self.embedding_item_xij0.weight
            xij_item_emb1 = self.embedding_item_xij1.weight
            
        
            users_emb = self.w_user(all_users)
            items_emb = self.w_item(all_items)
            #users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            users_emb = torch.cat([(1-hyper_x)*self.soft(users_emb), xij_user_emb], dim=1)
            items_emb0 = self.sig(torch.cat([items_emb, xij_item_emb0], dim=1))
            items_emb1 = self.sig(torch.cat([items_emb, xij_item_emb1], dim=1))
            return users_emb, items_emb0, items_emb1

    def getAllUsersEmbedding(self):
        with torch.no_grad():
            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]
            items_emb = all_items[items.long()]
            # xij_item_emb1 = self.embedding_item_xij1(items.long())
            # xij_item_emb0 = self.embedding_item_xij0(items.long())
            # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
            #xij_user_emb = self.embedding_user_xij(users.long())
            hyper_x = self.hyper_x
            xij_user_emb = torch.tensor([hyper_x]*len(users)).reshape(-1, 1).to(world.device)
        
            xij_item_emb = torch.zeros(len(xij), self.xij_dim).to(world.device)
            xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
            xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
            print('linear trans', self.w_user.weight, self.w_item.weight)
            print('lgn embedding1', users_emb[0:3], items_emb[0:3])
            users_emb = self.w_user(users_emb)
            items_emb = self.w_item(items_emb)
            print('lgn embedding2', users_emb[0:3], items_emb[0:3])
            #users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            users_emb = torch.cat([(1-hyper_x)*self.soft(users_emb), xij_user_emb], dim=1)
            items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))

            inner_pro = torch.mul(users_emb, items_emb)
            gamma = torch.sum(inner_pro, dim=1)
            return gamma


    def __dropout(self, keep_prob):
        size = self.Graph.size()
        index = self.Graph.indices().t()
        values = self.Graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.IntTensor(index.t(), values, size)
        return g

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        # if self.training:
        # g_droped = self.__dropout(self.keep_prob)
        if self.drop:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        #print('allgamma')
        with torch.no_grad():
            # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
            # gamma = self.f(gamma)
            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]
            #xij_user_emb = self.embedding_user_xij(users.long())
            hyper_x = self.hyper_x
            xij_user_emb = torch.tensor([hyper_x]*len(users)).reshape(-1, 1).to(world.device)
            users_emb = self.w_user(users_emb)
            #users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            users_emb = torch.cat([(1-hyper_x)*self.soft(users_emb), xij_user_emb], dim=1)
            xij_item_emb = self.embedding_item_xij0.weight
            all_items = self.w_item(all_items)
            all_items = self.sig(torch.cat([all_items, xij_item_emb], dim=1))

            gamma = torch.matmul(users_emb, all_items.t())
            return gamma

    def forwardWithReg(self, users, items, xij):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user.weight
        itemEmb_ego = self.embedding_item.weight
        w_user_ego = self.w_user.weight
        w_item_ego = self.w_item.weight
        #xij_user_emb = self.embedding_user_xij.weight
        xij_item_emb0 = self.embedding_item_xij0.weight
        xij_item_emb1 = self.embedding_item_xij1.weight
        
        reg_loss_emb = (1 / 2) * self.weight_decay1 * (torch.sum(userEmb_ego ** 2) 
                                                  +torch.sum(itemEmb_ego ** 2)
                                                  +torch.sum(w_user_ego ** 2)
                                                  +torch.sum(w_item_ego ** 2))
        
        reg_loss_x = (1 / 2) * self.weight_decay2 * (torch.sum(xij_item_emb0 ** 2)
                                                  +torch.sum(xij_item_emb1 ** 2))

        
                                                  

        reg_loss = reg_loss_emb + reg_loss_x                             
        print('decay', self.weight_decay1, self.weight_decay2)
        #reg_loss = reg_loss / float(len(users))
        
        return gamma, reg_loss

    def forward(self, users, items, xij):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        # xij_item_emb1 = self.embedding_item_xij1(items.long())
        # xij_item_emb0 = self.embedding_item_xij0(items.long())
        # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
        #xij_user_emb = self.embedding_user_xij(users.long())
        hyper_x = self.hyper_x
        xij_user_emb = torch.tensor([hyper_x]*len(users)).reshape(-1, 1).to(world.device)
        
        xij_item_emb = torch.zeros(len(xij), self.xij_dim).to(world.device)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
        print('linear trans', self.w_user.weight, self.w_item.weight)
        print('lgn embedding1', users_emb[0:3], items_emb[0:3])
        users_emb = self.w_user(users_emb)
        items_emb = self.w_item(items_emb)
        print('lgn embedding2', users_emb[0:3], items_emb[0:3])
        #users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
        users_emb = torch.cat([(1-hyper_x)*self.soft(users_emb), xij_user_emb], dim=1)
        items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))

        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class LightGCN_xij_item_personal_matrix_nohyper(nn.Module):
    def __init__(self, config):
        super(LightGCN_xij_item_personal_matrix_nohyper, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()
        print('LightGCN_xij_item_personal_matrix_nohyper!!!')

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.uniform_(self.embedding_user.weight, a=-1, b=1)
        nn.init.uniform_(self.embedding_item.weight, a=-1, b=1)
        print('use uniform initilizer')

        self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        nn.init.uniform_(self.embedding_user_xij.weight, a=-1, b=1)
        nn.init.uniform_(self.embedding_item_xij1.weight, a=-1, b=1)
        nn.init.uniform_(self.embedding_item_xij0.weight, a=-1, b=1)

        self.w_user = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.w_item = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.constant_(self.w_user.weight, 1)
        nn.init.constant_(self.w_item.weight, 1)
        
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.lgn_decay = self.config['var_weight_decay']
        #self.user_decay = self.config['user_decay']
        self.x_decay = self.config['x_weight_decay']
        #self.item_decay = self.config['item_decay']
        #self.weight_decay3 = self.config['w_weight_decay']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)
        
    def get_user_item_embedding(self):
        with torch.no_grad():
            all_users, all_items = self.computer()
            
            # xij_item_emb1 = self.embedding_item_xij1(items.long())
            # xij_item_emb0 = self.embedding_item_xij0(items.long())
            # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
            xij_user_emb = self.embedding_user_xij.weight
            
            xij_item_emb0 = self.embedding_item_xij0.weight
            xij_item_emb1 = self.embedding_item_xij1.weight
            
        
            users_emb = self.w_user(all_users)
            items_emb = self.w_item(all_items)
            users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
            items_emb0 = self.sig(torch.cat([items_emb, xij_item_emb0], dim=1))
            items_emb1 = self.sig(torch.cat([items_emb, xij_item_emb1], dim=1))
            return users_emb, items_emb0, items_emb1


    def __dropout(self, keep_prob):
        size = self.Graph.size()
        index = self.Graph.indices().t()
        values = self.Graph.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.IntTensor(index.t(), values, size)
        return g

    def computer(self):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        # if self.training:
        # g_droped = self.__dropout(self.keep_prob)
        if self.drop:
            if self.training:
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items
    
    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        #print('allgamma')
        with torch.no_grad():
            # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
            # gamma = self.f(gamma)
            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]
            xij_user_emb = self.embedding_user_xij(users.long())
            users_emb = self.w_user(users_emb)
            users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))

            xij_item_emb = self.embedding_item_xij0.weight
            all_items = self.w_item(all_items)
            all_items = self.sig(torch.cat([all_items, xij_item_emb], dim=1))

            gamma = torch.matmul(users_emb, all_items.t())
            return gamma

    def forwardWithReg(self, users, items, xij):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user.weight
        itemEmb_ego = self.embedding_item.weight
        w_user_ego = self.w_user.weight
        w_item_ego = self.w_item.weight
        xij_user_emb = self.embedding_user_xij.weight
        xij_item_emb0 = self.embedding_item_xij0.weight
        xij_item_emb1 = self.embedding_item_xij1.weight

        
        reg_loss_emb = (1 / 2) * self.lgn_decay * (torch.sum(userEmb_ego ** 2) 
                                                  +torch.sum(itemEmb_ego ** 2)
                                                  +torch.sum(w_user_ego ** 2)
                                                  +torch.sum(w_item_ego ** 2))

        reg_loss_x = (1 / 2) * self.x_decay * (torch.sum(xij_user_emb ** 2)
                                                  +torch.sum(xij_item_emb0 ** 2)
                                                  +torch.sum(xij_item_emb1 ** 2))

                                    
        print('decay', self.lgn_decay, self.x_decay)
        #reg_loss = reg_loss / float(len(users))
        reg_loss = reg_loss_emb + reg_loss_x
        return gamma, reg_loss


                                    
        #print('decay', self.user_decay, self.item_decay, self.x_decay)
        #reg_loss = reg_loss / float(len(users))
        #reg_loss = reg_loss_user + reg_loss_item + reg_loss_x_user + reg_loss_x_item
        #return gamma, reg_loss

    def forward(self, users, items, xij):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        # xij_item_emb1 = self.embedding_item_xij1(items.long())
        # xij_item_emb0 = self.embedding_item_xij0(items.long())
        # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        print('emb1', users_emb[:3], items_emb[:3])
        xij_user_emb = self.embedding_user_xij(users.long())
        
        xij_item_emb = torch.zeros(len(xij), self.xij_dim).to(world.device)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
        
        users_emb = self.w_user(users_emb)
        items_emb = self.w_item(items_emb)
        print('emb2', users_emb[:3], items_emb[:3])
        users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
        items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))

        print('emb3', users_emb[:3], items_emb[:3])

        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma




        
if __name__ == "__main__":
    import dataloader
    dataset = dataloader.LastFM()
    world.config['num_users'] = dataset.n_users
    world.config['num_items'] = dataset.m_items
    world.config['lightGCN_n_layers'] = 3
    world.config['keep_prob'] = 0.7
    model = LightGCN(world.config, dataset)
    model.train()
    users = torch.Tensor([1,2,3,4,5,6,7]).long()
    items = torch.Tensor([1,2,3,4,5,6,7]).long()
    gamma = model(users, items)
    print(gamma)
    
    
# =====================no using now====================
