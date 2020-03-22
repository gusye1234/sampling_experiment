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
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item.weight
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    def forwardNoSig(self, users, items):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        inner_pro = torch.mul(users_emb, items_emb)
        rating    = torch.sum(inner_pro, dim=1)
        # rating    = self.f(torch.sum(inner_pro, dim=1))
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
    
    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        users_emb = self.embedding_user(users.long())
        users_emb = self.sig(users_emb)
        allItems  = self.embedding_item.weight
        allItems  = self.soft(allItems)
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
            return self.sig(users_emb)

    def getAllUsersEmbedding(self):
        with torch.no_grad():
            users_emb = self.embedding_user.weight
            users_emb = self.sig(users_emb)
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
            allItems =  self.soft(allItems)
            # allItems shape (m, d)
            return allItems
    
    def forward(self, users, items):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(VAR)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users)
        users_emb = self.sig(users_emb)
        items_emb = self.embedding_item(items)
        items_emb = self.soft(items_emb)
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
class VarMF_xij(nn.Module):

    def __init__(self, config):
        super(VarMF_xij, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_xij = config['num_xij']
        self.latent_dim = config['latent_dim_var']
        self.xij_dim = config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_xij = torch.nn.Embedding(num_embeddings=self.num_xij, embedding_dim=self.xij_dim)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        users_emb = self.embedding_user(users.long())
        allItems  = self.embedding_item.weight
        xij_emb0 = self.embedding_xij.weight*(-0.3)
        # users_emb = self.sig(users_emb)
        users_emb = self.sig(torch.cat([users_emb, xij_emb0], dim=1))
        allItems  = self.soft(torch.cat([allItems, xij_emb0], dim=1))
        gamma     = torch.matmul(users_emb, allItems.t())
        # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
        # gamma = self.f(gamma)
        return gamma


    def forward(self, users, items, xij):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")

        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        xij_emb = self.embedding_xij.weight.repeat(len(users), 1)
        xij_emb = xij_emb*((xij - 0.3).reshape(-1, 1))

        users_emb = self.sig(torch.cat([users_emb, xij_emb], dim=1))
        items_emb = self.soft(torch.cat([items_emb, xij_emb], dim=1))
        inner_pro = torch.mul(users_emb, items_emb)

        rating = torch.sum(inner_pro, dim=1)

        return rating

class VarMF_xij2(nn.Module):

    def __init__(self, config):
        super(VarMF_xij2, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_xij = config['num_xij']
        self.latent_dim = config['latent_dim_var']
        self.xij_dim = config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_xij = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.xij_dim)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def forward(self, users, items, xij):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        xij_emb = self.embedding_xij(xij.long())

        users_emb = self.sig(torch.cat([users_emb, xij_emb], dim=1))
        items_emb = self.soft(torch.cat([items_emb, xij_emb], dim=1))
        inner_pro = torch.mul(users_emb, items_emb)

        rating = torch.sum(inner_pro, dim=1)

        return rating

class VarMF_xij_item_personal(nn.Module):
    def __init__(self, config):
        super(VarMF_xij_item_personal, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_xij = config['num_xij']
        self.latent_dim = config['latent_dim_var']
        self.xij_dim = config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim+self.xij_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        # self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        users_emb = self.embedding_user(users.long())
        allItems  = self.embedding_item.weight
        xij_emb0 = self.embedding_item_xij0.weight
        users_emb = self.sig(users_emb)
        allItems  = self.soft(torch.cat([allItems, xij_emb0], dim=1))
        gamma     = torch.matmul(users_emb, allItems.t())
        # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
        # gamma = self.f(gamma)
        return gamma

    def forward(self, users, items, xij):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        # xij_emb = self.embedding_xij(xij.long())
        xij_emb1 = self.embedding_item_xij1(items.long())
        xij_emb0 = self.embedding_item_xij0(items.long())
        xij_emb = xij_emb1 * xij.reshape(-1, 1) + xij_emb0 * (1 - xij.reshape(-1, 1))

        # xij_emb = torch.zeros(len(xij), self.xij_dim).to(world.device)
        # xij_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        # xij_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())

        # users_emb = self.sig(torch.cat([users_emb, xij_emb], dim=1))
        users_emb = self.sig(users_emb)
        items_emb = self.soft(torch.cat([items_emb, xij_emb], dim=1))
        inner_pro = torch.mul(users_emb, items_emb)

        rating = torch.sum(inner_pro, dim=1)

        return rating
 
class VarMF_xij_Symmetric_personal(nn.Module):
    def __init__(self, config):
        super(VarMF_xij_Symmetric_personal, self).__init__()
        self.num_users = config['num_users']
        self.num_items = config['num_items']
        self.num_xij = config['num_xij']
        self.latent_dim = config['latent_dim_var']
        self.xij_dim = config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_user_xij1 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.embedding_user_xij0 = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        # self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)

    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        users_emb = self.embedding_user(users.long())
        allItems  = self.embedding_item.weight
        xij_emb0 = self.embedding_item_xij0.weight
        xij_user_emb0 = self.embedding_user_xij0(users.long())
        users_emb = self.sig(torch.cat([users_emb, xij_user_emb0], dim=1))
        allItems  = self.soft(torch.cat([allItems, xij_emb0], dim=1))
        gamma     = torch.matmul(users_emb, allItems.t())
        # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
        # gamma = self.f(gamma)
        return gamma



    def forward(self, users, items, xij):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users.long())
        items_emb = self.embedding_item(items.long())
        
        xij_item_emb = torch.zeros(len(xij), self.xij_dim)
        xij_item_emb[xij.bool()]  = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())

        xij_user_emb = torch.zeros(len(xij), self.xij_dim)
        xij_user_emb[xij.bool()] = self.embedding_user_xij1(users[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_user_xij0(users[~xij.bool()].long())


        # users_emb = self.sig(torch.cat([users_emb, xij_emb], dim=1))
        users_emb = self.sig(torch.cat([users_emb, xij_user_emb], dim=1))
        items_emb = self.soft(torch.cat([items_emb, xij_item_emb], dim=1))
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

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.latent_dim = self.config['latent_dim_var']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use xavier initilizer')

    
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.weight_decay = self.config['var_weight_decay']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)

    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        with torch.no_grad():
            # gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
            # gamma = self.f(gamma)
            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]
            users_emb = self.sig(users_emb)
            gamma = torch.matmul(users_emb, self.soft(all_items).t())
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
            return self.sig(users_emb)

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
        userEmb_ego = self.embedding_user(users)
        itemEmb_ego = self.embedding_item(items)
        reg_loss = (1 / 2) * self.weight_decay * (1/self.num_items*torch.sum(userEmb_ego.reshape(1, -1) ** 2) + 1/self.num_users*torch.sum(itemEmb_ego.reshape(1, -1) ** 2))
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
        users_emb = self.sig(users_emb)
        items_emb = self.soft(items_emb)
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = torch.sum(inner_pro, dim=1)
        return gamma
    
        # return embeddings
        
class LightGCN_xij(nn.Module):
    def __init__(self, config):
        super(LightGCN_xij, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.num_xij = self.config['num_xij']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_xij = torch.nn.Embedding(num_embeddings=self.num_xij, embedding_dim=self.xij_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use xavier initilizer')
        
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop      = self.config['dropout']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)
        #print("save_txt")
        #np.savetxt('init_weight.txt', np.array(self.embedding_user.weight.detach()))

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

    def forwardWithReg(self,users, items, xij):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user(users)
        itemEmb_ego = self.embedding_item(items)
        xijEmb      = self.embedding_xij.weight
        reg_loss = (1/2)*self.weight_decay*(torch.norm(userEmb_ego, 2).pow(2) + 
                                            torch.norm(itemEmb_ego, 2).pow(2) + 
                                            torch.norm(xijEmb,      2).pow(2))
        reg_loss = reg_loss/float(len(users))
        return gamma, reg_loss
    

    def forward(self, users, items, xij):
        # compute embedding
        all_users, all_items = self.computer()
        #print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        xij_emb = self.embedding_xij.weight.repeat(len(users), 1)
        xij_emb = xij_emb * ((xij - 0.3).reshape(-1, 1))
        users_emb = self.sig(torch.cat([users_emb, xij_emb], dim=1))
        items_emb = self.soft(torch.cat([items_emb, xij_emb], dim=1))
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class LightGCN_xij2(nn.Module):
    def __init__(self, config):
        super(LightGCN_xij2, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.num_xij = self.config['num_xij']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_xij = torch.nn.Embedding(num_embeddings=2, embedding_dim=self.xij_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use xavier initilizer')
        
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop      = self.config['dropout']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)
        #print("save_txt")
        #np.savetxt('init_weight.txt', np.array(self.embedding_user.weight.detach()))

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
    
    def forwardWithReg(self, users, items, xij):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user(users)
        itemEmb_ego = self.embedding_item(items)
        xijEmb      = self.embedding_xij.weight
        reg_loss = (1/2)*self.weight_decay*(torch.norm(userEmb_ego, 2).pow(2) + 
                                            torch.norm(itemEmb_ego, 2).pow(2) + 
                                            torch.norm(xijEmb,      2).pow(2))
        reg_loss = reg_loss/float(len(users))
        return gamma, reg_loss

    def forward(self, users, items, xij):
        # compute embedding
        all_users, all_items = self.computer()
        #print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        xij_long = xij.long()
        xij_emb = self.embedding_xij(xij_long)
        users_emb = self.sig(torch.cat([users_emb, xij_emb], dim=1))
        items_emb = self.soft(torch.cat([items_emb, xij_emb], dim=1))
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

class LightGCN_xij_item_personal_single(nn.Module):
    def __init__(self, config):
        super(LightGCN_xij_item_personal_single, self).__init__()
        self.config = config
        self.dataset : dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use xavier initilizer')

        self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.w_user = torch.nn.Parameter(torch.Tensor(np.random.randn(1)))
        self.w_item = torch.nn.Parameter(torch.Tensor(np.random.randn(1)))

        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.weight_decay = self.config['var_weight_decay']
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
            xij_user_emb = self.embedding_user_xij(users.long())
            users_emb = self.w_user * users_emb
            users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))

            xij_item_emb = self.embedding_item_xij0.weight
            all_items = self.w_item * all_items
            all_items = self.sig(torch.cat([all_items, xij_item_emb], dim=1))

            gamma = torch.matmul(users_emb, all_items.t())
            return gamma

    def forwardWithReg(self, users, items, xij):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user(users)
        itemEmb_ego = self.embedding_item(items)
        xij_user_emb = self.embedding_user_xij(users.long())
        xij_item_emb = torch.zeros(len(xij), self.xij_dim)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
        reg_loss = (1 / 2) * self.weight_decay * (1/self.num_items*torch.sum(userEmb_ego  ** 2) 
                                                  +1/self.num_users*torch.sum(itemEmb_ego ** 2)
                                                  +1/self.num_items*torch.sum(xij_user_emb**2)
                                                  +1/self.num_users*torch.sum(xij_item_emb**2))
        #reg_loss = reg_loss / float(len(users))
        return gamma, reg_loss

    def forward(self, users, items, xij):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        # xij_item_emb1 = self.embedding_item_xij1(items.long())
        # xij_item_emb0 = self.embedding_item_xij0(items.long())
        # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
        xij_user_emb = self.embedding_user_xij(users.long())
        
        xij_item_emb = torch.zeros(len(xij), self.xij_dim)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
        
        print('user1',users_emb)
        print(self.w_user)
        users_emb = self.w_user*users_emb
        items_emb = self.w_item*items_emb
        print('user2', users_emb)
        users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
        items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))
        print('user3', users_emb)

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

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use xavier initilizer')

        self.embedding_user_xij = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.xij_dim)
        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.w_user = torch.nn.Linear(self.latent_dim, self.latent_dim)
        self.w_item = torch.nn.Linear(self.latent_dim, self.latent_dim)

        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.weight_decay = self.config['var_weight_decay']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)

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
        userEmb_ego = self.embedding_user(users)
        itemEmb_ego = self.embedding_item(items)
        xij_user_emb = self.embedding_user_xij(users.long())
        
        xij_item_emb = torch.zeros(len(xij), self.xij_dim)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
        reg_loss = (1 / 2) * self.weight_decay * (1/self.num_items*torch.sum(userEmb_ego ** 2) 
                                                  +1/self.num_users*torch.sum(itemEmb_ego ** 2)
                                                  +1/self.num_items*torch.sum(xij_user_emb **2)
                                                  +1/self.num_users*torch.sum(xij_item_emb **2)
                                                  +1/self.num_items*torch.sum(self.w_user.weight **2)
                                                  + 1/self.num_users* torch.sum(self.w_item.weight ** 2))
        #reg_loss = reg_loss / float(len(users))
        return gamma, reg_loss

    def forward(self, users, items, xij):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]
        # xij_item_emb1 = self.embedding_item_xij1(items.long())
        # xij_item_emb0 = self.embedding_item_xij0(items.long())
        # xij_item_emb = xij_item_emb1 * xij.reshape(-1, 1) + xij_item_emb0 * (1 - xij.reshape(-1, 1))
        
        xij_user_emb = self.embedding_user_xij(users.long())
        
        xij_item_emb = torch.zeros(len(xij), self.xij_dim)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())
        
        print('user1',users_emb)
        print(self.w_user)
        users_emb = self.w_user(users_emb)
        items_emb = self.w_item(items_emb)
        print('user2', users_emb)
        users_emb = self.soft(torch.cat([users_emb, xij_user_emb], dim=1))
        items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))
        print('user3', users_emb)

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
class LightGCN_xij_Symmetric_nopersonal(nn.Module):
    def __init__(self, config, dataset):
        super(LightGCN_xij_Symmetric_nopersonal, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = dataset
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.config['num_users']
        self.num_items = self.config['num_items']
        self.num_xij = self.config['num_xij']
        self.latent_dim = self.config['latent_dim_var']
        self.xij_dim = self.config['xij_dim']
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_user_xij2 = torch.nn.Embedding(2, embedding_dim=self.xij_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_item_xij2 = torch.nn.Embedding(2, embedding_dim=self.xij_dim)
        nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
        nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
        print('use xavier initilizer')
        
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop      = self.config['dropout']
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)
        #print("save_txt")
        #np.savetxt('init_weight.txt', np.array(self.embedding_user.weight.detach()))

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

    def forward(self, users, items, xij):
        # compute embedding
        all_users, all_items = self.computer()
        # print('forward')
        # all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        xij_long = xij.long()
        users_xij = self.embedding_user_xij2(xij_long)
        items_xij = self.embedding_item_xij2(xij_long)
        users_emb = self.sig(torch.cat([users_emb, users_xij], dim=1))
        items_emb = self.soft(torch.cat([items_emb,items_xij], dim=1))
        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma
