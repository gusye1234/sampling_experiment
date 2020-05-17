"""
Define models here
"""
import world
import torch
import dataloader
from torch import nn
import numpy as np

#Recommendation Model
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

        self.f = nn.Sigmoid()

        print('RecMF Init!')
    
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
    



#Posterior Model
class LightGCN_xij_item_personal_matrix(nn.Module):
    def __init__(self, config):
        super(LightGCN_xij_item_personal_matrix, self).__init__()
        self.config = config
        self.dataset: dataloader.BasicDataset = config['dataset']
        self.sig = nn.Sigmoid()
        self.soft = nn.Softmax(dim=1)
        self.__init_weight()
        print('LightGCN_xij_item_personal_matrix Init!')

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

        self.embedding_item_xij1 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)
        self.embedding_item_xij0 = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.xij_dim)

        self.embedding_item_xij1.weight.data = torch.tensor([2.0] * self.num_items).reshape(-1, 1)
        self.embedding_item_xij0.weight.data = torch.tensor([-2.0] * self.num_items).reshape(-1, 1)


        self.w_user = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        self.w_item = torch.nn.Linear(self.latent_dim, self.latent_dim, bias=False)
        nn.init.uniform_(self.w_user.weight, a=-1, b=1)
        nn.init.uniform_(self.w_item.weight, a=-1, b=1)

        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.drop = self.config['dropout']
        self.emb_decay = self.config['var_weight_decay']
        self.wdecay = self.config['w_weight_decay']

        self.xdecay = self.config['x_weight_decay']
        print('emb_decay, wdecay, xdecay', self.emb_decay, self.wdecay, self.xdecay)
        self.Graph = self.dataset.getSparseGraph().coalesce().to(world.device)


    def get_user_item_embedding(self):
        with torch.no_grad():
            all_users, all_items = self.computer()

            hyper_x = self.hyper_x
            xij_user_emb = torch.tensor([hyper_x] * self.num_users).reshape(-1, 1).to(world.device)

            xij_item_emb0 = self.embedding_item_xij0.weight
            xij_item_emb1 = self.embedding_item_xij1.weight

            users_emb = self.w_user(all_users)
            items_emb = self.w_item(all_items)

            users_emb = torch.cat([(1 - hyper_x) * self.soft(users_emb), xij_user_emb], dim=1)
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

        embs = [all_emb]

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

        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        return users, items


    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """

        with torch.no_grad():

            all_users, all_items = self.computer()
            users_emb = all_users[users.long()]

            hyper_x = self.hyper_x
            xij_user_emb = torch.tensor([hyper_x] * len(users)).reshape(-1, 1).to(world.device)
            users_emb = self.w_user(users_emb)

            users_emb = torch.cat([(1 - hyper_x) * self.soft(users_emb), xij_user_emb], dim=1)
            xij_item_emb = self.embedding_item_xij0.weight
            all_items = self.w_item(all_items)
            all_items = self.sig(torch.cat([all_items, xij_item_emb], dim=1))

            gamma = torch.matmul(users_emb, all_items.t())
            return gamma

    def forwardWithReg(self, users, items, xij, G, S):
        gamma = self.forward(users, items, xij)
        userEmb_ego = self.embedding_user.weight
        itemEmb_ego = self.embedding_item.weight

        xij_item_emb0 = self.embedding_item_xij0.weight
        xij_item_emb1 = self.embedding_item_xij1.weight

        w_user_ego = self.w_user.weight
        w_item_ego = self.w_item.weight

        reg_loss_emb = (1 / 2) * self.emb_decay * (torch.sum(userEmb_ego ** 2)
                                                   + torch.sum(itemEmb_ego ** 2)
                                                   + torch.sum(xij_item_emb0 ** 2)
                                                   + torch.sum(xij_item_emb1 ** 2))

        reg_loss_w = (1 / 2) * self.wdecay * (torch.sum(w_user_ego ** 2)
                                            + torch.sum(w_item_ego ** 2))


        reg_loss = reg_loss_emb + reg_loss_w
        print('reg_loss_emb, reg_loss_x', reg_loss_emb, reg_loss_w)
        #print('S, G', S, G, S/G)
        #reg_loss = S / G * reg_loss

        return gamma, reg_loss

    def forward(self, users, items, xij):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items[items.long()]

        hyper_x = self.hyper_x
        xij_user_emb = torch.tensor([hyper_x] * len(users)).reshape(-1, 1).to(world.device)

        xij_item_emb = torch.zeros(len(xij), self.xij_dim).to(world.device)
        xij_item_emb[xij.bool()] = self.embedding_item_xij1(items[xij.bool()].long())
        xij_item_emb[~xij.bool()] = self.embedding_item_xij0(items[~xij.bool()].long())

        users_emb = self.w_user(users_emb)
        items_emb = self.w_item(items_emb)


        users_emb = torch.cat([(1 - hyper_x) * self.soft(users_emb), xij_user_emb], dim=1)
        items_emb = self.sig(torch.cat([items_emb, xij_item_emb], dim=1))


        inner_pro = torch.mul(users_emb, items_emb)
        gamma = torch.sum(inner_pro, dim=1)
        return gamma

