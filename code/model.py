"""
Define models here
"""
import torch
from torch import nn

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
        self.f = nn.Sigmoid()
    
    def getUsersRating(self, users):
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
    
    
class VarMF(nn.Module):
    """
    create embeddings for variational inference
    input user-item pair to get the Probability.
    embedding normally initialized N(0,1)
    """
    def __init__(self, config):
        super(VarMF, self).__init__()
        self.num_users  = config['num_users']
        self.num_items  = config['num_items']
        self.latent_dim = config['latent_dim_var']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.f = nn.Sigmoid()
    
    def allGamma(self, users):
        """
        users : (batch_size, dim_var)
        """
        users_emb = self.embedding_user(users)
        gamma = torch.matmul(users_emb, self.embedding_item.weight.t())
        gamma = self.f(gamma)
        return gamma
        
    
    def forward(self, users, items):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(VAR)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        inner_pro = torch.mul(users_emb, items_emb)
        gamma     = self.f(torch.sum(inner_pro, dim=1))
        return gamma

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
        users_emb = self.embedding_user(users)
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