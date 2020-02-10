import torch
from torch import nn

class RecMF(nn.Module):
    """
    create embeddings for recommendation
    input user-item pair to get the rating.
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
    
    def forward(self, users, items):
        try:
            assert len(users) == len(items)
        except AssertionError:
            raise AssertionError(f"(Rec)users and items should be paired, \
                                 but we got {len(users)} users and {len(items)} items")
        users_emb = self.embedding_user(users)
        items_emb = self.embedding_item(items)
        inner_pro = torch.mul(users_emb, items_emb)
        rating    = self.f(torch.sum(inner_pro, dim=1))
        return rating
    
    
class VarMF(nn.Module):
    """
    create embeddings for variational inference
    input user-item pair to get the Probability.
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
        rating    = self.f(torch.sum(inner_pro, dim=1))
        return rating