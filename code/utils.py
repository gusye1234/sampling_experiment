"""
Including Sampling Algorithms
and help functions 
"""
import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from model import VarMF, VarMF_reg




class ELBO:
    """
    class for criterion L(theta, q; x_ij)
    hyperparameters: epsilon, eta
    details in *SamWalker: Social Recommendation with Informative Sampling Strategy*
    NOTE: multiply -1 to original ELBO here.
    forward:
        rating : shape(batch_size, 1) 
        gamma  : shape(batch_size, 1)
        xij    : shape(batch_size, 1)
    """
    eps = torch.Tensor([1e-8]).float()
    def __init__(self, config,
                 rec_model, var_model, rec_lr=0.003, var_lr=0.003):
        rec_model : nn.Module
        var_model : nn.Module
        self.epsilon = torch.Tensor([config['epsilon']])
        self.eta     = torch.Tensor([config['eta']])
        self.bce     = nn.BCELoss()
        self.optFortheta = optim.Adam(rec_model.parameters(), lr=rec_lr)
        self.optForvar   = optim.Adam(var_model.parameters(), lr=var_lr)
        
    def stageTwo(self, rating, gamma, xij, pij=None):
        """
        using the same data as stage one.
        we have r_{ij} \propto p_{ij}, so we need to divide pij for unbiased gradient.
        unbiased_loss = BCE(xij, rating_ij)*len(batch)
                        + (1-gamma_ij)/gamma_ij * cross(xij, epsilon)
                        + (cross(gamma_ij, eta_ij) + cross(gamma_ij, gamma_ij))/r_ij
        """
        rating : torch.Tensor
        gamma  : torch.Tensor
        xij    : torch.Tensor
        
        try:
            assert rating.size() == gamma.size() == xij.size()
            assert len(rating.size()) == 1
            if pij is not None:
                assert rating.size() == pij.size()
        except ValueError:
            print("input error!")
            print(f"Got rating{rating.size()}, gamma{gamma.size()}, xij{xij.size()}")
        gamma = gamma + self.eps
        if pij is None:
            pij = gamma.detach()
            
        same_term   = log(( 1-self.eta + self.eps)/(1-gamma+2*self.eps))
        part_one    = xij*log( (rating + self.eps)/self.epsilon ) \
                        + (1-xij)*log( (1-rating + self.eps)/(1-self.epsilon) ) \
                        + log(self.eta/gamma) \
                        - same_term
        out_one  = (part_one*gamma).detach()
        part_one    = part_one*gamma/pij 
        
        part_two = (self.cross(xij, self.epsilon) + same_term)
        out_two  = part_two.detach()
        part_two = part_two/pij
        
        out_loss = -(torch.sum(out_one) + torch.sum(out_two))
        
        loss1       = torch.sum(part_one)
        loss2       = torch.sum(part_two)
        loss: torch.Tensor = -(loss1+loss2)
        
        if torch.isnan(loss) or torch.isinf(loss):
            print("loss1:", loss1)
            print("loss2:", loss2)
            raise ValueError("nan or inf")
        cri = out_loss.item()
        
        self.optForvar.zero_grad()
        loss.backward()
        self.optForvar.step()

        return cri
             
    def stageOne(self, rating, xij, gamma=None,pij=None):
        """
        optimize recommender parameters
        same as samwalk, using BCE loss here
        p(user_i,item_j) = p(item_j|user_i)p(user_i).And p(user_i) \propto 1
        so if we divide loss by pij, then the rij coefficient will be eliminated.
        And we get BCE loss here(without `mean`)
        """
        if pij is None:
            loss: torch.Tensor = self.bce(rating, xij)*len(rating)
        else:
            if gamma is None:
                raise ValueError("You should input gamma and pij in the same time  \
                                 to calculate the loss for recommendation model")
            assert pij.size() == gamma.size()
            pij = pij + self.eps
            part_one = ELBO.cross(xij, rating)
            part_one = part_one*gamma/pij
            loss : torch.Tensor = -torch.sum(part_one)
        cri = loss.data
        self.optFortheta.zero_grad()
        loss.backward()
        self.optFortheta.step()
        return cri
        
    @staticmethod
    def cross(a, b):
        a : torch.Tensor
        b : torch.Tensor
        return a*torch.log(b + ELBO.eps) + (1-a)*torch.log(1-b + ELBO.eps) 

class BCE:
    """
    warp bce loss
    """
    def __init__(self, rec_model, lr=0.05):
        self.bce     = nn.BCELoss()
        self.model   = rec_model
        self.opt     = optim.Adam(rec_model.parameters(), lr=lr)
    
    def stageOne(self, rating, xij):
        loss = self.bce(rating, xij)
        cri = loss.data
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        return cri
        

        
# ==================Samplers=======================
# =================================================
# TODO


class Sample_MF:
    """
    implement the sample procedure \n
    we have: \n
        items: (m, d)
        user: (n, d)
        gamma: (n,m)
        D_k: sum(items, axis=0) => (d)
        S_i: sum(gamma, axis=1) => (n)
    NOTE:
        consider the huge dataset we may have, when calculate the probability,
        we turn off the gradient recording function, only return users and items list. 
        
    """
    def __init__(self, k, var_model:VarMF_reg):
        self.k = k
        self.model = var_model
        self.__compute = False
        self.__prob = {}
            
    def setK(self,k):
        self.k = k
    
    def sampleForEpoch(self, epoch_k, notcompute=False):
        """
        sample pairs for a whole epoch, `epoch_k` samples.
        the strategy changes
        """
        if self.__compute:
            pass
        elif notcompute and len(self.__prob) != 0:
            pass
        else:
            self.compute()
        self.__compute = False
        expected_Users = self.__prob['p(i)']*epoch_k
        Userbase       = expected_Users.int() 
        # int() will floor the numbers, like 1.4 -> 1
        AddOneProbs    = expected_Users - Userbase
        AddOnes        = torch.bernoulli(AddOneProbs)
        expected_Users = Userbase + AddOnes
        expected_Users = expected_Users.int()
        Samusers = None
        Samitems = None
        for user_id, sample_times in enumerate(expected_Users):
            items = self.sampleForUser(user_id, times=sample_times)
            users = torch.Tensor([user_id]*sample_times).long()
            if Samusers is None:
                Samusers = users
                Samitems = items
            else:
                Samusers = torch.cat([Samusers, users])
                Samitems = torch.cat([Samitems, items])
        return Samusers, Samitems
        
    def sampleForUser(self, user, times=1):
        candi_dim = self.__prob['p(k|i)'][user].squeeze()
        dims = torch.multinomial(candi_dim, times,replacement=True)
        candi_items = self.__prob['p(j|k)'][dims]
        items = torch.multinomial(candi_items, 1).squeeze(dim=1)
        return items.long()        

        
    
    def compute(self):
        """
        compute everything we need in the sampling.        
        """
        with torch.no_grad():
            u_emb : torch.Tensor = self.model.getAllUsersEmbedding() # shape (n,d)
            i_emb : torch.Tensor = self.model.getAllItemsEmbedding() # shape (m,d)
            # gamma = torch.matmul(u_emb, i_emb.t()) # shape (n,m)
            D_k = torch.sum(i_emb, dim=0) # shape (d)
            S_i = torch.sum(u_emb*D_k, dim=1) # shape (n)
            # S_i = torch.sum(gamma, dim=1) # shape (n)
            p_i = S_i/torch.sum(S_i) # shape (n)
            p_jk = (i_emb/D_k).t()  # shape (d,m)
            p_ki = ((u_emb*D_k)/S_i.unsqueeze(dim=1)) # shape (n, d)
            self.__compute = True
            self.__prob['u_emb']  = u_emb
            self.__prob['i_emb']  = i_emb
            # self.__prob['gamma']  = gamma
            self.__prob['D_k']    = D_k
            self.__prob['S_i']    = S_i
            self.__prob['p(i)']   = p_i
            self.__prob['p(k|i)'] = p_ki
            self.__prob['p(j|k)'] = p_jk
    
    def sample(self, notcompute=False):
        """
        sample self.k samples
        """
        if self.__compute:
            pass
        elif notcompute and len(self.__prob) != 0:
            pass
        else:
            self.compute()
        self.__compute = False
        users = torch.multinomial(self.__prob['p(i)'], self.k, replacement=True)
        candi_dim = self.__prob['p(k|i)'][users]
        dims = torch.multinomial(candi_dim, 1)
        dims = dims.squeeze(dim=1)
        candi_items = self.__prob['p(j|k)'][dims]
        items = torch.multinomial(candi_items, 1).squeeze(dim=1)
        # print(users.size(), items.size())
        return users, items


def UniformSample(users, dataset, k=1):
    """
    uniformsample k negative items and one positive item for one user
    return:
        np.array
    """
    dataset : BasicDataset
    allPos   = dataset.getUserPosItems(users)
    allNeg   = dataset.getUserNegItems(users)
    allItems = list(range(dataset.m_items))
    S = []
    for i, user in enumerate(users):
        posForUser = allPos[i]
        # negForUser = dataset.getUserNegItems([user])[0]
        negForUser = allNeg[i]
        onePos     = np.random.choice(posForUser, size=(1, ))
        kNeg       = np.random.choice(negForUser, size=(k, ))
        S.append(np.hstack([onePos, kNeg]))
    return np.array(S)
        
# ===================end samplers==========================
# =========================================================


def minibatch(*tensors, **kwargs):

    batch_size = kwargs.get('batch_size', world.config['batch_size'])

    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)


def shuffle(*arrays, **kwargs):

    require_indices = kwargs.get('indices', False)

    if len(set(len(x) for x in arrays)) != 1:
        raise ValueError('All inputs to shuffle must have '
                         'the same length.')

    shuffle_indices = np.arange(len(arrays[0]))
    np.random.shuffle(shuffle_indices)

    if len(arrays) == 1:
        result = arrays[0][shuffle_indices]
    else:
        result = tuple(x[shuffle_indices] for x in arrays)

    if require_indices:
        return result, shuffle_indices
    else:
        return result

# ====================Metrics==============================
# =========================================================

def recall_precisionATk(test_data, pred_data, k=5):
    """
    test_data should be a list? cause users may have different amount of pos items. shape (test_batch, k)
    pred_data : shape (test_batch, k) NOTE: pred_data should be pre-sorted
    k : top-k
    """
    assert len(test_data) == len(pred_data)
    recall_n    = []
    precis_n    = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK= pred_data[i][:k]
        bingo      = list(filter(lambda x: x in groundTrue, predictTopK))
        right_items = len(bingo)
        precis_n.append(float(bingo)/len(k))
        recall_n.append(float(bingo)/len(groundTrue))
    return {'recall': np.mean(recall_n), 'precision': np.mean(precis_n)}

def MRRatK(test_data, pred_data, k):
    """
    Mean Reciprocal Rank
    """
    MRR_n = len(test_data)
    scores = 0.
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        prediction = pred_data[i]
        for j, item in enumerate(prediction):
            if j >= k:
                break
            if item in groundTrue:
                scores += 1/(j+1)
                break
            
    return scores/MRR_n

def NDCGatK(test_data, pred_data, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    NOTE implementation is slooooow
    """
    pred_rel = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        pred = list(map(lambda x: x in groundTrue, pred_data[i][:k]))
        pred = np.array(pred).astype("float")
        # print(pred)
        pred_rel.append(pred)
    pred_rel = np.array(pred_rel)
    coefficients = np.log2(np.arange(2, k+2))
    # print(coefficients.shape, pred_rel.shape)
    # print(coefficients)
    assert len(coefficients) == pred_rel.shape[-1]
    
    pred_rel = pred_rel/coefficients
    idcg = np.sum(1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(pred_rel, axis=1)
    ndcg = dcg/idcg
    return np.mean(ndcg)
# ====================end Metrics=============================
# =========================================================


if __name__ == "__main__":
    config = {'epsilon':0.001, 'eta':0.5}
    test_rating = torch.rand(10,1)
    test_gamma  = torch.rand(10,1)
    test_xij    = torch.rand(10,1)
    # loss_test = ELBO(config)

    from pprint import pprint
    world.config['num_users'] = 4000
    world.config['num_items'] = 8000
    text_sample = VarMF_reg(world.config)
    sampler = Sample_MF(k=10, var_model=text_sample)
    for i in range(10):
        sampler.compute()
        users, items = sampler.sample()
        print("==")
        pprint(users)
        pprint(items)

        
