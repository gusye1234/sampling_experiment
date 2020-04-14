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
from model import VarMF_reg, RecMF
from time import time


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
    eps = torch.Tensor([1e-8]).float().to(world.device)

    def __init__(self, config,
                 rec_model, var_model):
        rec_model: nn.Module
        var_model: nn.Module
        self.epsilon = torch.Tensor([config['epsilon']]).to(world.device)
        self.eta = torch.Tensor([config['eta']]).to(world.device)
        self.bce = nn.BCELoss()
        rec_lr = config['rec_lr']
        var_lr = config['var_lr']
        self.gamma_KL = config['gamma_KL']
        self.optFortheta = optim.Adam(rec_model.parameters(), lr=rec_lr, weight_decay=config['rec_weight_decay'])
        self.optForvar = optim.Adam(var_model.parameters(), lr=var_lr)
        #self.no_var_decay = world.var_type.startswith('lgn')
          
        #if self.no_var_decay:
            #self.optForvar = optim.Adam(var_model.parameters(), lr=var_lr)    
        #else:
            #print('optimizer weight decay!')
            #self.optForvar = optim.Adam(var_model.parameters(), lr=var_lr, weight_decay=config['var_weight_decay'])

    def basic_GMF_loss(self, rating, xij):
        rating: torch.Tensor
        xij: torch.Tensor
        try:
            assert rating.size() == xij.size()
            assert len(rating.size()) == 1
        except ValueError:
            print("input error!")
            print(f"Got rating{rating.size()}, gamma{gamma.size()}, xij{xij.size()}")
        loss: torch.Tensor = self.bce(rating, xij) * len(rating)
        self.optFortheta.zero_grad()
        loss.backward()
        self.optFortheta.step()
        return loss.data


    def stageTwo_Prior(self, rating, gamma, xij, pij=None, reg_loss = None):
        """
        using the same data as stage one.
        we have r_{ij} \propto p_{ij}, so we need to divide pij for unbiased gradient.
        unbiased_loss = BCE(xij, rating_ij)*len(batch)
                        + (1-gamma_ij)/gamma_ij * cross(xij, epsilon)
                        + (cross(gamma_ij, eta_ij) + cross(gamma_ij, gamma_ij))/r_ij
        """
        rating: torch.Tensor
        gamma: torch.Tensor
        xij: torch.Tensor

        try:
            assert rating.size() == gamma.size() == xij.size()
            assert len(rating.size()) == 1
            if pij is not None:
                assert rating.size() == pij.size()
        except ValueError:
            print("input error!")
            print(f"Got rating{rating.size()}, gamma{gamma.size()}, xij{xij.size()}")

        gamma = gamma + self.eps

        
        print('gamma', gamma[:10])
        part_one =  self.cross(xij, rating)
        part_two =  self.cross(xij, self.epsilon)
        part_tre =  gamma*(log(self.eta)-log(gamma)) + (1-gamma)*(log(1-self.eta)-log(1-gamma + 2*ELBO.eps))
        print('part', part_one[:10], part_two[:10], part_tre[:10])
        if pij is None:      
            part_one = part_one * gamma
            part_two = part_two * (1 - gamma)
            print('part gamma', part_one[:10], part_two[:10],part_tre[:10])
            print('stagetwoNoPrior pij None!!!')
        else:
            pij = (pij + self.eps).detach()
            part_one = part_one * gamma / pij
            part_two = part_two * (1 - gamma)/ pij
            part_tre = part_tre/pij
        out_loss = -(torch.sum(part_one) + torch.sum(part_two)+self.gamma_KL*torch.sum(part_tre)).data

        loss1 = torch.sum(part_one)
        loss2 = torch.sum(part_two)
        loss3 = torch.sum(part_tre)
        print('loss1 2 3', loss1, loss2, loss3)
        print(self.gamma_KL)
        loss: torch.Tensor = -(loss1 + loss2 + self.gamma_KL*loss3)
        print('gamma_KL', self.gamma_KL)
        print('loss', loss)
        
        if reg_loss is not None:
            print(loss, reg_loss)
            loss = loss + reg_loss
            print('reg+loss', loss)
            
        
        if torch.isnan(loss) or torch.isinf(loss):
            print('part_one:', part_one)
            print('part_two:', part_two)
            print("loss1:", loss1)
            print("loss2:", loss2)
            raise ValueError("nan or inf")
        
        cri = out_loss

        self.optForvar.zero_grad()
        loss.backward()
        self.optForvar.step()

        return cri


    def stageOne(self, rating, xij, gamma=None, pij=None):
        """
        optimize recommendation parameters
        same as samwalk, using BCE loss here
        p(user_i,item_j) = p(item_j|user_i)p(user_i).And p(user_i) \propto 1
        so if we divide loss by pij, then the rij coefficient will be eliminated.
        And we get BCE loss here(without `mean`)
        """
        rating: torch.Tensor
        gamma: torch.Tensor
        xij: torch.Tensor

        if pij is None:
            if gamma is None:
                raise ValueError("You should input gamma and pij in the same time  \
                                 to calculate the loss for recommendation model")
            part_one = self.cross(xij, rating)
            part_one = part_one * gamma.detach()
            loss: torch.Tensor = -torch.sum(part_one)
            print('stageone pij None')

        else:
            pij = (pij + self.eps).detach()
            part_one = self.cross(xij, rating)
            part_one = part_one * gamma.detach() / pij
            loss: torch.Tensor = -torch.sum(part_one)

        cri = loss.data
        self.optFortheta.zero_grad()
        loss.backward()
        self.optFortheta.step()
        return cri

    def stageTwo(self, rating, gamma, xij, pij=None, reg_loss = None):
        """
        using the same data as stage one.
        we have r_{ij} \propto p_{ij}, so we need to divide pij for unbiased gradient.
        unbiased_loss = BCE(xij, rating_ij)*len(batch)
                        + (1-gamma_ij)/gamma_ij * cross(xij, epsilon)
                        + (cross(gamma_ij, eta_ij) + cross(gamma_ij, gamma_ij))/r_ij
        """
        rating: torch.Tensor
        gamma: torch.Tensor
        xij: torch.Tensor

        try:
            assert rating.size() == gamma.size() == xij.size()
            assert len(rating.size()) == 1
            if pij is not None:
                assert rating.size() == pij.size()
        except ValueError:
            print("input error!")
            print(f"Got rating{rating.size()}, gamma{gamma.size()}, xij{xij.size()}")

        gamma = gamma + self.eps

        same_term = log((1 - self.eta + self.eps) / (1 - gamma + 2 * self.eps))
        part_one = xij * log((rating + self.eps) / self.epsilon) \
                   + (1 - xij) * log((1 - rating + self.eps) / (1 - self.epsilon)) \
                   + log(self.eta / gamma) \
                   - same_term
        out_one = (part_one * gamma).detach()
        part_two = (self.cross(xij, self.epsilon) + same_term)
        out_two = part_two.detach()
        if pij is None:      
            part_one = part_one * gamma
            print('gamma', gamma[:1000])       
            part_two = part_two
            print('stagetwo pij None')
        else:
            pij = (pij + self.eps).detach()
            part_one = part_one * gamma / pij
            part_two = part_two / pij

        out_loss = -(torch.sum(out_one) + torch.sum(out_two))

        loss1 = torch.sum(part_one)
        loss2 = torch.sum(part_two)
        loss: torch.Tensor = -(loss1 + loss2)
        
        if reg_loss is not None:
            print(loss, reg_loss)
            loss = loss + reg_loss
            print('reg+loss', loss)
            
        
        if torch.isnan(loss) or torch.isinf(loss):
            print('part_one:', part_one)
            print('part_two:', part_two)
            print("loss1:", loss1)
            print("loss2:", loss2)
            raise ValueError("nan or inf")
        cri = out_loss.item()

        self.optForvar.zero_grad()
        loss.backward()
        self.optForvar.step()

        return cri

    @staticmethod
    def cross(a, b):
        a: torch.Tensor
        b: torch.Tensor
        return a * torch.log(b + ELBO.eps) + (1 - a) * torch.log(1 - b + ELBO.eps)


# ==================Samplers=======================
# =================================================


class Sample_positive_all:
    def __init__(self, dataset, var_model, prob=0.3):
        self.positive = prob
        self.model = var_model
        self.dataset = dataset
        self.__compute = False
        self.__prob = {}
        self.G = 0

    def sample_prob(self, gamma):
        pij = gamma[self.epoch_k_1:] / self.G * (1 - self.positive)
        pij = torch.cat([torch.tensor([1.0 / self.dataset.trainDataSize * self.positive] * self.epoch_k_1), pij])
        return pij

    def sampleForEpoch(self, epoch_k):
        self.epoch_k_1 = torch.tensor(self.positive * epoch_k)
        base = self.epoch_k_1.int()
        # int() will floor the numbers, like 1.4 -> 1
        AddOneProbs = self.epoch_k_1 - base
        # print(AddOneProbs)
        AddOnes = torch.bernoulli(AddOneProbs)
        self.epoch_k_1 = (base + AddOnes).int()
        self.epoch_k_2 = epoch_k - self.epoch_k_1
        res_users1, res_items1 = self.samplePositive()
        res_users2, res_items2 = self.sampleAll()
        res_users = torch.cat([res_users1, res_users2])
        res_items = torch.cat([res_items1, res_items2])
        return res_users, res_items

    def samplePositive(self):
        l = len(self.dataset.trainUser)
        candi_set = range(l)
        # print('len', len(self.dataset.trainUser))
        index = np.random.choice(candi_set, np.array(self.epoch_k_1), replace=True)
        users = self.dataset.trainUser.reshape(-1)[index]
        items = self.dataset.trainItem.reshape(-1)[index]
        # p_pos = torch.tensor([1.0/l]*epoch_k_1)
        return torch.tensor(users), torch.tensor(items)

    def sampleAll(self, notcompute=True):
        print(self.__prob, self.__compute)
        if self.__compute:
            pass
        elif notcompute and len(self.__prob) != 0:
            pass
        else:
            self.compute()
            print('have been computed')
        self.__compute = False
        expected_Users = self.__prob['p(i)'] * self.epoch_k_2
        Userbase = expected_Users.int()
        # int() will floor the numbers, like 1.4 -> 1
        AddOneProbs = expected_Users - Userbase
        # print(AddOneProbs)
        AddOnes = torch.bernoulli(AddOneProbs)
        expected_Users = Userbase + AddOnes
        expected_Users = expected_Users.int()
        Samusers = None
        Samitems = None
        for user_id, sample_times in enumerate(expected_Users):
            items = self.sampleForUser(user_id, times=sample_times)
            users = torch.Tensor([user_id] * sample_times).long()
            if Samusers is None:
                Samusers = users
                Samitems = items
            else:
                Samusers = torch.cat([Samusers, users])
                Samitems = torch.cat([Samitems, items])
        self.__compute = False
        self.__prob.clear()
        # p_all = torch.tensor([self.G]*epoch_k_2)
        return Samusers, Samitems

    def sampleForUser(self, user, times=1):
        candi_dim = self.__prob['p(k|i)'][user].squeeze()

        dims = torch.multinomial(candi_dim, times, replacement=True)
        candi_items = self.__prob['p(j|k)'][dims]
        items = torch.multinomial(candi_items, 1).squeeze(dim=1)
        return items.long()

    def compute(self):
        """
        compute everything we need in the sampling.
        """
        with torch.no_grad():
            u_emb: torch.Tensor = self.model.getAllUsersEmbedding()  # shape (n,d)
            i_emb: torch.Tensor = self.model.getAllItemsEmbedding()  # shape (m,d)
            # if self.save == True:
            # np.savetxt('get.txt', np.array(u_emb.detach()))
            # self.save = False
            # gamma = torch.matmul(u_emb, i_emb.t()) # shape (n,m)
            D_k = torch.sum(i_emb, dim=0)  # shape (d)
            S_i = torch.sum(u_emb * D_k, dim=1)  # shape (n)
            # S_i = torch.sum(gamma, dim=1) # shape (n)
            print(self.G)
            self.G = torch.sum(S_i)
            print(self.G)
            p_i = S_i / self.G  # shape (n)
            p_jk = (i_emb / D_k).t()  # shape (d,m)
            p_ki = ((u_emb * D_k) / S_i.unsqueeze(dim=1))  # shape (n, d)

            self.__compute = True
            self.__prob['u_emb'] = u_emb
            self.__prob['i_emb'] = i_emb
            # self.__prob['gamma']  = gamma
            self.__prob['D_k'] = D_k
            self.__prob['S_i'] = S_i
            self.__prob['p(i)'] = p_i
            print('pi', self.__prob['p(i)'])
            self.__prob['p(k|i)'] = p_ki
            print('pik', self.__prob['p(k|i)'])
            self.__prob['p(j|k)'] = p_jk
            print('pkj', self.__prob['p(j|k)'])

class sample_for_basic_GMF_loss:
    def __init__(self, k):
        self.k = k

    def sampleForEpoch(self, dataset, k=3):
        dataset: BasicDataset
        allPosItems = dataset.allPos
        posSamusers = None
        posSamitems = None
        for userID, posItems in enumerate(allPosItems):
            items = torch.tensor(posItems)
            users = torch.tensor([userID] * len(items)).long()
            # print(users)
            if posSamusers is None:
                posSamusers = users
                posSamitems = items
            else:
                posSamusers = torch.cat([posSamusers, users])
                posSamitems = torch.cat([posSamitems, items])

        negSamusers = None
        negSamitems = None
        itemNumUser = int(dataset.trainDataSize / dataset.n_users * k)
        allNegItems = dataset.allNeg
        for userID, negItems in enumerate(allNegItems):
            items = torch.tensor(np.random.choice(negItems, size=itemNumUser, replace=True))
            users = torch.tensor([userID] * itemNumUser).long()
            if negSamusers is None:
                negSamusers = users
                negSamitems = items
            else:
                negSamusers = torch.cat([negSamusers, users])
                negSamitems = torch.cat([negSamitems, items])

        # return posSamusers, posSamitems, negSamusers, negSamitems
        return torch.cat([posSamusers.long(), negSamusers.long()]), torch.cat([posSamitems.long(), negSamitems.long()])


def UniformSample(users, dataset, k=1):
    """
    uniformsample k negative items and one positive item for one user
    return:
        np.array
    """
    dataset: BasicDataset
    allPos = dataset.getUserPosItems(users)
    allNeg = dataset.getUserNegItems(users)
    # allItems = list(range(dataset.m_items))
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    total_start = time()
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[i]
        # negForUser = dataset.getUserNegItems([user])[0]
        negForUser = allNeg[i]
        sample_time2 += time() - start

        start = time()
        onePos_index = np.random.randint(0, len(posForUser))
        onePos = posForUser[onePos_index:onePos_index + 1]
        # onePos     = np.random.choice(posForUser, size=(1, ))
        kNeg_index = np.random.randint(0, len(negForUser), size=(k,))
        kNeg = negForUser[kNeg_index]
        end = time()
        sample_time1 += end - start
        S.append(np.hstack([onePos, kNeg]))
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]


def UniformSample_allpos(users, dataset, k=4):
    """
    uniformsample k negative items and one positive item for one user
    return:
        np.array
    """
    dataset: BasicDataset
    allPos = dataset.getUserPosItems(users)
    allNeg = dataset.getUserNegItems(users)
    # allItems = list(range(dataset.m_items))
    S = []
    sample_time1 = 0.
    sample_time2 = 0.
    total_start = time()
    for i, user in enumerate(users):
        start = time()
        posForUser = allPos[i]
        # negForUser = dataset.getUserNegItems([user])[0]
        negForUser = allNeg[i]
        sample_time2 += time() - start

        for positem in posForUser:
            start = time()
            # onePos_index = np.random.randint(0, len(posForUser))
            # onePos     = posForUser[onePos_index:onePos_index+1]
            # onePos     = np.random.choice(posForUser, size=(1, ))
            kNeg_index = np.random.randint(0, len(negForUser), size=(k,))
            kNeg = negForUser[kNeg_index]
            end = time()
            sample_time1 += end - start
            for negitemForpos in kNeg:
                S.append([user, positem, negitemForpos])
            # S.append(np.hstack([onePos, kNeg]))
    total = time() - total_start
    return np.array(S), [total, sample_time1, sample_time2]


def getAllData(dataset, gamma=None):
    """
    return all data (n_users X m_items)
    return:
        [u, i, x_ui]
    """
    # if gamma is not None:
    #     print(gamma.size())
    dataset: BasicDataset
    users = []
    items = []
    xijs = None
    allPos = dataset.allPos
    allxijs = np.array(dataset.UserItemNet.todense()).reshape(-1)
    items = np.tile(np.arange(dataset.m_items), (1, dataset.n_users)).squeeze()
    users = np.tile(np.arange(dataset.n_users), (dataset.m_items, 1)).T.reshape(-1)
    print(len(allxijs), len(items), len(users))
    assert len(allxijs) == len(items) == len(users)
    # for user in range(dataset.n_users):
    #     users.extend([user]*dataset.m_items)
    #     items.extend(range(dataset.m_items))
    if gamma is not None:
        return torch.Tensor(users).long(), torch.Tensor(items).long(), torch.from_numpy(allxijs).long(), gamma.reshape(
            -1)
    return torch.Tensor(users).long(), torch.Tensor(items).long(), torch.from_numpy(allxijs).long()


# ===================end samplers==========================
# =========================================================

def set_seed(seed):
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

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
    right_items = 0
    recall_n = 0
    precis_n = len(test_data) * k
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i][:k]
        bingo = list(filter(lambda x: x in groundTrue, predictTopK))
        right_items += len(bingo)
        recall_n += len(groundTrue)
    return {'recall': right_items / recall_n, 'precision': right_items / precis_n}


def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.mean(right_pred / recall_n)
    precis = np.mean(right_pred) / precis_n
    return {'recall': recall, 'precision': precis}


def NDCGatK_r(r, k):
    pred_data = r[:, :k]
    idcg = np.sum(1. / np.log2(np.arange(2, k + 2)))
    dcg = pred_data * (1. / np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    return np.mean(dcg) / idcg


def MRRatK_r(r, k):
    pred_data = r[:, :k]
    scores = 1. / np.arange(1, k + 1)
    pred_data = pred_data / scores
    pred_data = pred_data.sum(1)
    return np.mean(pred_data)


def getLabel(test_data, pred_data):
    r = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        r.append(pred)
    return np.array(r)


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
                scores += 1 / (j + 1)
                break

    return scores / MRR_n


def NDCGatK(test_data, pred_data, k):
    """
    Normalized Discounted Cumulative Gain
    rel_i = 1 or 0, so 2^{rel_i} - 1 = 1 or 0
    NOTE implementation is slooooow
    """
    pred_rel = []
    idcg = []
    for i in range(len(test_data)):
        groundTrue = test_data[i]
        predictTopK = pred_data[i][:k]
        pred = list(map(lambda x: x in groundTrue, predictTopK))
        pred = np.array(pred).astype("float")
        pred_rel.append(pred)

        if len(groundTrue) < k:
            coeForIdcg = np.log2(np.arange(2, len(groundTrue) + 2))
        else:
            coeForIdcg = np.log2(np.arange(2, k + 2))

        idcgi = np.sum(1. / coeForIdcg)
        idcg.append(idcgi)
        # print(pred)

    pred_rel = np.array(pred_rel)
    idcg = np.array(idcg)
    coefficients = np.log2(np.arange(2, k + 2))
    # print(coefficients.shape, pred_rel.shape)
    # print(coefficients)
    assert len(coefficients) == pred_rel.shape[-1]

    pred_rel = pred_rel / coefficients
    dcg = np.sum(pred_rel, axis=1)
    ndcg = dcg / idcg
    return np.mean(ndcg)


# ====================end Metrics=============================
# =========================================================


if __name__ == "__main__":
    config = {'epsilon': 0.001, 'eta': 0.5}
    test_rating = torch.rand(10, 1)
    test_gamma = torch.rand(10, 1)
    test_xij = torch.rand(10, 1)
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
