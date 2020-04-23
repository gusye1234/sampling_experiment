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

    def __init__(self, config, rec_model, var_model):
        rec_model: nn.Module
        var_model: nn.Module
        self.epsilon = torch.Tensor([config['epsilon']]).to(world.device)
        self.bce = nn.BCELoss()
        rec_lr = config['rec_lr']
        var_lr = config['var_lr']
        self.exprior = config['ex_prior']
        print('exprior', self.exprior)

        self.optFortheta = optim.Adam(rec_model.parameters(), lr=rec_lr, weight_decay=config['rec_weight_decay'])
        self.optForvar = optim.Adam(var_model.parameters(), lr=var_lr)
        print('ELBO Init!')

    def stageTwoPrior(self, rating, gamma, xij, pij=None, reg_loss=None):
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

        eta = xij * 1.0 + (1 - xij) * self.exprior
        print('eta', eta)
        same_term = log((1 - eta + self.eps) / (1 - gamma + 2 * self.eps))
        part_one = xij * log((rating + self.eps) / self.epsilon) \
                   + (1 - xij) * log((1 - rating + self.eps) / (1 - self.epsilon)) \
                   + log(eta / gamma) \
                   - same_term

        part_two = (self.cross(xij, self.epsilon) + same_term)

        if pij is None:
            part_one = part_one * gamma
            part_two = part_two

        else:
            print('Pro to gamma stagetwo!')
            pij = (pij + self.eps).detach()
            part_one = part_one * gamma / pij
            part_two = part_two / pij

        loss1 = torch.sum(part_one)
        loss2 = torch.sum(part_two)
        loss: torch.Tensor = -(loss1 + loss2)
        cri = loss.data

        if reg_loss is not None:
            print('loss regloss', loss, reg_loss)
            loss = loss + reg_loss
            print('reg+loss', loss)

        if torch.isnan(loss) or torch.isinf(loss):
            print('part_one:', part_one)
            print('part_two:', part_two)
            print("loss1:", loss1)
            print("loss2:", loss2)
            raise ValueError("nan or inf")


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
            print('pro to gamma stageone!')
            part_one = self.cross(xij, rating)
            loss: torch.Tensor = -torch.sum(part_one)

        cri = loss.data
        self.optFortheta.zero_grad()
        loss.backward()
        self.optFortheta.step()
        return cri


    @staticmethod
    def cross(a, b):
        a: torch.Tensor
        b: torch.Tensor
        return a * torch.log(b + ELBO.eps) + (1 - a) * torch.log(1 - b + ELBO.eps)


# ==================Samplers=======================
# =================================================


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

# ====================End Samplers=============================
# =========================================================


# ===================Train Tools===========================
# =========================================================

def set_seed(seed):
    np.random.seed(seed)   
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)


# ====================End Train Tools=============================
# =========================================================


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

