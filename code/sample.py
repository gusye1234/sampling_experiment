import world
import torch
from torch import nn, optim
import numpy as np
from torch import log
from dataloader import BasicDataset
from model import VarMF_reg, RecMF, LightGCN_xij_item_personal_matrix
from time import time


class SamplePersonal:
    def __init__(self,
                 varmodel:LightGCN_xij_item_personal_matrix,
                 dataset: BasicDataset):
        self.varmodel = varmodel
        self.dataset = dataset
        #self.allPos = dataset.allPos
        #self.allNeg = dataset.allNeg
        self.staPosUsers = dataset.posSampleUser
        self.staPosItems = dataset.posSampleItem
        self.numPosUsers = dataset.numPosUsers
        self.posSamplesNum = len(self.staPosUsers)
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.__prob = {}
        print("fast sample")

    def compute(self, epochk):
        with torch.no_grad():
            (user_emb, items_emb_0, items_emb_1) = self.varmodel.get_user_item_embedding()
            user_emb = user_emb.cpu()
            items_emb_0 = items_emb_0.cpu()
            items_emb_1 = items_emb_1.cpu()
            user_emb_pos = user_emb[self.staPosUsers]
            items_emb_pos_0 = items_emb_0[self.staPosItems]
            items_emb_pos_1 = items_emb_1[self.staPosItems]
            self.gammaPerPosSample = torch.mul(user_emb_pos, items_emb_pos_1).sum(1)
            #gammaPerUserPosSum = torch.zeros(self.n_users)
            #gammaPerUserPosSum = gammaPerUserPosSum.scatter_add_(0, self.staPosUsers, gammaPerPosSample)

        Sqk0 = torch.sum(items_emb_0, dim=0)
        staPosUsersTensor = self.staPosUsers.reshape(-1, 1).expand(-1, 21)

        Sqpik1 = torch.zeros(self.n_users, 21)
        Sqpik0 = torch.zeros(self.n_users, 21)
        Sqpik1.scatter_add_(0, staPosUsersTensor, items_emb_pos_1)
        Sqpik0.scatter_add_(0, staPosUsersTensor, items_emb_pos_0)
        Spi0 = torch.matmul(Sqk0, user_emb.t())


        Sppi1 = torch.mul(user_emb, Sqpik1).sum(1)
        Sppi0 = torch.mul(user_emb, Sqpik0).sum(1)
        Spni0 = Spi0 - Sppi0
        Spi = Sppi1 + Spi0 - Sppi0
        Sp = torch.sum(Spi)
        #print('Sqpik1 Sqpik0 Spi0 Sppi1 Sppi0 Spni0 Spi', Sqpik1.size(), Sqpik0.size(), Spi0.size(), Sppi1.size(), Sppi0.size(), Spni0.size(), Spi.size())
        self.__prob['negtrue/neg0'] = Spi0 / Spni0
        self.__prob['p(i)'] = Spi / Sp
        self.__prob['p(pos|i)'] = Sppi1 / Spi
        self.__prob['p(neg|i)'] = 1 - self.__prob['p(pos|i)']
        self.__prob['p(pos,i)'] = self.__prob['p(pos|i)']*self.__prob['p(i)']
        self.__prob['p(neg,i)'] = self.__prob['p(neg|i)']*self.__prob['p(i)']

        self.__prob['p(j|k)neg'] = (items_emb_0 / Sqk0).t()
        self.__prob['p(k|i)neg'] = (user_emb*Sqk0) / Spi0.unsqueeze(dim=1)

        self.numPerUserSample = self.round(self.__prob['p(i)'] * epochk)
        self.numPerUserPos = self.round(self.__prob['p(pos,i)'] * epochk)
        numPerUserNeg = self.numPerUserSample - self.numPerUserPos
        self.numPerUserNegTruth = self.round(numPerUserNeg * self.__prob['negtrue/neg0'])
        return Sp

    def sample(self, epochk):
        samplestart = time()
        compute_start = time()
        G = self.compute(epochk)
        print('compute time', time()-compute_start)

        Samusers = None
        Samitems = None
        Samxijs = None
        #print('sample start!!!')
        self.countPos = 0
        self.countNeg = 0
        candi_neg_items = torch.multinomial(self.__prob['p(j|k)neg'], epochk, replacement=True)
        all_sample_times = self.numPerUserSample
        #print('epochk', all_sample_times)
        for user_id, sample_times in enumerate(all_sample_times):
            if sample_times == 0:
                continue
            items, xijs = self.sampleForUser(user_id, candi_neg_items)
            users = torch.Tensor([user_id] * len(items)).long()
            if Samusers is None:
                Samusers = users
                Samitems = items
                Samxijs = xijs
            else:
                Samusers = torch.cat([Samusers, users])
                Samitems = torch.cat([Samitems, items])
                Samxijs = torch.cat([Samxijs, xijs])

        self.__prob.clear()
        print('samtime', time()-samplestart)
        return Samusers, Samitems, Samxijs, G.item()

    def sampleForUser(self, user, candi_neg_items):
        posi = self.numPerUserPos[user]
        #print('user', user)
        if posi == 0:
            posItems = None
        else:
            xijs1 = torch.tensor([1] * posi.item())
            posPerUserNum = self.numPosUsers[user]
            posGammaForUser = self.gammaPerPosSample[self.countPos:  self.countPos+posPerUserNum]
            #print('index', self.countPos, self.countPos+posPerUserNum)
            #print('posGammaForUser', posGammaForUser)

            posIndex = torch.multinomial(posGammaForUser, posi, replacement=True)
            #print('posIndex', posIndex)
            #print(self.staPosItems[self.countPos:self.countPos+posPerUserNum])
            posItems = self.staPosItems[self.countPos:self.countPos+posPerUserNum][posIndex]
            #print('posItems', posItems)
            self.countPos = self.countPos + posPerUserNum
        # =======================================
        negi = self.numPerUserNegTruth[user]
        candi_dim = self.__prob['p(k|i)neg'][user].squeeze()
        #print('candi_dim', candi_dim)
        dims = torch.multinomial(candi_dim, negi, replacement=True)
        #print('dims', dims)
        negItems = candi_neg_items[dims, self.countNeg:self.countNeg+negi]
        self.countNeg = self.countNeg+negi
        negItems = np.array(torch.diag(negItems))
        #print('negItems', negItems)
        userindex = np.array([user]*len(negItems))
        xij = 1 - self.dataset.getUserItemFeedback(userindex, negItems)
        #print('xij', xij)
        negitruth = xij.sum()
        negItemsTruthIndex = np.where(xij)
        negItems = torch.tensor(negItems[negItemsTruthIndex])
        #print('negtrue', negItems)

        xijs0 = torch.tensor([0] * negitruth)
        if posItems is None:
            return negItems.long(), xijs0
        return torch.cat([posItems.long(), negItems.long()]), torch.cat([xijs1, xijs0])





    @staticmethod
    def round(tensor):
        tensor_base = tensor.int()
        AddonePros = tensor - tensor_base
        AddONE = torch.bernoulli(AddonePros)
        return (tensor_base + AddONE).int()
        
        