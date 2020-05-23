#！/usr/local/bin/python
import world
import torch
import numpy as np
import dataloader
from dataloader import BasicDataset
from model import LightGCN_xij_item_personal_matrix
from time import time
import math
from time import  time
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
        self.exdim = world.config['latent_dim_var'] + 1
        self.staPosUsersTensor = dataset.staPosUsersTensor
        print('self.exdim', self.exdim)
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



        Sqk0 = torch.sum(items_emb_0, dim=0)

        Sqpik1 = torch.zeros(self.n_users, self.exdim)
        Sqpik0 = torch.zeros(self.n_users, self.exdim)
        Sqpik1.scatter_add_(0, self.staPosUsersTensor, items_emb_pos_1)
        Sqpik0.scatter_add_(0, self.staPosUsersTensor, items_emb_pos_0)


        Spi0 = torch.matmul(Sqk0, user_emb.t())


        Sppi1 = torch.mul(user_emb, Sqpik1).sum(1)
        Sppi0 = torch.mul(user_emb, Sqpik0).sum(1)
        Spni0 = Spi0 - Sppi0
        Spi = Sppi1 + Spi0 - Sppi0

        Sp = torch.sum(Spi)

        self.__prob['neg0/negtrue'] = Spi0 / Spni0
        self.__prob['p(i)'] = Spi / Sp
        self.__prob['p(pos|i)'] = Sppi1 / Spi
        self.__prob['p(neg|i)'] = 1 - self.__prob['p(pos|i)']
        self.__prob['p(pos,i)'] = self.__prob['p(pos|i)']*self.__prob['p(i)']
        self.__prob['p(neg,i)'] = self.__prob['p(neg|i)']*self.__prob['p(i)']

        self.__prob['p(j|k)neg'] = (items_emb_0 / Sqk0).t()
        self.__prob['p(k|i)neg'] = (user_emb * Sqk0) / Spi0.unsqueeze(dim=1)



        self.numPerUserPos = self.round(self.__prob['p(pos,i)'] * epochk)
        numPerUserNeg = self.round(self.__prob['p(neg,i)'] * epochk)
        self.numPerUserNegTruth = self.round(numPerUserNeg * self.__prob['neg0/negtrue'])
        self.numPerUserSample = self.numPerUserPos + self.numPerUserNegTruth



    def sample(self, epochk):
        print('sample start!')
        self.compute(epochk)

        Samusers = None
        Samitems = None
        Samxijs = None

        self.countPos = 0
        self.countNeg = 0
        candi_neg_items = torch.multinomial(self.__prob['p(j|k)neg'], epochk, replacement=True)
        all_sample_times = self.numPerUserSample

        for user_id, sample_times in enumerate(all_sample_times):
            if sample_times == 0:
                continue
            items, xijs = self.sampleForUser(user_id, candi_neg_items)
            users = torch.tensor([user_id] * len(items)).long()
            if Samusers is None:
                Samusers = users
                Samitems = items
                Samxijs = xijs
            else:
                Samusers = torch.cat([Samusers, users])
                Samitems = torch.cat([Samitems, items])
                Samxijs = torch.cat([Samxijs, xijs])

        self.__prob.clear()

        return Samusers, Samitems, Samxijs

    def sampleForUser(self, user, candi_neg_items):
        posi = self.numPerUserPos[user]
        posPerUserNum = self.numPosUsers[user]

        if posi == 0:
            posItems = None
        else:
            xijs1 = torch.LongTensor([1] * posi.item())

            posGammaForUser = self.gammaPerPosSample[self.countPos:  self.countPos+posPerUserNum]
            posIndex = torch.multinomial(posGammaForUser, posi, replacement=True)
            posItems = self.staPosItems[self.countPos:self.countPos+posPerUserNum][posIndex]

        self.countPos = self.countPos + posPerUserNum

        # =======================================
        negi = self.numPerUserNegTruth[user]
        candi_dim = self.__prob['p(k|i)neg'][user].squeeze()
        dims = torch.multinomial(candi_dim, negi, replacement=True)
        negItems = candi_neg_items[dims, self.countNeg:self.countNeg+negi]
        self.countNeg = self.countNeg+negi
        negItems = np.array(torch.diag(negItems))
        userindex = np.array([user]*len(negItems))
        xij = 1 - self.dataset.getUserItemFeedback(userindex, negItems)
        negitruth = xij.sum()
        negItemsTruthIndex = np.where(xij)
        negItems = torch.from_numpy(negItems[negItemsTruthIndex])
        xijs0 = torch.LongTensor([0] * negitruth)
        if posItems is None:
            return negItems, xijs0
        return torch.cat([posItems, negItems]), torch.cat([xijs1, xijs0])


    @staticmethod
    def round(tensor):
        tensor_base = tensor.long()
        AddonePros = tensor - tensor_base
        AddONE = torch.bernoulli(AddonePros)
        return (tensor_base + AddONE).long()


class sampleUniForVar():
    def __init__(self, dataset):
        self.dataset = dataset
        self.n_users = world.config['num_users']
        self.m_items = world.config['num_items']
        self.trainDataSize = world.config['trainDataSize']
        self.numItemsForUser = math.ceil(self.trainDataSize / self.n_users )
        users = np.arange(0, self.n_users)
        self.userRepeat = users.repeat(self.numItemsForUser)
        self.numItems = self.userRepeat.shape[0]
             
    def sample(self):
        items = np.random.choice(self.m_items, self.numItems, replace=True)
        xij = self.dataset.getUserItemFeedback(self.userRepeat, items)
        return torch.LongTensor(self.userRepeat), torch.LongTensor(items), torch.LongTensor(xij)
        


