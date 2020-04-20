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
        self.allPos = dataset.allPos
        self.allNeg = dataset.allNeg
        self.n_users = dataset.n_users
        self.m_items = dataset.m_items
        self.__prob = {}
        print("fast sample")
        
    def compute(self):
        compute_start = time()
        (user_emb,
         items_emb_0,
         items_emb_1) = self.varmodel.get_user_item_embedding()
        user_emb = user_emb.cpu()
        items_emb_0 = items_emb_0.cpu()
        items_emb_1 = items_emb_1.cpu()
        ################################
        # calculate Positive gamma
        with torch.no_grad():
            U_embs = []
            I_embs = []
            for user in range(self.n_users):
                posForuser = self.allPos[user]
                UserEmb = user_emb[user]
                U_embs.append(UserEmb.repeat((len(posForuser), 1)))
                I_embs.append(items_emb_1[posForuser])
            U_embs = torch.cat(U_embs, dim=0)
            I_embs = torch.cat(I_embs, dim=0)
            gamma  = torch.mul(U_embs, I_embs).sum(1)
        start = 0
        pos_gamma = []
        for user in range(self.n_users):
            pos_gamma.append(gamma[start : start+len(self.allPos[user])])
            start += len(self.allPos[user])   
        #################################
        # user_emb = user_emb.cpu().numpy()
        # items_emb_0 = items_emb_0.cpu().numpy()
        # items_emb_1 = items_emb_1.cpu().numpy()
        ################################
        Sq_k = torch.sum(items_emb_0, dim=0)
        Sq_ik1 = []
        Sq_ik0 = []
        Sp_i = []
        Sp_p_i = []
        for user in range(self.n_users):
            posForuser = self.allPos[user]
            UserEmb = user_emb[user]
            Emb1Foruser = items_emb_1[posForuser]
            Emb0Foruser = items_emb_0[posForuser]
            # ik_1 sqpik1; ik_0 sqpik0;
            ik_1 = torch.sum(Emb1Foruser, dim=0)
            ik_0 = torch.sum(Emb0Foruser, dim=0)
            p_pi = torch.matmul(UserEmb, ik_1)
            term2 = torch.matmul(UserEmb, Sq_k)
            term3 = torch.matmul(UserEmb, ik_0)
            #spi
            p_i = p_pi + term2 - term3
            Sq_ik1.append(ik_1)
            Sq_ik0.append(ik_0)
            Sp_p_i.append(p_pi)
            Sp_i.append(p_i)
        Sp_p_i = torch.Tensor(Sp_p_i)
        Sp_i   = torch.Tensor(Sp_i)
        sp = torch.sum(Sp_i)
        self.__prob['p(i)'] = Sp_i/sp
        print('pi', self.__prob['p(i)'])
        self.__prob['p(pos|i)'] = Sp_p_i/Sp_i
        self.__prob['p(neg|i)'] = 1 - self.__prob['p(pos|i)']
        self.__prob['p(j|pos,i)'] = pos_gamma
        # construct negative fast sample probs
        S_i =  torch.matmul(user_emb, Sq_k)
        self.__prob['p(i|neg)'] = S_i / torch.sum(S_i)
        self.__prob['p(k|neg, i)'] = (user_emb*Sq_k) / S_i.unsqueeze(dim=1)
        self.__prob['p(j|neg,i,k)'] = (items_emb_0 / Sq_k).t()
        print('compute time', time()-compute_start)
        return sp
        
    def sample(self, epoch_k):
        G = self.compute()
        expected_Users = self.__prob['p(i)'] * epoch_k
        expected_Users = self.round(expected_Users)
        Samusers = None
        Samitems = None
        Samxijs  = None
        print('sample start!!!')
        sample_start = time()
        for user_id, sample_times in enumerate(expected_Users):
            if sample_times == 0:
                continue
            items, xijs = self.sampleForUser(user_id, sample_times)
            users = torch.Tensor([user_id] * len(items)).long()
            if Samusers is None:
                Samusers = users
                Samitems = items
                Samxijs  = xijs
            else:
                Samusers = torch.cat([Samusers, users])
                Samitems = torch.cat([Samitems, items])
                Samxijs = torch.cat([Samxijs, xijs])
        print('final sample time ', time() - sample_start)
        self.__prob.clear()
        return Samusers, Samitems, Samxijs, G.item()
            
    def sampleForUser(self, user, times):
        pos_i = self.__prob['p(pos|i)'][user]
        neg_i = 1 - pos_i
        expected_pos = self.round(times*pos_i)
        if expected_pos == 0:
            posItems = None
        else:
            xijs1 = torch.Tensor([1]*expected_pos.item())
            posGammaForUser = self.__prob['p(j|pos,i)'][user]
            posIndex = torch.multinomial(posGammaForUser, expected_pos, replacement=True)
            posItems = torch.from_numpy(self.allPos[user][posIndex.cpu().numpy()])
        # =======================================
        expected_neg = self.round(times*neg_i)
        posForuser = torch.from_numpy(self.allPos[user]).long()
        index_pos = torch.zeros(self.m_items)
        index_pos[posForuser] = 1

        finalNegItems = []
        count_neg = 0
        
        neg_start = time()
        while True:
            candi_dim = self.__prob['p(k|neg, i)'][user].squeeze()
            dims = torch.multinomial(candi_dim, 1, replacement=True)
            condi_items = self.__prob['p(j|neg,i,k)'][dims]
            negItems = torch.multinomial(condi_items, 1, replacement=True).squeeze(dim=1)
            if index_pos[negItems] == 0:
                count_neg += 1
                finalNegItems.append(negItems.item())
            if count_neg == expected_neg:
                break
        negItems = torch.tensor(finalNegItems)
        
        xijs0 = torch.Tensor([0]*expected_neg.item())
        if posItems is None:
            return negItems.long(), xijs0
        return torch.cat([posItems.long(), negItems.long()]), torch.cat([xijs1, xijs0])

    @staticmethod
    def round(tensor):
        tensor_base = tensor.int()
        AddonePros = tensor - tensor_base
        AddONE = torch.bernoulli(AddonePros)
        return (tensor_base + AddONE).int()
        
        
        