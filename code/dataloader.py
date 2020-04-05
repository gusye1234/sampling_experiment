"""
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import sys
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
from scipy.sparse import csr_matrix
import world


class BasicDataset(Dataset):
    def __init__(self):
        self.n_users = None
        self.m_items = None
        self.__testDict = None
        
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    def getUserPosItems(self, users):
        raise NotImplementedError
    def getUserNegItems(self, users):
        raise NotImplementedError
    def getTestDict(self):
        raise NotImplementedError
    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A = 
            |0,   R|
            |R^T, 0|
        """
        raise NotImplementedError

class LastFM(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="/data/HungPaan/lastfm"):
        # train or test
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        self.n_users = 1892
        self.m_items = 4489
        if world.ontest:
            trainData = np.loadtxt(join(path, 'data1.txt'))    
            print("training data:",join(path, 'data1.txt'))
            testData  = np.loadtxt(join(path, 'test1.txt'))
            print("testing data:", join(path, 'test1.txt'))
        # trainData = pd.read_table(join(path, 'train.txt'), header=None)
        else:
            trainData = np.loadtxt(join(path, 'train.txt'))
            print("training data:",join(path, 'train.txt'))
            # testData  = pd.read_table(join(path, 'test.txt'), header=None)
            testData  = np.loadtxt(join(path, 'validation.txt'))
            print("testing data:", join(path, 'validation.txt'))
        trainData-= 1
        testData -= 1
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = trainData[:, 0]
        self.trainUniqueUsers = np.unique(self.trainUser)
        print(len(self.trainUniqueUsers))
        self.trainItem = trainData[:,1]
        print(len(np.unique(self.trainItem)))

        self.trainDataSize = len(self.trainUser)
        self.testUser  = testData[:, 0]
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = testData[:,1]
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        # self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self.allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self.allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
        
    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).float()


            sum1=torch.zeros(len(data))
            sum1.scatter_add_(0,index[0,:],data+1e-9)
            ssqrt=torch.sqrt(sum1*1.0)
            dnow=data/ssqrt[index[0,:]]/ssqrt[index[1,:]]
            self.Graph = torch.sparse.FloatTensor(index, dnow, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getTestDict(self):
        return self.__testDict
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    
    def __len__(self):
        return len(self.trainUniqueUsers)



class MovLens(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="/data/HungPaan/movlens"):
        # train or test
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        self.n_users = 6040
        self.m_items = 3952
        if world.ontest:
            trainData = np.loadtxt(join(path, 'data1.txt'))    
            print("training data:",join(path, 'data1.txt'))
            testData  = np.loadtxt(join(path, 'test1.txt'))
            print("testing data:", join(path, 'test1.txt'))
        # trainData = pd.read_table(join(path, 'train.txt'), header=None)
        else:
            trainData = np.loadtxt(join(path, 'train.txt'))
            print("training data:",join(path, 'train.txt'))
            # testData  = pd.read_table(join(path, 'test.txt'), header=None)
            testData  = np.loadtxt(join(path, 'validation.txt'))
            print("testing data:", join(path, 'validation.txt'))
        trainData-= 1
        testData -= 1
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = trainData[:, 0]
        self.trainUniqueUsers = np.unique(self.trainUser)
        print(len(self.trainUniqueUsers))
        self.trainItem = trainData[:,1]
        print(len(np.unique(self.trainItem)))

        self.trainDataSize = len(self.trainUser)
        self.testUser  = testData[:, 0]
        self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = testData[:,1]
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        # self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self.allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self.allPos[i])
            neg = allItems - pos
            self.allNeg.append(np.array(list(neg)))
        self.__testDict = self.__build_test()
        
    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim+self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).float()


            sum1=torch.zeros(len(data))
            sum1.scatter_add_(0,index[0,:],data+1e-9)
            ssqrt=torch.sqrt(sum1*1.0)
            dnow=data/ssqrt[index[0,:]]/ssqrt[index[1,:]]
            self.Graph = torch.sparse.FloatTensor(index, dnow, torch.Size([self.n_users+self.m_items, self.n_users+self.m_items]))
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
    def getTestDict(self):
        return self.__testDict
    
    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape((-1, ))
    
    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems
    
    def getUserNegItems(self, users):
        negItems = []
        for user in users:
            negItems.append(self.allNeg[user])
        return negItems
            
    
    
    def __getitem__(self, index):
        user = self.trainUniqueUsers[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    
    def __len__(self):
        return len(self.trainUniqueUsers) 
        
