"""
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
        
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    def getUserPosItems(self, users):
        raise NotImplementedError
    def getUserNegItems(self, users):
        raise NotImplementedError


class LastFM(BasicDataset):
    """
    Dataset type for pytorch
    Incldue graph information
    LastFM dataset
    """
    def __init__(self, path="../data/samwalk_data"):
        # train or test
        self.mode_dict = {'train':0, "test":1}
        self.mode    = self.mode_dict['train']
        self.n_users = 1892
        self.m_items = 4489
        trainData = pd.read_table(join(path, 'data1.txt'), header=None)
        # print(trainData.head())
        testData  = pd.read_table(join(path, 'test1.txt'), header=None)
        # print(testData.head())
        trustNet  = pd.read_table(join(path, 'trustnetwork.txt'), header=None).to_numpy()
        # print(trustNet[:5])
        trustNet -= 1
        trainData-= 1
        testData -= 1
        self.trustNet  = trustNet
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = np.array(trainData[:][0])
        self.trainItem = np.array(trainData[:][1])
        self.testUser  = np.array(testData[:][0])
        self.testUser  = np.array(testData[:][1])
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        
        # (users,users)
        self.socialNet    = csr_matrix((np.ones(len(trustNet)), (trustNet[:,0], trustNet[:,1]) ), shape=(self.n_users,self.n_users))
        # (users,items), bipartite graph
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items)) 
        
        # pre-calculate
        self.allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems    = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self.allPos[i])
            neg = allItems - pos
            self.allNeg.append(list(neg))

    
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
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape(-1)
    
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
        if self.mode == 0:
            user = self.trainUser[index]
        elif self.mode == 1:
            user = self.testUser[index]
        # return user_id and the positive items of the user
        return user
    
    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']
    
    
    def __len__(self):
        if self.mode == 0:
            return len(self.trainUser)
        else:
            return len(self.testUser)
        
        
class MovLens(BasicDataset):
    """
    Dataset type for pytorch
    MovLens-1M
    """
    def __init__(self, path="data/ml-1m"):
        #your own path
        self.mode_dict = {'train': 0, 'test': 1}
        self.mode = self.mode_dict['train']
        trainData = pd.read_table(join(path, 'ml_train.txt'), names=RATING_NAMES)
        testData = pd.read_table(join(path, 'ml_test.txt'), names=RATING_NAMES)
        #print(testData.loc[2])
        trainData -= 1
        testData -= 1
        self.trainData = trainData
        self.testData = testData
        #print(self.testData)
        self.users, self.items = self.extract_users_and_items(self.trainData.append(self.testData))
        self.n_users = self.get_n_users()
        self.m_items = self.get_n_items()
        #print(self.n_users, self.m_items)
        self.trainUser = np.array(self.trainData['user'])
        self.trainItem = np.array(self.trainData['item'])

        self.testUser = np.array(self.testData['user'])
        self.testItem = np.array(self.testData['item'])
        #print(self.testUser)
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))

        self.allPos = self.getUserPosItems(list(range(self.n_users)))
        self.allNeg = []
        allItems = set(range(self.m_items))
        for i in range(self.n_users):
            pos = set(self.allPos[i])
            neg = allItems - pos
            self.allNeg.append(list(neg))

    def extract_users_and_items(self, rating):
        """Extract users and items from rating data

        Args:
            rating: rating records in pandas DataFrame.

        Returns:
            users: users stored in np.ndarray
            items: items stored in np.ndarray
        """
        users = rating.user.unique()
        items = rating.item.unique()
        return users, items

    def get_n_users(self):
        return max(self.users)+1

    def get_n_items(self):
        return max(self.items)+1

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
        return np.array(self.UserItemNet[users, items]).astype('uint8').reshape(-1)

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
        if self.mode == 0:
            user = self.trainUser[index]
        elif self.mode == 1:
            user = self.testUser[index]
        # return user_id and the positive items of the user
        return user

    def switch2test(self):
        """
        change dataset mode to offer test data to dataloader
        """
        self.mode = self.mode_dict['test']

    def __len__(self):
        if self.mode == 0:
            return len(self.trainUser)
        else:
            return len(self.testUser)
