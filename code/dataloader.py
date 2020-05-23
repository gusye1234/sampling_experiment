"""
Design Dataset here
Every dataset's index has to start at 0
"""
import os
from os.path import join
import torch
import numpy as np
from torch.utils.data import Dataset
from scipy.sparse import csr_matrix
import world


class BasicDataset(Dataset):
    def __init__(self):
        self.n_users = None
        self.m_items = None
        self.allPos = None
        self.allNeg = None
        self.__testDict = None
        
    def getUserItemFeedback(self, users, items):
        raise NotImplementedError
    def getUserPosItems(self, users):
        raise NotImplementedError
    def getUserNegItems(self, users):
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
    def __init__(self, path="../data/lastfm/"):
        # train or test
        self.n_users = 1892
        self.m_items = 4489
        #load txt long
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
        trainData = (trainData - 1).astype(np.long)
        testData = (testData - 1).astype(np.long)
        self.trainData = trainData
        self.testData  = testData
        self.trainUser = trainData[:, 0]
        #self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = trainData[:, 1]

        self.trainDataSize = len(self.trainUser)
        self.testUser  = testData[:, 0]
        #self.testUniqueUsers = np.unique(self.testUser)
        self.testItem  = testData[:, 1]
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser))/self.n_users/self.m_items}")
        #train data matrix uint8
        self.UserItemNet  = csr_matrix((np.ones(len(self.trainUser), dtype=np.uint8), (self.trainUser, self.trainItem) ), shape=(self.n_users,self.m_items))
        testDict = self.__build_test()

        # pre-calculate for test 
        self.testUniqueUsers = list(testDict.keys())
        #postive items in train data
        self.exclude_index = []
        self.exclude_items = []
        allPos = self.getUserPosItems(self.testUniqueUsers)
        for range_i, items in enumerate(allPos):
            self.exclude_index.extend([range_i] * len(items))
            self.exclude_items.extend(items)
        #test postive
        self.groundTruth = list(testDict[user] for user in self.testUniqueUsers)
        #for sample
        self.posSampleUser, self.posSampleItem, self.numPosUsers = self.getAllUserPosItems()
        self.staPosUsersTensor = self.posSampleUser.reshape(-1, 1).expand(-1, world.config['latent_dim_var']+1)
        self.Graph = self.getSparseGraph()

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).float()

            sum1 = torch.zeros(self.n_users + self.m_items)
            sum1.scatter_add_(0, index[0, :], data + 1e-9)
            ssqrt = torch.sqrt(sum1 * 1.0)
            dnow = data / ssqrt[index[0, :]] / ssqrt[index[1, :]]
            self.Graph = torch.sparse.FloatTensor(index, dnow, torch.Size([self.n_users + self.m_items, self.n_users + self.m_items]))
        return self.Graph
        


    def __build_test(self):
        """
        return:
            dict: {user: [items]}
            int32
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data
    
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
        return np.array(self.UserItemNet[users, items]).reshape(-1)
    def getAllUserPosItems(self):
        staPosUsers = None
        staPosItems = None
        numPosUsers = []
        users = range(self.n_users)
        for user in users:
            positems = torch.LongTensor(self.UserItemNet[user].nonzero()[1])
            posusers = torch.LongTensor([user]*len(positems))
            numPosUsers.append(len(positems))

            if staPosUsers is None:
                staPosUsers = posusers
                staPosItems = positems

            else:
                staPosUsers = torch.cat([staPosUsers, posusers])
                staPosItems = torch.cat([staPosItems, positems])

        return staPosUsers, staPosItems, torch.LongTensor(numPosUsers)



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


class Ciao(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    LastFM dataset
    """

    def __init__(self, path="../data/lastfm/"):
        # train or test
        self.n_users = 5298
        self.m_items = 19301
        # load txt long
        if world.ontest:
            trainData = np.loadtxt(join(path, 'data1.txt'))
            print("training data:", join(path, 'data1.txt'))
            testData = np.loadtxt(join(path, 'test1.txt'))
            print("testing data:", join(path, 'test1.txt'))
        # trainData = pd.read_table(join(path, 'train.txt'), header=None)
        else:
            trainData = np.loadtxt(join(path, 'train.txt'))
            print("training data:", join(path, 'train.txt'))
            # testData  = pd.read_table(join(path, 'test.txt'), header=None)
            testData = np.loadtxt(join(path, 'validation.txt'))
            print("testing data:", join(path, 'validation.txt'))
        trainData = (trainData - 1).astype(np.long)
        testData = (testData - 1).astype(np.long)
        self.trainData = trainData
        self.testData = testData
        self.trainUser = trainData[:, 0]
        # self.trainUniqueUsers = np.unique(self.trainUser)
        self.trainItem = trainData[:, 1]

        self.trainDataSize = len(self.trainUser)
        self.testUser = testData[:, 0]
        # self.testUniqueUsers = np.unique(self.testUser)
        self.testItem = testData[:, 1]
        self.Graph = None
        print(f"LastFm Sparsity : {(len(self.trainUser) + len(self.testUser)) / self.n_users / self.m_items}")
        # train data matrix uint8
        self.UserItemNet = csr_matrix((np.ones(len(self.trainUser), dtype=np.uint8), (self.trainUser, self.trainItem)),
                                      shape=(self.n_users, self.m_items))
        testDict = self.__build_test()

        # pre-calculate for test
        self.testUniqueUsers = list(testDict.keys())
        # postive items in train data
        self.exclude_index = []
        self.exclude_items = []
        allPos = self.getUserPosItems(self.testUniqueUsers)
        for range_i, items in enumerate(allPos):
            self.exclude_index.extend([range_i] * len(items))
            self.exclude_items.extend(items)
        # test postive
        self.groundTruth = list(testDict[user] for user in self.testUniqueUsers)
        # for sample
        self.posSampleUser, self.posSampleItem, self.numPosUsers = self.getAllUserPosItems()
        self.staPosUsersTensor = self.posSampleUser.reshape(-1, 1).expand(-1, world.config['latent_dim_var'] + 1)
        self.Graph = self.getSparseGraph()

    def getSparseGraph(self):
        if self.Graph is None:
            user_dim = torch.LongTensor(self.trainUser)
            item_dim = torch.LongTensor(self.trainItem)

            first_sub = torch.stack([user_dim, item_dim + self.n_users])
            second_sub = torch.stack([item_dim + self.n_users, user_dim])
            index = torch.cat([first_sub, second_sub], dim=1)
            data = torch.ones(index.size(-1)).float()

            sum1 = torch.zeros(self.n_users + self.m_items)
            sum1.scatter_add_(0, index[0, :], data + 1e-9)
            ssqrt = torch.sqrt(sum1 * 1.0)
            dnow = data / ssqrt[index[0, :]] / ssqrt[index[1, :]]
            self.Graph = torch.sparse.FloatTensor(index, dnow, torch.Size(
                [self.n_users + self.m_items, self.n_users + self.m_items]))
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
            int32
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

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
        return np.array(self.UserItemNet[users, items]).reshape(-1)

    def getAllUserPosItems(self):
        staPosUsers = None
        staPosItems = None
        numPosUsers = []
        users = range(self.n_users)
        for user in users:
            positems = torch.LongTensor(self.UserItemNet[user].nonzero()[1])
            posusers = torch.LongTensor([user] * len(positems))
            numPosUsers.append(len(positems))

            if staPosUsers is None:
                staPosUsers = posusers
                staPosItems = positems

            else:
                staPosUsers = torch.cat([staPosUsers, posusers])
                staPosItems = torch.cat([staPosItems, positems])

        return staPosUsers, staPosItems, torch.LongTensor(numPosUsers)

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



if __name__ == '__main__':
    data = LastFM()