'''
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)
'''
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run sampling")
    parser.add_argument('--bpr_batch', type=int,default=4096,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--recdim', type=int,default=32,
                        help="the embedding size of lightGCN")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--dropout', type=bool,default=False,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--a_fold', type=int,default=100,
                        help="the fold num used to split large adj matrix, like gowalla")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--dataset', type=str,default='lastfm',
                        help="available datasets: [lastfm, gowalla]")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--topks', nargs='?',default="[20, 40, 60, 80, 100]",
                        help="@k test list")
    parser.add_argument('--tensorboard', type=bool,default="./checkpoints",
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=bool,default=False)
    parser.add_argument('--epochs', type=int,default=1000)
    
    return parser.parse_args()