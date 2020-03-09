"""
TODO
"""
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description="Run sampling")
    parser.add_argument('--recdim', type=int,default=16,
                        help="the embedding size of recmodel")
    parser.add_argument('--vardim', type=int,default=32,
                        help="the embedding size of varmodel")
    parser.add_argument('--reclr', type=float,default=0.01,
                        help="learning rate for rec model")
    parser.add_argument('--varlr', type=float,default=0.01,
                        help="learning rate for var model")
    parser.add_argument('--vardecay', type=float,default=0.001,
                        help="weight decay for var model")
    parser.add_argument('--recdecay', type=float,default=0.01,
                        help="weight decay for rec model")
    parser.add_argument('--layer', type=int,default=2,
                        help="the layer num of lightGCN")
    parser.add_argument('--dropout', type=bool,default=True,
                        help="using the dropout or not")
    parser.add_argument('--keepprob', type=float,default=0.6,
                        help="the batch size for bpr loss training procedure")
    parser.add_argument('--testbatch', type=int,default=100,
                        help="the batch size of users for testing")
    parser.add_argument('--path', type=str,default="./checkpoints",
                        help="path to save weights")
    parser.add_argument('--tensorboard', type=bool,default="./checkpoints",
                        help="enable tensorboard")
    parser.add_argument('--comment', type=str,default="lgn")
    parser.add_argument('--load', type=bool,default=False)
    parser.add_argument('--epochs', type=int,default=1000)
    
    return parser.parse_args()