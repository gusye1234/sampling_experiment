import torch
from torch import nn, optim




class ELBO(nn.Module):
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
    def __init__(self, config):
        super(ELBO, self).__init__()
        self.epsilon = torch.Tensor([config['epsilon']])
        self.eta     = torch.Tensor([config['eta']])
        self.bce     = nn.BCELoss()
        
    def forward(self, rating, gamma, xij):
        rating : torch.Tensor
        gamma  : torch.Tensor
        xij    : torch.Tensor
        assert len(rating) == len(gamma) and len(gamma) == len(xij)
        
        l_ab = ELBO.cross(xij, rating)
        first_term = gamma * l_ab
        
        second_term = (1-gamma)*ELBO.cross(xij, self.epsilon) + \
                        ELBO.cross(gamma, self.eta) - \
                        ELBO.cross(gamma, gamma)
        
        first_term   = torch.sum(first_term)
        second_term = torch.sum(second_term)
        
        return torch.neg(first_term + second_term)
        
    
    @staticmethod
    def cross(a, b):
        a : torch.Tensor
        b : torch.Tensor
        return a*torch.log(b) + (1-a)*torch.log(1-b) 
        

def SampleByGamma():


if __name__ == "__main__":
    config = {'epsilon':0.001, 'eta':0.5}
    test_rating = torch.rand(10,1)
    test_gamma  = torch.rand(10,1)
    test_xij    = torch.rand(10,1)
    loss_test = ELBO(config)
    print(loss_test(test_rating, test_gamma, test_xij))
    
        
