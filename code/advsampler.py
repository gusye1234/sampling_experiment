import torch
import torch.nn as nn
import torch.nn.functional as F

class AdvSampler(nn.Module):
    """Adversarial sampler for ExRec
    """

    def __init__(self, num_users, num_items, emb_dim, can_samples):
        """
        Args:
            num_users: The number of all users.
            num_items: The number of all items.
            emb_dim: Embedding dimension of users and items
            can_samples_: candidate negative samples (from real samples)
        """
        super(AdvSampler, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.emb_dim = emb_dim
        self.emb_users = torch.nn.Embedding(self.num_users, self.emb_dim)
        self.emb_items = torch.nn.Embedding(self.num_items, self.emb_dim)
        self.can_samples = can_samples
        nn.init.uniform_(self.emb_users.weight, -0.5, 0.5)
        nn.init.uniform_(self.emb_items.weight, -0.5, 0.5)

    def get_scores_for_candidate_samples(self):
        """Get scores of samples

        Returns:
             A tensor containing scores of all candidate samples
        """

        selected_emb_users = self.emb_users[self.can_samples[:, 0], :]
        selected_emb_items = self.emb_items[self.can_samples[:, 1], :]
        scores = torch.sum(torch.mul(selected_emb_users, selected_emb_items), 1)
        return scores #1 dismension

    def annealed_softmax(self, scores):
        """
        Args:
            scores:samples'scores

        Returns:
            Probabilities by applying annealed softmax on scores.
        """
        probs = torch.softmax(scores)
        return probs

    def generate_samples_for_Rec(self, k):
        """
        Args:
            k: the number of negative samples we want
            probs: the probability of each candidate samples

        Returns:
            A tensor containing k negative samples

        """
        scores = self.get_scores_for_candidate_samples()
        probs = self.annealed_softmax(scores)
        can_index = range(self.can_samples.shape[0])
        gen_neg = torch.multinomial(probs, k, replacement=True)
        gen_neg_prob = torch.gather(probs, dim=0, index=gen_neg)
        return gen_neg, self.can_samples[gen_neg], gen_neg_prob

    def forward(self, k):
        return self.generate_samples_for_Rec(k)