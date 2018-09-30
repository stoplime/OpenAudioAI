import torch
import torch.nn as nn
import torch.nn.functional as F

class DistanceClusterLoss(nn.Module):
    """ Cluster loss using basic distance formula
    """
    def __init__(self):
        pass

    def dist(self, p1, p2):
        return torch.dist(p1, p2)