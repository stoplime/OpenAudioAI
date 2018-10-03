import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistanceClusterLoss(nn.Module):
    """ Cluster loss using basic distance formula
    """
    def __init__(self):
        super(DistanceClusterLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def forward(self, preds, labels):
        """ Given a bunch of predictions, calculated the distance
            and negate it if the labels are different
            Params
            ------
            preds: List( Tensor(1, 1, 200) )
                A list of predictions in global order
            ------
            labels: List( id )
                A list of ids that correlate with the order of preds
            ------
            Return: Tensor( losses )
        """
        pred_loss = []
        for i, pred_i in enumerate(preds):
            # Start with a tensor of all zeros
            pred_loss.append(torch.tensor( np.zeros((1, 1, 200)) ).to(self.device))
            for j, pred_j in enumerate(preds):
                # Skip same node
                if i == j:
                    continue
                # Same speaker
                if labels[i] == labels[j]:
                    pred_loss[i] = pred_loss[i] + torch.dist(pred_i.double(), pred_j.double())
                # Different speaker
                else:
                    pred_loss[i] = pred_loss[i] - torch.dist(pred_i.double(), pred_j.double())
        
        pred_loss = torch.stack(pred_loss)

        return pred_loss.mean()