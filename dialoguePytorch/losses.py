import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class DistanceClusterLoss(nn.Module):
    """ Cluster loss using basic distance formula
    """
    def __init__(self, num_points):
        super(DistanceClusterLoss, self).__init__()
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        # Rememberst the number of points
        self.num_points = num_points
        self.points = []

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
        
        if isinstance(labels, (list,)):
            for i, pred in enumerate(preds):
                self.points.append((pred, labels[i]))
                if len(self.points) > self.num_points:
                    self.points.pop(0)
        else:
            self.points.append((preds, labels))
            if len(self.points) > self.num_points:
                self.points.pop(0)
        
        for i, pred_i in enumerate(self.points):
            # Start with a tensor of all zeros
            pred_loss.append(torch.tensor( np.zeros((1, 1, 200)) ).to(self.device))
            for j, pred_j in enumerate(self.points):
                # Skip same node
                if i == j:
                    continue
                # Same speaker
                if pred_i[1] == pred_j[1]:
                    pred_loss[i] = pred_loss[i] + torch.dist(pred_i[0].double(), pred_j[0].double())
                # Different speaker
                else:
                    pred_loss[i] = pred_loss[i] + ( 10 / torch.dist(pred_i[0].double(), pred_j[0].double()) )
        
        pred_loss = torch.stack(pred_loss)

        return pred_loss.mean()