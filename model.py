import torch
import torch.nn as nn
import torch.nn.functional as F 


class DMF(nn.Module):
    def __init__(self, n_user, n_item):
        super(DMF, self).__init__()

        self.user_proj = nn.Sequential(
                        nn.Linear(n_item, 128, bias=False),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU())

        self.item_proj = nn.Sequential(
                        nn.Linear(n_user, 128, bias=False),
                        nn.ReLU(),
                        nn.Linear(128, 64),
                        nn.ReLU())

        for layer in self.user_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                try:
                    layer.bias.data.normal_(0.0, 0.01)
                except:
                    pass

        for layer in self.item_proj:
            if isinstance(layer, nn.Linear):
                nn.init.normal_(layer.weight, 0, 0.01)
                try:
                    layer.bias.data.normal_(0.0, 0.01)
                except:
                    pass
        
        self.cosine = nn.CosineSimilarity(dim=1)
        self.treshold = nn.Threshold(0.00001, 0.00001)

    def forward(self, user, item):
        user = F.normalize(user)
        item = F.normalize(item)
        p = self.user_proj(user)
        q = self.item_proj(item)

        pred = self.cosine(p, q)
        return self.treshold(pred)
