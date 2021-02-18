import torch
import torch.nn as nn
import torch.nn.functional as F 


class DMF(nn.Module):
    def __init__(self, layers, n_user, n_item):
        super(DMF, self).__init__()

        user_MLP = []
        for i in range(len(layers)):
            if i == 0:
                user_MLP.append(nn.Linear(n_item, layers[i], bias=False))
                user_MLP.append(nn.ReLU())
            else:
                user_MLP.append(nn.Linear(layers[i-1], layers[i]))
                user_MLP.append(nn.ReLU())
        self.user_proj = nn.Sequential(*user_MLP)

        item_MLP = []
        for i in range(len(layers)):
            if i == 0:
                item_MLP.append(nn.Linear(n_user, layers[i], bias=False))
                item_MLP.append(nn.ReLU())
            else:
                item_MLP.append(nn.Linear(layers[i-1], layers[i]))
                item_MLP.append(nn.ReLU())
        self.item_proj = nn.Sequential(*item_MLP)

        self.cosine = nn.CosineSimilarity(dim=1)

        self.init_weights()

    
    def init_weights(self):
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
        
        
    def forward(self, user, item):
        p = self.user_proj(user)
        q = self.item_proj(item)

        pred = self.cosine(p, q)
        return torch.min(torch.ones_like(pred), torch.max(torch.ones_like(pred) * 1e-6, pred))