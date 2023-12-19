from torch import nn
from torchvision import models

# Vision Transformer Model class
class PretrainViT(nn.Module):
    def __init__(self):
        super(PretrainViT, self).__init__()
        model = models.vit_b_16(weights=None)
        num_classifier_feature = model.heads.head.in_features
        model.heads.head = nn.Sequential(
            nn.Linear(num_classifier_feature, 120)
        )
        self.model = model
        for param in self.model.parameters():
                param.requires_grad = False

    def forward(self, x):
        return self.model(x)