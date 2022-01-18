import torch
import torch.nn as nn
import timm
import clip


class Baseline(nn.Module):
    def __init__(self, backbone='resnet50', drop=0.5, loss='BCE', fc_nonlinear=False):
        super().__init__()
        self.loss = loss

        feat_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_extractor = feat_extractor
        self.dropout = nn.Dropout(p=drop)
        if not fc_nonlinear:
            self.fc = nn.Linear(self.feat_extractor.num_features, 1)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feat_extractor.num_features, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(self, batch):
        img = batch['img']
        x = self.feat_extractor(img)
        x = self.dropout(x)
        x = self.fc(x)
        if self.loss == 'MSE':
            x = torch.sigmoid(x)*100
        elif self.loss == 'BCE':
            x = torch.sigmoid(x)
        return x

class Baseline2(nn.Module):
    def __init__(self, backbone='resnet50', drop=0.5, loss='MSE', meta_dim=12):
        super().__init__()
        self.loss = loss

        feat_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_extractor = feat_extractor
        self.batchnorm = nn.BatchNorm1d(self.feat_extractor.num_features + meta_dim)
        self.dropout = nn.Dropout(p=drop)
        self.fc = nn.Linear(self.feat_extractor.num_features + meta_dim, 1)

    def forward(self, batch):
        img = batch['img']
        meta = batch['meta']
        
        img = self.feat_extractor(img)
        x = torch.cat((img, meta), 1)
        x = self.batchnorm(x)
        x = self.dropout(x)
        x = self.fc(x)
        if self.loss == 'MSE':
            x = torch.sigmoid(x)*100
        elif self.loss == 'BCE':
            x = torch.sigmoid(x)
        return x

class Baseline3(nn.Module):
    def __init__(self, backbone='resnet50', drop=0.5, loss='MSE', meta_dim=12):
        super().__init__()
        self.loss = loss

        feat_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_extractor = feat_extractor
        self.layernorm = nn.LayerNorm(self.feat_extractor.num_features + meta_dim)
        self.dropout = nn.Dropout(p=drop)
        self.fc = nn.Linear(self.feat_extractor.num_features + meta_dim, 1)

    def forward(self, batch):
        img = batch['img']
        meta = batch['meta']
        
        img = self.feat_extractor(img)
        x = torch.cat((img, meta), 1)
        x = self.layernorm(x)
        x = self.dropout(x)
        x = self.fc(x)
        if self.loss == 'MSE':
            x = torch.sigmoid(x)*100
        elif self.loss == 'BCE':
            x = torch.sigmoid(x)
        return x


class Baseline4(nn.Module):
    def __init__(self, backbone='resnet50', drop=0.5, loss='BCE', fc_nonlinear=False):
        super().__init__()
        self.loss = loss

        feat_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_extractor = feat_extractor

        # Blur embedding.
        self.blur_embedding = nn.Embedding(2, 32)

        self.dropout = nn.Dropout(p=drop)
        if not fc_nonlinear:
            self.fc = nn.Linear(self.feat_extractor.num_features + 32, 1)
        else:
            self.fc = nn.Sequential(
                nn.Linear(self.feat_extractor.num_features + 32, 128),
                nn.ReLU(),
                nn.Linear(128, 1)
            )

    def forward(self, batch):
        img = batch['img']
        x = self.feat_extractor(img) # (bs, 2048)
        blur_emb = self.blur_embedding(batch['blur']) # (bs, 32)
        x = torch.cat((x, blur_emb), dim=1)

        x = self.dropout(x)
        x = self.fc(x)
        if self.loss == 'MSE':
            x = torch.sigmoid(x)*100
        elif self.loss == 'BCE':
            x = torch.sigmoid(x)
        return x

class Baseline5(nn.Module):
    def __init__(self, backbone='resnet50', drop=0.5, loss='MSE'):
        super().__init__()
        self.loss = loss

        feat_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_extractor = feat_extractor
        # self.layernorm = nn.LayerNorm(self.feat_extractor.num_features + 4)
        self.dropout = nn.Dropout(p=drop)
        self.fc = nn.Linear(self.feat_extractor.num_features + 4, 1)

    def forward(self, batch, animal_probs):
        img = batch['img']

        img = self.feat_extractor(img)
        x = torch.cat((img, animal_probs), 1)
        img = self.dropout(img)
        # x = self.layernorm(x)
        x = self.fc(x)
        if self.loss == 'MSE':
            x = torch.sigmoid(x)*100
        elif self.loss == 'BCE':
            x = torch.sigmoid(x)
        return x


class Baseline6(nn.Module):
    def __init__(self, backbone='resnet50', drop=0.5, loss='MSE'):
        super().__init__()
        self.loss = loss

        feat_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_extractor = feat_extractor
        self.dropout = nn.Dropout(p=drop)
        self.fc1 = nn.Sequential(
            nn.Linear(self.feat_extractor.num_features, 64),
            nn.ReLU())
        self.fc2 = nn.Linear(64 + 2, 1)

    def forward(self, batch, animal_probs):
        img = batch['img']

        img = self.feat_extractor(img)
        img = self.dropout(img)
        img = self.fc1(img)

        x = torch.cat((img, animal_probs), 1)
        x = self.fc2(x)
        if self.loss == 'MSE':
            x = torch.sigmoid(x)*100
        elif self.loss == 'BCE':
            x = torch.sigmoid(x)
        return x