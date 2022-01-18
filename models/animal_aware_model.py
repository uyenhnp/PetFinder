import torch
import torch.nn as nn
import timm


class AnimalAwareModel(nn.Module):
    def __init__(self, backbone='resnet50', drop=0.5, loss='BCE', use_embedding=False):
        super().__init__()
        self.loss = loss

        feat_extractor = timm.create_model(backbone, pretrained=True, num_classes=0)
        self.feat_extractor = feat_extractor
        self.dropout = nn.Dropout(p=drop)

        self.use_embedding = use_embedding
        if use_embedding:
            self.emb_dim = 8
            self.animal_embedding = nn.Embedding(2, self.emb_dim)
            self.color_embedding = nn.Embedding(3, self.emb_dim)
            self.maturity_embedding = nn.Embedding(2, self.emb_dim)

            self.fc = nn.Linear(self.feat_extractor.num_features + self.emb_dim*3, 1)
        else:
            self.fc = nn.Linear(self.feat_extractor.num_features + 5, 1)

    def forward(self, batch, animal_probs, color_probs, maturity_probs):
        img = batch['img']
        img = self.feat_extractor(img)

        if self.use_embedding:
            if self.training:
                animal_class = torch.multinomial(animal_probs, num_samples=1).squeeze()
                color_class = torch.multinomial(color_probs, num_samples=1).squeeze()
                maturity_class = torch.multinomial(maturity_probs, num_samples=1).squeeze()
            else:
                _, animal_class = torch.max(animal_probs, dim=1)
                _, color_class = torch.max(color_probs, dim=1)
                _, maturity_class = torch.max(maturity_probs, dim=1)
            
            animal_emb = self.animal_embedding(animal_class) # (N, 8)
            color_emb = self.color_embedding(color_class) # (N, 8)
            maturity_emb = self.maturity_embedding(maturity_class) # (N, 8)

            img = torch.cat((img, animal_emb, color_emb, maturity_emb), dim=1)
        else:
            img = torch.cat((
                img,
                animal_probs[:,0].unsqueeze(1),
                color_probs,
                maturity_probs[:,0].unsqueeze(1),
            ), dim=1)

        x = self.dropout(img)
        x = self.fc(x)
        if self.loss == 'MSE':
            x = torch.sigmoid(x)*100
        elif self.loss == 'BCE':
            x = torch.sigmoid(x)
        return x
