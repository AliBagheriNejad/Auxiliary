import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F



class FeatureExtractor(nn.Module):
    def __init__(self, drop=0.1, input_channels=1):
        super(FeatureExtractor, self).__init__()

        self.conv1 = nn.Conv1d(input_channels, 16, kernel_size=128)
        self.bn1 = nn.BatchNorm1d(16)
        self.dropout1 = nn.Dropout(drop)
        self.pool1 = nn.MaxPool1d(kernel_size=4)

        self.conv2 = nn.Conv1d(16, 32, kernel_size=64)
        self.bn2 = nn.BatchNorm1d(32)
        self.dropout2 = nn.Dropout(drop)
        self.pool2 = nn.MaxPool1d(kernel_size=4)

        self.conv3 = nn.Conv1d(32, 64, kernel_size=16)
        self.bn3 = nn.BatchNorm1d(64)
        self.dropout3 = nn.Dropout(drop)
        self.pool3 = nn.MaxPool1d(kernel_size=2)

        self.conv4 = nn.Conv1d(64, 128, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(128)
        self.dropout4 = nn.Dropout(drop)
        self.pool4 = nn.MaxPool1d(kernel_size=2)

        self.conv5 = nn.Conv1d(128, 256, kernel_size=2)
        self.bn5 = nn.BatchNorm1d(256)
        self.dropout5 = nn.Dropout(drop)

    def forward(self, x):
        x = self.pool1(self.dropout1(F.relu(self.bn1(self.conv1(x)))))
        x = self.pool2(self.dropout2(F.relu(self.bn2(self.conv2(x)))))
        x = self.pool3(self.dropout3(F.relu(self.bn3(self.conv3(x)))))
        x = self.pool4(self.dropout4(F.relu(self.bn4(self.conv4(x)))))
        x = self.dropout5(F.relu(self.bn5(self.conv5(x))))


        x = torch.flatten(x, 1)
        return x

class Classifier(nn.Module):
    def __init__(self, num_classes, drop=0.2):
        super(Classifier, self).__init__()

        self.fc1 = nn.Linear(1024, 128)
        self.dropout1 = nn.Dropout(drop)

        self.fc2 = nn.Linear(128, 64)
        self.dropout2 = nn.Dropout(drop)

        self.fc3 = nn.Linear(256, 64)
        self.dropout2 = nn.Dropout(drop)

        self.fcc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        latent = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = F.softmax(self.fcc(latent), dim=1)
        return x, latent

class Network(nn.Module):
    def __init__(self, num_classes, in_channels = 2):
        super().__init__()
        self.feature_extractor = FeatureExtractor(input_channels = in_channels)
        self.classifier = Classifier(num_classes)

        self.best_acc = 0
        self.save_path = 'model_weights.pth'
        self.patience = 10
        self.e_ratio = 100
        self.in_ch = in_channels
        self.weight_dic = {
            'train_loss':None,
            'train_acc': None,
            'val_acc': None,
            'val_loss': None
        }
        self.metrics_now = {
            'train_loss':None,
            'train_acc': None,
            'val_acc': None,
            'val_loss': None
        }
        self.metrics_best = {
            'train_loss':-np.inf,
            'train_acc': 0,
            'val_acc': 0,
            'val_loss': -np.inf
        }

    def forward(self, x):
        if self.in_ch == 1:
            x = x.view(x.shape[0],  1, x.shape[1])  # Reshape input to (batch_size, channels, length)
        else:
            x = x.view(x.shape[0], x.shape[2] , x.shape[1])
        features = self.feature_extractor(x)
        x, latent = self.classifier(features)
        embed = torch.concat([features.reshape(x.shape[0],-1), latent.reshape(x.shape[0],-1)], dim=1) ## change this if needed

        return x, embed


    def early_stopping(self,thing,epoch):

        '''
        Incase you wanted to use best loss
        just use "-loss"

        '''
        self.check_weight()
        # Early stopping
        if (thing > self.best_acc) and (np.abs(thing-self.best_acc) > np.abs(self.best_acc)/self.e_ratio):
        # if thing > self.best_acc :


            self.best_acc = thing
            self.best_epoch = epoch
            self.current_patience = 0

            # Save the model's weights
            torch.save(self.state_dict(), self.save_path)
            print("<<<<<<<  !Model saved!  >>>>>>>")
            return False
        else:
            self.current_patience += 1
            # Check if the patience limit is reached
            if self.current_patience >= self.patience:
                print("Early stopping triggered!")
                return True
            else:
                return False
    
    def check_weight(self):

        for k in self.weight_dic.keys():

            if  (self.metrics_now[k] > self.metrics_best[k]):
                self.metrics_best[k] = self.metrics_now[k]
                self.weight_dic[k] = self.state_dict()

class Model2(nn.Module):
    def __init__(self,num_classes, in_channels=2, aux_feat=1089):
        super().__init__()
        self.cls = Network(num_classes,in_channels)
        self.calc_feat_dim(num_classes)
        self.aux = nn.Sequential(
            nn.Linear(self.aux_dim,512),
            nn.ReLU(),
            nn.Linear(512, 64),
            nn.ReLU(),
            nn.Linear(64,num_classes)
        )
        self.label_coder = lambda y:F.one_hot(y, num_classes=num_classes)

        self.best_acc = 0
        self.save_path = 'model_weights.pth'
        self.patience = 10
        self.e_ratio = 100
        self.in_ch = in_channels
        self.weight_dic = {
            'train_loss':None,
            'train_acc': None,
            'val_acc': None,
            'val_loss': None
        }
        self.metrics_now = {
            'train_loss':None,
            'train_acc': None,
            'val_acc': None,
            'val_loss': None
        }
        self.metrics_best = {
            'train_loss':-np.inf,
            'train_acc': 0,
            'val_acc': 0,
            'val_loss': -np.inf
        }

    def forward(self,x,y):

        red_embed = self.other_emb(x,y)

        x_mid = x[:,2,:,:] # Fix indexing for dyanimc model training
        y_mid = y[:,2]
        y_mid_ohe = self.label_coder(y_mid)

        x_cls, embed = self.cls(x_mid)
        x = self.concat_embd(x_cls,embed)

        x = torch.concat([x,red_embed], dim=1)
        x = x.reshape(x.shape[0], -1)
        x = self.aux(x)

        return F.softmax(x, dim=1), F.softmax(x_cls, dim=1)


    def other_emb(self,x,y):
        x = x[:,[0,1,3,4],:,:]
        y_ohe = self.label_coder(y)
        xshape = x.shape
        x = x.reshape(xshape[0]*xshape[1], xshape[2], xshape[3])
        x, embed = self.cls(x)

        if len(x.shape)==2:
            x = x.reshape(xshape[0], xshape[1], -1)
        else:
            raise('x shape has problems')
        
        if len(embed.shape)==2:
            embed = embed.reshape(xshape[0], xshape[1], -1)
        else:
            raise('embed shape has problems')
        
        embedding = self.concat_embd(x, embed)

        return embedding
    

    def concat_embd(self, x, embed):

        if len(x.shape)==2:
            x = x.unsqueeze(1)
            embed = embed.unsqueeze(1)
        _, y_hat = torch.max(x, 2)
        y_hat_ohe = self.label_coder(y_hat)
        # y_hat_ohe = y_hat_ohe.reshape(-1,1)
        if len(x.shape)==3:
            embedding = torch.concat([y_hat_ohe, embed], dim=2)
        if len(x.shape)==2:
            embedding = torch.concat([y_hat_ohe, embed], dim=2)

        return embedding


    def calc_feat_dim(self, n, n_sample=5):
        x = torch.randn(1,1024,2)
        _, embed = self.cls(x)
        dim = (embed.shape[1] + n) * n_sample
        self.aux_dim = dim

    
    def early_stopping(self,thing,epoch):

        '''
        Incase you wanted to use best loss
        just use "-loss"

        '''
        self.check_weight()
        # Early stopping
        if (thing > self.best_acc) and (np.abs(thing-self.best_acc) > np.abs(self.best_acc)/self.e_ratio):
        # if thing > self.best_acc :


            self.best_acc = thing
            self.best_epoch = epoch
            self.current_patience = 0

            # Save the model's weights
            torch.save(self.state_dict(), self.save_path)
            print("<<<<<<<  !Model saved!  >>>>>>>")
            return False
        else:
            self.current_patience += 1
            # Check if the patience limit is reached
            if self.current_patience >= self.patience:
                print("Early stopping triggered!")
                return True
            else:
                return False
    
    def check_weight(self):

        for k in self.weight_dic.keys():

            if  (self.metrics_now[k] > self.metrics_best[k]):
                # print(f'Replacing weight for best \'{k}\'')
                self.metrics_best[k] = self.metrics_now[k]
                self.weight_dic[k] = self.state_dict()









