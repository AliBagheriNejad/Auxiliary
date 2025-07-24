import pickle
from sklearn.preprocessing import StandardScaler
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def read_pkl(path):
    
    with open(path, 'rb') as file:
        data = pickle.load(file)

    return data

def scale_it (*X):

    X_reshape = tuple(x.reshape(x.shape[0],-1,x.shape[3]) for x in X)
    scaler_channel = []
    X_scaled_list = [np.zeros_like(x) for x in X_reshape]

    for i in range(X_reshape[0].shape[-1]):

        scaler = StandardScaler()
        scaler.fit(X_reshape[0][:,:,i])
        for j in range(len(X_reshape)):

            X_scaled_list[j][:,:,i] = scaler.transform(X_reshape[j][:,:,i])
        scaler_channel.append(scaler)

    X_scaled_final = [x.reshape(x1.shape) for x,x1 in zip(X_scaled_list, X)]

    return X_scaled_final, scaler_channel

def tensor_it(X,y):

    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    y_tensor = torch.tensor(y, dtype=torch.long).to(device)

    return X_tensor, y_tensor

def label_encoder(y):

    labels = np.unique(y)
    codes = np.arange(0,len(labels))
    y_encoded = np.zeros_like(y)

    for c,l in zip(codes,labels):
        
        y_encoded[y==l] = c

    return y_encoded.astype(np.uint8), dict(zip(labels, codes))

def make_loader(X, y, bs=None):

    dataset = TensorDataset(X,y)

    if bs:
        data_loader = DataLoader(dataset, batch_size=bs, shuffle=False)
    else:
        data_loader = DataLoader(dataset, shuffle=False)

    return data_loader


