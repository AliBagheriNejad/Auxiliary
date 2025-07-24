import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import numpy as np
import tqdm
import os
import shutil
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

def tensor_it(X, y, for_aux=True):

    if for_aux:
        X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y, dtype=torch.long).to(device)
    else:
        X_tensor = torch.tensor(X[:,2,:,:], dtype=torch.float32).to(device)
        y_tensor = torch.tensor(y[:,2], dtype=torch.long).to(device)

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

def  train_classifier(
        model,
        criterion,
        optimizer,
        train_dataloader,
        val_dataloader,
        epochs = 100,
        early_stopping = 'val_loss',
        mode = 'aux'
):

    '''
    This function will train a classifier model

    '''
    # Ensure 'temp' folder exists and is empty
    def fix_temp():
        temp_dir = 'temp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        else:
            for filename in os.listdir(temp_dir):
                file_path = os.path.join(temp_dir, filename)
                if os.path.isfile(file_path):
                    os.remove(file_path)
    def save_weight_dic():
        for k,v in zip(model.weight_dic.keys(), model.weight_dic.values()):
            weight_name = f'{mode}_{k}_{np.abs(model.metrics_best[k]):.6f}.pth'
            weight_path = os.path.join('temp', weight_name)
            torch.save(v, weight_path)
            print(f'Weight <{weight_path}> saved successfully')

    fix_temp()
    # Training loop (example, not complete)
    train_losses, train_accs, valid_losses, valid_accs = [], [], [], []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        correct_train = 0
        total_train = 0

        progress_bar = tqdm.tqdm(enumerate(train_dataloader), total=len(train_dataloader), desc=f'Epoch {epoch + 1}/{epochs}')

        for i, (batch_data, batch_labels) in progress_bar:
            batch_data = batch_data.to(device)
            batch_labels = batch_labels.to(device)
            optimizer.zero_grad()

            if mode == 'aux':
                outputs = model(batch_data, batch_labels)
            else:
                outputs = model(batch_data)
            batch_labels = batch_labels[:,2]
            loss = criterion(outputs, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += batch_labels.size(0)
            correct_train += (predicted == batch_labels).sum().item()

            progress_bar.set_postfix(train_loss=train_loss / (i + 1), train_acc=100 * correct_train / total_train)
        
        train_loss_log = train_loss / len(train_dataloader)
        train_acc_log = 100 * correct_train / total_train
        train_losses.append(train_loss_log)
        train_accs.append(train_acc_log)

        # Validation
        model.eval()
        valid_loss = 0.0
        correct_valid = 0
        total_valid = 0

        with torch.no_grad():
            for batch_data, batch_labels in val_dataloader:
                batch_data = batch_data.to(device)
                batch_labels = batch_labels.to(device)
                if mode == 'aux':
                    outputs = model(batch_data, batch_labels)
                else:
                    outputs = model(batch_data)
                    
                batch_labels = batch_labels[:,2]
                loss = criterion(outputs, batch_labels)

                valid_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                total_valid += batch_labels.size(0)
                correct_valid += (predicted == batch_labels).sum().item()

        val_loss_log = valid_loss / len(val_dataloader)
        val_acc_log = 100 * correct_valid / total_valid
        valid_losses.append(val_loss_log)
        valid_accs.append(val_acc_log)
        print(f'validation_acc: {valid_accs[-1]:.1f}, validation_loss: {valid_losses[-1]:.4f}', end='\n')


        model.metrics_now = {
            'train_loss': -train_loss_log,
            'train_acc': train_acc_log,
            'val_acc': val_acc_log,
            'val_loss': -val_loss_log
        }

        # Early stopping
        if early_stopping == 'val_acc':
            do_break = model.early_stopping(valid_accs[-1],epoch)
        elif early_stopping == 'val_loss':
            do_break = model.early_stopping(-valid_losses[-1],epoch)
        elif early_stopping == 'train_acc':
            do_break = model.early_stopping(train_accs[-1],epoch)
        elif early_stopping == 'train_loss':
            do_break = model.early_stopping(-train_losses[-1],epoch)

        if do_break:
            save_weight_dic()
            break

    if not do_break:
        save_weight_dic()

    return {'train_loss':train_losses, 'train_acc': train_accs, 'val_loss':valid_losses, 'val_acc':valid_accs}

def show_report(model, X, y, label_names, split='Train'):
    predicted = model_forward(model, X, y)

    y_pred = predicted.cpu().numpy()
    y_true = y.cpu().numpy()[:,2]

    report = classification_report(
        y_true,
        y_pred,
        target_names=label_names
    )
    return report

def model_forward(model,X,y):
    model.eval()
    with torch.no_grad():
        try:
            outputs = model(X)
        except TypeError:
            outputs = model(X,y)
        _, predicted = torch.max(outputs, 1)
    return predicted

