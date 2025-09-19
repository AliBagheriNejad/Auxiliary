import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import tqdm

from torch import device as ddevice
from torch.cuda import is_available
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import set_detect_anomaly

import src.utils as utils
import src.models as models

# import mlflow


device = ddevice("cuda" if is_available() else "cpu")
data_dir = r'F:\thesis\Articles\2nd\cod\Data'
file_name = 'input.pkl'
file_name_label = 'output.pkl'
file_path = os.path.join(data_dir, file_name)
file_path_label = os.path.join(data_dir, file_name_label)

data = utils.read_pkl(file_path)
label = utils.read_pkl(file_path_label)

label_encoded, label_map = utils.label_encoder(label)

X_train, X_test, y_train, y_test = train_test_split(
    data,
    label_encoded,
    test_size = 0.2,
    random_state = 69,
    shuffle = True,
)

(X_train_scaled, X_test_scaled),scaler_list = utils.scale_it(X_train, X_test)

X_train_scaled_tensor, y_train_tensor = utils.tensor_it(X_train_scaled,y_train)
X_test_scaled_tensor, y_test_tensor = utils.tensor_it(X_test_scaled,y_test)

# X_train_scaled_tensor = utils.read_pkl('Data/1024/features_train.pkl')
# X_test_scaled_tensor = utils.read_pkl('Data/1024/features_test.pkl')
# y_train_tensor = utils.read_pkl('Data/1024/label_train.pkl')
# y_test_tensor = utils.read_pkl('Data/1024/label_test.pkl')

train_loader = utils.make_loader(X_train_scaled_tensor,y_train_tensor, 128)
test_loader = utils.make_loader(X_test_scaled_tensor,y_test_tensor, 128)


weight_dir = r'F:\thesis\Articles\2nd\mlruns\994478961421787748\5945e3b605184dd4866fcccf6edc6ace\artifacts'
weight_dir = os.path.join(weight_dir,'test_weight.pth')
network = utils.load_model(models.Network(26), weight_dir)

weight_dir = r'F:\thesis\Articles\2nd\cod\others\aux_weight_1.pth'
# weight_dir = os.path.join(weight_dir,'test_weight.pth')
auxiliary = utils.load_model(
    models.AuxNet(n_layer=1, in_dim=1024*5, out_dim=1024), 
    weight_dir)

feature_extractor = network.feature_extractor
classifier = network.classifier
# auxiliary = models.AuxNet(n_layer=1, in_dim=1024*5, out_dim=1024)
discriminator = models.Classifier(num_classes=2)
generator = models.FeatureExtractor(input_channels=2)

## Training GAN
# ===============================================================
MODE = 'aux_gan'
MODEL_AUX = auxiliary
MODEL_G = generator
MODEL_D = discriminator
MODEL_C = classifier
EPOCHS = 10
TRAIN_DATALOADER = train_loader
TEST_DATALOADER = test_loader
OPTIMIZER_D = optim.Adam(MODEL_D.parameters(), lr=0.00001)
OPTIMIZER_G = optim.Adam(MODEL_G.parameters(), lr = 0.00001)
CRITERION = nn.CrossEntropyLoss()
EARLY_STOPPING = 'test_loss'
SHOW_GRAD = False
P_D = 1
STD = 0.01
MEAN = 10

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
        for k,v in zip(MODEL.weight_dic.keys(), MODEL.weight_dic.values()):
            weight_name = f'{MODE}_{k}_{np.abs(MODEL.metrics_best[k]):.6f}.pth'
            weight_path = os.path.join('temp', weight_name)
            torch.save(v, weight_path)
            print(f'Weight <{weight_path}> saved successfully')

def calc_features_d(batch_data, disc_label):
    
    x = calc_features_g(batch_data)
    fake_features = x[:,2,:]
    if MODEL_AUX.include_y:
        y_one_hot = MODEL_AUX.label_coder(disc_label)
        x = torch.concat([y_one_hot,x], dim=2)
    batch_data = x.flatten(1,2)
    real_features = MODEL_AUX(batch_data)

    return real_features, fake_features

def calc_features_g(batch_data):
    x_fe = batch_data.reshape(batch_data.shape[0]*batch_data.shape[1], batch_data.shape[2], batch_data.shape[3])
    x_fe = MODEL_G(x_fe)
    x = x_fe.reshape(batch_data.shape[0], batch_data.shape[1], -1)
    return x

def add_gaussian_noise(tensor, std=0.05, mean = 2):
    noise = torch.randn_like(tensor) * std + mean
    return tensor + noise

fix_temp()

train_losses_d, train_accs_d, test_losses_d, test_accs_d = [], [], [], []
train_losses_g, train_accs_g, test_losses_g, test_accs_g = [], [], [], []
    
for epoch in range(EPOCHS):
    train_loss_d = 0.0
    correct_train_d = 0
    total_train_d = 0
    train_loss_g = 0.0
    correct_train_g = 0
    total_train_g = 0

    progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'Epoch {epoch + 1}/{EPOCHS}')

    for i,(batch_data, batch_labels) in progress_bar:
        batch_data = batch_data.to(device)
        batch_data = batch_data.permute(0,1,3,2)
        batch_labels = batch_labels.to(device)
        batch_label = batch_labels[:,2].to(device)


        ###############
        # Discriminator
        ###############
        if epoch%P_D == 0:
            real_label = torch.ones_like(batch_label)
            fake_label = torch.zeros_like(batch_label)
            disc_label = torch.concat([real_label,fake_label], dim=0)
            
            MODEL_D.train()
            
            OPTIMIZER_D.zero_grad()

            # Extract V
            with torch.no_grad():
                real_features, fake_features = calc_features_d(batch_data, disc_label)

            features = torch.concat([real_features, fake_features], dim=0)
            features = add_gaussian_noise(features, std=STD, mean=MEAN)
            outputs = MODEL_D(features)
            loss = CRITERION(outputs, disc_label)

            loss.backward()
            OPTIMIZER_D.step()

            train_loss_d += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train_d += disc_label.size(0)
            correct_train_d += (predicted == disc_label).sum().item()

        ########
        # Generator
        #########

        MODEL_G.train()
        gen_label = torch.ones_like(batch_label)
        
        OPTIMIZER_G.zero_grad()

        # gen_features = calc_features_g(batch_data)
        # gen_features = gen_features.flatten(1,2)
        gen_features = MODEL_G(batch_data[:,2,:,:])

        gen_features = add_gaussian_noise(gen_features, std=STD, mean=MEAN)
        outputs = MODEL_D(gen_features)
        loss = CRITERION(outputs, gen_label)

        loss.backward()
        OPTIMIZER_G.step()

        train_loss_g += loss.item()
        _, predicted = torch.max(outputs, 1)
        total_train_g += batch_labels.size(0)
        correct_train_g += (predicted == gen_label).sum().item()

        try:
            progress_bar.set_postfix_str(
                f'train_loss_d={train_loss_d / (i + 1):.4f}\
, train_acc_d={100 * correct_train_d / total_train_d:.4f}\
, train_loss_g={train_loss_g / (i+1):.4f}\
, train_acc_g={100 * correct_train_g / total_train_g:.4f}')
        except ZeroDivisionError:
            progress_bar.set_postfix_str(
                f'train_loss_g={train_loss_g / (i+1):.4f}\
, train_acc_g={100 * correct_train_g / total_train_g:.4f}')
    if epoch%P_D == 0:
        train_loss_log_d = train_loss_d / len(TRAIN_DATALOADER) /2
        train_acc_log_d = 100 * correct_train_d / total_train_d
        train_losses_d.append(train_loss_log_d)
        train_accs_d.append(train_acc_log_d)

    train_loss_log_g = train_loss_g / len(TRAIN_DATALOADER)
    train_acc_log_g = 100 * correct_train_g / total_train_g
    train_losses_g.append(train_loss_log_g)
    train_accs_g.append(train_acc_log_g)

    MODEL_D.eval()
    MODEL_G.eval()
    test_loss_d = 0.0
    correct_test_d = 0
    total_test_d = 0
    test_loss_g = 0.0
    correct_test_g = 0
    total_test_g = 0

    with torch.no_grad():
        for batch_data, batch_labels in TEST_DATALOADER:
            batch_data = batch_data.to(device)
            batch_data = batch_data.permute(0,1,3,2)
            batch_labels = batch_labels.to(device)
            batch_label = batch_labels[:,2]

            
            ###############
            # Discriminator
            ###############

            real_label = torch.ones_like(batch_label)
            fake_label = torch.zeros_like(batch_label)
            disc_label = torch.concat([real_label,fake_label], dim=0)

        # Extract V
            real_features, fake_features = calc_features_d(batch_data, disc_label)

            features = torch.concat([real_features, fake_features], dim=0)
            features = add_gaussian_noise(features, std=STD, mean=MEAN)
            outputs = MODEL_D(features)
            loss = CRITERION(outputs, disc_label)


            test_loss_d += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test_d += disc_label.size(0)
            correct_test_d += (predicted == disc_label).sum().item()
    
            ##########
            # Generator
            ##########
            gen_label = torch.ones_like(batch_label)

            # gen_features = calc_features_g(batch_data)
            # gen_features = gen_features.flatten(1,2)
            gen_features = MODEL_G(batch_data[:,2,:,:])

            gen_features = add_gaussian_noise(gen_features, std=STD, mean=MEAN)
            outputs = MODEL_D(gen_features)
            loss = CRITERION(outputs, gen_label)

            test_loss_g += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_test_g += batch_labels.size(0)
            correct_test_g += (predicted == gen_label).sum().item()     


        # test_loss += loss.item()
        # _, predicted = torch.max(outputs, 1)
        # total_test += batch_labels.size(0)
        # correct_test += (predicted == batch_label).sum().item()

        # test_loss += loss.item()
        # _, predicted = torch.max(outputs,1)
        # total_test += batch_labels.size(0)
        # correct_test += (predicted == batch_label).sum().item()
    # if epoch%P_D == 0:
    test_loss_log_d = test_loss_d / len(TEST_DATALOADER) /2
    test_acc_log_d = 100 * correct_test_d / total_test_d
    test_losses_d.append(test_loss_log_d)
    test_accs_d.append(test_acc_log_d)

    test_loss_log_g = test_loss_g / len(TEST_DATALOADER)
    test_acc_log_g = 100 * correct_test_g / total_test_g
    test_losses_g.append(test_loss_log_g)
    test_accs_g.append(test_acc_log_g)

    # if epoch%P_D == 0:
    print(f'val_loss_D: {test_losses_d[-1]:.4f}, val_acc: {test_accs_d[-1]:.1f}\
, val_loss_g: {test_losses_g[-1]:.4f}, val_accs_g:{test_accs_g[-1]:.1f}', end='\n')
    # else:
    #     print(f'val_loss_g: {test_losses_g[-1]:.4f}, val_accs_g:{test_accs_g[-1]:.1f}', end='\n')

    # MODEL.metrics_now = {
    #             'train_loss': -train_loss_log,
    #             'train_acc': train_acc_log,
    #             'val_acc': test_acc_log,
    #             'val_loss': -test_loss_log,
    #         }

    if SHOW_GRAD:
        print('Grad for D')
        for name, param in MODEL_D.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.mean().item():.10f}")
        print('Grad for G')
        for name, param in MODEL_G.named_parameters():
            if param.grad is not None:
                print(f"{name}: {param.grad.mean().item():.10f}")

    # if EARLY_STOPPING == 'test_acc':
    #         do_break = MODEL.early_stopping(test_accs[-1],epoch)
    # elif EARLY_STOPPING == 'test_loss':
    #     do_break = MODEL.early_stopping(-test_losses[-1],epoch)
    # elif EARLY_STOPPING == 'train_acc':
    #     do_break = MODEL.early_stopping(train_accs[-1],epoch)
    # elif EARLY_STOPPING == 'train_loss':
    #     do_break = MODEL.early_stopping(-train_losses[-1],epoch)
        
    # if do_break:
    #     break
# ===============================================================





## Training just auxiliary
# ===============================================================
# MODE = 'aux_gan'
# MODEL = auxiliary
# EPOCHS = 10
# TRAIN_DATALOADER = train_loader
# TEST_DATALOADER = test_loader
# OPTIMIZER = optim.Adam(auxiliary.parameters(), lr=0.00001)
# CRITERION = nn.CrossEntropyLoss()
# EARLY_STOPPING = 'test_loss'
# SHOW_GRAD = True

# def fix_temp():
#     temp_dir = 'temp'
#     if not os.path.exists(temp_dir):
#         os.makedirs(temp_dir)
#     else:
#         for filename in os.listdir(temp_dir):
#             file_path = os.path.join(temp_dir, filename)
#             if os.path.isfile(file_path):
#                 os.remove(file_path)

# def save_weight_dic():
#         for k,v in zip(MODEL.weight_dic.keys(), MODEL.weight_dic.values()):
#             weight_name = f'{MODE}_{k}_{np.abs(MODEL.metrics_best[k]):.6f}.pth'
#             weight_path = os.path.join('temp', weight_name)
#             torch.save(v, weight_path)
#             print(f'Weight <{weight_path}> saved successfully')

# fix_temp()

# train_losses, train_accs, test_losses, test_accs = [], [], [], []
    
# for epoch in range(EPOCHS):
#     MODEL.train()
#     train_loss = 0.0
#     correct_train = 0
#     total_train = 0

#     progress_bar = tqdm.tqdm(enumerate(TRAIN_DATALOADER), total=len(TRAIN_DATALOADER), desc=f'Epoch {epoch + 1}/{EPOCHS}')

#     for i,(batch_data, batch_labels) in progress_bar:
         
#         batch_data = batch_data.to(device)
#         batch_labels = batch_labels.to(device)
#         batch_label = batch_labels[:,2].to(device)
        
#         OPTIMIZER.zero_grad()

#         # Extract V
#         # with torch.no_grad():
#         #     x_fe = batch_data.reshape(batch_data.shape[0]*batch_data.shape[1], batch_data.shape[2], batch_data.shape[3])
#         #     x_fe = feature_extractor(x_fe)
#         #     x = x_fe.reshape(batch_data.shape[0], batch_data.shape[1], -1)
#         #     if MODEL.include_y:
#         #         y_one_hot = MODEL.label_coder(batch_label)
#         #         x = torch.concat([y_one_hot,x], dim=2)
#         #     batch_data = x.flatten(1,2)
#         batch_data = batch_data.flatten(1,2)
#         features = MODEL(batch_data)
#         outputs = classifier(features)
#         loss = CRITERION(outputs, batch_label)

#         loss.backward()
#         OPTIMIZER.step()

#         train_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total_train += batch_labels.size(0)
#         correct_train += (predicted == batch_label).sum().item()

#         progress_bar.set_postfix_str(
#             f'train_loss={train_loss / (i + 1):.4f}\
#             , train_acc={100 * correct_train / total_train:.4f}')
        
#     train_loss_log = train_loss / len(TRAIN_DATALOADER)
#     train_acc_log = 100 * correct_train / total_train

#     train_losses.append(train_loss_log)
#     train_accs.append(train_acc_log)

#     MODEL.eval()
#     test_loss = 0.0
#     correct_test = 0
#     total_test = 0

#     with torch.no_grad():
#         for batch_data, batch_labels in TEST_DATALOADER:
#             batch_data = batch_data.to(device)
#             batch_labels = batch_labels.to(device)
#             batch_label = batch_labels[:,2]
            
#         batch_data = batch_data.flatten(1,2)
#         features = MODEL(batch_data)
#         outputs = classifier(features)
#         loss = CRITERION(outputs, batch_label)


#         train_loss += loss.item()
#         _, predicted = torch.max(outputs, 1)
#         total_train += batch_labels.size(0)
#         correct_train += (predicted == batch_label).sum().item()

#         test_loss += loss.item()
#         _, predicted = torch.max(outputs,1)
#         total_test += batch_labels.size(0)
#         correct_test += (predicted == batch_label).sum().item()

#     test_loss_log = test_loss / len(TEST_DATALOADER)
#     test_acc_log = 100 * correct_test / total_test
#     test_losses.append(test_loss_log)
#     test_accs.append(test_acc_log)

#     print(f'val_loss: {test_losses[-1]:.4f}, val_acc: {test_accs[-1]:.1f}', end='\n')

#     MODEL.metrics_now = {
#                 'train_loss': -train_loss_log,
#                 'train_acc': train_acc_log,
#                 'val_acc': test_acc_log,
#                 'val_loss': -test_loss_log,
#             }

#     if SHOW_GRAD:
#         for name, param in MODEL.named_parameters():
#             if param.grad is not None:
#                 print(f"{name}: {param.grad.mean().item():.10f}")

#     if EARLY_STOPPING == 'test_acc':
#             do_break = MODEL.early_stopping(test_accs[-1],epoch)
#     elif EARLY_STOPPING == 'test_loss':
#         do_break = MODEL.early_stopping(-test_losses[-1],epoch)
#     elif EARLY_STOPPING == 'train_acc':
#         do_break = MODEL.early_stopping(train_accs[-1],epoch)
#     elif EARLY_STOPPING == 'train_loss':
#         do_break = MODEL.early_stopping(-train_losses[-1],epoch)
        
#     if do_break:
#         break
# ===============================================================