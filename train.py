import numpy as np
import os
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torch import device as ddevice
from torch.cuda import is_available
import torch.nn as nn
import torch.optim as optim

import src.utils as utils
import src.models as models

import mlflow


device = ddevice("cuda" if is_available() else "cpu")
data_dir = r'F:\thesis\Articles\2nd\code\Data'
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

train_loader = utils.make_loader(X_train_scaled_tensor,y_train_tensor, 128)
test_loader = utils.make_loader(X_test_scaled_tensor,y_test_tensor, 128)


experiment_name = 'Auxilixary'
experiment = mlflow.get_experiment_by_name(experiment_name)
if experiment is None:
    mlflow.create_experiment(experiment_name)
mlflow.set_experiment(experiment_name)

mlflow.start_run()
mlflow.log_param('label map',label_map)
mlflow.log_param('data_shape', data.shape)

model = models.Model2(26).to(device)
model.save_path = r'F:\thesis\Articles\2nd\code\temp\test_weight.pth'
model.patience = 50
model.best_acc = -100
model.e_ratio = 1000

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
a = utils.train_classifier(
    model,
    criterion,
    optimizer,
    train_loader,
    test_loader,
    epochs=3,
    early_stopping='val_loss',
    alpha = 0.5,
)



train_cr = utils.show_report(model, X_train_scaled_tensor, y_train_tensor, list(label_map.keys()))
test_cr = utils.show_report(model, X_test_scaled_tensor, y_test_tensor, list(label_map.keys()), split='Test')
train_cm = utils.calc_cm(model, X_train_scaled_tensor, y_train_tensor)
test_cm = utils.calc_cm(model, X_test_scaled_tensor, y_test_tensor)

utils.save_cm(train_cm, list(label_map.keys()))
utils.save_cm(test_cm, list(label_map.keys()), split="Test")



with open(r'F:\thesis\Articles\2nd\code\temp\train_cr.txt', 'w', encoding='utf-8') as f:
    f.write(train_cr)
with open(r'F:\thesis\Articles\2nd\code\temp\test_cr.txt', 'w', encoding='utf-8') as f:
    f.write(train_cr)

mlflow.log_artifacts(r'F:\thesis\Articles\2nd\code\temp')
mlflow.end_run()




