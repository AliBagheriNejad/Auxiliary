from src.models import *
import src.utils as utils
import os
from sklearn.model_selection import train_test_split
import torch
import pickle

from mlflow.tracking import MlflowClient

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def test_models():
    m_type = 'classic'

    if m_type == 'aux':
        model = Model2(30,2)
        print(model)
        model.to('cpu')

        num = 1
        input_test = torch.randn(num, 5, 1024, 2)
        label_test = torch.ones(size=(num,5)).long()
        output = model(input_test, label_test)

        print(output.shape)

    else:
        model = Network(30,2)
        print(model)
        model.to('cpu')

        num = 10
        input_test = torch.randn(num,1024,2)
        output_test = torch.ones(size=(num,1)).long()
        output,_ = model(input_test)

        print(output.shape)

def get_data():
    global label_map
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

    return X_train_scaled_tensor, y_train_tensor, X_test_scaled_tensor, y_test_tensor, label_map

def show_feat_dist():
    # weight_dir = r'F:\thesis\Articles\2nd\code\temp\test_weight.pth'
    art_dir2 = r'F:\thesis\Articles\2nd\mlruns-20250726T033440Z-1-001\mlruns\994478961421787748\5945e3b605184dd4866fcccf6edc6ace\artifacts'
    art_dir = r'F:\thesis\Articles\2nd\mlruns-20250726T033440Z-1-001\mlruns\623922822895938578\26f61da787ea448cb55e49f781245ffe\artifacts'
    weight_dir = os.path.join(art_dir,'test_weight.pth')
    weight_dir2 = os.path.join(art_dir2,'test_weight.pth')

    X, y, X_t, y_t, lm = get_data()

    model = Model2(26)
    model2 = Network(26)
    model.load_state_dict(torch.load(weight_dir, map_location=device))
    model2.load_state_dict(torch.load(weight_dir2, map_location=device))

    featus = utils.get_features(model.cls, X_t)
    featus2 = utils.get_features(model2, X_t)

    embeding = utils.reduce_dim(featus[:,:1025])
    embeding2 = utils.reduce_dim(featus2[:,:1025])
    utils.plot_dist(embeding, y_t, list(label_map.keys()), True, 'Auxiliary features')
    utils.plot_dist(embeding2, y_t, list(label_map.keys()), False, 'Usual feature')
    report = utils.show_report(model, X_t, y_t, list(lm.keys()))
    report2 = utils.show_report(model2, X_t, y_t, list(lm.keys()))
    print(report)
    print(report2)


art_dir2 = r'F:\thesis\Articles\2nd\mlruns-20250726T033440Z-1-001\mlruns\994478961421787748\5945e3b605184dd4866fcccf6edc6ace\artifacts'
art_dir = r'F:\thesis\Articles\2nd\mlruns-20250726T033440Z-1-001\mlruns\623922822895938578\26f61da787ea448cb55e49f781245ffe\artifacts'
weight_dir = os.path.join(art_dir,'test_weight.pth')
weight_dir2 = os.path.join(art_dir2,'test_weight.pth')

X, y, X_t, y_t, lm = get_data()

# model = utils.load_model(Model2(26), weight_dir)
model2 = utils.load_model(Network(26), weight_dir2)
label_coder = lambda x: torch.nn.functional.one_hot(x,num_classes=26)

# y_hat = utils.model_forward(model, X_t, y_t)
# y_hat2 = utils.model_forward(model2, X_t, y_t)

# pred_dic2 = utils.pred_mat(X_t, y_t, y_hat2, list(lm.values()))

features_train = torch.zeros(len(X),5,1088)
features_test = torch.zeros(len(X_t),5,1088)
model2.eval()
with torch.no_grad():
    for i, input_train in enumerate(X):

        output, embed = model2(input_train)
        embed = embed[:,:-64] # Delete classifier features (keep FE features)
        output,_ = torch.max(output, 1)
        output = output.long()
        y_code = label_coder(output)

        features_train[i] = torch.concat([y_code, embed], dim=1)
    for i, input_train in enumerate(X_t):

        output, embed = model2(input_train)
        embed = embed[:,:-64] # Delete classifier features (keep FE features)
        output,_ = torch.max(output, 1)
        output = output.long()
        y_code = label_coder(output)

        features_test[i] = torch.concat([y_code, embed], dim=1)

with open('features_train.pkl', 'wb') as file:
    pickle.dump(features_train, file)
with open('features_test.pkl', 'wb') as file:
    pickle.dump(features_test, file)
with open('y_train.pkl', 'wb') as file:
    pickle.dump(y, file)
with open('y_test.pkl', 'wb') as file:
    pickle.dump(y_t, file)




