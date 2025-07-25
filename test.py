from src.models import *
import src.utils as utils
import os
from sklearn.model_selection import train_test_split
import torch

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

    return X_train_scaled_tensor, y_train_tensor, X_test_scaled_tensor, y_test_tensor

def show_feat_dist():
    weight_dir = r'F:\thesis\Articles\2nd\code\temp\test_weight.pth'

    X, y, X_t, y_t = get_data()

    model = Model2(26)
    model.load_state_dict(torch.load(weight_dir, map_location=device))

    featus = utils.get_features(model, X)
    embeding = utils.reduce_dim(featus)
    utils.plot_dist(embeding, y, list(label_map.keys()))




