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

def make_features(li=False):
    art_dir2 = r'F:\thesis\Articles\2nd\mlruns\994478961421787748\5945e3b605184dd4866fcccf6edc6ace\artifacts'
    weight_dir2 = os.path.join(art_dir2,'test_weight.pth')

    X, y, X_t, y_t, lm = get_data()

    model2 = utils.load_model(Network(26), weight_dir2)
    label_coder = lambda x: torch.nn.functional.one_hot(x,num_classes=26)

    features_train = torch.zeros(len(X),5,1024+26)
    features_test = torch.zeros(len(X_t),5,1024+26)
    model2.eval()
    with torch.no_grad():
        for i, input_train in enumerate(X):

            output, embed = model2(input_train, latent_in=False)
            output,_ = torch.max(output, 1)
            output = output.long()
            y_code = label_coder(output)
            y_code[2,:]  = 0

            if li:
                features_train[i] = torch.concat([y_code, embed], dim=1)
            else :
                features_train[i] = embed

        for i, input_train in enumerate(X_t):

            output, embed = model2(input_train, latent_in=False)
            output,_ = torch.max(output, 1)
            output = output.long()
            y_code = label_coder(output)
            y_code[2,:]  = 0
            if li:
                features_test[i] = torch.concat([y_code, embed], dim=1)
            else :
                features_test[i] = embed

    print(features_train.shape)
    print(features_test.shape)
    print(y.shape)
    print(y_t.shape)

    with open('Data/1024y/features_train.pkl', 'wb') as file:
        pickle.dump(features_train, file)
    with open('Data/1024y/features_test.pkl', 'wb') as file:
        pickle.dump(features_test, file)
    with open('Data/1024y/label_train.pkl', 'wb') as file:
        pickle.dump(y, file)
    with open('Data/1024y/label_test.pkl', 'wb') as file:
        pickle.dump(y_t, file)



def kl_divergence_builtin(tensor1, tensor2, epsilon=1e-8):
    """
    Calculate KL divergence using PyTorch's built-in KL divergence function.
    """
    assert tensor1.shape == tensor2.shape, "Tensors must have the same shape"
    
    # Add epsilon and normalize
    tensor1_safe = tensor1 + epsilon
    tensor2_safe = tensor2 + epsilon
    
    p = tensor1_safe / tensor1_safe.sum(dim=1, keepdim=True)
    q = tensor2_safe / tensor2_safe.sum(dim=1, keepdim=True)
    
    # Use PyTorch's KL divergence (returns per-sample KL, then take mean)
    kl_div = F.kl_div(torch.log(q), p, reduction='none').sum(dim=1)
    return kl_div.mean()

X, y, X_t, y_t, lm = get_data()
X = X.permute(0,1,3,2)
X_t = X_t.permute(0,1,3,2)


weight_dir = r'F:\thesis\Articles\2nd\mlruns\994478961421787748\5945e3b605184dd4866fcccf6edc6ace\artifacts'
weight_dir = os.path.join(weight_dir,'test_weight.pth')
network = utils.load_model(Network(26), weight_dir)

weight_dir = r'F:\thesis\Articles\2nd\cod\others\aux_weight_1.pth'
# weight_dir = os.path.join(weight_dir,'test_weight.pth')
auxiliary = utils.load_model(
    AuxNet(n_layer=1, in_dim=1024*5, out_dim=1024), 
    weight_dir)

feature_extractor = network.feature_extractor
classifier = network.classifier
# auxiliary = models.AuxNet(n_layer=1, in_dim=1024*5, out_dim=1024)
discriminator = Classifier(num_classes=2)
generator = FeatureExtractor(input_channels=2)

def calc_features_g(batch_data):
    x_fe = batch_data.flatten(0,1)
    x_fe = generator(x_fe)
    x = x_fe.reshape(batch_data.shape[0], batch_data.shape[1], -1)
    return x

batch_data = X
x = calc_features_g(batch_data)
fake_features = x[:,2,:]
batch_data = x.flatten(1,2)
real_features = auxiliary(batch_data)






