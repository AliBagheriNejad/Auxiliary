from src.models import *


model = Model2(10,2)
print(model)
model.to('cpu')

num = 128
input_test = torch.randn(num, 5, 1024, 2)
label_test = torch.ones(size=(num,5)).long()
output = model(input_test, label_test)

print(output.shape)