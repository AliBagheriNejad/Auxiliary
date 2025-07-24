from src.models import *

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