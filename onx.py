import csv

import onnx
from onnx2pytorch import ConvertModel

import torch.utils.data
import torchvision.datasets
import projectogon

onnx_model = onnx.load('/home/aasgarik/Desktop/Code/ERAN/nets/onnx/mnist/mnist_relu_3_50.onnx')
model = ConvertModel(onnx_model, experimental=True)
epsilon = 0.1


criterion = torch.nn.CrossEntropyLoss()
transform = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()])

tests = csv.reader(open('/home/aasgarik/Desktop/Code/ERAN/data/mnist_test.csv'))

total = 0
correct = 0
verified = 0
for test in tests:
    x = torch.FloatTensor([[int(i) for i in test[1:len(test)]]]) / 255
    y = torch.LongTensor([int(test[0])])
    output = model(x)
    current_correct = int(torch.sum(torch.topk(output, 1).indices.reshape(y.shape) == y))
    if current_correct:
        if projectogon.verify(x[0], y[0], [m for m in model.modules() if isinstance(m, torch.nn.Linear)],  epsilon):
            verified += 1
    correct += current_correct
    total += y.nelement()
print(verified, verified / correct, correct, correct / total, total)
