import csv

import onnx
from onnx2pytorch import ConvertModel

import torch.utils.data
import torchvision.datasets
import projectogon

onnx_model = onnx.load('/home/aasgarik/Desktop/Code/ERAN/nets/onnx/mnist/mnist_relu_3_50.onnx')
model = ConvertModel(onnx_model, experimental=True)
epsilon = 0.02


tests = csv.reader(open('/home/aasgarik/Desktop/Code/ERAN/data/mnist_test.csv'))

total = 0
correct = 0
verified = 0
for i, test in enumerate(tests):
    x = torch.FloatTensor([[int(j) for j in test[1:len(test)]]]) / 255
    y = torch.LongTensor([int(test[0])])
    output = model(x)
    current_correct = int(torch.sum(torch.topk(output, 1).indices.reshape(y.shape) == y))
    current_verified = 0
    if current_correct:
        if projectogon.verify(x[0], y[0], [m for m in model.modules() if isinstance(m, torch.nn.Linear)],  epsilon):
            current_verified = 1
    verified += current_verified
    correct += current_correct
    status = "Wrong"
    if current_correct:
        status = "Correct"
    if current_verified:
        status = "Verified"
    print("Image: {}\tStatus: {}".format(i, status))
    total += y.nelement()
print(verified, verified / correct, correct, correct / total, total)
