import csv
import os
import sys
import time
import traceback

import onnx
import torch.utils.data
from onnx2pytorch import ConvertModel

import projectogon

onnx_model = onnx.load('/home/aasgarik/Desktop/Code/ERAN/nets/onnx/mnist/mnist_relu_3_50.onnx')
model = ConvertModel(onnx_model, experimental=True)
epsilon = float(os.environ.get("EPSILON"))


tests = csv.reader(open('/home/aasgarik/Desktop/Code/ERAN/data/mnist_test.csv'))

total = 0
correct = 0
verified = 0
error = 0
for i, test in enumerate(tests):
    x = torch.FloatTensor([[int(j) for j in test[1:len(test)]]]) / 255
    y = torch.LongTensor([int(test[0])])
    output = model(x)
    current_correct = int(torch.sum(torch.topk(output, 1).indices.reshape(y.shape) == y))
    current_verified = 0
    current_error = 0
    if current_correct:
        start = time.time()
        try:
            if projectogon.verify(x[0], y[0], [m for m in model.modules() if isinstance(m, torch.nn.Linear)],  epsilon):
                current_verified = 1
        except Exception as e:
            traceback.print_exc(file=sys.stdout)
            current_error = 1
        duration = time.time() - start

    verified += current_verified
    correct += current_correct
    error += current_error
    status = "Wrong"
    if current_correct:
        status = "Correct"
    if current_verified:
        status = "Verified"
    if current_error:
        status = "Error"

    print("Image: {}\tStatus: {}\tDuration: {}".format(i, status, duration))
    total += y.nelement()
    print(verified, verified / correct, correct, correct / total, total)
    sys.stdout.flush()
