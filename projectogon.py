from time import sleep

import numpy
import torch.nn
from matplotlib import pyplot
from skgeom import Polygon as SKPolygon, PolygonSet, PolygonWithHoles
from skgeom.draw import draw
from skgeom import minkowski
from sympy import Point2D
from sympy.geometry.polygon import Polygon as SymPolygon
from sympy.geometry.line import Line, Line2D, Point
from skgeom import Sign
from sympy.abc import x, y

affine = torch.nn.Linear(6, 4)
relu = torch.nn.ReLU()

with torch.no_grad():
    affine.bias[:] = 0

epsilon = 0.1
over_approximation = 0.0001 * epsilon

input_data = numpy.zeros(6)


def get_box(input_data, epsilon):
    result_list = []
    for i, j in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        result_list.append((
            input_data[0] + i * epsilon,
            input_data[1] + j * epsilon
        ))
    return numpy.array(result_list)


def plot_polygon(numpy_polygon):
    draw(SKPolygon(numpy_polygon))
    # pyplot.show()


for j in range(0, affine.out_features, 2):
    accumulation_polygon = None
    for i in range(0, affine.in_features, 2):
        sub_input = input_data[i: i + 2]
        input_box = get_box(sub_input, epsilon)
        plot_polygon(input_box)
        sub_affine = torch.nn.Linear(2, 2, affine.bias is not None)
        with torch.no_grad():
            sub_affine.bias[:] = affine.bias[j: j + 2]
            sub_affine.weight[:] = affine.weight[j: j + 2, i: i + 2]
            after_affine = sub_affine(torch.FloatTensor(input_box))
        result_polygon = SKPolygon(after_affine)
        if result_polygon.orientation() == Sign.NEGATIVE:
            result_polygon.reverse_orientation()
        if accumulation_polygon is None:
            accumulation_polygon = result_polygon
        else:
            accumulation_polygon = minkowski.minkowski_sum(accumulation_polygon, result_polygon)  # type: PolygonWithHoles
            accumulation_polygon = accumulation_polygon.outer_boundary()  # type: SKPolygon

    after_affine = accumulation_polygon.coords
    augmented_polygon = []
    for i in range(len(after_affine)):
        current_point = after_affine[i]
        next_point = after_affine[(i + 1) % len(after_affine)]
        between = []
        if current_point[0] * next_point[0] < 0:
            between.append((0,
                            current_point[1] + (
                                    next_point[1] - current_point[1]
                            ) * abs(current_point[0]) / abs(next_point[0] - current_point[0])))
        if current_point[1] * next_point[1] < 0:
            between.append((current_point[0] + (
                    next_point[0] - current_point[0]
            ) * abs(current_point[1]) / abs(next_point[1] - current_point[1]),
                            0))
        between = sorted(between, key=lambda p: (p[0] - current_point[0]) ** 2 + (p[0] - current_point[0]) ** 2)
        augmented_polygon.append(tuple(current_point))
        augmented_polygon.extend(between)
    augmented_polygon = torch.FloatTensor(augmented_polygon)
    with torch.no_grad():
        after_relu_polygon = relu(augmented_polygon + over_approximation) - over_approximation
        after_relu_polygon = SymPolygon(*after_relu_polygon).vertices
    plot_polygon(after_relu_polygon)


def verify(single_input, single_label, linear_layers, radius=epsilon):

    return False
