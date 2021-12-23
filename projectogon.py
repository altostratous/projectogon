import random

import numpy
import torch.nn
from skgeom import Polygon as SKPolygon, PolygonWithHoles
from skgeom import Sign
from skgeom import minkowski
from skgeom import boolean_set
from skgeom.draw import draw
import skgeom
from sympy import Line2D, Point, Segment2D
from sympy.geometry.polygon import Polygon as SymPolygon
from skgeom import simplify
from matplotlib import pyplot

relu = torch.nn.ReLU()

epsilon = 0.1
over_approximation = 0.0001 * epsilon
simplification_count = 32


def get_box(input_data, radius):
    result_list = []
    for i, j in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        result_list.append((
            min(1, max(0, input_data[0] + i * radius)),
            min(1, max(0, input_data[1] + j * radius))
        ))
    return numpy.array(result_list)


def plot_polygon(numpy_polygon):
    sk_polygon = SKPolygon(numpy_polygon)
    draw(sk_polygon)
    print("simplicity", sk_polygon.is_simple())
    pyplot.show()


def input2projectogon(single_input, radius):
    projectogon = []
    for i in range(0, len(single_input), 2):
        sub_input = single_input[i: i + 2]
        input_box = get_box(sub_input, radius)
        projectogon.append(input_box)
    return projectogon


def forward_to_hidden_projectogon(projectogon, affine):
    output_projectogon = []
    for j in range(0, affine.out_features, 2):
        output_indices = [j, j + 1]
        after_relu_polygon = forward_one_output_polygon(affine, output_indices, projectogon)
        output_projectogon.append(after_relu_polygon)
    return output_projectogon


def forward_one_output_polygon(affine, output_indices, projectogon, activate=True):
    transformed_polygons = []
    for projectogon_index, input_polygon in enumerate(projectogon):
        if input_polygon is None:
            continue
        i = projectogon_index * 2
        # plot_polygon(input_polygon)
        sub_affine = torch.nn.Linear(2, 2, False)
        with torch.no_grad():
            # TODO trivial test on epsilon very small
            sub_affine.weight[:] = affine.weight[output_indices, i: i + 2]
            after_affine = sub_affine(torch.FloatTensor(input_polygon))
        # after_affine = thicken_if_needed(after_affine)
        # if after_affine is None:
        #     continue
        result_polygon = SKPolygon(after_affine)
        if not result_polygon.is_simple():
            result_polygon = SKPolygon(skgeom.convex_hull.graham_andrew(after_affine))
            if result_polygon.orientation() == Sign.ZERO:
                result_polygon = SKPolygon(thicken_if_needed(result_polygon.coords))
        if result_polygon.orientation() == Sign.NEGATIVE:
            result_polygon.reverse_orientation()
        transformed_polygons.append(result_polygon)

    accumulation_polygon = custom_reduce(transformed_polygons, reduce_minkowski)

    after_affine = torch.FloatTensor(accumulation_polygon.coords)
    after_affine += affine.bias[output_indices]
    if not activate:
        return after_affine
    augmented_polygon = augment_polygon_at_zero(after_affine)
    with torch.no_grad():
        after_relu_polygon = relu(augmented_polygon + over_approximation) - over_approximation
        after_relu_polygon = thicken_if_needed(after_relu_polygon)
    return after_relu_polygon


def custom_reduce(l, func, reduction='linear'):
    if len(l) == 0:
        return None
    if len(l) == 1:
        return l[0]
    if reduction == 'logarithmic':
        split_index = len(l) // 2
    elif reduction == 'linear':
        split_index = 1
    else:
        assert False
    first = custom_reduce(l[:split_index], func, reduction)
    second = custom_reduce(l[split_index:], func, reduction)
    result = reduce_minkowski(first, second)
    return result


def reduce_minkowski(a, b):
    if a is None:
        return b
    if b is None:
        return a
    accumulation_polygon = minkowski.minkowski_sum(a, b)  # type: PolygonWithHoles
    accumulation_polygon = accumulation_polygon.outer_boundary()  # type: SKPolygon
    # TODO make sure simplify over-approximates
    accumulation_polygon = simplify(accumulation_polygon, simplification_count, "count")
    return accumulation_polygon


def thicken_if_needed(after_relu_polygon):
    supposedly_polygon = SymPolygon(*after_relu_polygon)
    if isinstance(supposedly_polygon, SymPolygon):
        after_relu_polygon = supposedly_polygon.vertices
    elif isinstance(supposedly_polygon, Segment2D):
        thickened = thicken(supposedly_polygon)
        after_relu_polygon = thickened.vertices
    else:
        after_relu_polygon = None
    return after_relu_polygon


def thicken(supposedly_polygon):
    points = [tuple(arg) for arg in supposedly_polygon.args]
    for point in reversed(supposedly_polygon.args):
        x, y = point.args
        ratio = random.random()
        y += (1 - ratio) * over_approximation
        x += ratio * over_approximation
        points.append((x, y))
    thickened = SymPolygon(*points)
    return thickened


def augment_polygon_at_zero(after_affine):
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
        between = sorted(between, key=lambda p: (p[0] - current_point[0]) ** 2 + (p[1] - current_point[1]) ** 2)
        augmented_polygon.append(tuple(current_point))
        augmented_polygon.extend(between)
    augmented_polygon = torch.FloatTensor(augmented_polygon)
    return augmented_polygon


def forward_to_logit_projectogon(projectogon, affine, single_label):
    single_label = int(single_label)
    output_projectogon = []
    for j in range(affine.out_features):
        if j == single_label:
            continue
        output_indices = [single_label, j]
        after_relu_polygon = forward_one_output_polygon(affine, output_indices, projectogon, False)
        output_projectogon.append(after_relu_polygon)
    return output_projectogon


def verify_logit_projectogon(projectogon):
    classification_boundary = Line2D(Point(0, 0), slope=1)
    for polygon_points in projectogon:
        # plot_polygon(polygon_points)
        polygon = SymPolygon(*polygon_points)
        if not polygon.intersect(classification_boundary).is_empty:
            return False
    return True


def verify(single_input, single_label, linear_layers, radius=epsilon):
    projectogon = input2projectogon(single_input, radius)
    for layer in linear_layers[:-1]:
        projectogon = forward_to_hidden_projectogon(projectogon, layer)
        # print('done with one layer')
    projectogon = forward_to_logit_projectogon(projectogon, linear_layers[-1], single_label)
    return verify_logit_projectogon(projectogon)
