import numpy
import torch.nn
from skgeom import Polygon as SKPolygon, PolygonWithHoles
from skgeom import Sign
from skgeom import minkowski
from skgeom.draw import draw
from sympy import Line2D, Point
from sympy.geometry.polygon import Polygon as SymPolygon

relu = torch.nn.ReLU()

epsilon = 0.1
over_approximation = 0.0001 * epsilon


def get_box(input_data, radius):
    result_list = []
    for i, j in ((1, -1), (1, 1), (-1, 1), (-1, -1)):
        result_list.append((
            input_data[0] + i * radius,
            input_data[1] + j * radius
        ))
    return numpy.array(result_list)


def plot_polygon(numpy_polygon):
    return
    draw(SKPolygon(numpy_polygon))
    # pyplot.show()


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


def forward_one_output_polygon(affine, output_indices, projectogon):
    accumulation_polygon = None
    for projectogon_index, input_box in enumerate(projectogon):
        i = projectogon_index * 2
        plot_polygon(input_box)
        sub_affine = torch.nn.Linear(2, 2, affine.bias is not None)
        with torch.no_grad():
            sub_affine.bias[:] = affine.bias[output_indices]
            sub_affine.weight[:] = affine.weight[output_indices, i: i + 2]
            after_affine = sub_affine(torch.FloatTensor(input_box))
        result_polygon = SKPolygon(after_affine)
        if result_polygon.orientation() == Sign.NEGATIVE:
            result_polygon.reverse_orientation()
        if accumulation_polygon is None:
            accumulation_polygon = result_polygon  # type: PolygonWithHoles
        else:
            accumulation_polygon = minkowski.minkowski_sum(accumulation_polygon,
                                                           result_polygon)  # type: PolygonWithHoles
            accumulation_polygon = accumulation_polygon.outer_boundary()  # type: SKPolygon
    after_affine = accumulation_polygon.coords
    augmented_polygon = augment_polygon_at_zero(after_affine)
    with torch.no_grad():
        after_relu_polygon = relu(augmented_polygon + over_approximation) - over_approximation
        after_relu_polygon = SymPolygon(*after_relu_polygon).vertices
    plot_polygon(after_relu_polygon)
    return after_relu_polygon


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
        between = sorted(between, key=lambda p: (p[0] - current_point[0]) ** 2 + (p[0] - current_point[0]) ** 2)
        augmented_polygon.append(tuple(current_point))
        augmented_polygon.extend(between)
    augmented_polygon = torch.FloatTensor(augmented_polygon)
    return augmented_polygon


def forward_to_logit_projectogon(projectogon, affine, single_label):
    output_projectogon = []
    for j in range(affine.out_features):
        if j == single_label:
            continue
        output_indices = [single_label, j]
        after_relu_polygon = forward_one_output_polygon(affine, output_indices, projectogon)
        output_projectogon.append(after_relu_polygon)
    return output_projectogon


def verify_logit_projectogon(projectogon):
    classification_boundary = Line2D(Point(0, 0), slope=1)
    for polygon_points in projectogon:
        polygon = SymPolygon(*polygon_points)
        if not polygon.intersect(classification_boundary).is_empty:
            return False
    return True


def verify(single_input, single_label, linear_layers, radius=epsilon):
    projectogon = input2projectogon(single_input, radius)
    for layer in linear_layers[:-1]:
        projectogon = forward_to_hidden_projectogon(projectogon, layer)
    projectogon = forward_to_logit_projectogon(projectogon, linear_layers[-1], single_label)
    return verify_logit_projectogon(projectogon)
