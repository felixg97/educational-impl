import math


def eucledian_distance(x_1, x_2):
    sum_squared_differences = 0

    # summation of the squared differences
    for i in range(len(x_1)):
        sum_squared_differences += (x_1[i] - x_2[i])**2

    # taking the square root of the sum of squared differences
    distance = math.sqrt(sum_squared_differences)

    return distance


def manhattan_distance(x_1, x_2):
    sum_absolute_differences = 0

    # summation of the absolute differences
    for i in range(len(x_1)):
        sum_absolute_differences += abs(x_1[i] - x_2[i])

    return sum_absolute_differences 