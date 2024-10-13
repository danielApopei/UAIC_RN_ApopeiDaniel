def convert_linear(linear_equation: str) -> list:
    x_coefficient, rest = linear_equation.split("x")
    y_coefficient, rest = rest.split("y")
    z_coefficient, rest = rest.split("z")
    preliminary_result = [x_coefficient, y_coefficient, z_coefficient]
    result = []
    for el in preliminary_result:
        if el == "" or el == "+":
            result.append(1)
        elif el == "-":
            result.append(-1)
        else:
            result.append(int(el))
    return result


def read_from_file(file_path: str) -> tuple[list, list]:
    matrix = []
    vector = []

    with open(file_path, "r") as file:
        for line in file:
            line = line.strip().replace(" ", "")
            left_side, right_side = line.split("=")
            matrix.append(left_side)
            vector.append(right_side)

    matrix = [convert_linear(equation) for equation in matrix]
    vector = list(map(int, vector))

    return matrix, vector
