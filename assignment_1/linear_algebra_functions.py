def determinant(matrix: list[list[float]]) -> float:
    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[0][1] * matrix[1][0]
    a = matrix[0][0] * matrix[1][1] * matrix[2][2]
    b = matrix[0][1] * matrix[1][2] * matrix[2][0]
    c = matrix[1][0] * matrix[2][1] * matrix[0][2]
    d = matrix[0][2] * matrix[1][1] * matrix[2][0]
    e = matrix[0][1] * matrix[1][0] * matrix[2][2]
    f = matrix[1][2] * matrix[2][1] * matrix[0][0]
    return a+b+c-d-e-f


def trace(matrix: list[list[float]]) -> float:
    return matrix[0][0] * matrix[1][1] * matrix[2][2]


def norm(vector: list[float]) -> float:
    return (vector[0] ** 2 + vector[1] ** 2 + vector[2] ** 2) ** 0.5


def transpose(matrix: list[list[float]]) -> list[list[float]]:
    return [row for row in zip(*matrix)]


def multiply(matrix: list[list[float]], vector: list[float]) -> list[float]:
    # abc  d
    # def  e
    # ghi  f
    term_0 = matrix[0][0] * vector[0] + matrix[0][1] * vector[1] + matrix[0][2] * vector[2]
    term_1 = matrix[1][0] * vector[0] + matrix[1][1] * vector[1] + matrix[1][2] * vector[2]
    term_2 = matrix[2][0] * vector[0] + matrix[2][1] * vector[1] + matrix[2][2] * vector[2]
    return [term_0, term_1, term_2]


def replace_column(matrix, vector, col):
    result = []
    for i, row_in in enumerate(matrix):
        new_row = []
        for j, el in enumerate(row_in):
            if j == col:
                new_row.append(vector[i])
            else:
                new_row.append(el)
        result.append(new_row)
    return result


def solve_cramer(matrix: list[list[float]], vector: list[float]) -> list[float]:

    det_a = determinant(matrix)
    trans = transpose(matrix)
    if det_a == 0:
        raise ValueError("Determinant is 0!")
    ax = replace_column(matrix.copy(), vector, 0)
    ay = replace_column(matrix.copy(), vector, 1)
    az = replace_column(matrix.copy(), vector, 2)

    solution = [determinant(ax) / det_a, determinant(ay) / det_a, determinant(az) / det_a]
    return solution


def minor(matrix, i, j) -> list[list[float]]:
    new_matrix = []
    for ii, row in enumerate(matrix):
        if ii == i:
            continue
        new_row = []
        for jj, el in enumerate(row):
            if jj == j:
                continue
            new_row.append(el)
        new_matrix.append(new_row)
    return new_matrix


def element_cofactor(matrix, i, j):
    return (-1) ** (i + j) * determinant(minor(matrix, i, j))


def cofactor(matrix):
    n = len(matrix)
    m = len(matrix[0])
    cofactor_matrix = []
    for i in range(n):
        row = []
        for j in range(m):
            row.append(element_cofactor(matrix, i, j))
        cofactor_matrix.append(row)
    return cofactor_matrix


def adjoint(matrix):
    return transpose(cofactor(matrix))


def inverse(matrix):
    adj = adjoint(matrix)
    coefficient = 1 / determinant(matrix)
    inv = []
    for row in adj:
        new_row = []
        for el in row:
            new_row.append(el * coefficient)
        inv.append(new_row)
    return inv


def solve_inversion(matrix, vector):
    return multiply(inverse(matrix), vector)


matrix = [[2, 3, -1], [1, -1, 4], [3, 1, 2]]
vector = [5, 6, 7]

inv = inverse(matrix)
for row in inv:
    print(row)
print(f"\n{vector}")
print(multiply(inv, vector))