from math import sqrt

def intersecting_points(line: tuple[float, float], circle: tuple[float, float, float]) -> tuple[float,float]: 
    result: list[tuple[float,float]] = []

    a1, b1 = line
    a2, b2, r = circle

    def get_y(x: float) -> float:
        return a1*x+b1

    a = a1 + 1 
    b = 2*b1*a1 - 2*b2*a1 - 2*a2
    c = b2**2 + b1**2 + a2**2 - r**2 - 2*b2*b1

    # determinent
    D = b**2 - 4*a*c

    if D == 0:
        # one solution
        res_x = ((-1*b) + sqrt(b**2 - 4*a*c))/(2*a)
        result.append((res_x, get_y(res_x)))

    elif D >= 1:
        # two solutions
        res_x1 = ((-1*b) - sqrt(b**2 - 4*a*c))/(2*a)
        res_x2 = ((-1*b) + sqrt(b**2 - 4*a*c))/(2*a)
        
        result.append((res_x1, get_y(res_x1)))
        result.append((res_x2, get_y(res_x2)))

    # D <= -1 -> default no solutions

    return result



def zero_intersecting():
    line = (0, -3)
    circle = (1, 1, 1)
    expected = []

    result = intersecting_points(line, circle)

    assert result == expected, f"Expected {expected}, got {result}"


def one_intersecting():
    line = (0, 0)
    circle = (1, 1, 1)
    expected = [(1, 0)]

    result = intersecting_points(line, circle)

    assert len(result) == 1
    assert abs(result[0][0] - expected[0][0]) < 1e-6
    assert abs(result[0][1] - expected[0][1]) < 1e-6


def two_intersecting():
    line = (0, 1)
    circle = (1, 1, 1)
    expected = [(0, 1), (2, 1)]

    result = intersecting_points(line, circle)

    assert len(result) == 2
    assert abs(result[0][0] - expected[0][0]) < 1e-6
    assert abs(result[0][1] - expected[0][1]) < 1e-6
    assert abs(result[1][0] - expected[1][0]) < 1e-6
    assert abs(result[1][1] - expected[1][1]) < 1e-6


def run_tests():
    zero_intersecting()
    one_intersecting()
    two_intersecting()
    print("All tests passed")


def main():
    run_tests()


if __name__ == "__main__":
    main()
