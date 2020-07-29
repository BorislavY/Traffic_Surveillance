import numpy as np

"""
Pass two lines defined by a point at each end of the line and the function will return True if the intersect.



def calc_params(point1, point2):
    A = (point1[1] - point2[1])
    B = (point2[0] - point1[0])
    C = (point1[0]*point2[1] - point2[0]*point1[1])
    return A, B, -C


def intersection(line1, line2):
    L1 = calc_params(line1[0], line1[1])
    L2 = calc_params(line1[0], line1[1])
    D  = L1[0] * L2[1] - L1[1] * L2[0]

    if D != 0:
        x = Dx / D
        y = Dy / D
        return x,y
    else:
        return False

"""

def intersection(line1, line2):
    xdiff = (line1[0][0] - line1[1][0], line2[0][0] - line2[1][0])
    ydiff = (line1[0][1] - line1[1][1], line2[0][1] - line2[1][1])

