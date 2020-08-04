import cv2
import numpy as np

'''
Functions used in the main script.

"intersects", "orientation" and "on_segment" are used for determining if two line segments intersect.
The code is taken from https://www.kite.com/python/answers/how-to-check-if-two-line-segments-intersect-in-python

"scale" uses linear interpolation to scale a line segment by a given factor.
The code is taken from https://stackoverflow.com/questions/28825461/how-to-extend-a-line-segment-in-both-directions

"on_mouse" is a mouse callback function for OpenCV, used to display the coordinates of a mouse click in the console.

'''


def on_segment(p, q, r):
    if max(p[0], q[0]) >= r[0] >= min(p[0], q[0]) and max(p[1], q[1]) >= r[1] >= min(p[1], q[1]):
        return True
    return False


def orientation(p, q, r):
    val = ((q[1] - p[1]) * (r[0] - q[0])) - ((q[0] - p[0]) * (r[1] - q[1]))
    if val == 0: return 0
    return 1 if val > 0 else -1


def intersects(seg1, seg2):
    p1, q1 = seg1
    p2, q2 = seg2

    o1 = orientation(p1, q1, p2)
    o2 = orientation(p1, q1, q2)
    o3 = orientation(p2, q2, p1)
    o4 = orientation(p2, q2, q1)

    if o1 != o2 and o3 != o4:
        return True

    if o1 == 0 and on_segment(p1, q1, p2):
        return True

    if o2 == 0 and on_segment(p1, q1, q2):
        return True

    if o3 == 0 and on_segment(p2, q2, p1):
        return True

    if o4 == 0 and on_segment(p2, q2, q1):
        return True

    return False


def scale(line, factor):
    t0 = 0.5 * (1.0-factor)
    t1 = 0.5 * (1.0+factor)

    x_p1 = line[0][0]
    y_p1 = line[0][1]
    x_p2 = line[1][0]
    y_p2 = line[1][1]

    x1 = x_p1 + (x_p2 - x_p1) * t0
    y1 = y_p1 + (y_p2 - y_p1) * t0
    x2 = x_p1 + (x_p2 - x_p1) * t1
    y2 = y_p1 + (y_p2 - y_p1) * t1

    return (int(x1), int(y1)), (int(x2), int(y2))


def on_mouse(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        print('x = %d, y = %d'%(x, y))