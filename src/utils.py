from enum import Enum
import cv2
import math

class DetectionMode(Enum):
    DETECTION_5 = 0
    DETECTION_106 = 1

def change_reference(old_x, old_y, new_x, new_y):
    delta_x = old_x + new_x
    delta_y = old_y + new_y
    return lambda coord: (coord[0]-delta_x, coord[1]-delta_y)

def resize_and_border(img, desired_size):
    old_size = img.shape[:2] # old_size - (height, width)

    ratio = float(desired_size)/max(old_size)
    new_size = tuple([int(x*ratio) for x in old_size]) # new_size - (width, height)

    img = cv2.resize(img, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h//2, delta_h-(delta_h//2)
    left, right = delta_w//2, delta_w-(delta_w//2)

    borders = {
        'top': top,
        'bottom': bottom,
        'left': left,
        'right': right
    }

    color = [0, 0, 0]
    new_img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return new_img, borders, (new_size[1], new_size[0])

def resize_point(point, old_shape, new_shape, border_left, border_top):
    Ry = new_shape[1]/old_shape[1]
    Rx = new_shape[0]/old_shape[0]
    return (border_left + Rx * point[0], border_top + Ry * point[1])

def affine_transform(img, src_points, dst_points):
    rows, cols = img.shape[:2]
    affine_matrix = cv2.getAffineTransform(src_points, dst_points)
    img_output = cv2.warpAffine(img, affine_matrix, (cols,rows))
    return img_output, affine_matrix

def equilateral_triangle(point1, point2):
    x1, y1 = point1
    x2, y2 = point2

    dx = x2 - x1
    dy = y2 - y1

    alpha = 60./180*math.pi

    xp = x1 + math.cos( alpha)*dx + math.sin(alpha)*dy
    yp = y1 + math.sin(-alpha)*dx + math.cos(alpha)*dy

    return (xp, yp)