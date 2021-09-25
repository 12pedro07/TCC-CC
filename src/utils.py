from enum import Enum

class DetectionMode(Enum):
    DETECTION_5 = 0
    DETECTION_106 = 1

def change_reference(old_x, old_y, new_x, new_y):
    delta_x = old_x + new_x
    delta_y = old_y + new_y
    return lambda coord: (coord[0]-delta_x, coord[1]-delta_y)