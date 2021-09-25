import numpy as np
from shapely import geometry

def espelha(x1, y1, x2, y2, x3, y3):
    """
    (x1, y1) = Ponto que sera espelhado
    (x2, y2), (x3, y3) = Pontos que definem a reta "espelho"

    Retorna (x4, y4) = Ponto espelhado
    """
    # y = mx + c
    m = (y3-y2)/(x3-x2)
    c = (x3*y2-x2*y3)/(x3-x2)
    d = (x1 + (y1 - c)*m)/(1 + m**2)

    x4 = 2*d - x1
    y4 = 2*d*m - y1 + 2*c

    return (int(x4), int(y4))

def infla_poligono(poligono, fator=0.10, inflar=True):
        """
        Inputs:
            - poligono = poligono a ser inflado;
            - fator = taxa que deve inflar em porcentagem;
            - inflar = True para inflar, False para comprimir;

        Retorna np.array com os pontos do poligono inflado
        """
        polygon = geometry.Polygon(list(poligono))
        xs = list(polygon.exterior.coords.xy[0])
        ys = list(polygon.exterior.coords.xy[1])
        x_center = 0.5 * min(xs) + 0.5 * max(xs)
        y_center = 0.5 * min(ys) + 0.5 * max(ys)
        min_corner = geometry.Point(min(xs), min(ys))
        center = geometry.Point(x_center, y_center)
        shrink_distance = center.distance(min_corner)*fator
        if inflar:
            polygon_resized = polygon.buffer(shrink_distance) #expand
        else:
            polygon_resized = polygon.buffer(-shrink_distance) #shrink
        return np.array(list(zip(polygon_resized.exterior.coords.xy[0], polygon_resized.exterior.coords.xy[1])))