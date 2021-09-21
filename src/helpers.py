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
