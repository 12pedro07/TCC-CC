import math

def testa(coordenadas):
    def espelha(x1,y1, x2, y2, x3, y3):
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

    # Linha de divisao da testa (y = mx + c)
    # Pontos da linha de divisao: (x1, y1), (x2, y2)
    canto_face_esquerdo = coordenadas[1]
    canto_face_direito = coordenadas[17]
    indices_contorno = list(range(9,15))
    indices_contorno.extend(list(range(30, 24, -1))) # [9, 10, ... 16, 32, 31, ... 25]
    coordenadas_espelhadas = []

    for indice in indices_contorno:
        x1, y1 = coordenadas[indice]
        coordenadas_espelhadas.append(espelha(x1, y1, *canto_face_esquerdo, *canto_face_direito))

    coordenadas_espelhadas.insert(0, canto_face_esquerdo)
    coordenadas_espelhadas.append(canto_face_direito)

    return coordenadas_espelhadas


def sulcoEsquerdo(coordenadas):
    def espelha(x1,y1, x2, y2, x3, y3):
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

    # Linha de divisao da testa (y = mx + c)
    # Pontos da linha de divisao: (x1, y1), (x2, y2)
    canto_face_esquerdo = coordenadas[77]
    canto_face_direito = coordenadas[78]
    indices_contorno = list(range(76,78))
    indices_contorno.extend(list(range(77, 78, 1))) # [9, 10, ... 16, 32, 31, ... 25]
    coordenadas_espelhadas = []

    for indice in indices_contorno:
        x1, y1 = coordenadas[indice]
        coordenadas_espelhadas.append(espelha(x1, y1, *canto_face_esquerdo, *canto_face_direito))

    coordenadas_espelhadas.insert(0, canto_face_esquerdo)
    coordenadas_espelhadas.append(canto_face_direito)

    return coordenadas_espelhadas

def sulcoDireito(coordenadas):
    def espelha(x1,y1, x2, y2, x3, y3):
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

    # Linha de divisao da testa (y = mx + c)
    # Pontos da linha de divisao: (x1, y1), (x2, y2)
    canto_face_esquerdo = coordenadas[83]
    canto_face_direito = coordenadas[84]
    indices_contorno = list(range(82,84))
    indices_contorno.extend(list(range(83, 84, 1))) # [9, 10, ... 16, 32, 31, ... 25]
    coordenadas_espelhadas = []

    for indice in indices_contorno:
        x1, y1 = coordenadas[indice]
        coordenadas_espelhadas.append(espelha(x1, y1, *canto_face_esquerdo, *canto_face_direito))

    coordenadas_espelhadas.insert(0, canto_face_esquerdo)
    coordenadas_espelhadas.append(canto_face_direito)

    return coordenadas_espelhadas