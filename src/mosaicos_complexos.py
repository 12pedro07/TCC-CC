import math
from helpers import *

def testa(coordenadas):
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
    x1, y1 = coordenadas[86] # Ponto que sera espelhado
    x2, y2 = coordenadas[76] # Primeiro ponto para definir a reta
    x3, y3 = coordenadas[78] # Segundo ponto para definir a reta
    
    # Ponto novo criado artificialmente pelo espelhamento de (x1, y1) na reta definida por [(x2,y2), (x3,y3)]
    ponto_espelhado = espelha(x1, y1, x2, y2, x3, y3)

    roi = [76, 77, 52, 6, 5, 4] # Indices da maioria dos pontos da roi (falta o imaginario)
    roi = [*map(lambda x: coordenadas[x], roi)] # Converte os indices para coordenadas (x, y)
    roi.append(ponto_espelhado) # Adiciona o ponto espelhado

    return roi

def sulcoDireito(coordenadas):
    x1, y1 = coordenadas[86] # Ponto que sera espelhado
    x2, y2 = coordenadas[82] # Primeiro ponto para definir a reta
    x3, y3 = coordenadas[84] # Segundo ponto para definir a reta
    
    # Ponto novo criado artificialmente pelo espelhamento de (x1, y1) na reta definida por [(x2,y2), (x3,y3)]
    ponto_espelhado = espelha(x1, y1, x2, y2, x3, y3)

    roi = [82, 83, 61, 22, 21, 20] # Indices da maioria dos pontos da roi (falta o imaginario)
    roi = [*map(lambda x: coordenadas[x], roi)] # Converte os indices para coordenadas (x, y)
    roi.append(ponto_espelhado) # Adiciona o ponto espelhado

    return roi
    
def bochechaDireita(coordenadas):
    x1, y1 = coordenadas[86] # Ponto que sera espelhado
    x2, y2 = coordenadas[82] # Primeiro ponto para definir a reta
    x3, y3 = coordenadas[84] # Segundo ponto para definir a reta
    
    # Ponto novo criado artificialmente pelo espelhamento de (x1, y1) na reta definida por [(x2,y2), (x3,y3)]
    ponto_espelhado = espelha(x1, y1, x2, y2, x3, y3)

    roi = [82, 30, 31, 32, 18, 19, 20] # Indices da maioria dos pontos da roi (falta o imaginario)
    roi = [*map(lambda x: coordenadas[x], roi)] # Converte os indices para coordenadas (x, y)
    roi.append(ponto_espelhado) # Adiciona o ponto espelhado

    return roi

def bochechaEsquerda(coordenadas):
    x1, y1 = coordenadas[86] # Ponto que sera espelhado
    x2, y2 = coordenadas[76] # Primeiro ponto para definir a reta
    x3, y3 = coordenadas[78] # Segundo ponto para definir a reta
    
    # Ponto novo criado artificialmente pelo espelhamento de (x1, y1) na reta definida por [(x2,y2), (x3,y3)]
    ponto_espelhado = espelha(x1, y1, x2, y2, x3, y3)

    roi = [76, 14, 15, 16, 2, 3, 4] # Indices da maioria dos pontos da roi (falta o imaginario)
    roi = [*map(lambda x: coordenadas[x], roi)] # Converte os indices para coordenadas (x, y)
    roi.append(ponto_espelhado) # Adiciona o ponto espelhado

    return roi

