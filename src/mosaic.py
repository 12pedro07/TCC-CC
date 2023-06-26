# ====== AUX FUNCTIONS ====== #
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

# ====== MOSAIC FUNCTIONS ====== #
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

# ====== MOSAIC DICTIONARY ====== #
MOSAIC = {
    # === SIMPLE === #
    "Face base": {
        "coords": [1,9,10,11,12,13,14,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,31,30,29,28,27,26,25,17],
        "color": (255,255,255),
        "priority": 0,
        "inflation": 0
    },
    "Sombrancelha esquerda": {
        "coords": [43,48,49,51,50,46,47,45,44],
        "color": (0,255,255),
        "priority": 10,
        "inflation": 0
    },
    "Sombrancelha direita": {
        "coords": [101,100,99,98,97,102,103,104,105],
        "color": (0,255,255),
        "priority": 10,
        "inflation": 0
    },
    "Olho esquerdo": {
        "coords": [35,36,33,37,39,75,46,47,45,44,43],
        "color": (255,0,255),
        "priority": 9,
        "inflation": 0
    },
    "Olho direito": {
        "coords": [81,89,90,87,91,93,101,100,99,98,97],
        "color": (255,0,255),
        "priority": 9,
        "inflation": 0
    },
    "Nariz": {
        "coords": [72,75,76,77,78,79,80,85,84,83,82,81],
        "color": (0,255,0),
        "priority": 8,
        "inflation": 0
    },
    "Entre olhos": {
        "coords": [49,51,50,46,39,75,72,81,89,97,102,103,104],
        "color": (255,0,0),
        "priority": 8,
        "inflation": 0
    },
    "Boca": {
        "coords": [52,55,56,53,59,58,61,68,67,63,64],
        "color": (0,0,255),
        "priority": 10,
        "inflation": 0
    },
    # === COMPLEX === #
    "Testa": {
        "function": testa,
        "color": (63,63,255),
        "priority": 5,
        "inflation": 0
    },
    "Sulco Esquerda": {
        "function": sulcoEsquerdo,
        "color": (1, 0, 0),
        "priority": 20,
        "inflation": 0
    },
    "Sulco Direito": {
        "function": sulcoDireito,
        "color": (1, 0, 0),
        "priority": 20,
        "inflation": 0
    },
    "Bochecha esquerda": {
        "function": bochechaEsquerda,
        "color": (255, 0, 0),
        "priority": 20,
        "inflation": 0
    },
    "Bochecha direita": {
        "function": bochechaDireita,
        "color": (255, 0, 0),
        "priority": 20,
        "inflation": 0
    }
}