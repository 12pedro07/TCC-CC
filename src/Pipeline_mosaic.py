import cv2
import pickle
import numpy as np
from pathlib import Path

from mosaicos_complexos import *
from helpers import infla_poligono

# === PARAMETROS DE DADOS
RESULT_DATE = "2021-08-19_19-12-13"
MOSAICO = {
    "Face base": {
        "coords": [1,9,10,11,12,13,14,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,31,30,29,28,27,26,25,17],
        "color": (255,255,255),
        "priority": 10,
        "inflation": 0.1
    },
    "Sombrancelha esquerda": {
        "coords": [43,48,49,51,50,46,47,45,44],
        "color": (0,255,255),
        "priority": 10,
        "inflation": 0.1
    },
    "Sombrancelha direita": {
        "coords": [101,100,99,98,97,102,103,104,105],
        "color": (0,255,255),
        "priority": 10,
        "inflation": 0.1
    },
    "Olho esquerdo": {
        "coords": [35,36,33,37,39,75,46,47,45,44,43],
        "color": (255,0,255),
        "priority": 10,
        "inflation": 0.1
    },
    "Olho direito": {
        "coords": [81,89,90,87,91,93,101,100,99,98,97],
        "color": (255,0,255),
        "priority": 10,
        "inflation": 0.1
    },
    "Nariz": {
        "coords": [72,75,76,77,78,79,80,85,84,83,82,81],
        "color": (0,255,0),
        "priority": 10,
        "inflation": 0.1
    },
    "Entre olhos": {
        "coords": [49,51,50,46,39,75,72,81,89,97,102,103,104],
        "color": (255,0,0),
        "priority": 10,
        "inflation": 0.1
    },
    "Boca": {
        "coords": [52,55,56,53,59,58,61,68,67,63,64],
        "color": (0,0,255),
        "priority": 10,
        "inflation": 0.1
    }
}
MOSAICO_COMPLEXO = { 
    # Partes que nao sao apenas ligar pontos, devem ter um nome e apontar para uma funcao
    # esta funcao deve ter como entrada as coordenadas dos pontos do rosto na imagem e
    # deve devolver as coordenadas do poligono a ser desenhado
    "Testa": {
        "function": testa,
        "color": (63,63,255),
        "priority": 5,
        "inflation": 0.1
    },
    "Sulco Esquerda": {
        "function": sulcoEsquerdo,
        "color": (1, 0, 0),
        "priority": 20,
        "inflation": 0.1
    },
    "Sulco Direito": {
        "function": sulcoDireito,
        "color": (1, 0, 0),
        "priority": 20,
        "inflation": 0.1
    },
    "Bochecha esquerda": {
        "function": bochechaEsquerda,
        "color": (255, 0, 0),
        "priority": 20,
        "inflation": 0.1
    },
    "Bochecha direita": {
        "function": bochechaDireita,
        "color": (255, 0, 0),
        "priority": 20,
        "inflation": 0.1
    }
}

RESULT_PATH = Path("..", "Results", RESULT_DATE, "mosaic")

# === PARAMETROS DE DESENHO
COLOR = (255,127,0)
APHA = 0.4 # Em porcentagem
LINE_THICKNESS = 2

# === Cria a pasta dos resultados
try:
    RESULT_PATH.mkdir(parents=False, exist_ok=True)
except FileNotFoundError:
    print(f"Path {RESULT_PATH.parent} não existe... finalizando")
    exit(100)


# === Carrega os dados das faces dos arquivos binarios
faces_data = {}
for file_path in Path("..", "Results", RESULT_DATE, "faces").glob("*"):
    with open(file_path, 'rb') as face_file:
        try:
            faces_data[file_path.stem] = pickle.load(face_file)[0] # Primeira imagem que tiver
        except Exception as e:
            print(f"Pulando {file_path}")

# === Analisa cada face
for face, data in faces_data.items():
    img_sulco_esquerdo = []
    print("Criando mosaico da face ", face)
    # Procura a imagem correspondente no dataset
    face_path = Path("..", "Dataset").glob(f"**/*{face}*")
    try: 
        face_path = next(face_path)
    except StopIteration: 
        print(f"Face {face} não encontrado no dataset...")
        continue
    # Carrega a imagem em memoria
    img = cv2.imread(str(face_path))
    overlay = img.copy()
    if img is None:
        print(f"Erro ao carregar imagem {face_path}... pulando")
        continue
    regions = {**MOSAICO, **MOSAICO_COMPLEXO} # Merge dos dois dicionarios
    regions = list(regions.items()) # Quebra o dicionario em uma lista com chave e valores
    regions.sort(key=lambda item1: item1[1]['priority']) # Ordena por prioridade
    # Itera em todas as partes do mosaico e desenha (ordenado por prioridade, prioridade < vai para o background)
    for label, region_info in regions:
        print(f"\t|-> {label.upper()}... ", end="")
        # Separa a cor
        color = region_info["color"]
        # Separa os pontos
        try:
            # Tenta executar como um mosaico
            points = region_info["coords"]
            # Separando e filtrando coordenadas da regiao do mosaico
            points_filtered = [data['landmark_2d_106'][index] for index in points] # Pega a coordenada dos landmarks referente ao label atual
        except KeyError:
            # Se nao conseguir, entao eh um mosaico complexo
            points_filtered = region_info["function"](data['landmark_2d_106'])
        
        points_filtered = infla_poligono(points_filtered, fator=region_info['inflation'], inflar=True)

        # Reestrutura os dados conforme requisitado pela funcao de poligono e converte para inteiros
        points_filtered = np.array(points_filtered, dtype=np.int32).reshape((-1,1,2))
        # Draw the region
        overlay = cv2.fillPoly(
            overlay,            # Imagem
            [points_filtered],  # Vertices do poligono
            color,              # Cor da linha
            cv2.LINE_AA         # Tipo de linha
        )
        # Adiciona o overlay com alpha por cima da imagem
        print("OK")
    img = cv2.addWeighted(overlay, APHA, img, 1 - APHA, 0)
    # Salva o resultado
    result_path = str(Path(RESULT_PATH, face+face_path.suffix))
    cv2.imwrite(result_path, img)
    