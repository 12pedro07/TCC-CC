import cv2
import pickle
import numpy as np
from pathlib import Path
from masks import Mask
import json

from utils import change_reference
from mosaicos_complexos import *

# === PARAMETROS DE DADOS
IDENTIFIER = "UNIFESP_Completo"
MOSAICO = {
    "Face base": {
        "coords": [1,9,10,11,12,13,14,15,16,2,3,4,5,6,7,8,0,24,23,22,21,20,19,18,32,31,30,29,28,27,26,25,17],
        "color": (255,255,255),
        "priority": 10
    },
    "Sombrancelha esquerda": {
        "coords": [43,48,49,51,50,46,47,45,44],
        "color": (0,255,255),
        "priority": 10
    },
    "Sombrancelha direita": {
        "coords": [101,100,99,98,97,102,103,104,105],
        "color": (0,255,255),
        "priority": 10
    },
    "Olho esquerdo": {
        "coords": [35,36,33,37,39,75,46,47,45,44,43],
        "color": (255,0,255),
        "priority": 10
    },
    "Olho direito": {
        "coords": [81,89,90,87,91,93,101,100,99,98,97],
        "color": (255,0,255),
        "priority": 10
    },
    "Nariz": {
        "coords": [72,75,76,77,78,79,80,85,84,83,82,81],
        "color": (0,255,0),
        "priority": 10
    },
    "Entre olhos": {
        "coords": [49,51,50,46,39,75,72,81,89,97,102,103,104],
        "color": (255,0,0),
        "priority": 10
    },
    "Boca": {
        "coords": [52,55,56,53,59,58,61,68,67,63,64],
        "color": (0,0,255),
        "priority": 10
    }
}
MOSAICO_COMPLEXO = { 
    # Partes que nao sao apenas ligar pontos, devem ter um nome e apontar para uma funcao
    # esta funcao deve ter como entrada as coordenadas dos pontos do rosto na imagem e
    # deve devolver as coordenadas do poligono a ser desenhado
    "Testa": {
        "function": testa,
        "color": (63,63,255),
        "priority": 5
    },
    "Sulco Esquerda": {
        "function": sulcoEsquerdo,
        "color": (1, 0, 0),
        "priority": 20
    },
    "Sulco Direito": {
        "function": sulcoDireito,
        "color": (1, 0, 0),
        "priority": 20
    },
    "Bochecha esquerda": {
        "function": bochechaEsquerda,
        "color": (255, 0, 0),
        "priority": 20
    },
    "Bochecha direita": {
        "function": bochechaDireita,
        "color": (255, 0, 0),
        "priority": 20
    }
}


# === PARAMETROS DE DESENHO
COLOR = (255,127,0)
APHA = 0.4 # Em porcentagem
LINE_THICKNESS = 2

FACE_PATH = Path("..", "Results", IDENTIFIER)

# === Carrega os dados das faces dos arquivos binarios
faces = {}
for file_path in FACE_PATH.glob("*"):
    try:
        with open(file_path / "face_data.json") as f:
            data = json.load(f)
            faces[file_path.stem] = data
    except FileNotFoundError:
        pass

# === Analisa cada face
for face, data in faces.items():
    print("Criando mosaico da face ", face)
    # Procura a imagem correspondente no dataset
    face_found = False
    face_path = Path("..", "Dataset", data['dataset']).glob(f"**/*{face}*")
    try: 
        face_path = next(face_path)
        face_found = True
    except StopIteration: 
        print(f"Face {face} não encontrado no dataset...")
        continue
    # Carrega a imagem em memoria
    img = cv2.imread(str(face_path))
    # Carrega a imagem da face em memoria
    face_img = cv2.imread(str(FACE_PATH / face / "bbox_crop.png"))
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
        # Translada a origem para o canto superior esquerdo da bbox e converte as coordenadas
        x_left, y_top, x_right, y_bottom = map(int, data['bbox'])
        mask2origin = change_reference(0,0, x_left, y_top) # Funcao para converter as coordenadas da imagem p/ relativo a face
        points_filtered_bbox = np.array(list(map(mask2origin, points_filtered)), dtype=np.int32).reshape((-1,1,2)) # Para usar com o rosto recortado
        # Reestrutura os dados conforme requisitado pela funcao de poligono e converte para inteiros
        points_filtered = np.array(points_filtered, dtype=np.int32).reshape((-1,1,2)) # Para usar na imagem completa
        # Cria o caminho para salvar a mascara e gera o diretorio se necessario
        mask_file_path = FACE_PATH / face / "masks"
        crop_file_path = FACE_PATH / face / "crops"
        mask_file_path.mkdir(parents=True, exist_ok=True)
        crop_file_path.mkdir(parents=True, exist_ok=True)
        # Gera a mascara (e salva como arquivo para registro)
        mask = Mask.generate(
            face_img.shape, # Altura x Largura da mascara
            points_filtered_bbox, # Pontos do poligono para criar a mascara
            mask_file_path / f"{label}.jpg")
        # Salva as estatisticas da mascara
        px_count, px_pct = Mask.statistics(mask)
        with Path(FACE_PATH, face, "face_data.json").open('r+') as json_file:
            data = json.load(json_file)
            data['mask'] = {
                'pixel_count': int(px_count),
                'pixel_count_pct': float(px_pct)
            }
            json_file.seek(0) # Retorna o cursor para o inicio
            json.dump(data, json_file) # Sobreescreve
            json_file.truncate() # Trunca o arquivo para tirar o conteudo antigo que permaneceu
        # Gera o recorte usando a mascara
        masked_image = Mask.apply(
            face_img,
            mask,
            crop_file_path / f"{label}.jpg")
        # Desenha a regiao recortada na imagem para registro
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
    result_path = str(Path(FACE_PATH, face, "mosaic"+face_path.suffix))
    cv2.imwrite(result_path, img)
    