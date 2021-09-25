import cv2                                  # Processamento de imagem
import numpy as np                          # Matematica
import pickle                               # Salvar e carregar arquivos binarios (faces)
from datetime import datetime, timedelta    # Data e hora
from pathlib import Path                    # Facilitar o uso de paths
import json
import insightface                          # Detectar face e pontos fiduciais
from insightface.app import FaceAnalysis
from insightface.data import get_image as ins_get_image
from utils import *

# ====== PARAMETROS ====== #
IDENTIFIER = "UNIFESP_Completo"      # Identificador, usado para gerar a pasta com resultados
DATASET = "Dataset_UNIFESP-SEM_APARATO-COMPLETO-360_IMAGENS-30_RN" # Dataset que deve ser executado
INPUT_IMG_EXTENSION = "bmp"          # Extensao das imagens que serão carregadas
OUTPUT_IMG_EXTENSION = "png"         # Extensao das imagens que serão geradas como saída
DETECTION_APPROACH = DetectionMode.DETECTION_106 # Detector que sera utilizado para achar as faces dos bebes
DOT_SIZE = 1
SHOW_NUMBERS = False

# ====== PATHS PARA AS PASTAS ====== #
# === DEFINIDOS PELO USUARIO
dst_path = Path("..", "Results", IDENTIFIER)                     # Path de destino, onde serão salvos os resultados
Dataset_path = Path("..", "Dataset", DATASET) # Path para o diretorio onde estao as imagens
# === GERADOS AUTOMATICAMENTE
# Carrega o path para as imagens com e sem dor em dois generators
path_cdor = Dataset_path.joinpath("com_dor").glob(f"*.{INPUT_IMG_EXTENSION}")
path_sdor = Dataset_path.joinpath("sem_dor").glob(f"*.{INPUT_IMG_EXTENSION}")

# ====== MODELO DA REDE NEURAL ====== #
if DETECTION_APPROACH == DetectionMode.DETECTION_5:
    # === R50 - So detectar os 5 pts do rosto
    # 1 -> Carregando do model_zoo
    # model = insightface.model_zoo.get_model("retinaface_r50_v1") # Carrega o modelo da rede neural
    # print(model)
    # model.prepare(ctx_id = 1, nms=0.4)                      # Prepara o modelo com os parametros ctx_id 
    #                                                         # [gpu (>0) ou cpu (<0)] e nms [threashold]
    # 2 -> Usando FaceAnalysis
    app = FaceAnalysis(allowed_modules=['detection']) # Habilitando apenas modelos de detecção
    app.prepare(ctx_id=1, det_size=(640, 640)) # det precisa ser de tamanho 2^x para x > 0 E {I}
elif DETECTION_APPROACH == DetectionMode.DETECTION_106:
    # === R50 + 2d106 - Detectar o rosto + Regressao de 106 pontos
    app = FaceAnalysis(allowed_modules=['detection', 'landmark_2d_106'])
    app.prepare(ctx_id=1, det_size=(640, 640))
else:
    print("Detector escolhido é invalido, consultar em utils.DetectionMode as opções disponiveis...")
    exit(-1)

# ====== FUNCOES ====== #
def detect_face(img, mode=DETECTION_APPROACH, show_numbers=SHOW_NUMBERS):
    """
    Carrega e processa uma dada imagem, marcando o bounding box 
    da primeira face detectada e seus pontos fiduciais

    Input:
        - img_path: Path para a imagem que sera analisada pela rede;
    Output:
        - Imagem com a bounding box e pontos fiduciais marcados; 
        - Faces: Lista com pontos resultantes de cada face detectada;
    """
    faces = app.get(img) # Detecta as faces
    rimg = img.copy()
    for face in faces:
        if mode == DetectionMode.DETECTION_5:
            lmk = face.kps.astype(int)
        elif mode == DetectionMode.DETECTION_106:
            lmk = face.landmark_2d_106
            lmk = np.round(lmk).astype(int)
        else:
            faces = []
            rimg = []
            break
        for i in range(lmk.shape[0]):
            p = tuple(lmk[i])
            if show_numbers:
                cv2.putText(
                    rimg,                       # Imagem 
                    str(i),                     # Texto
                    p,                          # Coordenadas da extremidade inferior esquerda do texto
                    cv2.FONT_HERSHEY_SIMPLEX,   # Fonte
                    0.2*DOT_SIZE,               # Multiplicador p/ tamanho da fonte
                    (255,0,0),                  # Cor
                    1,                          # Grossura da linha em pixels
                    cv2.LINE_AA                 # Tipo de linha
                )
            cv2.circle(rimg, p, DOT_SIZE, (255,0,0), -1, cv2.LINE_AA)
    return (rimg, faces)

# ====== MAIN ====== #
# Carrega cada par de imagens de bebes com e sem dor
for pain_class, imgs_path in [("sem dor", path_sdor), ("com dor", path_cdor)]:
    for img_path in imgs_path:
        print(">>> Imagem ", str(img_path.stem))
        save_path = dst_path / img_path.stem
        save_path.mkdir(parents=True, exist_ok=True)
        # === INICIALIZAÇÃO
        print("\t|-> Carregando imagem...")
        img = cv2.imread(str(img_path)) # Carrega a imagem com dor
        # === DETECÇÂO
        print("\t|-> Detectando faces...")
        # Encontra as faces
        img_marked, faces = detect_face(img)
        # Salva a imagem na pasta de destino com a data atual
        print("\t|-> Salvando imagem com as faces detectadas... ", end='')
        if len(faces) > 0:
            cv2.imwrite(
                str(save_path / f"detection.{OUTPUT_IMG_EXTENSION}"), # Path e nome do arquivo
                img_marked # Imagem
            ) # Salva imagem completa
            print("OK")
        else:
            print("Nao houveram deteccoes")
        # === SALVANDO PONTOS DA FACES E METADADOS
        print(f"\t|-> Salvando faces e metadados (.json)...", end='')
        if len(faces) > 0:
            with (save_path / "face_data.json").open('w') as ffp:
                face = faces[0]
                json.dump(
                    {
                        "timestamp": (datetime.utcnow()-timedelta(hours=3)).strftime("%Y-%m-%d_%H-%M-%S"), # Data_hora atual no Brasil
                        "label": pain_class,
                        "dataset": DATASET,
                        "identifier": IDENTIFIER,
                        **{key: value.astype(float).tolist() if isinstance(value, np.ndarray) else value.astype(float) for key, value in dict(face).items()}
                    }, 
                    ffp
                ) # Faz o dump da lista de dicionarios
            print("OK")
        else:
            print("Nao houveram deteccoes")
        # === CROP DAS FACES / SALVANDO CROP
        print("\t|-> Recortando faces da imagem")
        # Itera pelas faces encontradas
        for idx in range(len(faces)):
            print(f"\t\t|-> Salvando Face {idx+1} de {len(faces)}... ", end='')
            face = faces[idx] # Extrai o objeto que contem os pontos da face
            bboxes = face.bbox.astype(int) # Separa os pontos do bounding box e faz o cast para int (eles sao float)
            try:
                cv2.imwrite(
                    str(save_path / f"bbox_crop.{OUTPUT_IMG_EXTENSION}"), # Path e nome do arquivo
                    img[bboxes[1]:bboxes[3], # Y 
                        bboxes[0]:bboxes[2]] # X
                )
                print("OK")
            except Exception as ex:
                print(f"Falha - {repr(ex)}")
        print()