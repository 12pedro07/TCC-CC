import json
import shutil
from pathlib import Path
import tqdm

RESULTS_PATH = Path('..', 'Results')
DATASET_ORIGINS = ['RESULTS-UNIFESP_REFERENCE', 'RESULTS-UNIFESP_TEST']
DATASET_NAME = 'MOSAIC_PAIN_DATASET-0.1'

# Varre todas as pastas que vao originar o dataset
for origin in tqdm.tqdm(DATASET_ORIGINS, desc='FILE ORIGINS', leave=False, position=0):
    origin_path = RESULTS_PATH / origin
    # Varre todas os diretorios de faces dentro do diretorio origem 
    for face_dir in tqdm.tqdm(list(origin_path.glob('*')), desc='FACE DIRS', leave=False, position=1):
        with open(face_dir/'face_data.json') as f:
            json_data = json.load(f)
        # Separa a classe de dor
        pain_class = json_data.get('label', 'unknown')
        # Copia os crops para a pasta correta
        for crop in tqdm.tqdm(list((face_dir/'crops').glob('*')), desc='CROPS', leave=False, position=2):
            # Local onde o dataset deste crop esta localizado
            dataset_crop_path = RESULTS_PATH / DATASET_NAME / pain_class / crop.stem
            # Cria se necessario
            dataset_crop_path.mkdir(parents=True, exist_ok=True)
            # Nome do arquivo no dataset
            file_name = face_dir.stem + crop.suffix
            # Copia
            shutil.copy(str(crop), str(dataset_crop_path/file_name))