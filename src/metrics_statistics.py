import matplotlib.pyplot as plt
import numpy as np
import statistics
import cv2
from masks import Mask
from tqdm import tqdm

# Importando as bibliotecas das medidas de similaridade
from similarity_functions import MSE, MI, PCC

from pathlib import Path

def plotarGraficos(list_of_metrics, fig_path):
    for metrics, metric_dict in list_of_metrics.items():
        for region, values in metric_dict.items():
            fig, axs = plt.subplots(1, 1, figsize=(10,7))
            media = statistics.mean(values)
            std = statistics.stdev(values)

            samples = len(values)
            textstr = '\n'.join((
                r'$samples=%d$' % (samples, ),
                r'$\mu=%{}$'.format(formatacoes[metrics]) % (media, ),
                r'$\sigma=%{}$'.format(formatacoes[metrics]) % (std, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            axs.text(0.75, 0.95, textstr, transform=axs.transAxes, fontsize=14,
            verticalalignment='top', bbox=props)

            axs.hist(values, bins=10, color="g")
            axs.set_title(f"{region} - {metrics}")
            if(escalas_das_metricas[metrics] is not None):
                axs.set_xlim(escalas_das_metricas[metrics])
            axs.grid()
            fig.savefig(fig_path / f"{metrics}-{region}.jpg")

# Armazenando os caminhos
DATA_PATH = Path("..", "Results", "RESULTS-UNIFESP_TEST")

# Local onde será salvo os gráficos
fig_path = Path("..", "Results", "plots-finals_results")
fig_path.mkdir(parents=True, exist_ok=True)

# Carregando a imagem da face média
face_media_geral = Path("..", "Results", "face_media-sem_filtro.jpg")
img_face_media_geral = cv2.imread(str(face_media_geral), 0)

# Listas onde estará sendo armazenados os valores das métricas de cada região
mi_res_arr = []
mse_res_arr = []
pcc_res_arr = []

regioes_metricas = {
    "PCC": {},
    "MI": {},
    "MSE": {}
}

escalas_das_metricas = {
    "PCC": None,#(-1, 1),
    "MI": None,
    "MSE": None#(0, 255),
}

formatacoes = {
    "PCC": ".2f",
    "MI": ".2e",
    "MSE": ".2f"  
}

# Itera nas pastas de recém-nascidos com dor e sem dor na base teste
for dir in tqdm(list(DATA_PATH.glob("*")), position=0):
    img_trans_afim_rosto = cv2.imread(str(dir / "bbox_crop-affine.png"), 0) # Rosto após transformação afim
    mascara_regiao_da_face = None

    # Itera na pasta masks onde se encontra as mascaras de cada região da face
    for regions in tqdm(list((dir / "masks" ).glob("*")), position=1):
        mascara_regiao_da_face = (cv2.imread(str(regions), 0)/255).astype(np.uint8)

        # Resultado das medidas de similaridade
        masked_img = Mask.apply(img_face_media_geral, mascara_regiao_da_face)
        masked_avg_face_img = Mask.apply(img_trans_afim_rosto, mascara_regiao_da_face)

        mi_res = MI(masked_img, masked_avg_face_img, mascara_regiao_da_face)
        mse_res = MSE(masked_avg_face_img, mascara_regiao_da_face)
        pcc_res = PCC(masked_avg_face_img, mascara_regiao_da_face)

        # Adicionando aos dicionarios
        try: regioes_metricas["MI"][regions.stem].append(mi_res)
        except KeyError: regioes_metricas["MI"][regions.stem] = [mi_res]

        try: regioes_metricas["MSE"][regions.stem].append(mse_res)
        except KeyError: regioes_metricas["MSE"][regions.stem] = [mse_res]

        try: regioes_metricas["PCC"][regions.stem].append(pcc_res)
        except KeyError: regioes_metricas["PCC"][regions.stem] = [pcc_res]
        
plotarGraficos(regioes_metricas, fig_path)