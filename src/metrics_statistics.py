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
            # samples = len(values)
            # textstr = '\n'.join((
            #     r'$samples=%d$' % (samples, ),
            #     r'$\mu=%{}$'.format(formatacoes[metrics]) % (media, ),
            #     r'$\sigma=%{}$'.format(formatacoes[metrics]) % (std, )))
            # props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

            # axs.text(0.75, 0.90, textstr, transform=axs.transAxes, fontsize=14,
            # verticalalignment='top', bbox=props)

            axs.hist(values, bins=20, color="b")
            # axs.set_title(f"{region} - {metrics}")
            if(escalas_das_metricas[metrics] is not None):
                axs.set_xlim(escalas_das_metricas[metrics])

            try:
                axs.axvline(artefatos[metrics][region], color='r', linestyle='dashed', linewidth=2, label=f"Artefato {metrics}={artefatos[metrics][region]}")
                axs.legend()
            except: 
                plt.close('all')
                continue

            axs.grid()
            fig.savefig(fig_path / f"{metrics}-{region}.jpg")
            plt.close('all')

# Armazenando os caminhos
DATA_PATH = Path("..", "Results", "RESULTS-UNIFESP_TEST")

# Local onde será salvo os gráficos
fig_path = Path("..", "Results", "ARTEFATO-plots-finals_UNIFESP")
fig_path.mkdir(parents=True, exist_ok=True)

# Carregando a imagem da face média
face_media_geral = Path("..", "Results", "face_mediana.jpg")
img_face_media_geral = cv2.imread(str(face_media_geral), 0)

# Listas onde estará sendo armazenados os valores das métricas de cada região
mi_res_arr = []
mse_res_arr = []
pcc_res_arr = []

artefatos = {
    "PCC": {
        "Nariz": 0.0111,
        "Olho esquerdo": -0.00388,
        "Olho direito": -0.015
    },
    "MI": {
        "Nariz": 0.343,
        "Olho esquerdo": 0.415,
        "Olho direito": 0.568
    },
    "MSE": {
        "Nariz": 3500,
        "Olho esquerdo": 2980,
        "Olho direito": 4865
    }
}
escalas_das_metricas = {
    "PCC": (-0.1, 0.1),
    "MI": (0,1),
    "MSE": (0, 15000),
}
formatacoes = {
    "PCC": ".5f",
    "MI": ".3f",
    "MSE": ".2f"  
}

regioes_metricas = {
    "PCC": {},
    "MI": {},
    "MSE": {}
}

# Itera nas pastas de recém-nascidos com dor e sem dor na base teste
for dir in tqdm(list(DATA_PATH.glob("*")), position=0):
    img_trans_afim_rosto = cv2.imread(str(dir / "bbox_crop-affine.png"), 0) # Rosto após transformação afim
    mascara_regiao_da_face = None

    # Itera na pasta masks onde se encontra as mascaras de cada região da face
    for regions in tqdm(list((dir / "masks" ).glob("*")), position=1):
        mascara_regiao_da_face = (cv2.imread(str(regions), 0)/255).astype(np.uint8)

        masked_avg = Mask.apply(img_face_media_geral, mascara_regiao_da_face)
        masked_img = Mask.apply(img_trans_afim_rosto, mascara_regiao_da_face)

        mi_res = MI(img_trans_afim_rosto, img_face_media_geral, mascara_regiao_da_face)
        mse_res = MSE(img_trans_afim_rosto, img_face_media_geral, mascara_regiao_da_face)
        pcc_res = PCC(img_trans_afim_rosto, img_face_media_geral, mascara_regiao_da_face)

        # Adicionando aos dicionarios
        try: regioes_metricas["MI"][regions.stem].append(mi_res)
        except KeyError: regioes_metricas["MI"][regions.stem] = [mi_res]

        try: regioes_metricas["MSE"][regions.stem].append(mse_res)
        except KeyError: regioes_metricas["MSE"][regions.stem] = [mse_res]

        try: regioes_metricas["PCC"][regions.stem].append(pcc_res)
        except KeyError: regioes_metricas["PCC"][regions.stem] = [pcc_res]
        
plotarGraficos(regioes_metricas, fig_path)