from similarity_functions import *
from pathlib import Path
import cv2
import numpy as np
from masks import Mask

dir = Path("..", "..", "Results", "ARTEFATOS", "20190322_152946")
img_trans_afim_rosto = cv2.imread(str(dir / "bbox_crop-affine.png"), 0)

face_media_geral = Path("..", "..", "Results", "face_mediana.jpg")
img_face_media_geral = cv2.imread(str(face_media_geral), 0)

for regions in list((dir / "masks" ).glob("*")):
    mascara_regiao_da_face = (cv2.imread(str(regions), 0)/255).astype(np.uint8)
    mi_res = MI(img_trans_afim_rosto, img_face_media_geral, mascara_regiao_da_face)
    mse_res = MSE(img_trans_afim_rosto, img_face_media_geral, mascara_regiao_da_face)
    pcc_res = PCC(img_trans_afim_rosto, img_face_media_geral, mascara_regiao_da_face)
    print(regions.stem, " - MI={} / MSE={} / PCC={}".format(mi_res, mse_res, pcc_res))
    
    
    masked_avg = Mask.apply(img_face_media_geral, mascara_regiao_da_face)
    masked_img = Mask.apply(img_trans_afim_rosto, mascara_regiao_da_face)

    # cv2.imshow(f"AVG - {regions.stem}", masked_avg)
    # cv2.imshow(f"IMG - {regions.stem}", masked_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()
