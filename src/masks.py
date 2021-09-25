import numpy as np
import cv2

class Mask:
    def generate(shape, polygon, dst_path=None):
        mask = np.zeros([shape[0], shape[1]]) # Cria imagem com altura e largura corretas e apenas 1 canal
        mask = cv2.fillPoly(mask, [polygon], 255) # Aplica o poligono na mascara
        mask = mask.astype(np.uint8)
        if dst_path is not None:
            cv2.imwrite(str(dst_path), mask) # Salva a imagem
        return mask
    def apply(image, mask, dst_path=None):
        masked_img = cv2.bitwise_and(image, image, mask=mask) # Aplica a mascara a imagem
        if dst_path is not None:
            cv2.imwrite(str(dst_path), masked_img) # Salva a imagem
        return masked_img
    def statistics(mask):
        mask = mask/255 if mask.max() == 255 else mask # Garante que os valores da mascara sao apenas 0 e 1
        mask = np.uint8(mask) # Converte os valores para uint8
        mask_pixels = mask.sum() # Soma a imagem para ter quantos pixels pertencem a mascara
        mask_pct = mask_pixels / np.prod(mask.shape) # Porcentagem de pixels da imagem que pertencem a mascara
        return mask_pixels, mask_pct