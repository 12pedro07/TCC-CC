import numpy as np
import cv2
def MSE(img1, img2):
    """
    MSE - Mean Squared Error
    """
    return ((img1 - img2 )**2).mean(axis=None)

def PCC(img1, img2):

    """
    PCC - Pearson Correlation Coefficient
    """

    I1 = img1 - (img1).mean(axis=None)
    I2 = img2 - (img2).mean(axis=None)

    numerador = (I1*I2).sum()
    print(numerador)

    I1_ = np.sqrt((I1**2).sum())
    I2_ = np.sqrt((I2**2).sum())

    denominador = I1_ * I2_
    print(denominador)

    return numerador/denominador



def MI():
    """
    MI - Mutual Information
    """
    return 0

img1 = cv2.imread(r"../RESULTS/UNIFESP/UNIFESP/01_113852/bbox_crop-affine.png", 0)

img2 = cv2.imread(r"../RESULTS/face_media-sem_filtro.jpg", 0)

print(MSE(img1, img2))