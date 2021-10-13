import numpy as np
import cv2
# from scipy.stats import entropy

def entropy(pdf):
    entropy_ = 0
    for row in pdf:
        for value in row:
            if value != 0:
                entropy_ += value*np.log2(1/value)
    return entropy_

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

    I1_ = np.sqrt((I1**2).sum())
    I2_ = np.sqrt((I2**2).sum())

    denominador = I1_ * I2_

    return numerador/denominador



def MI(img1, img2, mask):
    """
    MI - Mutual Information
    """
    masked_pixel_count = sum(sum(mask))
    
    hist1 = cv2.calcHist(
        [img1],     # Imagem (Imagem encapsulada em uma lista)
        [0],        # Canais (Apenas o canal 0)
        mask,       # Mascara
        [256],      # Shape do Histograma (1 canal 256 bins)
        [0,256]     # Range das intensidades dos pixels
    )
    hist2 = cv2.calcHist(
        [img2],
        [0],
        mask,
        [256],
        [0,256]
    )
    
    pdf1 = hist1/masked_pixel_count
    pdf2 = hist2/masked_pixel_count
    pdf_joint = pdf1*pdf2.T

    entropy1 = entropy(pdf1)
    entropy2 = entropy(pdf2)
    entropy_joint = entropy(pdf_joint)

    return entropy1 + entropy2 - entropy_joint