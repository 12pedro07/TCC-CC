import numpy as np
from scipy.stats import entropy

def entropy_(pdf):
    entropy_ = 0
    for row in pdf:
        for value in row:
            if value != 0:
                entropy_ += value*np.log2(1/value)
    return entropy_

def MSE(img1, img2, mask):
    """
    MSE - Mean Squared Error
    """
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    squared_img = (img1-img2)**2
    masked_np = np.ma.array(squared_img, mask=mask==0)
    return masked_np.mean()

def PCC(img1, img2, mask):
    """
    PCC - Pearson Correlation Coefficient
    """

    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    masked_array1 = np.ma.array(img1, mask=mask==0) # Filtrando regiao 0 da mascara na imagem 1
    masked_array2 = np.ma.array(img2, mask=mask==0)
    I1 = masked_array1 - masked_array1.mean(axis=None)
    I2 = masked_array2 - masked_array2.mean(axis=None)
    numerador = (I1*I2).sum(axis=None)

    I1_ = np.sqrt((masked_array1**2).sum(axis=None))
    I2_ = np.sqrt((masked_array2**2).sum(axis=None))
    denominador = I1_ * I2_

    return numerador/denominador

def MI(img1, img2, mask):
    """
    MI - Mutual Information
    """

    hist_2d, _, _ = np.histogram2d(
        img1.ravel(),
        img2.ravel(),
        bins=20,
        weights=mask.ravel())

    pxy = hist_2d / float(np.sum(hist_2d))
    px = np.sum(pxy, axis=1) # marginal for x over y
    py = np.sum(pxy, axis=0) # marginal for y over x
    px_py = px[:, None] * py[None, :] # Broadcast to multiply marginals
    # Now we can do the calculation using the pxy, px_py 2D arrays
    nzs = pxy > 0 # Only non-zero pxy values contribute to the sum
    return np.sum(pxy[nzs] * np.log(pxy[nzs] / px_py[nzs]))
