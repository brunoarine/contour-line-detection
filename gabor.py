# -*- coding: utf-8 -*-
"""
Created on Thu Jul 26 15:16:11 2018

@author: soldeace
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt


def gabor_filter(image,
                 ksize=(11, 11),
                 sigma=4.0,
                 theta=3.2*np.pi/2,
                 lam=10.0,
                 gamma=0.5,
                 psi=0,
                 ktype=cv2.CV_32F):
    '''
    cv2.getGaborKernel(ksize, sigma, theta, lambda, gamma, psi, ktype)
    ksize : size of gabor filter (n, n)
    sigma : standard deviation of the gaussian function
    theta : orientation of the normal to the parallel stripes
    lam : wavelength of the sunusoidal factor
    gamma : spatial aspect ratio
    psi : phase offset
    ktype : type of values that each pixel in the gabor kernel can hold
    '''

    g_kernel = cv2.getGaborKernel(ksize, sigma, theta, lam, gamma, psi, ktype)

    filtered_img = cv2.filter2D(image, cv2.CV_8UC3, g_kernel)

    cv2.imshow('image', image)
    cv2.imshow('filtered image', filtered_img)

    h, w = g_kernel.shape[:2]
    g_kernel = cv2.resize(g_kernel, (3*w, 3*h), interpolation=cv2.INTER_CUBIC)
    cv2.imshow('gabor kernel (resized)', g_kernel)

    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return filtered_img


def build_filters():
    """ returns a list of kernels in several orientations
    """
    filters = []
    ksize = 21
    for theta in np.arange(0, np.pi, np.pi/32):
        params = {'ksize': (ksize, ksize), 'sigma': 8.0, 'theta': theta,
                  'lambd': 15.0,
                  'gamma': 0.5, 'psi': 0, 'ktype': cv2.CV_32F}
        kern = cv2.getGaborKernel(**params)
        kern /= 1.5*kern.sum()
        filters.append((kern, params))
        return filters


def process(img, filters):
    """
    returns the img filtered by the filter list
    """
    accum = np.zeros_like(img)
    for kern, params in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        np.maximum(accum, fimg, accum)
    return accum


def find_orientation(matrix):
    '''
    Returns the orientation angle of a blob
    '''
    y, x = np.nonzero(matrix)
    x = x - np.mean(x)
    y = y - np.mean(y)
    coords = np.vstack([x, y])
    cov = np.cov(coords)
    evals, evecs = np.linalg.eig(cov)
    sort_indices = np.argsort(evals)[::-1]
    # Eigenvector with largest eigenvalue
    x_v, y_v = evecs[:, sort_indices[0]]
    angle = np.arctan(y_v/x_v)
    # plt.figure(figsize=(20,15))
    #plt.imshow(matrix, cmap='gray')
    #scale = 200
    # plt.plot([0, x_v*scale],
    #     [0, y_v*scale], color='red')
    # plt.gca().invert_yaxis()
    # plt.show()
    return angle


def fft_enhance(img, debug=False):

    rows, cols = img.shape
    nrows = cv2.getOptimalDFTSize(rows)
    ncols = cv2.getOptimalDFTSize(cols)
    nimg = np.zeros((nrows, ncols))
    nimg[:rows, :cols] = img

    dft = cv2.dft(np.float32(nimg), flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    magnitude = np.log(cv2.magnitude(dft_shift[:, :, 0],
                                     dft_shift[:, :, 1]))
    magnitude_threshold = magnitude.mean() + 5*magnitude.std()
    mask = (magnitude > magnitude_threshold).astype('uint8')
    angle = find_orientation(mask)
    fshift = dft_shift
    fshift[:, :, 0] *= mask
    fshift[:, :, 1] *= mask

    f_ishift = np.fft.ifftshift(fshift)
    img_back = cv2.idft(f_ishift)
    img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])
    img_back = cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX)

    if debug:
        cv2.imshow('imagem', img)
        cv2.imshow('mask', mask*250)
        cv2.imshow('imagem back', img_back.astype('uint8'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    return (img_back, angle)


if __name__ == "__main__":
    img_rgb = cv2.imread('resources/talhao.jpg')
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_BGR2HSV)[:, :, 0]
    #img_gray = cv2.bitwise_not(img_gray)
    img_gray = cv2.normalize(img_gray, None, 0, 255, cv2.NORM_MINMAX)
    # img_binary = cv2.adaptiveThreshold(img_gray, 255,
    #                                   cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    #                                   cv2.THRESH_BINARY, 119, 8)
    _, img_binary = cv2.threshold(img_gray, 0, 255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    img_enhanced, img_angle = fft_enhance(img_binary, debug=False)

    _, img_enhanced_binary = cv2.threshold(img_enhanced.astype('uint8'), 0, 255,
                                           cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#    plt.figure(figsize=(20,15))
 #   plt.imshow(img_enhanced_binary, cmap='gray')
  #  plt.show()
    img_gabor = gabor_filter(img_gray, ksize=(
        21, 21), theta=img_angle, sigma=8, lam=15)
    #filters = build_filters()
    #img = process(img_gray, filters)
    cv2.imshow('imagem', img_gabor)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
