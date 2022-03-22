import cv2
import os
import numpy as np
#from PIL import Image
from itertools import product
import math
def NormalisationDataset(imagesPath):
 for image in os.listdir(imagesPath):
     dim = (64, 64)
     imgPath = os.path.join(imagesPath, image)
     img = cv2.imread(imgPath)
     img = cv2.GaussianBlur(img, (3, 3), 0)
     edge = cv2.Canny(img,50, 200)
     contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
     for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        resized_img = img[y:y + h, x:x + w]
     resized_img = cv2.resize(resized_img,dim,interpolation = cv2.INTER_AREA)
     cv2.imwrite(imgPath, resized_img)
def NormalisationImage(imgPath):
    dim = (64, 64)
    img = cv2.imread(imgPath)
    img = cv2.GaussianBlur(img, (3, 3), 0)
    edge = cv2.Canny(img, 50, 200)
    contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    for (i, c) in enumerate(sorted_contours):
        x, y, w, h = cv2.boundingRect(c)
        resized_img = img[y:y + h, x:x + w]
    resized_img = cv2.resize(resized_img, dim, interpolation=cv2.INTER_AREA)
    cv2.imwrite(imgPath, resized_img)
def BinarisationDataset(imagesPath):
    for image in os.listdir(imagesPath):
        imgPath = os.path.join(imagesPath, image)
        img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)  # grey image
        (thresh, bin_img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # binary image
        imgArray = np.array(bin_img)
        edge = cv2.Canny(bin_img, 50, 200)  # number's edges
        contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
        x = sorted_contours[0][0][0][0]
        y = sorted_contours[0][0][0][1]
        cv2.imwrite(imgPath, bin_img)  # save l'image
        if imgArray[0][0] == 0 and imgArray[63][63] == 0:  # we check a point of the img edge, if it's white so the bg is black
            bin_img = cv2.imread(imgPath)
            # make mask of white pixels / black pixels
            black = np.where((bin_img[:, :, 0] == 0) & (bin_img[:, :, 1] == 0) & (bin_img[:, :, 2] == 0))
            white = np.where((bin_img[:, :, 0] == 255) & (bin_img[:, :, 1] == 255) & (bin_img[:, :, 2] == 255))
            # Turn black pixels to white and vice versa
            bin_img[black] = (255, 255, 255)
            bin_img[white] = (0, 0, 0)
            cv2.imwrite(imgPath, bin_img)  # save image
def BinarisationImage(imgPath):
    img = cv2.imread(imgPath, cv2.IMREAD_GRAYSCALE)  # grey image
    (thresh, bin_img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)  # binary image
    imgArray = np.array(bin_img)
    bincopy = cv2.GaussianBlur(bin_img, (3, 3), 0)
    edge = cv2.Canny(bincopy, 50, 200)  # number's edges
    contours, hierarchy = cv2.findContours(edge.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    cv2.imwrite(imgPath, bin_img)  # save l'image
    if imgArray[0][0] == 0 and imgArray[63][63] == 0 :
        bin_img = cv2.imread(imgPath)
        # make mask of white pixels / black pixels
        black = np.where((bin_img[:, :, 0] == 0) & (bin_img[:, :, 1] == 0) & (bin_img[:, :, 2] == 0))
        white = np.where((bin_img[:, :, 0] == 255) & (bin_img[:, :, 1] == 255) & (bin_img[:, :, 2] == 255))
        # Turn black pixels to white and vice versa
        bin_img[black] = (255, 255, 255)
        bin_img[white] = (0, 0, 0)
        cv2.imwrite(imgPath, bin_img)  # save image
def imgToMatrix(path):
    img = cv2.imread(path)
    dim = np.shape(img)
    w = dim[0]
    h = dim[1]
    matrix = np.ones((h, w, 1), dtype="uint8")
    matrix[np.where((img == [0, 0, 0]).all(axis=2))] = [0]
    return matrix
def img_moyen (path):
    matrix = imgToMatrix(path)
    dim = np.shape(matrix)
    w = dim[0]
    h = dim[1]
    nbpixels = w*h
    somme = 0
    for i in range(h):
        for j in range(w):
            pixel = matrix[i][j][0]
            somme = somme + pixel
    return (somme / nbpixels)
def imgbin_ecart(path):
    moyenne = img_moyen(path)
    return( math.sqrt(moyenne-(moyenne*moyenne)))
def img_Centree(path):
    img = cv2.imread(path)
    img_matrix = imgToMatrix(path)
    dim = np.shape(img_matrix)
    w = dim[0]
    h = dim[1]
    imgCentre = np.ones((h, w, 1))
    moy = img_moyen(path)
    ecart = imgbin_ecart(path)
    for i in range(h):
        for j in range(w):
            imgCentre[i][j][0] = ((img_matrix[i][j][0] - moy) / ecart)
    return imgCentre
def matrix_moyen (matrix):
    dim = np.shape(matrix)
    w = dim[0]
    h = dim[1]
    nbpixels = w*h
    somme = 0
    for i in range(h):
        for j in range(w):
            pixel = matrix[i][j][0]
            somme = somme + pixel
    return (somme / nbpixels)
def correlationMatrix(datasetPath , imgPath):
    #NormalisationImage(imgPath)
    BinarisationImage(imgPath)
    imgG = img_Centree(imgPath)
    corrMatrix = np.ones((12, 12, 1))
    moyenneMax = []
    stop = False
    jump = False
    cpt = 0
    MAXmoy = 0
    for image in os.listdir(datasetPath):
       if stop == False:
         reffPath = os.path.join(datasetPath, image)
         cor = 0
         cpt += 1
         if jump == False:
           imgF = img_Centree(reffPath)
           for k, L in product(range(-6,6), range(-6,6)):
              for i, j in product(range(10,48), range(10,48)):
                  cor += (imgF[i][j])*(imgG[i-k][j-L])
              cor = ( cor / (12*12))
              corrMatrix[k+6][L+6] = cor
           moy = matrix_moyen(corrMatrix)
         if MAXmoy < moy:
             MAXmoy = moy
         if MAXmoy > 4 :
           stop = True
           moyenneMax.append(MAXmoy)
           print(moyenneMax.index(MAXmoy),": Le MAX:", MAXmoy)
         if moy < 0 :
             jump = True
         if ( cpt == 67 ):
           jump = False
           moyenneMax.append(MAXmoy)
           print(moyenneMax.index(MAXmoy),": Le MAX: ", MAXmoy)
           cpt = 0
           MAXmoy = 0
    MAX = max(moyenneMax)
    index = moyenneMax.index(MAX)
    if( MAX == 0 ):
        index = -1
    return index
def decisionChiffre(nombre):
    if nombre != -1 :
        print("Le chiffre est: ",nombre)
    else:
        print("Ce n'est pas un chiffre!")