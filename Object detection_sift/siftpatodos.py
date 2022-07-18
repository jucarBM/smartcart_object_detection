
import cv2
from matplotlib import pyplot as plt
import numpy as np
import time

t0=time.time()
# ------------------
sift = cv2.SIFT_create() # SE CREA EL SIFT
# ----------------
import rawpy
import imageio
import time

def reading_raw (path):
    # path = 'RITZ1.dng'
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess()

    resizedFrame = cv2.resize(rgb, (0, 0), fx=0.20, fy=0.20)
    BGRR = cv2.cvtColor(resizedFrame, cv2.COLOR_RGB2BGR)
    notsizebgr=cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return BGRR,notsizebgr


def get_index_positions(list_of_elems, element):
    ''' Returns the indexes of all occurrences of give element in
    the list- listOfElements '''
    index_pos_list = []
    index_pos = 0
    while True:
        try:
            # Search for item in list from indexPos to the end of list
            index_pos = list_of_elems.index(element, index_pos)
            # Add the index position in list
            index_pos_list.append(index_pos)
            index_pos += 1
        except ValueError as e:
            break
    return index_pos_list


# import cv2
# cv2.imshow('', BGRR)
# cv2.waitKey()
# cv2.destroyAllWindows()

# ----------------------RITZ
visualizar11,I11rtiz = reading_raw('RITZ1.dng')
visualizar12,I12rtiz = reading_raw('RITZ2.dng')
visualizar13,I13rtiz = reading_raw('RITZ3.dng')

 
# ----
keypts11, descriptores11 = sift.detectAndCompute(I11rtiz, None)
keypts12, descriptores12 = sift.detectAndCompute(I12rtiz, None)
keypts13, descriptores13 = sift.detectAndCompute(I13rtiz, None)






# ----------------------chips
visualizar21,I21rtiz = reading_raw('CHIPS1.dng')
visualizar22,I22rtiz = reading_raw('CHIPS2.dng')
visualizar23,I23rtiz = reading_raw('CHIPS3.dng')


# ----
keypts21, descriptores21 = sift.detectAndCompute(I21rtiz, None)
keypts22, descriptores22 = sift.detectAndCompute(I22rtiz, None)
keypts23, descriptores23 = sift.detectAndCompute(I23rtiz, None)






# ----------------------laive
visualizar31,I31rtiz = reading_raw('LAIVE1.dng')
visualizar32,I32rtiz = reading_raw('LAIVE2.dng')
visualizar33,I33rtiz = reading_raw('LAIVE3.dng')

# ----
keypts31, descriptores31 = sift.detectAndCompute(I31rtiz, None)
keypts32, descriptores32 = sift.detectAndCompute(I32rtiz, None)
keypts33, descriptores33 = sift.detectAndCompute(I33rtiz, None)






# ----------------------RITZ
visualizar41,I41rtiz = reading_raw('jabon1.dng')
visualizar42,I42rtiz = reading_raw('jabon2.dng')
visualizar43,I43rtiz = reading_raw('jabon3.dng')

# ----
keypts41, descriptores41 = sift.detectAndCompute(I41rtiz, None)
keypts42, descriptores42 = sift.detectAndCompute(I42rtiz, None)
keypts43, descriptores43 = sift.detectAndCompute(I43rtiz, None)




# ----------------------RITZ
visualizar51,I51rtiz = reading_raw('inka1.dng')
visualizar52,I52rtiz = reading_raw('inka2.dng')
visualizar53,I53rtiz = reading_raw('inka3.dng')

# ----
keypts51, descriptores51 = sift.detectAndCompute(I51rtiz, None)
keypts52, descriptores52 = sift.detectAndCompute(I52rtiz, None)
keypts53, descriptores53 = sift.detectAndCompute(I53rtiz, None)

# -----------------------
IIIS  = [I11rtiz,I12rtiz,I13rtiz,I21rtiz,I22rtiz,I23rtiz,I31rtiz,I32rtiz,I33rtiz,I41rtiz,I42rtiz,I43rtiz,I51rtiz,I52rtiz,I53rtiz]

Descriptors = [descriptores11,descriptores12,descriptores13,descriptores21,descriptores22,descriptores23,descriptores31,descriptores32,descriptores33,descriptores41,descriptores42,descriptores43,descriptores51,descriptores52,descriptores53]

Keyys = [keypts11,keypts12,keypts13,keypts21,keypts22,keypts23,keypts31,keypts32,keypts33,keypts41,keypts42,keypts43,keypts51,keypts52,keypts53]

Names=['RITZ','RITZ','RITZ','CHIPS','CHIPS','CHIPS','LAIVE','LAIVE','LAIVE','JABON','JABON','JABON','INKA','INKA','INKA']


# Parámetros de FLANN para ser usados con SIFT
FLANN_INDEX_KDTREE = 1
index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
search_params = dict(checks=50) # A más chequeos, más exactitud (más lento)

# Correspondencias con FLANN
flann = cv2.FlannBasedMatcher(index_params, search_params)
t1=time.time()


import PIL
# import Image
import glob

for filename in glob.glob("*jpeg"):

    # cap =cv2.imread(filename,1)

  


    # -------------ingresamos imagne objetivo
    I2 = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    T4 = time.time()
    keypts2, descriptores2 = sift.detectAndCompute(I2, None)
    t5 = time.time()

    # --------------------------

    # # Parámetros de FLANN para ser usados con SIFT
    # FLANN_INDEX_KDTREE = 1
    # index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    # search_params = dict(checks=50) # A más chequeos, más exactitud (más lento)
    # --------------------------



    # # Correspondencias con FLANN
    # flann = cv2.FlannBasedMatcher(index_params, search_params)

    # --------------Se realizan los matches
    M=[]
    for i in range(len(IIIS)):

        M.append(flann.knnMatch(Descriptors[i], descriptores2, k=2))


    MASK_MATCHADOR=[]
    # ---------------------
    # Máscara vacía para dibujar
    for i in range(len(IIIS)):
        MASK_MATCHADOR.append([[0, 0] for i in range(len(M[i]))])



    # ---------------

    BUENOS_MATCH_NUM=[]
    # Llenar valores usando el ratio
    NNDR = 0.4
    for k in range(len(IIIS)):

        for i, (m, n) in enumerate(M[k]):
            if m.distance < NNDR * n.distance:
                MASK_MATCHADOR[k][i]=[1, 0]
        
        BUENOS_MATCH_NUM.append(np.count_nonzero(MASK_MATCHADOR[k]))


    # --------------------aplico criterio 

    # buscamos el mayor de todos, si se repiten, sería el mismo número

    Ojeto_detectado=max(BUENOS_MATCH_NUM)

    # Encontamos los onjetos detectados



    index_pos_list = get_index_positions(BUENOS_MATCH_NUM, Ojeto_detectado)

    print(index_pos_list)

    t2=time.time()
    img_matches = cv2.drawMatchesKnn(IIIS[index_pos_list[0]], Keyys[index_pos_list[0]], I2, keypts2, M[index_pos_list[0]], None,
                                    matchColor=(0, 255, 0), 
                                    singlePointColor=(255, 0, 0),
                                    matchesMask=MASK_MATCHADOR[index_pos_list[0]], flags=0)

    # Mostrar las correspondencias
    plt.figure(figsize=(16,16))
    plt.imshow(img_matches)
    plt.show()

    t3=time.time()

    print(t1-t0)
    print(t2-t1)
    print(t3-t2)
    print(t5-T4)
      # GUARDAR DIRECTORIO
    with open ('datatest1.txt','a') as f:
        f.write(str(index_pos_list))
        f.write('\n')
        f.write(str(Names[index_pos_list[0]]))
        f.write('\n')
        f.close()




# t2=time.time()

# # Dibujo de las correspondencias


# # for i in range(len(index_pos_list)):


# img_matches = cv2.drawMatchesKnn(IIIS[index_pos_list[0]], Keyys[index_pos_list[0]], I2, keypts2, M[index_pos_list[0]], None,
#                                 matchColor=(0, 255, 0), 
#                                 singlePointColor=(255, 0, 0),
#                                 matchesMask=MASK_MATCHADOR[index_pos_list[0]], flags=0)

# # Mostrar las correspondencias
# plt.figure(figsize=(16,16))
# plt.imshow(img_matches)
# plt.show()

# t3=time.time()

# print(t1-t0)
# print(t2-t1)
# print(t3-t2)


