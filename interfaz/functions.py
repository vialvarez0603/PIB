import os
import matplotlib.pyplot as plt
import numpy as np
import random
from queue import Queue
import cv2
from scipy import signal
import pandas as pd


import skimage
from skimage import exposure, restoration
from skimage.restoration import denoise_nl_means, estimate_sigma
from skimage.filters import threshold_otsu
from skimage.morphology import skeletonize

### Mejoramiento y Segmentación

def gauss(x,y, sigma):
  return (1/(2*np.pi*sigma**2))*np.exp(-(x**2+y**2)/(2*sigma**2))

def kernel_gaussiano(n, sigma = 1):
  kernel = np.zeros((n,n))

  for i in range(len(kernel)):
    for j in range(len(kernel[0])):
      kernel[i,j] = gauss(i-n//2,j-n//2, sigma)

  return kernel

def preprocess_image(im, white_extr = False, white_thresh = 250):
  image = im.copy()

  if white_extr:
    image[image>white_thresh] = 0

  gaussian_filtered = cv2.GaussianBlur(image, (5, 5), 0)

  sigma_est = np.mean(estimate_sigma(gaussian_filtered))
  denoised = denoise_nl_means(gaussian_filtered, h=0.6 * sigma_est, sigma = sigma_est, fast_mode=False, patch_size=3, patch_distance=1)

  denoised = (255 * denoised).astype(np.uint8)

  equalized_image = cv2.equalizeHist(denoised)
  equalized_image[:,0] = 0
  equalized_image[:,-1] = 0
  equalized_image[0,:] = 0
  equalized_image[-1,:] = 0

  return denoised, equalized_image

def binarizar(imagen, umbral, gray_levels = 256):
  binaria = np.zeros_like(imagen)
  filas, columnas = imagen.shape

  for i in range(filas):
    for j in range(columnas):
      f = imagen[i, j]
      if f >= umbral: binaria[i, j] = gray_levels - 1

  return binaria

def cumulated_probability(image, clase_sup, clase_inf=0):
  cum_hist = cumulated_histogram(image, normalize = True)
  cum_probability = cum_hist[clase_sup] - clase_inf

  return cum_probability

def class_mean(image, clase_sup, clase_inf=0):
  cum_hist = cumulated_histogram(image, normalize = True)
  class_av = 0
  for i in range(clase_inf, clase_sup):
    class_av += i * cum_hist[i]

  return class_av

def inter_class_var(image, clase_sup, clase_inf=0, gray_levels = 256):
  global_av = class_mean(image, gray_levels-1)
  class_av = class_mean(image, clase_sup, clase_inf=clase_inf)
  class_prob = cumulated_probability(image, clase_sup, clase_inf=clase_inf)

  if class_prob*(1-class_prob) != 0: var = (global_av*class_prob - class_av)**2/(class_prob*(1-class_prob))
  else: var = 0

  return var

def Otsu(image, gray_levels = 256):
  varianza = np.zeros(gray_levels)

  for t in range(1,gray_levels):
    varianza[t] = inter_class_var(image, t, gray_levels = gray_levels)

  t_opt = np.argmax(varianza)
  bin_image = binarizar(image, t_opt, gray_levels = gray_levels)

  return t_opt, bin_image

def neighbours_analizer(seed, values, val, analyzed_pos, max_fil, max_col, num = 4):
  i, j = seed
  vecinos = [[i-1, j], [i+1, j], [i, j-1], [i, j+1]]

  if num == 8:
    vecinos.extend([[i-1, j-1], [i+1, j+1], [i-1, j+1], [i+1, j-1]])

  # Filtramos vecinos que están fuera de los límites de la imagen o ya fueron evaluados
  vecinos = [[x, y] for x, y in vecinos if 0 <= x < max_fil and 0 <= y < max_col and (x, y) not in analyzed_pos]
  vecinos = [[x, y] for x, y in vecinos if values[x,y] == val]

  return vecinos

def etiquetado(image, value = 0, bin_val = 200, gray_levels = 256, bin = True, size_thresh = 0):
  if bin:
    # Paso 1: Binarizamos la imagen.
    umbral = threshold_otsu(image)
    bin = binarizar(image, umbral)

  else:
    bin = binarizar(image, bin_val)

  # Paso 2: Identificamos las coordenadas de los píxeles que son iguales a 0 o 255 (value)
  coord = np.argwhere(bin == value)
  # Estas coordenadas van a ser las distintas semillas que evaluamos para el etiquetado

  k = 1
  evaluated = set()  # En evaluated vamos a poner los pixeles que ya etiquetamos, para no re-evaluarlos
  result = np.zeros_like(image, dtype=np.uint64)

  while len(coord) != 0:
      seed = coord[-1]  # Tomamos el último elemento
      coord = coord[:-1]  # Eliminamos la semilla de la lista

      if tuple(seed) in evaluated:
          continue  # Pasamos a la siguiente semilla si ya ha sido evaluada

      neighbours = neighbours_analizer(seed, bin, value, evaluated, len(bin), len(bin[0]))

      if not neighbours:
          continue  # Pasamos a la siguiente semilla si no tiene vecinos

      while len(neighbours) != 0:
          i, j = seed
          result[i, j] = k

          seed = neighbours.pop()
          evaluated.add(tuple(seed))

          new_neighbours = neighbours_analizer(seed, bin, value, evaluated, len(bin), len(bin[0]))
          for neighbour in new_neighbours:
              evaluated.add(tuple(neighbour))
          neighbours.extend(new_neighbours)

      if size_thresh != 0:
        length = len(result[result == k])
        if length < size_thresh:
          result[result == k] = 0
          k -=1

      k += 1  # Cambiamos de etiqueta

  return(result)

def fill(bin_im, im):
  new = np.zeros_like(bin_im)

  fils, cols = new.shape

  for j in range(cols):
    lim_sup = -1
    lim_inf = -1
    for i in range(fils):
      if bin_im[i,j] == 255 and lim_sup == -1:
        lim_sup = i

      elif bin_im[i,j] == 0 and lim_inf == -1:
        if np.all(bin_im[i:,j]==0):
          lim_inf = i

    new[lim_sup:lim_inf,j] = 1


  return new.astype("uint8")

def calcular_bounding_boxes(original, matriz_etiquetada, gray_levels = 256, Plot=True, color=(0, 255, 0), alpha=0.5):
    objetos = np.unique(matriz_etiquetada)[1:]  # Ignoramos el valor 0 que representa el fondo
    bounding_boxes = []
    image_bounding_boxes = np.zeros_like(matriz_etiquetada, dtype=np.uint8)
    white = gray_levels - 1

    for objeto in objetos:
        # Encontrar las coordenadas de los píxeles pertenecientes al objeto
        coords_objeto = np.argwhere(matriz_etiquetada == objeto)

        # Calcular los límites de la bounding box
        fila_min = np.min(coords_objeto[:, 0])
        fila_max = np.max(coords_objeto[:, 0])
        col_min = np.min(coords_objeto[:, 1])
        col_max = np.max(coords_objeto[:, 1])

        bounding_boxes.append(((col_min, fila_min), (col_max, fila_max)))  # Formato: ((x1, y1), (x2, y2))

    superposicion = np.array(original.copy())
    superposicion = superposicion.astype(np.uint8)

    for box in bounding_boxes:
      cv2.rectangle(superposicion, box[0], box[1], (white, white, white), 2)  # Le hacemos un borde blanco a cada bounding box
      cv2.rectangle(superposicion, box[0], box[1], color, -1)  # Dibujamos cada bounding box sobre la imagen original

    # Combinamos las dos imágenes con transparencia
    original = original.astype(np.uint8)
    imagen_superpuesta = cv2.addWeighted(original, 1 - alpha, superposicion, alpha, 0)

    if Plot:
      plt.figure(figsize=(10, 10))
      plt.imshow(cv2.cvtColor(imagen_superpuesta, cv2.COLOR_BGR2RGB))
      plt.axis('off')
      plt.show()

    return imagen_superpuesta, bounding_boxes


def bounding_box_size(bounding_box, nombre = 'objeto', show = True):
  # bounding_box es una tupla de la forma ((x1, y1), (x2, y2))
  x1, y1 = bounding_box[0]  # Coordenadas del vértice superior izquierdo
  x2, y2 = bounding_box[1]  # Coordenadas del vértice inferior derecho
  # Calculamos el ancho y el alto del rectángulo
  ancho = abs(x2+1 - x1)
  alto = abs(y2+1 - y1)

  if show:
    print(f'Ancho del {nombre}: {ancho} píxeles \nAlto del {nombre}: {alto} píxeles \nTamaño del {nombre}: {ancho*alto} píxeles cuadrados')
  return (ancho, alto, ancho*alto)

def calculate_size(objects):
  sizes = []
  for obj in objects:
    mini = obj[0]
    maxi = obj[1]

    size = (maxi[0]-mini[0])*(maxi[1]-mini[1])
    sizes.append(size)

  return sizes

def distancia_euclidea(v1, v2):
  return np.sqrt(np.sum((v1-v2)**2))

def k_means(img, k, min_dsv = 10, max_iter = 100, gray_levels = 256, random_state = None, print_message = True):
  pixeles = np.ravel(img).astype('float')

  if random_state == None:
    centroides = np.random.choice(range(gray_levels), k, replace=False)
  else:
    random_generator = np.random.RandomState(random_state)
    centroides = random_generator.choice(range(gray_levels), k, replace=False)


  centroides = sorted(centroides)

  asignaciones = np.zeros(img.shape)

  for n in range(max_iter):

    for i in range(len(img)):
      for j in range(len(img[0])):
        distancias = [distancia_euclidea(img[i,j], centroide) for centroide in centroides]
        asignaciones[i,j] = np.argmin(distancias)

    nuevos_centroides = []
    for i in range(k):
        cluster_puntos = img[asignaciones == i]
        if len(cluster_puntos) != 0:
            nuevo_centroide = np.mean(cluster_puntos)
        else:
            nuevo_centroide = np.random.choice(pixeles)
        nuevos_centroides.append(nuevo_centroide)

    distancia_centroides = [distancia_euclidea(centroides[i], nuevos_centroides[i]) for i in range(k)]

    centroides = nuevos_centroides

    if np.max(distancia_centroides) < min_dsv:
      if print_message: print(f'Se cumplio la distancia minima {min_dsv}')
      break

  new_img = asignaciones

  return new_img

def y_coord(etiquetas):
  etiquetas_unicas = np.unique(etiquetas)
  etiquetas_unicas = etiquetas_unicas[etiquetas_unicas != 0]  # Excluye la etiqueta de fondo (0)
  filas_promedio = [-1]

  for etiqueta in etiquetas_unicas:
    coordenadas = np.argwhere(etiquetas == etiqueta)

    filas = coordenadas[:, 0]
    fila_promedio = np.mean(filas)

    filas_promedio.append(fila_promedio)

  return filas_promedio

def get_RPE(k_im, thresh = 500, show = False):
  k_im[:5,:] = 0
  k_im[-5:, :] = 0
  k_im[:,:5] = 0
  k_im[:,-5:] = 0
  EPR = np.zeros_like(k_im)
  EPR[k_im== np.max(np.unique(k_im))] = 255

  et_EPR = etiquetado(EPR, value=255, size_thresh = thresh)
  filas = y_coord(et_EPR)

  et_max = np.argmax(filas)
  EPR_fin = np.zeros_like(et_EPR)
  EPR_fin[et_EPR == et_max] = 255

  if show:
    show_image(EPR, titles = 'Centroide máximo')
    show_tagged(EPR, et_EPR)
    show_image(EPR_fin, titles = 'EPR detectado')

  return EPR_fin

def skeletonize_image(image):
    skeleton = skeletonize(image // 255)  # Convertir a binario si no lo está
    return (skeleton * 255).astype(np.uint8)

def best_fit_line(image):
  coords = np.column_stack(np.where(image == 255))
  if coords.size == 0:
    raise ValueError("No se encontraron píxeles con valor 255 en la imagen.")
  X = coords[:, 1]
  Y = coords[:, 0]
  A = np.vstack([X, np.ones(len(X))]).T
  m, c = np.linalg.lstsq(A, Y, rcond=None)[0]

  return X, Y, m, c

def draw_best_fit_line(image, m, c, grey_val=150):
  line_image = np.copy(image)
  rows, cols = line_image.shape
  for x in range(cols):
    y = int(m * x + c)
    if 0 <= y < rows:
      line_image[y, x] = grey_val
  return line_image

def best_fit_parabola(image):
  coords = np.column_stack(np.where(image == 255))
  if coords.size == 0:
    raise ValueError("No se encontraron píxeles con valor 255 en la imagen.")
  X = coords[:, 1]
  Y = coords[:, 0]

  A = np.vstack([X**2, X, np.ones(len(X))]).T
  a, b, c = np.linalg.lstsq(A, Y, rcond=None)[0]

  return X, Y, a, b, c

def draw_best_fit_parabola(image, a, b, c, grey_val=150):
  parabola_image = np.copy(image)
  rows, cols = parabola_image.shape
  for x in range(cols):
    y = int(a * x**2 + b * x + c)
    if 0 <= y < rows:
      parabola_image[y, x] = grey_val
  return parabola_image

def calculate_R2(X, Y, predicted_Y):
  ss_res = np.sum((Y - predicted_Y) ** 2)
  ss_tot = np.sum((Y - np.mean(Y)) ** 2)
  r_squared = 1 - (ss_res / ss_tot)
  return r_squared

### Generalización

def pre_process(imagen, show_steps = False):
  den1, im1 = preprocess_image(imagen, white_extr = True)

  et1 = etiquetado(den1, value=255, bin = True)

  lengths1 = [len(et1[et1 == val]) for val in np.unique(et1[1:])]

  im2 = np.zeros_like(et1)
  val = np.unique(et1)[np.argmax(lengths1[1:])+1]
  im2[et1 == val] = 255

  im3 = fill(im2, imagen)*imagen

  if show_steps:
    show_image(den1, titles = 'Primer procesamiento')

    plt.figure(figsize=(15,10))
    plt.subplot(121)
    plt.imshow(den1, cmap='gray', vmin=0, vmax=255)
    plt.title('Imagen',fontsize=15)
    plt.subplot(122)
    plt.imshow(et1, cmap='nipy_spectral')
    plt.title('Etiquetada',fontsize=15)
    plt.show()

    show_image(im2, titles = 'Imagen binarizada')

    show_image(im3, titles = 'Imagen rellenada')

  im_obj, obj = calcular_bounding_boxes(im3,et1, Plot = show_steps)

  val_max = np.argmax(calculate_size(obj))
  bb1 = obj[val_max]

  im4 = im3[bb1[0][1]:bb1[1][1],bb1[0][0]:bb1[1][0]]

  if show_steps:
    show_image(im4, titles = 'Imagen final')

  return im4

def vessels(im, thresh = 75, k_num = 3, max_iterat = 50, inform = False, show = False):
  kmeans_imagen = k_means(im, k_num, min_dsv = 4, max_iter = max_iterat, print_message = False)
  if np.max(np.unique(kmeans_imagen)) != 0:
    kk = kmeans_imagen*int(255/np.max(np.unique(kmeans_imagen)))
    kk[0,:] = 0
    kk[-1,:] = 0
    kk[:,0] = 0
    kk[:,-1] = 0
    etiquetado1 = etiquetado(kk, value=0, bin_val = 1, bin = False, size_thresh = thresh)
    if show:
      show_tagged(kk,etiquetado1)
    if inform:
      print(f'Se detectaron {(len(np.unique(etiquetado1))-2)} vasos en la retina.')
    return kk, etiquetado1,(len(np.unique(etiquetado1))-2)  #Restamos 1 por el fondo del etiquetado (objeto - retina) y 1 por el etiquetado normal (etiquetado del fondo)

  else:
    if inform:
      print(f'No se pudo realizar el procesamiento con éxito.')
    return kk, etiquetado1, -1

def best_fit_corr(k_im, mode = 'straight', plot = False, inform = False):
  EPR = get_RPE(k_im, show = plot)
  skeleton = skeletonize_image(EPR)

  if plot: show_image(skeleton, titles = 'Esqueleto')

  if mode == 'straight':
    X, Y, m, c = best_fit_line(skeleton)
    predicted_Y = m * X + c
    line_sk = draw_best_fit_line(skeleton, m, c)
    line_epr = draw_best_fit_line(EPR, m, c)
    title_sk = "Esqueleto con Línea de Mejor Ajuste"
    title_epr = "Original con Línea de Mejor Ajuste"
  elif mode == 'parabolic':
    X, Y, a, b, c = best_fit_parabola(skeleton)
    predicted_Y = a * X**2 + b * X + c
    line_sk = draw_best_fit_parabola(skeleton, a, b, c)
    line_epr = draw_best_fit_parabola(EPR, a, b, c)
    title_sk = "Esqueleto con Parábola de Mejor Ajuste"
    title_epr = "Original con Parábola de Mejor Ajuste"

  if plot:
    show_image(line_sk, titles = title_sk)
    show_image(line_epr, titles = title_epr)

  r_squared = calculate_R2(X, Y, predicted_Y)

  if inform: print(f'Coeficiente de determinación (R2): {np.round(r_squared,3)}')

  return line_epr, r_squared


### Clasificación y Evaluación del Rendimiento

def classify(cant, R, umbral_r = 0.725, umbral_vessels = 0):
  if cant > umbral_vessels:
    if R < umbral_r:
      diagnostico = "CNV"
    else:
      diagnostico = "DME"
  else:
    if R < umbral_r:
      diagnostico = "Patología indefinida"
    else:
      diagnostico = "retina normal"

  return diagnostico