import numpy as np
import imutils
import time
import dlib
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from imutils.video import VideoStream
from imutils.video import FPS
from utils.centroidtracker import CentroidTracker
from utils.trackableobject import TrackableObject
import tensorflow as tf

PATH_TO_MODEL_DIR = "fine_tuned_model_bars"
PATH_TO_SAVE_MODEL = PATH_TO_MODEL_DIR + "/saved_model"

detect_fn = tf.saved_model.load(PATH_TO_SAVE_MODEL)

# Ruta del video (Se debe cargar de manera manual)
PATH_VIDEO = "videos/bars.mp4"

# Ruta del video en donde almacenaremos los resultados
PATH_OUTPUT = "videos/video_out_st_bars.mp4"

# Cuántos frames vamos a saltarnos (Durante estos frames nuestro algoritmo de seguimiento funciona)
SKIP_FPS = 1

# Cuál será el umbral mínimo par que se considere una detección
TRESHOLD = 0.5

# Cargamos el video
vs = cv2.VideoCapture(PATH_VIDEO)

# Inicializamos el writer para poder guardar el video
writer = None

# Definimos ancho y alto
W = int(vs.get(cv2.CAP_PROP_FRAME_WIDTH))
H = int(vs.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Inicializamos la clase centroid tracker con dos variable fundamentales
# maxDissapared (Si pasa ese tiempo y no se detecta más el centroide lo elimina)
# Si la distancia es mayor a maxDistance no lo podra asociar como si fuera el mismo objecto.
ct = CentroidTracker(maxDisappeared=40, maxDistance=50)

# Inicializamos variables principales
trackers = []
trackableObjects = {}

totalFrame = 0
totalDown = 0
totalUp = 0

DIRECTION_PEOPLE = True

# Creamos un umbral para sabre si el carro paso de izquierda a derecha o viceversa
# En este caso lo deje fijo pero se pudiese configurar según la ubicación de la cámara.
POINT = [0, int((H/2)-H*0.1), W, int(H*0.1)]

# Los FPS nos van a permitir ver el rendimiento de nuestro modelo y si funciona en tiempo real.
fps = FPS().start()

# Definimos el formato del archivo resultante y las rutas.
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
writer = cv2.VideoWriter(PATH_OUTPUT, fourcc, 20.0, (W, H), True)

# Bucle que recorre todo el video
while True:
    # Leemos el primer frame
    ret, frame = vs.read()

    # Si ya no hay más frame, significa que el video termino y por tanto se sale del bucle
    if frame is None:
        break

    status = "Waiting"
    rects = []

    # Deteccion de imagenes, la salida es un array de rectangulos
    # Nos saltamos los frames especificados.
    if totalFrame % SKIP_FPS == 0:
        status = "Detecting"
        trackers = []
        # Tomamos la imagen la convertimos a array luego a tensor
        image_np = np.array(frame)

        input_tensor = tf.convert_to_tensor(image_np)
        input_tensor = input_tensor[tf.newaxis, ...]

        # Predecimos los objectos y clases de la imagen
        detections = detect_fn(input_tensor)

        detection_scores = np.array(detections["detection_scores"][0])
        # Realizamos una limpieza para solo obtener las clasificaciones mayores al umbral.
        detection_clean = [x for x in detection_scores if x >= TRESHOLD]

        # Recorremos las detecciones
        for x in range(len(detection_clean)):
            idx = int(detections['detection_classes'][0][x])
            # Tomamos los bounding box
            ymin, xmin, ymax, xmax = np.array(
                detections['detection_boxes'][0][x])
            box = [xmin, ymin, xmax, ymax] * np.array([W, H, W, H])

            (startX, startY, endX, endY) = box.astype("int")

            # Con la función de dlib empezamos a hacer seguimiento de los boudiung box obtenidos
            tracker = dlib.correlation_tracker()
            rect = dlib.rectangle(startX, startY, endX, endY)
            tracker.start_track(frame, rect)

            trackers.append(tracker)
    else:
        # En caso de que no hagamos detección haremos seguimiento
        # Recorremos los objetos que se les está realizando seguimiento
        for tracker in trackers:
            status = "Tracking"
            # Actualizamos y buscamos los nuevos bounding box
            tracker.update(frame)
            pos = tracker.get_position()

            startX = int(pos.left())
            startY = int(pos.top())
            endX = int(pos.right())
            endY = int(pos.bottom())

            rects.append((startX, startY, endX, endY))

    # Dibujamos el umbral de conteo
    cv2.rectangle(frame, (POINT[0], POINT[1]), (POINT[0] +
                  POINT[2], POINT[1] + POINT[3]), (255, 0, 255), 2)

    # Se hace update al tracker de centroides
    objects = ct.update(rects)

    # Recorremos cada una de las detecciones
    for (objectID, centroid) in objects.items():
        # Revisamos si el objeto ya se ha contado
        to = trackableObjects.get(objectID, None)
        if to is None:
            to = TrackableObject(objectID, centroid)

        else:
            # Si no se ha contado, analizamos la dirección del objeto
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)
            if not to.counted:
                if centroid[0] > POINT[0] and centroid[0] < (POINT[0] + POINT[2]) and centroid[1] > POINT[1] and centroid[1] < (POINT[1]+POINT[3]):
                    if DIRECTION_PEOPLE:
                        if direction > 0:
                            totalUp += 1
                            to.counted = True
                        else:
                            totalDown += 1
                            to.counted = True
                    else:
                        if direction < 0:
                            totalUp += 1
                            to.counted = True
                        else:
                            totalDown += 1
                            to.counted = True

        trackableObjects[objectID] = to

        # Dibujamos el centroide y el ID de la detección encontrada
        text = "ID {}".format(objectID)
        cv2.putText(frame, text, (centroid[0]-10, centroid[1]-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)

    # Totalizamos los resultados finales
    info = [
        ("Subiendo", totalUp),
        ("Bajando", totalDown),
        ("Estado", status),
    ]

    for (i, (k, v)) in enumerate(info):
        text = "{}: {}".format(k, v)
        cv2.putText(frame, text, (10, H - ((i*20) + 20)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Almacenamos el framme en nuestro video resultante.
    writer.write(frame)
    totalFrame += 1
    fps.update()

# Terminamos de analizar FPS y mostramos resultados finales
fps.stop()

print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

# Cerramos el stream the almacenar video y de consumir el video.
writer.release()
vs.release()
