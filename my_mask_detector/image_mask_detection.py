import os

import cv2
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

# receber e tratar os parâmetros
# ap = argparse.ArgumentParser()
# ap.add_argument("-i", "--image", required=True, help="caminho para a imagem")
# ap.add_argument("-c", "--confidence", type=float, default=0.5, help="probabilidade para filtrar detecções")
# args = vars(ap.parse_args())
# img_path = args['image']
# acepted_confidence = args['confidence']
images_path = os.path.join('test_images', 'test')
predictions_path = os.path.join('test_images', 'predicted')
img_path = 'test_images/test/test (1).jpg'
acepted_confidence = 0.5

# carregar dnn
print("[INFO] carregando modelo de detecção facial...")
prototxtPath = os.path.join('face_detector', 'deploy.prototxt')
weightsPath = os.path.join('face_detector', "res10_300x300_ssd_iter_140000.caffemodel")
net = cv2.dnn.readNet(prototxtPath, weightsPath)

# carregar detector de máscaras
print("[INFO] carregando modelo de detecção de máscaras...")
model = load_model('my_trained_model.model')


def predict_image_with_face_detection(img_path, pred_path = predictions_path, show_pred = False):

    # carregar imagem
    image = cv2.imread(img_path)
    img_orig = image.copy()
    (h, w) = image.shape[:2]

    # passar o blob pela rede e obter detecções
    print("[INFO] computando detecções de faces...")
    blob = cv2.dnn.blobFromImage(image, 1.0, (300, 300), (104.0, 177.0, 123.0))
    net.setInput(blob)
    detections = net.forward()

    # iterar sobre detecções, aceitando acima de determinado nível de confiança
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > acepted_confidence:
            # pega as coordenadas da imagem
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            # garante que o box esteja dentro das dimensões
            (startX, startY) = (max(0, startX), max(0, startY))
            (endX, endY) = (min(w - 1, endX), min(h - 1, endY))

            # extrai a face da imagem, redimensiona para 224x224
            face = image[startY:endY, startX:endX]

            #face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

            # prepara imagem para predição, redimensionando para 224x224 e aumentando uma dimensão no array
            face = cv2.resize(face, (224, 224))
            face = img_to_array(face)
            face = preprocess_input(face)
            face = np.expand_dims(face, axis=0)

            # faz a predição propriamente dita e salva a probabilidade nas variáveis
            (mask, withoutMask) = model.predict(face)[0]

            # determina o rótulo e a respectiva cor baseado na predição feita
            label = "Com mascara" if mask > withoutMask else "SEM mascara"
            label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
            color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)

            # adiciona o rótulo e um retângulo com a cor na imagem
            cv2.rectangle(img_orig, (0, h - 60), (w, h), (0, 0, 0), -1)
            #cv2.putText(img_orig, label, (startX, startY - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.putText(img_orig, label, (w + 20, h + 150), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)
            cv2.rectangle(img_orig, (startX, startY), (endX, endY), color, 2)

            if show_pred:
                # após a iteração e predições, mostra a imagem e salva em disco se apertar 's'
                cv2.imshow("Output", img_orig)
                k = cv2.waitKey(0)
                if k == 27:  # aguarda ESC para sair do programa
                    cv2.destrosyWindow('Output')
                    continue
                elif k == ord('s'):  # aguarda 's' para salvar imagem e sair do programa
                    filename = os.path.split(img_path)[-1]
                    cv2.imwrite(os.path.join('test_images', 'predicted', filename), img_orig)
                    cv2.destroyWindow('Output')
                    continue
            else:
                filename = os.path.split(img_path)[-1]
                cv2.imwrite(os.path.join('test_images', 'predicted', filename), img_orig)

def predict_image_without_face_detection(img_path, pred_path = predictions_path, show_pred = False):

    # carregar imagem
    image = cv2.imread(img_path)
    img_orig = image.copy()
    (h, w) = image.shape[:2]

    face = cv2.resize(image, (224, 224))
    face = img_to_array(face)
    face = preprocess_input(face)
    face = np.expand_dims(face, axis=0)

    # faz a predição propriamente dita e salva a probabilidade nas variáveis
    (mask, withoutMask) = model.predict(face)[0]

    # determina o rótulo e a respectiva cor baseado na predição feita
    label = "Com mascara" if mask > withoutMask else "SEM mascara"
    label = "{}: {:.2f}%".format(label, max(mask, withoutMask) * 100)
    color = (0, 255, 0) if mask > withoutMask else (0, 0, 255)
    print(f'imagem {img_path} - {label}')

    # adiciona o rótulo e um retângulo com a cor na imagem
    cv2.putText(image, label, (40, h - 100), cv2.FONT_HERSHEY_SIMPLEX, 0.85, color, 2)

    if show_pred:
        # após a iteração e predições, mostra a imagem e salva em disco se apertar 's'
        cv2.imshow("Output", image)
        k = cv2.waitKey(0)
        if k == 27:  # aguarda ESC para sair do programa
            cv2.destroyWindow('Output')
        elif k == ord('s'):  # aguarda 's' para salvar imagem e sair do programa
            filename = os.path.split(img_path)[-1]
            cv2.imwrite(os.path.join('test_images', 'predicted', filename), image)
            cv2.destroyWindow('Output')

    else:
        filename = os.path.split(img_path)[-1]
        cv2.imwrite(os.path.join('test_images', 'predicted', filename), image)

for img in os.listdir(images_path):
    print(f'imagem {img} sendo detectada')
    predict_image_with_face_detection(os.path.join(images_path, img), show_pred=True)

print('fim')