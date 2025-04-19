import cv2

def contar_camaras_disponibles(max_camaras=10):
    disponibles = 0
    for i in range(max_camaras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            disponibles += 1
            cap.release()
    return disponibles

cantidad = contar_camaras_disponibles()
print(f"Cámaras disponibles: {cantidad}")

def listar_camaras_disponibles(max_camaras=10):
    camaras = []
    for i in range(max_camaras):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            camaras.append(i)
            cap.release()
    return camaras

camaras_disponibles = listar_camaras_disponibles()
print(f"Cámaras disponibles: {camaras_disponibles}")
