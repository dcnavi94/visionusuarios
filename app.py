import cv2 as cv   
import numpy as np  
import os   

# Crear carpeta de almacenamiento de rostros   
os.makedirs("user_faces", exist_ok=True)  

# Función de registrar rostros en la base de datos   
def registrar_rostros():  
    dni = input("Ingrese su DNI: ")  
    # crear subcarpetas donde se registren las imagenes del dni   
    user_folder = f"user_faces/{dni}"  
    os.makedirs(user_folder, exist_ok=True)   
    # Iniciar el capturador de video   
    cap = cv.VideoCapture(0)  # cámara por default de tu dispositivo   
    # detector de rostros   
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")  
    contador = 0   

    while contador < 50:  # Cambiado a 50 para capturar 50 imágenes  
        ret, frame = cap.read()  
        if not ret:  
            print("No se pudo capturar el video.")  
            break  
        # filtro gray   
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)  
        faces = face_detector.detectMultiScale(gray, 1.3, 5)  
        
        for (x, y, w, h) in faces:  
            face = gray[y:y+h, x:x+w]  
            # normalizar las imagenes  --> 150 x 150 px   
            face_resize = cv.resize(face, (150, 150), interpolation=cv.INTER_AREA)  
            cv.imwrite(f"{user_folder}/{contador}.jpg", face_resize)  # Corregido  
            contador += 1  
            
            # dibujar el rectangulo  
            cv.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Corregido  
            
            cv.imshow("captura de rostros", frame)  
            
            if contador >= 50:  # Cambiado para salir después de 50 imágenes  
                break  
            
        if cv.waitKey(1) & 0xFF == ord("q"):  
            break  

    print("Rostros registrados")        
    cap.release()  
    cv.destroyAllWindows()       

    # train
def train_model():
    #dataset --> label y data 
    #entrenar con LBPH FACE
    print("......entrenando modelo ........")
    # reconocedor de rostros
    # deteccion --->
    #deteccion y reconocimientos 
    face_recognizer = cv.face.LBPHFaceRecognizer.create()
    data = []
    labels = []
    #ruta de usuarios 
    base_dir = "user_faces"

    for user_folder in os.listdir(base_dir):
        user_path= os.path.join(base_dir,user_folder)
        if os.path.isdir(user_path):
            #extraer el label de la ruta de la user-path 
            label = int (user_folder)
            for image_file in os.listdir(user_path):
                image_path = os.path.join(user_path,image_file)
                #leer cada una de las imagenes 
                image = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
                if image is not None:
                    data.append(image)
                    labels.append(label)

        #Entreno mi modelo
    face_recognizer.train(data,np.array(labels))
    face_recognizer.write("model/ModeloLBPHFace.xml")
    print("Modelo entrenado con exito")
    


if __name__ == '__main__':  
    while True:  
        print("Sistema de Registro de Asistencias")  
        print("1. Registrar Usuarios")  
        print("2. Salir")  
        opcion = input("Seleccione opción: ")  
        
        if opcion == '1':  
            registrar_rostros()  
            train_model()
        elif opcion == '2':  
            break  
        else:  
            print("Opción no válida")