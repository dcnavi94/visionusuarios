import cv2 as cv 
import numpy as np 
import os 

#users_faces   ---> UserFaces ----> userFaces
os.makedirs("user_faces", exist_ok=True)
#Funcion de registrar rostros 
def registrar_rostros():
    dni = input("Ingrese su DNI:")
    #Crear un subcarpeta donde se registre las iamgenes de ese dni 
    user_folder = f"user_faces/{dni}"
    os.makedirs(user_folder,exist_ok=True)
    #Iniciar el capturador de video 
    cap = cv.VideoCapture(0) # camra por default de tu dispotivo  
    #Detector de rostros 
    contador = 0
    face_detector = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml") 
    while contador < 100 : 
        ret,frame = cap.read()
        if not ret:
            break
        #Filtro gray 
        gray = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray,1.3,5)
        for (x,y,w,h) in faces:
            face = gray[y:y+h,x:x+w]
            #Normalizar las imagenes ---> 150 px x 150 px 
            face_resize = cv.resize(face,(150,150),interpolation=cv.INTER_AREA)
            cv.imwrite(f"{user_folder}/{contador}.jpg",face_resize)
            contador += 1 
            #Dibujar el rectangulo
            cv.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
            if contador == 50:
                break
            cv.imshow("Captura de Rostros",frame)

            if cv.waitKey(1) & 0xFF == ord("q"):
                break
    print("Rostro Capturado")
    cap.release()
    cv.destroyAllWindows()

#Entrenar el modelo con esos rostros 
def train_model():
    #Training LBPHFace 
    print("Entrenando el modelo...")
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    #Dataset: 
    # imagnes y son etiquetas 
    data = []
    labels = []
    #Ruta de Usuarios
    base_dir = "users_faces"
    for user_folder in os.listdir(base_dir):
        user_path = os.path.join(base_dir,user_folder)
        if os.path.isdir(user_path):
            #Extraer el dni del nombre de la carpeta para convertirlo en label y asignarlo a su conjunto de imagenes
            label = int(user_folder)
            for image_file in os.listdir(user_path):
                image_path = os.path.join(user_path,image_file)
                image = cv.imread(image_path,cv.IMREAD_GRAYSCALE)
                if image is not None:
                    data.append(image)
                    labels.append(label)
        
    #Entrenar el modelo
    face_recognizer.train(data, np.array(labels))
    #Ruta de guardado del modelo
    face_recognizer.write("model/ModeloLBPHFace.xml")
    print("Modelo entrenado con éxito")

#registro de asistencias 
def registrar_asistencias():
    face_recognizer = cv.face.LBPHFaceRecognizer_create()
    try:
        face_recognizer.read("model/ModeloLBPHFace.xml")
    except Exception as e:
        print("Error al cargar el modelo:", e)
        return
    cap = cv.VideoCapture(0)
    face_cascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")
    #COnfidencia es el valor que mepermite decidir si la predcicion que arrojo mim modelo es veridica 
    while True:
        ret, frame = cap.read()
        if not ret:
            print("No se puede acceder a la cámara")
            break
        gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)
        for(x,y,w,h) in faces:
            face = gray[y:y+h, x:x+w]
            face_rize = cv.resize(face, (150,150), interpolation=cv.INTER_CUBIC)
            label, confidencia = face_recognizer.predict(face_rize)

            print("Label:", label)
            print("Confidencia:", confidencia)
            cv.rectangle(frame, (x,y), (x+w, y+h), (0,255,0),2) 
            cv.putText(frame, f"{label}", (x,y-10), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2, cv.LINE_AA)
        cv.imshow("Registro de Asistencias", frame)
        if cv.waitKey(1) & 0xFF == ord("q"):
            break
    cap.release()
    

#Define main fnction 
if __name__ == "__main__":
    while True : 
        print("Sistema de Registro de Asistencias")
        print("1.Registrar Usuarios")
        print("2.Registrar Asistencias")
        print("3.Salir")
        opcion = input("Seleccione una opcion:")
        if opcion == "1":
            registrar_rostros()
            train_model()
        elif opcion == "2":
            registrar_asistencias()
        elif opcion == "3":
            print("Saliendo del sistema...")
            break
        else:
            print("Opcion no valida")

    # users_faces
    # - 727382382 ----> label ---> 0.jpg
         # 0.jpg 

    # - kjwsdksdksjdk 
    # -djkwldjsojdksjd