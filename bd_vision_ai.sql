CREATE DATABASE IF NOT EXISTS bd_vision_ai_g02;
use bd_vision_ai_g02;
  
-- Tabla de usuarios
CREATE TABLE Usuarios(
    id_usuario INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    dni varchar(9) NOT NULL,
    path_foto varchar(255) NOT NULL,
    nombres varchar(50) NOT NULL,
    apellidos varchar(50) NOT NULL,
    correo varchar(50) NOT NULL
);                                                

-- Tabla de Horarios 
CREATE TABLE Horarios(
    id_horario INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    hora_inicio TIME NOT NULL,
    hora_maxima TIME NOT NULL
);

-- Tabla de Asistencias 
CREATE TABLE Asistencias(
    id_asistencia INT NOT NULL AUTO_INCREMENT PRIMARY KEY,
    id_usuario INT NOT NULL,
    id_horario INT NOT NULL,
    fecha_asistencia DATE NOT NULL,
    hora_asistencia TIME NOT NULL,
    hora_asistencia_salida TIME NOT NULL,
    estado_asistencia ENUM('PRESENTE', 'AUSENTE', 'TARDANZA') NOT NULL,
    FOREIGN KEY (id_usuario) REFERENCES Usuarios(id_usuario),
    FOREIGN KEY (id_horario) REFERENCES Horarios(id_horario)
);
