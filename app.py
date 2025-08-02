from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import pickle
import face_recognition
import numpy as np
import base64
import io
from PIL import Image

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'

# Configuração correta do SocketIO
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   logger=True,
                   engineio_logger=True)

# Carregar encodings
try:
    with open('data/encodings.pickle', 'rb') as f:
        data = pickle.loads(f.read())
        known_encodings = data['encodings']
        known_names = data['names']
except FileNotFoundError:
    known_encodings = []
    known_names = []

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    print('Cliente conectado')
    emit('status', {'msg': 'Conectado ao servidor'})

@socketio.on('disconnect')
def handle_disconnect():  # Removido parâmetro que estava causando erro
    print('Cliente desconectado')

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # Processar frame de vídeo
        image_data = data['image'].split(',')[1]
        image_bytes = base64.b64decode(image_data)
        image = Image.open(io.BytesIO(image_bytes))
        frame = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Processamento de reconhecimento facial
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        face_locations = face_recognition.face_locations(rgb_frame)
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        results = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(known_encodings, face_encoding)
            name = "Desconhecido"
            
            if True in matches:
                match_index = matches.index(True)
                name = known_names[match_index]
            
            results.append(name)
        
        emit('recognition_result', {'names': results, 'locations': face_locations})
        
    except Exception as e:
        print(f"Erro no processamento: {e}")
        emit('error', {'message': str(e)})

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
