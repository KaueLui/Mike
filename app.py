from flask import Flask, render_template, request, jsonify, Response
import cv2
import face_recognition
import pickle
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

ARQUIVO_ENCODINGS = 'encodings.pickle'

# Criar diretório de uploads se não existir
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def carregar_encodings_conhecidos():
    """Carrega os encodings e nomes do arquivo ou cria um dicionário vazio."""
    try:
        # Verificar se o arquivo existe e não está vazio
        if os.path.exists(ARQUIVO_ENCODINGS) and os.path.getsize(ARQUIVO_ENCODINGS) > 0:
            with open(ARQUIVO_ENCODINGS, 'rb') as f:
                return pickle.load(f)
        else:
            return {"nomes": [], "encodings": []}
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        # Se houver qualquer erro, retorna dicionário vazio
        return {"nomes": [], "encodings": []}

def salvar_encodings(dados):
    """Salva os encodings no arquivo."""
    with open(ARQUIVO_ENCODINGS, 'wb') as f:
        pickle.dump(dados, f)

@app.route('/')
def index():
    """Página inicial."""
    dados = carregar_encodings_conhecidos()
    pessoas_cadastradas = dados.get('nomes', [])
    return render_template('index.html', pessoas=pessoas_cadastradas)

@app.route('/cadastro')
def cadastro():
    """Página de cadastro de novos rostos."""
    return render_template('cadastro.html')

@app.route('/reconhecimento')
def reconhecimento():
    """Página de reconhecimento em tempo real."""
    return render_template('reconhecimento.html')

@app.route('/api/cadastrar', methods=['POST'])
def cadastrar_rosto():
    """API para cadastrar um novo rosto."""
    try:
        # Verificar se é JSON
        if not request.is_json:
            return jsonify({'erro': 'Content-Type deve ser application/json'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'erro': 'Dados JSON inválidos'}), 400
            
        nome = data.get('nome', '').strip()
        imagem_base64 = data.get('imagem', '')
        
        # Validações mais específicas
        if not nome:
            return jsonify({'erro': 'Nome é obrigatório'}), 400
            
        if len(nome) < 2:
            return jsonify({'erro': 'Nome deve ter pelo menos 2 caracteres'}), 400
            
        if not imagem_base64:
            return jsonify({'erro': 'Imagem é obrigatória'}), 400
        
        # Verificar se já existe pessoa com mesmo nome
        dados_conhecidos = carregar_encodings_conhecidos()
        if nome.lower() in [n.lower() for n in dados_conhecidos.get("nomes", [])]:
            return jsonify({'erro': f'Já existe uma pessoa cadastrada com o nome "{nome}"'}), 400
        
        # Decodificar imagem base64
        try:
            if ',' in imagem_base64:
                image_data = base64.b64decode(imagem_base64.split(',')[1])
            else:
                image_data = base64.b64decode(imagem_base64)
                
            image = Image.open(BytesIO(image_data))
        except Exception as e:
            return jsonify({'erro': f'Erro ao decodificar imagem: {str(e)}'}), 400
        
        # Converter para array numpy
        frame = np.array(image)
        
        # Verificar se a imagem não está vazia
        if frame.size == 0:
            return jsonify({'erro': 'Imagem está vazia'}), 400
            
        # Converter cores se necessário
        if len(frame.shape) == 3 and frame.shape[2] == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # Detectar rostos
        print(f"Processando imagem para {nome}...")
        caixas_rosto = face_recognition.face_locations(rgb_frame, model='hog')
        encodings_rosto = face_recognition.face_encodings(rgb_frame, caixas_rosto)
        
        print(f"Rostos detectados: {len(caixas_rosto)}")
        
        if len(encodings_rosto) == 1:
            # Adicionar novo rosto
            dados_conhecidos["encodings"].append(encodings_rosto[0])
            dados_conhecidos["nomes"].append(nome)
            
            # Salvar
            salvar_encodings(dados_conhecidos)
            print(f"✅ {nome} cadastrado com sucesso!")
            
            return jsonify({'sucesso': f'{nome} foi cadastrado com sucesso!'})
            
        elif len(encodings_rosto) > 1:
            return jsonify({'erro': f'Detectados {len(encodings_rosto)} rostos na imagem. Certifique-se de que apenas um rosto está visível.'}), 400
        else:
            return jsonify({'erro': 'Nenhum rosto foi detectado na imagem. Tente capturar novamente com melhor iluminação.'}), 400
            
    except Exception as e:
        print(f"❌ Erro no cadastro: {str(e)}")
        return jsonify({'erro': f'Erro interno do servidor: {str(e)}'}), 500

@app.route('/api/reconhecer', methods=['POST'])
def reconhecer_rosto():
    """API para reconhecer rostos em uma imagem."""
    try:
        data = request.get_json()
        imagem_base64 = data.get('imagem')
        
        if not imagem_base64:
            return jsonify({'erro': 'Imagem é obrigatória'}), 400
        
        # Carregar dados conhecidos
        dados_conhecidos = carregar_encodings_conhecidos()
        known_face_encodings = dados_conhecidos.get("encodings", [])
        known_face_names = dados_conhecidos.get("nomes", [])
        
        if not known_face_encodings:
            return jsonify({'rostos': []})
        
        # Decodificar imagem
        image_data = base64.b64decode(imagem_base64.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = np.array(image)
        
        # Converter para RGB
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # Usar configurações otimizadas para tempo real
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model='hog',  # Mais rápido que 'cnn'
            number_of_times_to_upsample=0  # Sem upsampling para velocidade
        )
        
        # Apenas processar se houver rostos
        if not face_locations:
            return jsonify({'rostos': []})
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        resultados = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            # Usar tolerância maior para melhor performance
            matches = face_recognition.compare_faces(
                known_face_encodings, 
                face_encoding, 
                tolerance=0.6
            )
            name = "Desconhecido"
            
            if True in matches:
                first_match_index = matches.index(True)
                name = known_face_names[first_match_index]
            
            resultados.append({
                'nome': name,
                'localizacao': {
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left)
                }
            })
        
        return jsonify({'rostos': resultados})
        
    except Exception as e:
        print(f"Erro no reconhecimento: {str(e)}")
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500

@app.route('/api/pessoas')
def listar_pessoas():
    """API para listar pessoas cadastradas."""
    dados = carregar_encodings_conhecidos()
    pessoas = dados.get('nomes', [])
    return jsonify({'pessoas': pessoas})

@app.route('/api/detectar_rosto', methods=['POST'])
def detectar_rosto():
    """API otimizada para detectar rostos sem reconhecer."""
    try:
        data = request.get_json()
        imagem_base64 = data.get('imagem')
        
        if not imagem_base64:
            return jsonify({'erro': 'Imagem é obrigatória'}), 400
        
        # Decodificar imagem
        image_data = base64.b64decode(imagem_base64.split(',')[1])
        image = Image.open(BytesIO(image_data))
        frame = np.array(image)
        
        # Converter para RGB
        if len(frame.shape) == 3:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
        else:
            rgb_frame = frame
        
        # Usar configurações ultra-rápidas para detecção apenas
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model='hog',
            number_of_times_to_upsample=0
        )
        
        resultados = []
        for (top, right, bottom, left) in face_locations:
            resultados.append({
                'localizacao': {
                    'top': int(top),
                    'right': int(right),
                    'bottom': int(bottom),
                    'left': int(left)
                }
            })
        
        return jsonify({'rostos': resultados})
        
    except Exception as e:
        print(f"Erro na detecção: {str(e)}")
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)