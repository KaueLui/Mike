# servidor-central/app.py
from flask import Flask, render_template, request, jsonify
from flask_socketio import SocketIO, emit, join_room, leave_room
import cv2
import face_recognition
import pickle
import os
import numpy as np
import base64
from io import BytesIO
from PIL import Image
import json
import time
from datetime import datetime
import threading

app = Flask(__name__)
app.config['SECRET_KEY'] = 'facial-security-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, cors_allowed_origins="*")

# Configurações
ARQUIVO_ENCODINGS = 'data/encodings.pickle'
ARQUIVO_NODES = 'data/nodes.json'
ARQUIVO_ALERTS = 'data/alerts.json'

# Criar diretórios necessários
os.makedirs('data', exist_ok=True)
os.makedirs('uploads', exist_ok=True)
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Estado global do sistema
sistema = {
    'nodes': {},           # Nós conectados
    'alerts': [],          # Alertas recentes
    'stats': {
        'total_detections': 0,
        'active_nodes': 0,
        'known_faces': 0
    }
}

# Funções originais (mantidas do seu código)
def carregar_encodings_conhecidos():
    """Carrega os encodings e nomes do arquivo ou cria um dicionário vazio."""
    try:
        if os.path.exists(ARQUIVO_ENCODINGS) and os.path.getsize(ARQUIVO_ENCODINGS) > 0:
            with open(ARQUIVO_ENCODINGS, 'rb') as f:
                return pickle.load(f)
        else:
            return {"nomes": [], "encodings": []}
    except (FileNotFoundError, EOFError, pickle.UnpicklingError):
        return {"nomes": [], "encodings": []}

def salvar_encodings(dados):
    """Salva os encodings no arquivo."""
    with open(ARQUIVO_ENCODINGS, 'wb') as f:
        pickle.dump(dados, f)

# Funções do sistema distribuído
def carregar_nodes():
    """Carrega configurações dos nós."""
    try:
        if os.path.exists(ARQUIVO_NODES):
            with open(ARQUIVO_NODES, 'r') as f:
                return json.load(f)
        return {}
    except:
        return {}

def salvar_nodes():
    """Salva configurações dos nós."""
    with open(ARQUIVO_NODES, 'w') as f:
        json.dump(sistema['nodes'], f, indent=2)

def adicionar_alert(alert_data):
    """Adiciona novo alerta ao sistema."""
    alert = {
        'id': len(sistema['alerts']) + 1,
        'timestamp': datetime.now().isoformat(),
        'node_id': alert_data.get('node_id'),
        'location': alert_data.get('location'),
        'detected_faces': alert_data.get('faces', []),
        'type': alert_data.get('type', 'detection'),
        'severity': alert_data.get('severity', 'info')
    }
    
    sistema['alerts'].insert(0, alert)  # Mais recente primeiro
    
    # Manter apenas últimos 1000 alertas
    if len(sistema['alerts']) > 1000:
        sistema['alerts'] = sistema['alerts'][:1000]
    
    # Salvar alertas
    try:
        with open(ARQUIVO_ALERTS, 'w') as f:
            json.dump(sistema['alerts'][:100], f, indent=2)
    except:
        pass
    
    return alert

# =================== ROTAS DO SISTEMA DISTRIBUÍDO ===================

@app.route('/')
def dashboard():
    """Dashboard principal do sistema distribuído."""
    dados = carregar_encodings_conhecidos()
    sistema['stats']['known_faces'] = len(dados.get('nomes', []))
    sistema['stats']['active_nodes'] = len([n for n in sistema['nodes'].values() if n.get('status') == 'online'])
    
    return render_template('dashboard.html', 
                         stats=sistema['stats'],
                         nodes=sistema['nodes'],
                         recent_alerts=sistema['alerts'][:10])

@app.route('/nodes')
def nodes_page():
    """Página de gerenciamento de nós."""
    return render_template('nodes.html', nodes=sistema['nodes'])

@app.route('/alerts')
def alerts_page():
    """Página de histórico de alertas."""
    return render_template('alerts.html', alerts=sistema['alerts'][:50])

# =================== ROTAS DO SISTEMA WEB ORIGINAL ===================

@app.route('/web')
def web_index():
    """Página inicial do sistema web original."""
    dados = carregar_encodings_conhecidos()
    pessoas_cadastradas = dados.get('nomes', [])
    return render_template('web/index.html', pessoas=pessoas_cadastradas)

@app.route('/web/cadastro')
def web_cadastro():
    """Página de cadastro de novos rostos."""
    return render_template('web/cadastro.html')

@app.route('/web/reconhecimento')
def web_reconhecimento():
    """Página de reconhecimento em tempo real."""
    return render_template('web/reconhecimento.html')

# =================== APIs INTEGRADAS ===================

@app.route('/api/cadastrar', methods=['POST'])
def cadastrar_rosto():
    """API para cadastrar um novo rosto (integrada)."""
    try:
        # Verificar se é JSON
        if not request.is_json:
            return jsonify({'erro': 'Content-Type deve ser application/json'}), 400
            
        data = request.get_json()
        if not data:
            return jsonify({'erro': 'Dados JSON inválidos'}), 400
            
        nome = data.get('nome', '').strip()
        imagem_base64 = data.get('imagem', '')
        
        # Validações
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
        
        # Processar imagem
        try:
            if ',' in imagem_base64:
                image_data = base64.b64decode(imagem_base64.split(',')[1])
            else:
                image_data = base64.b64decode(imagem_base64)
                
            image = Image.open(BytesIO(image_data))
            frame = np.array(image)
            
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                rgb_frame = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB)
            else:
                rgb_frame = frame
            
            print(f"Processando imagem para {nome}...")
            caixas_rosto = face_recognition.face_locations(rgb_frame, model='hog')
            encodings_rosto = face_recognition.face_encodings(rgb_frame, caixas_rosto)
            
            print(f"Rostos detectados: {len(caixas_rosto)}")
            
            if len(encodings_rosto) == 1:
                # Adicionar novo rosto
                dados_conhecidos["encodings"].append(encodings_rosto[0])
                dados_conhecidos["nomes"].append(nome)
                salvar_encodings(dados_conhecidos)
                
                # Atualizar estatísticas
                sistema['stats']['known_faces'] = len(dados_conhecidos["nomes"])
                
                # INTEGRAÇÃO: Notificar nós sobre novo cadastro
                socketio.emit('face_database_updated', {
                    'action': 'added',
                    'name': nome,
                    'total_faces': len(dados_conhecidos["nomes"])
                }, room='nodes')
                
                # INTEGRAÇÃO: Notificar dashboard
                socketio.emit('system_update', {
                    'type': 'new_face_registered',
                    'data': {'name': nome}
                }, room='dashboard')
                
                print(f"✅ {nome} cadastrado com sucesso!")
                return jsonify({'sucesso': f'{nome} foi cadastrado com sucesso!'})
                
            elif len(encodings_rosto) > 1:
                return jsonify({'erro': f'Detectados {len(encodings_rosto)} rostos na imagem. Use apenas um rosto.'}), 400
            else:
                return jsonify({'erro': 'Nenhum rosto detectado. Tente com melhor iluminação.'}), 400
                
        except Exception as e:
            return jsonify({'erro': f'Erro ao processar imagem: {str(e)}'}), 400
            
    except Exception as e:
        print(f"❌ Erro no cadastro: {str(e)}")
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500

@app.route('/api/reconhecer', methods=['POST'])
def reconhecer_rosto():
    """API para reconhecer rostos (do sistema original)."""
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
            model='hog',
            number_of_times_to_upsample=0
        )
        
        if not face_locations:
            return jsonify({'rostos': []})
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        resultados = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
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

# =================== APIs DO SISTEMA DISTRIBUÍDO ===================

@app.route('/api/nodes', methods=['GET'])
def listar_nodes():
    """Lista todos os nós registrados."""
    return jsonify({'nodes': sistema['nodes']})

@app.route('/api/alerts', methods=['GET'])
def listar_alerts():
    """Lista alertas recentes."""
    limit = request.args.get('limit', 50, type=int)
    return jsonify({'alerts': sistema['alerts'][:limit]})

# =================== WEBSOCKET EVENTS ===================

@socketio.on('connect')
def handle_connect():
    print(f'Cliente conectado: {request.sid}')

@socketio.on('disconnect')
def handle_disconnect():
    print(f'Cliente desconectado: {request.sid}')
    
    # Verificar se era um nó sensor
    for node_id, node_data in sistema['nodes'].items():
        if node_data.get('session_id') == request.sid:
            node_data['status'] = 'offline'
            node_data['last_seen'] = datetime.now().isoformat()
            
            # Notificar dashboard
            socketio.emit('node_status_changed', {
                'node_id': node_id,
                'status': 'offline'
            }, room='dashboard')
            break

@socketio.on('register_node')
def handle_node_registration(data):
    """Registrar novo nó sensor."""
    node_id = data.get('node_id')
    location = data.get('location', 'Localização não informada')
    node_type = data.get('type', 'camera')
    
    if not node_id:
        emit('registration_error', {'error': 'node_id é obrigatório'})
        return
    
    # Registrar nó
    sistema['nodes'][node_id] = {
        'id': node_id,
        'location': location,
        'type': node_type,
        'status': 'online',
        'session_id': request.sid,
        'registered_at': datetime.now().isoformat(),
        'last_seen': datetime.now().isoformat(),
        'stats': {
            'detections_today': 0,
            'total_detections': 0
        }
    }
    
    salvar_nodes()
    join_room('nodes')
    
    emit('registration_success', {
        'node_id': node_id,
        'message': f'Nó {node_id} registrado com sucesso'
    })
    
    # Notificar dashboard
    socketio.emit('node_registered', {
        'node': sistema['nodes'][node_id]
    }, room='dashboard')
    
    print(f'✅ Nó registrado: {node_id} em {location}')

@socketio.on('detection_event')
def handle_detection_event(data):
    """Processar evento de detecção de um nó."""
    node_id = data.get('node_id')
    faces = data.get('faces', [])
    timestamp = data.get('timestamp', datetime.now().isoformat())
    
    if node_id not in sistema['nodes']:
        emit('error', {'message': 'Nó não registrado'})
        return
    
    # Atualizar status do nó
    sistema['nodes'][node_id]['last_seen'] = datetime.now().isoformat()
    sistema['nodes'][node_id]['stats']['total_detections'] += len(faces)
    sistema['nodes'][node_id]['stats']['detections_today'] += len(faces)
    
    # Criar alerta
    alert = adicionar_alert({
        'node_id': node_id,
        'location': sistema['nodes'][node_id]['location'],
        'faces': faces,
        'type': 'face_detection',
        'severity': 'info' if any(f.get('nome') != 'Desconhecido' for f in faces) else 'warning'
    })
    
    # Atualizar estatísticas globais
    sistema['stats']['total_detections'] += len(faces)
    
    # Notificar dashboard em tempo real
    socketio.emit('new_detection', {
        'alert': alert,
        'node': sistema['nodes'][node_id],
        'stats': sistema['stats']
    }, room='dashboard')
    
    emit('detection_processed', {'alert_id': alert['id']})
    
    print(f'📸 Detecção em {node_id}: {len(faces)} rosto(s)')

@socketio.on('join_dashboard')
def handle_join_dashboard():
    """Cliente quer receber atualizações do dashboard."""
    join_room('dashboard')
    emit('dashboard_joined', {
        'stats': sistema['stats'],
        'active_nodes': len([n for n in sistema['nodes'].values() if n.get('status') == 'online']),
        'recent_alerts': sistema['alerts'][:5]
    })

@socketio.on('heartbeat')
def handle_heartbeat(data):
    """Heartbeat dos nós sensores."""
    node_id = data.get('node_id')
    if node_id in sistema['nodes']:
        sistema['nodes'][node_id]['last_seen'] = datetime.now().isoformat()
        sistema['nodes'][node_id]['status'] = 'online'

# =================== MONITORAMENTO ===================

def monitor_nodes():
    """Monitorar status dos nós."""
    while True:
        try:
            now = datetime.now()
            for node_id, node_data in sistema['nodes'].items():
                if node_data.get('status') == 'online':
                    last_seen = datetime.fromisoformat(node_data.get('last_seen', now.isoformat()))
                    # Se não teve heartbeat há mais de 30 segundos, marcar como offline
                    if (now - last_seen).seconds > 30:
                        node_data['status'] = 'offline'
                        socketio.emit('node_status_changed', {
                            'node_id': node_id,
                            'status': 'offline'
                        }, room='dashboard')
            
            time.sleep(10)  # Verificar a cada 10 segundos
        except:
            time.sleep(10)

def init_system():
    """Inicializar sistema distribuído."""
    global sistema
    
    # Carregar dados salvos
    sistema['nodes'] = carregar_nodes()
    
    try:
        if os.path.exists(ARQUIVO_ALERTS):
            with open(ARQUIVO_ALERTS, 'r') as f:
                sistema['alerts'] = json.load(f)
    except:
        sistema['alerts'] = []
    
    # Marcar todos os nós como offline inicialmente
    for node_data in sistema['nodes'].values():
        node_data['status'] = 'offline'
    
    # Atualizar estatísticas
    dados = carregar_encodings_conhecidos()
    sistema['stats']['known_faces'] = len(dados.get('nomes', []))
    
    print("🚀 Sistema distribuído inicializado!")

if __name__ == '__main__':
    init_system()
    
    # Iniciar thread de monitoramento
    monitor_thread = threading.Thread(target=monitor_nodes, daemon=True)
    monitor_thread.start()
    
    print("🌐 Servidor Central iniciado em http://0.0.0.0:5000")
    print("📱 Sistema Web em http://0.0.0.0:5000/web")
    print("📊 Dashboard em http://0.0.0.0:5000/")
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)