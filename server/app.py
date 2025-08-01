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
# Configurações melhoradas para o SocketIO
app.config['SECRET_KEY'] = 'facial-security-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   ping_timeout=120,      # Aumentar timeout para 2 minutos
                   ping_interval=30,      # Ping a cada 30 segundos
                   logger=False,
                   engineio_logger=False,
                   transports=['websocket', 'polling'])  # Adicionar transports

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

# =================== APIs DOS NÓS ===================

@app.route('/api/nodes', methods=['GET'])
def api_get_nodes():
    """API para listar todos os nós."""
    return jsonify({
        'nodes': sistema['nodes'],
        'total': len(sistema['nodes']),
        'online': len([n for n in sistema['nodes'].values() if n.get('status') == 'online']),
        'offline': len([n for n in sistema['nodes'].values() if n.get('status') == 'offline'])
    })

@app.route('/api/nodes', methods=['POST'])
def api_add_node():
    """API para adicionar novo nó."""
    data = request.get_json()
    
    node_id = data.get('node_id')
    if not node_id:
        return jsonify({'erro': 'ID do nó é obrigatório'}), 400
    
    if node_id in sistema['nodes']:
        return jsonify({'erro': 'Nó já existe'}), 400
    
    # Criar novo nó
    novo_no = {
        'id': node_id,
        'location': data.get('location', ''),
        'type': data.get('type', 'camera'),
        'url': data.get('url'),
        'status': 'offline',
        'last_seen': datetime.now().isoformat(),
        'registered_at': datetime.now().isoformat(),
        'stats': {
            'total_detections': 0,
            'last_detection': None
        }
    }
    
    sistema['nodes'][node_id] = novo_no
    salvar_nodes()
    
    # Notificar dashboard
    socketio.emit('node_registered', {'node': novo_no}, room='dashboard')
    
    return jsonify({'sucesso': 'Nó adicionado com sucesso', 'node': novo_no})

@app.route('/api/nodes/<node_id>', methods=['PUT'])
def api_update_node(node_id):
    """API para atualizar configurações do nó."""
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    data = request.get_json()
    node = sistema['nodes'][node_id]
    
    # Atualizar campos permitidos
    if 'location' in data:
        node['location'] = data['location']
    if 'url' in data:
        node['url'] = data['url']
    if 'type' in data:
        node['type'] = data['type']
    
    node['updated_at'] = datetime.now().isoformat()
    salvar_nodes()
    
    # Notificar dashboard
    socketio.emit('node_updated', {'node': node}, room='dashboard')
    
    return jsonify({'sucesso': 'Nó atualizado com sucesso', 'node': node})

@app.route('/api/nodes/<node_id>', methods=['DELETE'])
def api_delete_node(node_id):
    """API para remover nó."""
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    del sistema['nodes'][node_id]
    salvar_nodes()
    
    # Notificar dashboard
    socketio.emit('node_removed', {'node_id': node_id}, room='dashboard')
    
    return jsonify({'sucesso': 'Nó removido com sucesso'})

@app.route('/api/nodes/<node_id>/stream')
def api_node_stream(node_id):
    """API para obter stream da câmera do nó."""
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    node = sistema['nodes'][node_id]
    camera_url = node.get('url')
    
    if not camera_url:
        return jsonify({'erro': 'URL da câmera não configurada'}), 400
    
    # Para IP Webcam, precisamos adicionar o endpoint correto
    if 'video' not in camera_url and ':8080' in camera_url:
        # IP Webcam Android - adicionar endpoint de video
        if not camera_url.endswith('/'):
            camera_url += '/'
        camera_url += 'video'
    
    try:
        # Testar conexão básica primeiro
        import requests
        test_response = requests.get(camera_url.replace('/video', '/'), timeout=3)
        
        # ATUALIZAR STATUS DO NÓ PARA ONLINE
        node['status'] = 'online'
        node['last_seen'] = datetime.now().isoformat()
        salvar_nodes()
        
        # NOTIFICAR DASHBOARD SOBRE MUDANÇA DE STATUS
        socketio.emit('node_status_changed', {
            'node_id': node_id,
            'status': 'online',
            'node': node
        }, room='dashboard')
        
        return jsonify({
            'status': 'online',
            'stream_url': camera_url,
            'node_id': node_id,
            'proxy_url': f'/api/nodes/{node_id}/proxy_stream'
        })
        
    except Exception as e:
        # MARCAR COMO OFFLINE SE FALHAR
        node['status'] = 'offline'
        salvar_nodes()
        
        socketio.emit('node_status_changed', {
            'node_id': node_id,
            'status': 'offline',
            'node': node
        }, room='dashboard')
        
        return jsonify({'erro': f'Câmera não acessível: {str(e)}'}), 503

@app.route('/api/nodes/<node_id>/proxy_stream')
def api_proxy_stream(node_id):
    """Proxy para stream da câmera (evita CORS)."""
    if node_id not in sistema['nodes']:
        return "Nó não encontrado", 404
    
    node = sistema['nodes'][node_id]
    camera_url = node.get('url')
    
    if not camera_url:
        return "URL não configurada", 400
    
    # Ajustar URL para IP Webcam
    if 'video' not in camera_url and ':8080' in camera_url:
        if not camera_url.endswith('/'):
            camera_url += '/'
        camera_url += 'video'
    
    try:
        import requests
        from flask import Response
        
        # CONFIGURAÇÕES OTIMIZADAS PARA STREAM
        resp = requests.get(
            camera_url, 
            stream=True, 
            timeout=30,
            headers={
                'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                'Connection': 'keep-alive'
            }
        )
        
        def generate():
            try:
                for chunk in resp.iter_content(chunk_size=8192):  # Chunks maiores
                    if chunk:
                        yield chunk
            except Exception as e:
                print(f"Erro no stream: {e}")
                yield b''  # Finalizar graciosamente
        
        return Response(
            generate(),
            content_type=resp.headers.get('content-type', 'multipart/x-mixed-replace'),
            headers={
                'Cache-Control': 'no-cache, no-store, must-revalidate',
                'Pragma': 'no-cache',
                'Expires': '0'
            }
        )
        
    except Exception as e:
        return f"Erro no stream: {str(e)}", 503

@app.route('/api/nodes/<node_id>/restart', methods=['POST'])
def api_restart_node(node_id):
    """API para reiniciar nó."""
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    try:
        # Marcar nó como reiniciando
        sistema['nodes'][node_id]['status'] = 'restarting'
        sistema['nodes'][node_id]['last_seen'] = datetime.now().isoformat()
        salvar_nodes()
        
        # Notificar sobre mudança de status
        socketio.emit('node_status_changed', {
            'node_id': node_id,
            'status': 'restarting',
            'node': sistema['nodes'][node_id]
        }, room='dashboard')
        
        # Simular comando de reinicialização
        socketio.emit('restart_command', {'node_id': node_id})
        
        # Após 5 segundos, marcar como offline (simulando reinício)
        def marcar_offline():
            time.sleep(5)
            if node_id in sistema['nodes']:
                sistema['nodes'][node_id]['status'] = 'offline'
                salvar_nodes()
                socketio.emit('node_status_changed', {
                    'node_id': node_id,
                    'status': 'offline',
                    'node': sistema['nodes'][node_id]
                }, room='dashboard')
        
        # Executar em thread separada
        restart_thread = threading.Thread(target=marcar_offline, daemon=True)
        restart_thread.start()
        
        return jsonify({'sucesso': f'Comando de reinicialização enviado para {node_id}'})
        
    except Exception as e:
        return jsonify({'erro': f'Erro ao reiniciar nó: {str(e)}'}), 500

@app.route('/api/nodes/<node_id>/toggle_status', methods=['POST'])
def api_toggle_node_status(node_id):
    """API para alternar status do nó manualmente."""
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    node = sistema['nodes'][node_id]
    current_status = node.get('status', 'offline')
    
    # Alternar status
    new_status = 'offline' if current_status == 'online' else 'online'
    node['status'] = new_status
    node['last_seen'] = datetime.now().isoformat()
    salvar_nodes()
    
    # Notificar mudança
    socketio.emit('node_status_changed', {
        'node_id': node_id,
        'status': new_status,
        'node': node
    }, room='dashboard')
    
    return jsonify({
        'sucesso': f'Status do nó alterado para {new_status}',
        'node': node
    })

# =================== WEBSOCKET EVENTS ===================

@socketio.on('connect')
def handle_connect():
    try:
        print(f'✅ Cliente conectado: {request.sid}')
        
        # Enviar dados iniciais para diferentes tipos de clientes
        emit('connection_confirmed', {
            'status': 'connected',
            'timestamp': datetime.now().isoformat(),
            'server_time': time.time()
        })
        
    except Exception as e:
        print(f'❌ Erro na conexão: {e}')

@socketio.on('disconnect')
def handle_disconnect():
    try:
        print(f'❌ Cliente desconectado: {request.sid}')
        
        # Verificar se era um nó sensor
        for node_id, node_data in sistema['nodes'].items():
            if node_data.get('session_id') == request.sid:
                node_data['status'] = 'offline'
                node_data['last_seen'] = datetime.now().isoformat()
                salvar_nodes()  # Salvar mudança
                
                # Notificar dashboard
                socketio.emit('node_status_changed', {
                    'node_id': node_id,
                    'status': 'offline',
                    'node': node_data
                }, room='dashboard')
                print(f'📡 Nó {node_id} marcado como offline')
                break
                
    except Exception as e:
        print(f'❌ Erro na desconexão: {e}')

# Adicionar evento de ping personalizado
@socketio.on('ping')
def handle_ping():
    """Responder ping dos clientes."""
    emit('pong', {'timestamp': datetime.now().isoformat()})

@socketio.on('keep_alive')
def handle_keep_alive(data):
    """Manter conexão ativa."""
    node_id = data.get('node_id')
    if node_id and node_id in sistema['nodes']:
        sistema['nodes'][node_id]['last_seen'] = datetime.now().isoformat()
        sistema['nodes'][node_id]['status'] = 'online'
    
    emit('keep_alive_response', {'status': 'ok'})

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
    
    # Notificar dashboard in tempo real
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
    
    # CONFIGURAÇÕES OTIMIZADAS PARA O SERVIDOR
    socketio.run(app, 
                debug=False, 
                host='0.0.0.0', 
                port=5000,
                use_reloader=False,  # Evitar recarregamento duplo
                log_output=False)    # Reduzir logs verbosos