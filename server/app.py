# servidor-central/app.py
from flask import Flask, render_template, request, jsonify, Response
from flask_socketio import SocketIO, emit, join_room
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
import requests

# Configuração da aplicação
app = Flask(__name__)
app.config.update({
    'SECRET_KEY': 'facial-security-2024',
    'UPLOAD_FOLDER': 'uploads'
})

# SOLUÇÃO: Configuração WebSocket ultra-estável
socketio = SocketIO(app, 
                   cors_allowed_origins="*",
                   async_mode='threading',
                   ping_timeout=120,           # Aumentado
                   ping_interval=60,           # Aumentado  
                   logger=False,
                   engineio_logger=False,
                   transports=['polling'],     # APENAS polling
                   always_connect=False,       # Não forçar conexão
                   allow_upgrades=False)       # Não permitir upgrade para websocket

# Configurações de arquivos
ARQUIVOS = {
    'encodings': 'data/encodings.pickle',
    'nodes': 'data/nodes.json',
    'alerts': 'data/alerts.json'
}

# Criar diretórios necessários
for pasta in ['data', 'uploads', app.config['UPLOAD_FOLDER']]:
    os.makedirs(pasta, exist_ok=True)

# Estado global do sistema
sistema = {
    'nodes': {},
    'alerts': [],
    'stats': {'total_detections': 0, 'active_nodes': 0, 'known_faces': 0}
}

# =================== FUNÇÕES UTILITÁRIAS ===================

def carregar_json(arquivo, default=None):
    """Carrega arquivo JSON ou retorna valor padrão."""
    try:
        if os.path.exists(arquivo) and os.path.getsize(arquivo) > 0:
            with open(arquivo, 'r') as f:
                return json.load(f)
    except:
        pass
    return default if default is not None else {}

def salvar_json(arquivo, dados):
    """Salva dados em arquivo JSON."""
    try:
        with open(arquivo, 'w') as f:
            json.dump(dados, f, indent=2)
    except Exception as e:
        print(f"Erro ao salvar {arquivo}: {e}")

def carregar_encodings():
    """Carrega encodings de rostos conhecidos."""
    try:
        if os.path.exists(ARQUIVOS['encodings']) and os.path.getsize(ARQUIVOS['encodings']) > 0:
            with open(ARQUIVOS['encodings'], 'rb') as f:
                return pickle.load(f)
    except:
        pass
    return {"nomes": [], "encodings": []}

def salvar_encodings(dados):
    """Salva encodings de rostos."""
    with open(ARQUIVOS['encodings'], 'wb') as f:
        pickle.dump(dados, f)

def processar_imagem_base64(imagem_base64):
    """Converte imagem base64 para array numpy RGB com validação melhorada."""
    try:
        if ',' in imagem_base64:
            image_data = base64.b64decode(imagem_base64.split(',')[1])
        else:
            image_data = base64.b64decode(imagem_base64)
        
        image = Image.open(BytesIO(image_data))
        
        # CORREÇÃO 1: Redimensionar imagem se muito grande
        width, height = image.size
        if width > 800 or height > 600:
            # Manter proporção mas reduzir tamanho
            ratio = min(800/width, 600/height)
            new_size = (int(width * ratio), int(height * ratio))
            image = image.resize(new_size, Image.Resampling.LANCZOS)
            print(f"Imagem redimensionada de {width}x{height} para {new_size[0]}x{new_size[1]}")
        
        # CORREÇÃO 2: Converter para RGB se necessário
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        frame = np.array(image)
        return frame
        
    except Exception as e:
        print(f"Erro ao processar imagem: {e}")
        raise

def criar_alert(dados_alert):
    """Cria e salva um novo alerta."""
    alert = {
        'id': len(sistema['alerts']) + 1,
        'timestamp': datetime.now().isoformat(),
        **dados_alert
    }
    
    sistema['alerts'].insert(0, alert)
    sistema['alerts'] = sistema['alerts'][:1000]  # Manter apenas 1000
    
    salvar_json(ARQUIVOS['alerts'], sistema['alerts'][:100])
    return alert

def atualizar_stats():
    """Atualiza estatísticas do sistema."""
    dados = carregar_encodings()
    sistema['stats'].update({
        'known_faces': len(dados.get('nomes', [])),
        'active_nodes': len([n for n in sistema['nodes'].values() if n.get('status') == 'online'])
    })

# =================== ROTAS PRINCIPAIS ===================

@app.route('/')
def dashboard():
    atualizar_stats()
    return render_template('dashboard.html', 
                         stats=sistema['stats'],
                         nodes=sistema['nodes'],
                         recent_alerts=sistema['alerts'][:10])

@app.route('/nodes')
def nodes_page():
    return render_template('nodes.html', nodes=sistema['nodes'])

@app.route('/alerts')
def alerts_page():
    return render_template('alerts.html', alerts=sistema['alerts'][:50])

@app.route('/web')
def web_index():
    dados = carregar_encodings()
    return render_template('web/index.html', pessoas=dados.get('nomes', []))

@app.route('/web/cadastro')
def web_cadastro():
    return render_template('web/cadastro.html')

@app.route('/web/reconhecimento')
def web_reconhecimento():
    return render_template('web/reconhecimento.html')

# =================== APIs INTEGRADAS ===================

@app.route('/api/cadastrar', methods=['POST'])
def cadastrar_rosto():
    try:
        print("=== INICIANDO CADASTRO ===")
        
        if not request.is_json:
            print("Erro: Content-Type não é JSON")
            return jsonify({'erro': 'Content-Type deve ser application/json'}), 400
        
        data = request.get_json()
        nome = data.get('nome', '').strip()
        imagem_base64 = data.get('imagem', '')
        
        print(f"Nome recebido: {nome}")
        print(f"Imagem recebida: {'Sim' if imagem_base64 else 'Não'}")
        
        # Validações
        if not nome or len(nome) < 2:
            print("Erro: Nome inválido")
            return jsonify({'erro': 'Nome deve ter pelo menos 2 caracteres'}), 400
        if not imagem_base64:
            print("Erro: Imagem não fornecida")
            return jsonify({'erro': 'Imagem é obrigatória'}), 400
        
        print("Carregando encodings existentes...")
        dados_conhecidos = carregar_encodings()
        
        if nome.lower() in [n.lower() for n in dados_conhecidos.get("nomes", [])]:
            print(f"Erro: Nome {nome} já existe")
            return jsonify({'erro': f'Já existe pessoa com nome "{nome}"'}), 400
        
        print("Processando imagem...")
        # CORREÇÃO 3: Processar imagem com melhor tratamento
        rgb_frame = processar_imagem_base64(imagem_base64)
        print(f"Tamanho da imagem processada: {rgb_frame.shape}")
        
        print("Detectando rostos...")
        # CORREÇÃO 4: Usar parâmetros mais restritivos para detecção
        caixas_rosto = face_recognition.face_locations(
            rgb_frame, 
            model='hog',  # Mais rápido e menos falsos positivos
            number_of_times_to_upsample=0  # Não aumentar resolução
        )
        print(f"Rostos detectados: {len(caixas_rosto)}")
        
        # CORREÇÃO 5: Se detectar muitos rostos, tentar com CNN (mais preciso)
        if len(caixas_rosto) > 5:
            print("Muitos rostos detectados, tentando com modelo CNN...")
            try:
                caixas_rosto = face_recognition.face_locations(
                    rgb_frame, 
                    model='cnn',  # Mais preciso
                    number_of_times_to_upsample=0
                )
                print(f"Rostos detectados com CNN: {len(caixas_rosto)}")
            except Exception as e:
                print(f"Erro com CNN, usando resultado HOG: {e}")
        
        # CORREÇÃO 6: Se ainda muitos rostos, filtrar por tamanho
        if len(caixas_rosto) > 3:
            print("Filtrando rostos por tamanho...")
            # Filtrar rostos muito pequenos (prováveis falsos positivos)
            caixas_filtradas = []
            for (top, right, bottom, left) in caixas_rosto:
                width = right - left
                height = bottom - top
                area = width * height
                # Só considerar rostos com área mínima (evita ruído)
                if area > 2000:  # Área mínima em pixels
                    caixas_filtradas.append((top, right, bottom, left))
            
            caixas_rosto = caixas_filtradas
            print(f"Rostos após filtragem: {len(caixas_rosto)}")
        
        if len(caixas_rosto) == 0:
            print("Erro: Nenhum rosto detectado")
            return jsonify({'erro': 'Nenhum rosto detectado. Melhore a iluminação ou aproxime o rosto.'}), 400
        elif len(caixas_rosto) > 1:
            print(f"Erro: Múltiplos rostos detectados ({len(caixas_rosto)})")
            return jsonify({'erro': f'Detectados {len(caixas_rosto)} rostos. Certifique-se de que há apenas uma pessoa na imagem.'}), 400
        
        print("Gerando encodings...")
        encodings_rosto = face_recognition.face_encodings(rgb_frame, caixas_rosto)
        
        if len(encodings_rosto) == 1:
            print("Salvando dados...")
            dados_conhecidos["encodings"].append(encodings_rosto[0])
            dados_conhecidos["nomes"].append(nome)
            salvar_encodings(dados_conhecidos)
            
            print("Enviando notificações...")
            # Notificações (com tratamento de erro)
            try:
                socketio.emit('face_database_updated', {
                    'action': 'added', 'name': nome, 'total_faces': len(dados_conhecidos["nomes"])
                }, room='nodes')
                
                socketio.emit('system_update', {
                    'type': 'new_face_registered', 'data': {'name': nome}
                }, room='dashboard')
            except Exception as e:
                print(f"Erro nas notificações: {e}")
            
            print(f"=== CADASTRO CONCLUÍDO: {nome} ===")
            return jsonify({'sucesso': f'{nome} cadastrado com sucesso!'})
        
        else:
            print("Erro: Falha ao gerar encoding")
            return jsonify({'erro': 'Erro ao processar rosto detectado.'}), 400
            
    except Exception as e:
        print(f"=== ERRO GERAL NO CADASTRO: {e} ===")
        import traceback
        traceback.print_exc()
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500

@app.route('/api/reconhecer', methods=['POST'])
def reconhecer_rosto():
    try:
        data = request.get_json()
        imagem_base64 = data.get('imagem')
        
        if not imagem_base64:
            return jsonify({'erro': 'Imagem é obrigatória'}), 400
        
        dados_conhecidos = carregar_encodings()
        known_encodings = dados_conhecidos.get("encodings", [])
        known_names = dados_conhecidos.get("nomes", [])
        
        if not known_encodings:
            return jsonify({'rostos': []})
        
        rgb_frame = processar_imagem_base64(imagem_base64)
        face_locations = face_recognition.face_locations(rgb_frame, model='hog', number_of_times_to_upsample=0)
        
        if not face_locations:
            return jsonify({'rostos': []})
        
        face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)
        
        resultados = []
        for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
            matches = face_recognition.compare_faces(known_encodings, face_encoding, tolerance=0.6)
            name = "Desconhecido"
            
            if True in matches:
                name = known_names[matches.index(True)]
            
            resultados.append({
                'nome': name,
                'localizacao': {'top': int(top), 'right': int(right), 'bottom': int(bottom), 'left': int(left)}
            })
        
        return jsonify({'rostos': resultados})
        
    except Exception as e:
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500

@app.route('/api/detectar_rosto', methods=['POST'])
def detectar_rosto():
    try:
        data = request.get_json()
        imagem_base64 = data.get('imagem')
        
        if not imagem_base64:
            return jsonify({'erro': 'Imagem é obrigatória'}), 400
        
        rgb_frame = processar_imagem_base64(imagem_base64)
        
        # Usar mesmos parâmetros melhorados
        face_locations = face_recognition.face_locations(
            rgb_frame, 
            model='hog', 
            number_of_times_to_upsample=0
        )
        
        # Filtrar rostos pequenos
        if len(face_locations) > 3:
            face_locations_filtradas = []
            for (top, right, bottom, left) in face_locations:
                width = right - left
                height = bottom - top
                area = width * height
                if area > 2000:
                    face_locations_filtradas.append((top, right, bottom, left))
            face_locations = face_locations_filtradas
        
        resultados = [{'localizacao': {'top': int(top), 'right': int(right), 'bottom': int(bottom), 'left': int(left)}} for (top, right, bottom, left) in face_locations]
        
        return jsonify({'rostos': resultados})
        
    except Exception as e:
        return jsonify({'erro': f'Erro interno: {str(e)}'}), 500

@app.route('/api/pessoas')
def listar_pessoas():
    dados = carregar_encodings()
    return jsonify({'pessoas': dados.get('nomes', [])})

# =================== APIs DOS NÓS ===================

@app.route('/api/nodes', methods=['GET'])
def api_get_nodes():
    return jsonify({
        'nodes': sistema['nodes'],
        'total': len(sistema['nodes']),
        'online': len([n for n in sistema['nodes'].values() if n.get('status') == 'online']),
        'offline': len([n for n in sistema['nodes'].values() if n.get('status') == 'offline'])
    })

@app.route('/api/nodes', methods=['POST'])
def api_add_node():
    data = request.get_json()
    node_id = data.get('node_id')
    
    if not node_id:
        return jsonify({'erro': 'ID do nó é obrigatório'}), 400
    if node_id in sistema['nodes']:
        return jsonify({'erro': 'Nó já existe'}), 400
    
    novo_no = {
        'id': node_id,
        'location': data.get('location', ''),
        'type': data.get('type', 'camera'),
        'url': data.get('url'),
        'status': 'offline',
        'last_seen': datetime.now().isoformat(),
        'registered_at': datetime.now().isoformat(),
        'stats': {'total_detections': 0, 'last_detection': None}
    }
    
    sistema['nodes'][node_id] = novo_no
    salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
    
    socketio.emit('node_registered', {'node': novo_no}, room='dashboard')
    return jsonify({'sucesso': 'Nó adicionado com sucesso', 'node': novo_no})

@app.route('/api/nodes/<node_id>', methods=['PUT'])
def api_update_node(node_id):
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    data = request.get_json()
    node = sistema['nodes'][node_id]
    
    for campo in ['location', 'url', 'type']:
        if campo in data:
            node[campo] = data[campo]
    
    node['updated_at'] = datetime.now().isoformat()
    salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
    
    socketio.emit('node_updated', {'node': node}, room='dashboard')
    return jsonify({'sucesso': 'Nó atualizado com sucesso', 'node': node})

@app.route('/api/nodes/<node_id>', methods=['DELETE'])
def api_delete_node(node_id):
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    del sistema['nodes'][node_id]
    salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
    
    socketio.emit('node_removed', {'node_id': node_id}, room='dashboard')
    return jsonify({'sucesso': 'Nó removido com sucesso'})

@app.route('/api/nodes/<node_id>/stream')
def api_node_stream(node_id):
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    node = sistema['nodes'][node_id]
    camera_url = node.get('url')
    
    if not camera_url:
        return jsonify({'erro': 'URL da câmera não configurada'}), 400
    
    if 'video' not in camera_url and ':8080' in camera_url:
        camera_url = camera_url.rstrip('/') + '/video'
    
    try:
        requests.get(camera_url.replace('/video', '/'), timeout=3)
        
        node.update({'status': 'online', 'last_seen': datetime.now().isoformat()})
        salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
        
        socketio.emit('node_status_changed', {'node_id': node_id, 'status': 'online', 'node': node}, room='dashboard')
        
        return jsonify({
            'status': 'online',
            'stream_url': camera_url,
            'node_id': node_id,
            'proxy_url': f'/api/nodes/{node_id}/proxy_stream'
        })
        
    except Exception as e:
        node['status'] = 'offline'
        salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
        
        socketio.emit('node_status_changed', {'node_id': node_id, 'status': 'offline', 'node': node}, room='dashboard')
        return jsonify({'erro': f'Câmera não acessível: {str(e)}'}), 503

@app.route('/api/nodes/<node_id>/proxy_stream')
def api_proxy_stream(node_id):
    if node_id not in sistema['nodes']:
        return "Nó não encontrado", 404
    
    camera_url = sistema['nodes'][node_id].get('url')
    if not camera_url:
        return "URL não configurada", 400
    
    if 'video' not in camera_url and ':8080' in camera_url:
        camera_url = camera_url.rstrip('/') + '/video'
    
    try:
        print(f"🎥 Conectando à câmera: {camera_url}")
        resp = requests.get(camera_url, stream=True, timeout=10, 
                          headers={
                              'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
                              'Accept': 'multipart/x-mixed-replace,image/jpeg,*/*'
                          })
        
        if resp.status_code != 200:
            return f"Câmera retornou status {resp.status_code}", resp.status_code
        
        def generate():
            try:
                chunk_count = 0
                for chunk in resp.iter_content(chunk_size=8192):
                    if chunk:
                        chunk_count += 1
                        yield chunk
                        # Limitar chunks para evitar travamento
                        if chunk_count > 10000:  # Reset após muitos chunks
                            break
            except GeneratorExit:
                print(f"📹 Stream finalizado para {node_id}")
            except Exception as e:
                print(f"❌ Erro no stream {node_id}: {e}")
            finally:
                try:
                    resp.close()
                except:
                    pass
        
        return Response(generate(),
                       content_type=resp.headers.get('content-type', 'multipart/x-mixed-replace; boundary=frame'),
                       headers={
                           'Cache-Control': 'no-cache, no-store, must-revalidate',
                           'Pragma': 'no-cache',
                           'Expires': '0',
                           'Connection': 'close',
                           'X-Content-Type-Options': 'nosniff'
                       })
        
    except requests.exceptions.ConnectTimeout:
        return "⏱️ Timeout na conexão com câmera (10s)", 504
    except requests.exceptions.ConnectionError as e:
        return f"🔌 Câmera não acessível: {str(e)}", 503
    except Exception as e:
        print(f"❌ Erro no proxy stream {node_id}: {e}")
        return f"💥 Erro no stream: {str(e)}", 503

@app.route('/api/nodes/<node_id>/toggle_status', methods=['POST'])
def api_toggle_node_status(node_id):
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    node = sistema['nodes'][node_id]
    new_status = 'offline' if node.get('status') == 'online' else 'online'
    
    node.update({'status': new_status, 'last_seen': datetime.now().isoformat()})
    salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
    
    socketio.emit('node_status_changed', {'node_id': node_id, 'status': new_status, 'node': node}, room='dashboard')
    return jsonify({'sucesso': f'Status alterado para {new_status}', 'node': node})

@app.route('/api/nodes/<node_id>/restart', methods=['POST'])
def api_restart_node(node_id):
    if node_id not in sistema['nodes']:
        return jsonify({'erro': 'Nó não encontrado'}), 404
    
    node = sistema['nodes'][node_id]
    
    # Marcar como reiniciando
    node.update({
        'status': 'restarting', 
        'last_seen': datetime.now().isoformat()
    })
    salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
    
    # Notificar mudança de status
    socketio.emit('node_status_changed', {
        'node_id': node_id, 
        'status': 'restarting', 
        'node': node
    }, room='dashboard')
    
    # Simular reinicialização (em 5 segundos volta para offline)
    def reset_status():
        time.sleep(5)
        if node_id in sistema['nodes']:
            sistema['nodes'][node_id].update({
                'status': 'offline',
                'last_seen': datetime.now().isoformat()
            })
            salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
            socketio.emit('node_status_changed', {
                'node_id': node_id, 
                'status': 'offline', 
                'node': sistema['nodes'][node_id]
            }, room='dashboard')
    
    # Executar em thread separada
    import threading
    threading.Thread(target=reset_status, daemon=True).start()
    
    return jsonify({'sucesso': f'Nó {node_id} será reiniciado', 'node': node})

# =================== WEBSOCKET EVENTS ===================

@socketio.on('connect')
def handle_connect():
    try:
        print(f"🔗 Cliente conectado via {request.transport}: {request.sid}")
        emit('connection_confirmed', {
            'status': 'connected', 
            'timestamp': datetime.now().isoformat(),
            'sid': request.sid,
            'transport': 'polling'  # Sempre polling
        })
    except Exception as e:
        print(f"❌ Erro na conexão: {e}")

@socketio.on('disconnect')
def handle_disconnect():
    try:
        print(f"🔌 Cliente desconectado: {request.sid}")
        # Marcar nó como offline se desconectou
        for node_id, node_data in list(sistema['nodes'].items()):
            if node_data.get('session_id') == request.sid:
                node_data.update({
                    'status': 'offline', 
                    'last_seen': datetime.now().isoformat()
                })
                salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
                try:
                    socketio.emit('node_status_changed', {
                        'node_id': node_id, 
                        'status': 'offline', 
                        'node': node_data
                    }, room='dashboard')
                except:
                    pass  # Ignorar erros de emissão durante desconexão
                break
    except Exception as e:
        print(f"❌ Erro na desconexão: {e}")

# NOVO: Event para manter conexão viva
@socketio.on('ping')
def handle_ping():
    try:
        emit('pong', {'timestamp': datetime.now().isoformat()})
    except Exception as e:
        print(f"❌ Erro no ping: {e}")

@socketio.on('join_dashboard')
def handle_join_dashboard():
    try:
        join_room('dashboard')
        atualizar_stats()
        emit('dashboard_joined', {
            'stats': sistema['stats'],
            'nodes': sistema['nodes'],
            'alerts': sistema['alerts'][:10]
        })
        print(f"📊 Cliente {request.sid} entrou no dashboard")
    except Exception as e:
        print(f"❌ Erro ao entrar no dashboard: {e}")

@socketio.on_error_default
def default_error_handler(e):
    print(f"🚨 Erro WebSocket: {e}")
    # Não reenviar erro para cliente para evitar loops
    return False

# CORREÇÃO: Monitor de nós mais estável
def monitor_nodes():
    """Monitor de status dos nós com melhor tratamento de erros."""
    consecutive_errors = 0
    max_errors = 5
    
    while True:
        try:
            now = datetime.now()
            nodes_to_update = []
            
            for node_id, node_data in list(sistema['nodes'].items()):  # Usar list() para evitar RuntimeError
                if node_data.get('status') == 'online':
                    try:
                        last_seen_str = node_data.get('last_seen', now.isoformat())
                        last_seen = datetime.fromisoformat(last_seen_str.replace('Z', '+00:00'))
                        
                        # Timeout aumentado para 90 segundos
                        if (now - last_seen.replace(tzinfo=None)).seconds > 90:
                            node_data['status'] = 'offline'
                            nodes_to_update.append((node_id, node_data.copy()))
                            
                    except (ValueError, AttributeError) as e:
                        print(f"Erro ao processar timestamp do nó {node_id}: {e}")
                        # Reset timestamp em caso de erro
                        node_data['last_seen'] = now.isoformat()
            
            # Salvar mudanças
            if nodes_to_update:
                salvar_json(ARQUIVOS['nodes'], sistema['nodes'])
            
            # Notificar mudanças (com tratamento de erro)
            for node_id, node_data in nodes_to_update:
                try:
                    socketio.emit('node_status_changed', {
                        'node_id': node_id, 
                        'status': 'offline', 
                        'node': node_data
                    }, room='dashboard')
                except Exception as e:
                    print(f"Erro ao emitir mudança de status para {node_id}: {e}")
            
            # Reset contador de erros em caso de sucesso
            consecutive_errors = 0
            time.sleep(20)  # Intervalo de 20 segundos
            
        except Exception as e:
            consecutive_errors += 1
            print(f"Erro no monitor de nós ({consecutive_errors}/{max_errors}): {e}")
            
            # Se muitos erros consecutivos, parar temporariamente
            if consecutive_errors >= max_errors:
                print(f"⚠️ Monitor pausado por muitos erros consecutivos")
                time.sleep(60)  # Pausa de 1 minuto
                consecutive_errors = 0
            else:
                time.sleep(10)  # Pausa menor para tentar novamente

# CORREÇÃO: Inicialização com thread daemon
def init_system():
    """Inicialização do sistema com monitor de nós."""
    try:
        sistema['nodes'] = carregar_json(ARQUIVOS['nodes'])
        sistema['alerts'] = carregar_json(ARQUIVOS['alerts'], [])
        
        # Marcar todos os nós como offline na inicialização
        for node_data in sistema['nodes'].values():
            node_data['status'] = 'offline'
            node_data['last_seen'] = datetime.now().isoformat()
        
        atualizar_stats()
        
        # Iniciar monitor em thread daemon
        monitor_thread = threading.Thread(target=monitor_nodes, daemon=True)
        monitor_thread.start()
        
        print("🚀 Sistema distribuído inicializado com monitor!")
    except Exception as e:
        print(f"Erro na inicialização: {e}")

# CORREÇÃO: Inicialização mais robusta
if __name__ == '__main__':
    try:
        print("=== INICIANDO SISTEMA MELHORADO ===")
        init_system()
        
        print("🌐 Servidor: http://127.0.0.1:5000")
        print("📱 Web: http://127.0.0.1:5000/web")
        print("📊 Dashboard: http://127.0.0.1:5000/")
        print("📡 Nós: http://127.0.0.1:5000/nodes")
        print("⚠️  WebSocket: APENAS polling (máxima estabilidade)")
        
        # Configuração otimizada para estabilidade
        socketio.run(app, 
                    debug=False,
                    host='127.0.0.1',
                    port=5000, 
                    use_reloader=False,
                    log_output=False,
                    allow_unsafe_werkzeug=True)
                    
    except KeyboardInterrupt:
        print("\n=== SERVIDOR PARADO ===")
    except Exception as e:
        print(f"=== ERRO CRÍTICO: {e} ===")
        import traceback
        traceback.print_exc()

@app.route('/api/test')
def test_endpoint():
    """Endpoint de teste para verificar se o servidor está funcionando."""
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'nodes_count': len(sistema['nodes']),
        'alerts_count': len(sistema['alerts'])
    })

@app.route('/.well-known/appspecific/com.chrome.devtools.json')
def chrome_devtools():
    """Rota para evitar logs 404 do Chrome DevTools"""
    return jsonify({'error': 'Not implemented'}), 404