// Funções JavaScript compartilhadas para o sistema de reconhecimento facial

// Função para exibir mensagens de sucesso ou erro
function exibirMensagem(elemento, tipo, mensagem) {
    const alertClass = tipo === 'sucesso' ? 'alert-success' : 'alert-danger';
    elemento.innerHTML = `<div class="alert ${alertClass}">${mensagem}</div>`;
}

// Função para limpar mensagens
function limparMensagem(elemento) {
    elemento.innerHTML = '';
}

// Função para desabilitar/habilitar botões
function alterarEstadoBotao(botaoId, desabilitado) {
    document.getElementById(botaoId).disabled = desabilitado;
}

// Função para parar stream de vídeo
function pararVideoStream(video) {
    if (video && video.srcObject) {
        video.srcObject.getTracks().forEach(track => {
            track.stop();
        });
        video.srcObject = null;
    }
}

// Função para validar se um nome é válido
function validarNome(nome) {
    return nome && nome.trim().length >= 2;
}

// Função para testar conectividade com o servidor
async function testarConectividade() {
    try {
        const response = await fetch('/api/pessoas', {
            method: 'GET',
            headers: {
                'Accept': 'application/json'
            }
        });
        
        if (response.ok) {
            console.log('✅ Servidor conectado');
            return true;
        } else {
            console.warn('⚠️ Servidor respondeu com erro:', response.status);
            return false;
        }
    } catch (error) {
        console.error('❌ Erro de conectividade:', error);
        return false;
    }
}

// Função melhorada para fazer requisições à API
async function fazerRequisicaoAPI(url, dados) {
    try {
        // Testar conectividade primeiro
        const conectado = await testarConectividade();
        if (!conectado) {
            throw new Error('Sem conexão com o servidor. Verifique se o Flask está rodando.');
        }
        
        const response = await fetch(url, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            },
            body: JSON.stringify(dados)
        });
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP ${response.status}: ${errorText}`);
        }
        
        const contentType = response.headers.get('content-type');
        if (!contentType || !contentType.includes('application/json')) {
            throw new Error('Resposta do servidor não é JSON válido');
        }
        
        return await response.json();
        
    } catch (error) {
        console.error('Erro na requisição:', error);
        throw error;
    }
}

// Função para redimensionar canvas mantendo proporção
function redimensionarCanvas(canvas, larguraMax = 640, alturaMax = 480) {
    const ctx = canvas.getContext('2d');
    const { width: larguraOriginal, height: alturaOriginal } = canvas;
    
    let novaLargura = larguraOriginal;
    let novaAltura = alturaOriginal;
    
    if (larguraOriginal > larguraMax) {
        novaLargura = larguraMax;
        novaAltura = (alturaOriginal * larguraMax) / larguraOriginal;
    }
    
    if (novaAltura > alturaMax) {
        novaAltura = alturaMax;
        novaLargura = (novaLargura * alturaMax) / novaAltura;
    }
    
    // Criar novo canvas com tamanho redimensionado
    const canvasRedimensionado = document.createElement('canvas');
    canvasRedimensionado.width = novaLargura;
    canvasRedimensionado.height = novaAltura;
    
    const ctxRedimensionado = canvasRedimensionado.getContext('2d');
    ctxRedimensionado.drawImage(canvas, 0, 0, novaLargura, novaAltura);
    
    return canvasRedimensionado;
}

console.log('Sistema de Reconhecimento Facial carregado!');