# Documentacao do Projecto Pratico
# Sistema de Deteccao de Fadiga e Sonolencia em Condutores de Veiculos
**Trabalho de Fim de Curso - Universidade Oscar Ribas**
**Autor:** Romao Henrique Albano Quilamiquiza
**Orientador:** MSc. Helder Chissingui

---

## 1. Visao Geral do Sistema

O sistema implementa um **Driver Monitoring System (DMS)** em tempo real que detecta fadiga e sonolencia do condutor atraves de visao computacional. Utiliza a webcam do computador para capturar o rosto do condutor e analisar indicadores biometricos visuais, classificando o estado em tres niveis:

| Estado | Significado | Indicador Visual |
|--------|-------------|------------------|
| **NORMAL** | Condutor atento e responsivo | Circulo verde |
| **FADIGA MODERADA** | Sinais iniciais de cansaco | Borda amarela + aviso |
| **SONOLENCIA CRITICA** | Risco iminente de adormecer | Borda vermelha piscante + alarme sonoro |

---

## 2. Arquitectura do Sistema

```
CAMERA (1280x720, 30 FPS)
    |
    v
[detector.py] - MediaPipe Face Mesh (468 landmarks)
    |
    |-- Calculo EAR (Eye Aspect Ratio)
    |-- Calculo MAR (Mouth Aspect Ratio)
    |-- Calculo PERCLOS (Percentage of Eye Closure)
    |-- Calculo HPE (Head Pose Estimation)
    |
    v
[classificador.py] - Classificacao do Estado
    |-- Rede Neural MLP (modelo treinado)
    |-- Regras baseadas em limiares (fallback)
    |
    v
[alerta.py] - Sistema de Alertas
    |-- Alertas sonoros (beeps + audio MP3)
    |-- Alertas visuais (bordas coloridas)
    |
    v
[ui.py] - Interface Visual
    |-- Paineis de metricas em tempo real
    |-- Landmarks faciais coloridos
    |-- Barras de progresso
    |
    v
ECRÃ (Janela OpenCV)
```

---

## 3. Estrutura de Ficheiros

```
project_pratico/
|-- main.py                    # Ponto de entrada principal
|-- config.py                  # Configuracoes e limiares
|-- requirements.txt           # Dependencias Python
|-- alert.mp3                  # Som de alerta
|
|-- src/
|   |-- __init__.py
|   |-- detector.py            # Extracao de atributos (EAR, MAR, PERCLOS, HPE)
|   |-- classificador.py       # Classificacao por rede neural + regras
|   |-- alerta.py              # Sistema de alertas sonoros e visuais
|   |-- ui.py                  # Interface visual (overlay sobre o video)
|
|-- modelo/
|   |-- __init__.py
|   |-- treinar_modelo.py      # Script de treinamento do modelo MLP
|   |-- modelo_fadiga.pkl      # Modelo treinado (serializado)
|   |-- scaler.pkl             # Normalizador de features
|
|-- venv/                      # Ambiente virtual Python
```

---

## 4. Indicadores Biometricos Implementados

### 4.1 EAR - Eye Aspect Ratio (Soukupova & Cech, 2016)

**Formula:**
```
EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
```

Mede a abertura dos olhos usando 6 landmarks por olho. Quando o olho fecha, o EAR diminui.

- **Valor normal (olhos abertos):** 0.18 - 0.28
- **Limiar de olho fechado:** < 0.16
- **Landmarks olho direito:** 33, 160, 158, 133, 153, 144
- **Landmarks olho esquerdo:** 362, 385, 387, 263, 373, 380

### 4.2 MAR - Mouth Aspect Ratio (Abbas et al., 2022)

**Formula:**
```
MAR = (||sup1-inf1|| + ||sup2-inf2|| + ||sup3-inf3||) / (3 * ||esq-dir||)
```

Detecta bocejos medindo a abertura vertical da boca em relacao a horizontal.

- **Valor normal:** 0.0 - 0.20
- **Limiar de bocejo:** > 0.55
- **Duracao minima para contar como bocejo:** 0.7 segundos
- **Bocejos para alerta de fadiga:** >= 3

### 4.3 PERCLOS - Percentage of Eye Closure (Wierwille et al., 1994)

**Formula:**
```
PERCLOS = (frames_olhos_fechados / total_frames) * 100
```

Mede o percentual de tempo com as palpebras fechadas numa janela temporal de 30 segundos.

- **Normal:** < 15%
- **Fadiga moderada:** 15% - 30%
- **Sonolencia critica:** > 30%

### 4.4 HPE - Head Pose Estimation (Murphy-Chutorian & Trivedi, 2009)

Estima a orientacao 3D da cabeca usando `cv2.solvePnP()` com 6 pontos faciais.

- **Pitch** (inclinacao frente/tras): Limiar > 20 graus
- **Yaw** (rotacao esquerda/direita): Limiar > 30 graus
- **Roll** (inclinacao lateral): Limiar > 20 graus
- **Tempo minimo para alerta:** 2.0 segundos

---

## 5. Logica de Classificacao

### 5.1 Classificacao Baseada em Regras (Detector)

O detector usa regras directas baseadas nos indicadores medidos:

**SONOLENCIA CRITICA** (pelo menos uma condicao):
- Olhos fechados >= 15 frames consecutivos (~0.5s a 30 FPS)
- PERCLOS >= 30%
- Cabeca inclinada por mais de 2 segundos

**FADIGA MODERADA** (pelo menos uma condicao):
- >= 3 bocejos detectados
- PERCLOS entre 15% e 30%
- Bocejando actualmente
- Cabeca inclinada

**NORMAL:**
- Nenhuma das condicoes acima verificada

### 5.2 Rede Neural MLP (Classificador)

Modelo treinado com scikit-learn MLPClassifier:

**Arquitectura:**
```
Input(5) -> Dense(64, ReLU) -> Dense(128, ReLU) -> Dense(64, ReLU) -> Dense(32, ReLU) -> Output(3)
```

**Features de entrada:**
1. EAR medio
2. MAR
3. PERCLOS (normalizado 0-1)
4. |Pitch| / 90 (normalizado)
5. |Roll| / 90 (normalizado)

**Metricas de desempenho:**
- Acuracia: 98.2%
- Precisao: 98.3%
- Sensibilidade: 98.2%
- F-Score: 98.2%

---

## 6. Sistema de Alertas

| Estado | Alerta Visual | Alerta Sonoro |
|--------|---------------|---------------|
| NORMAL | Nenhum | Nenhum |
| FADIGA MODERADA | Borda amarela + mensagem de atencao | Beep suave (800 Hz, 300 ms) |
| SONOLENCIA CRITICA | Borda vermelha piscante + mensagem de alerta | Alarme intenso (1500 Hz, 800 ms) + audio MP3 |

O sistema de alertas opera numa thread separada para nao bloquear o processamento de video.

---

## 7. Interface Visual

A interface sobrepoem-se ao video da camera em tempo real:

- **Painel superior:** Titulo do sistema + FPS actual
- **Painel esquerdo:** Metricas biometricas (EAR, MAR, PERCLOS, HPE) com barras de progresso
- **Painel direito:** Estado do condutor com indicador circular colorido e barra de confianca
- **Landmarks faciais:** Pontos coloridos por regiao (verde=olho direito, laranja=olho esquerdo, magenta=boca, ciano=pose)
- **Legenda:** Canto inferior esquerdo
- **Alertas em tela cheia:** Bordas e mensagens para estados criticos

---

## 8. Correcoes e Ajustes Realizados

### 8.1 Resolucao do Erro TensorFlow

**Problema:** TensorFlow 2.21.0 falhava ao inicializar (DLL initialization failed) quando importado pelo MediaPipe.

**Causa:** O projecto nao utiliza TensorFlow directamente. O MediaPipe tentava importa-lo opcionalmente e crashava porque a instalacao estava corrompida.

**Solucao:** Desinstalacao do TensorFlow (`pip uninstall tensorflow`), ja que nao e uma dependencia necessaria do projecto.

### 8.2 Incompatibilidade da Versao do MediaPipe

**Problema:** MediaPipe 0.10.33 removeu a API `mp.solutions` que o codigo utiliza.

**Solucao:** Downgrade para MediaPipe 0.10.14 (`pip install mediapipe==0.10.14`) que mantem a API `mp.solutions.face_mesh`.

### 8.3 Erro de Caminho Unicode no MediaPipe

**Problema:** O backend C++ do MediaPipe nao conseguia resolver caminhos com caracteres nao-ASCII (o `a` com til em "Romao" no caminho do utilizador), causando `FileNotFoundError`.

**Solucao:** Patch no ficheiro `venv/.../mediapipe/python/solution_base.py` para converter o caminho para o formato curto do Windows (8.3) usando `GetShortPathNameW` quando detecta caracteres nao-ASCII.

**Nota:** Este patch e local ao venv e precisa de ser reaplicado se o MediaPipe for reinstalado.

### 8.4 Eliminacao de Falsos Alertas na Inicializacao

**Problema:** O sistema disparava alertas de fadiga/sonolencia imediatamente ao iniciar, antes de ter dados estaveis.

**Causas identificadas:**
1. As metricas comecavam a zero (EAR=0 interpretado como olhos fechados)
2. Os primeiros frames do MediaPipe produzem valores instaveis
3. O historico de EAR era poluido com dados do periodo de warm-up
4. O PERCLOS era calculado com frames instaveis

**Solucao - Mecanismo de Warm-up:**
- Adicionado contador de frames e flag `warmup_completo` ao detector
- Os primeiros 60 frames (2 segundos) sao descartados
- Quando o warm-up termina, o historico de EAR e limpo completamente
- A classificacao de estado forca NORMAL durante o warm-up

### 8.5 Recalibracao do Modelo MLP

**Problema:** O modelo treinado com dados sinteticos usava EAR "alerta" = 0.32, mas o EAR real medido pelo MediaPipe com olhos abertos e ~0.20. O modelo interpretava o estado normal como fadiga/sonolencia.

**Solucao:**
- Recalibracao dos dados sinteticos de treino para reflectir valores reais do MediaPipe:
  - Estado alerta: EAR ~ N(0.23, 0.04) em vez de N(0.32, 0.04)
  - Estado fadiga: EAR ~ N(0.18, 0.04) em vez de N(0.26, 0.05)
  - Estado sonolencia: EAR ~ N(0.10, 0.04) em vez de N(0.15, 0.05)
- Ajuste do limiar EAR de 0.22 para 0.16
- Re-treinamento do modelo com os novos dados

### 8.6 Simplificacao da Classificacao

**Problema:** Dois sistemas de classificacao paralelos (modelo MLP + regras do detector) podiam divergir, causando comportamento inconsistente.

**Solucao:** O estado final do condutor e agora determinado directamente pelas regras do detector (`_classificar_estado`), que reflectem exactamente os indicadores medidos. O modelo MLP e usado apenas para fornecer o valor de confianca.

### 8.7 Renomeacao do Estado Normal

**Problema:** O estado normal chamava-se "ALERTA", o que causava confusao porque parecia um aviso.

**Solucao:** Renomeado de "ALERTA" para "NORMAL" no dicionario `NOMES_ESTADO` do `config.py`.

---

## 9. Controlos do Sistema

| Tecla | Accao |
|-------|-------|
| **ESC** | Encerrar o sistema |
| **S** | Silenciar alerta sonoro |
| **R** | Reiniciar contadores (bocejos, PERCLOS, frames) |

---

## 10. Dependencias

```
opencv-python >= 4.7.0      # Processamento de video e imagem
mediapipe == 0.10.14         # Deteccao de landmarks faciais (468 pontos)
numpy >= 1.24.0              # Computacao numerica
scikit-learn >= 1.2.0        # Modelo de rede neural MLP
pygame >= 2.5.0              # Reproducao de audio (alertas)
joblib                       # Serializacao do modelo treinado
```

---

## 11. Como Executar

```bash
# 1. Activar o ambiente virtual
venv\Scripts\activate

# 2. Instalar dependencias (se necessario)
pip install -r requirements.txt

# 3. Treinar o modelo (opcional, ja esta treinado)
python modelo/treinar_modelo.py

# 4. Executar o sistema
python main.py
```

---

## 12. Referencias Bibliograficas

- **Soukupova, T. & Cech, J.** (2016). Real-time eye blink detection using facial landmarks. *21st Computer Vision Winter Workshop.*
- **Abbas, Q. et al.** (2022). A comprehensive approach for driver drowsiness detection. *IEEE Access.*
- **Wierwille, W.W. et al.** (1994). Research on vehicle-based driver status/performance monitoring. *NHTSA Report.*
- **Murphy-Chutorian, E. & Trivedi, M.M.** (2009). Head pose estimation in computer vision: A survey. *IEEE TPAMI.*
- **LeCun, Y., Bengio, Y. & Hinton, G.** (2015). Deep learning. *Nature.*
