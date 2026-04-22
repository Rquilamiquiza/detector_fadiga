# Sistema de Deteccao de Fadiga e Sonolencia em Condutores

Sistema de monitorizacao de condutores (Driver Monitoring System - DMS) que detecta fadiga e sonolencia em tempo real atraves da webcam, utilizando visao computacional e aprendizado de maquina.

> Trabalho de Fim de Curso - Universidade Oscar Ribas
> **Autor:** Romao Henrique Albano Quilamiquiza
> **Orientador:** MSc. Helder Chissingui

---

## Funcionalidades

- Deteccao de 468 landmarks faciais com **MediaPipe Face Mesh**
- Calculo de indicadores biometricos em tempo real:
  - **EAR** (Eye Aspect Ratio) - abertura dos olhos
  - **MAR** (Mouth Aspect Ratio) - deteccao de bocejos
  - **PERCLOS** (Percentage of Eye Closure) - percentual de olhos fechados
  - **HPE** (Head Pose Estimation) - orientacao 3D da cabeca
- Classificacao em tres estados: **NORMAL**, **FADIGA MODERADA**, **SONOLENCIA CRITICA**
- Rede Neural MLP treinada (scikit-learn) com acuracia ~98%
- Alertas sonoros e visuais
- Interface visual com metricas sobrepostas ao video

---

## Estrutura do Projecto

```
project_pratico/
├── main.py                    # Ponto de entrada
├── config.py                  # Configuracoes e limiares
├── requirements.txt           # Dependencias
├── alert.mp3                  # Som de alerta
├── DOCUMENTACAO_PROJETO.md    # Documentacao tecnica completa
│
├── src/
│   ├── detector.py            # Extracao de atributos (EAR, MAR, PERCLOS, HPE)
│   ├── classificador.py       # Classificacao por rede neural + regras
│   ├── alerta.py              # Sistema de alertas sonoros e visuais
│   └── ui.py                  # Interface visual
│
└── modelo/
    ├── treinar_modelo.py      # Script de treinamento
    ├── modelo_fadiga.pkl      # Modelo MLP treinado
    └── scaler.pkl             # Normalizador de features
```

---

## Requisitos

- **Python 3.9 - 3.11** (recomendado 3.10)
- **Webcam** funcional
- **Sistema Operativo:** Windows, Linux ou macOS

---

## Instalacao

### 1. Clonar o repositorio

```bash
git clone https://github.com/<teu-utilizador>/<nome-do-repo>.git
cd <nome-do-repo>
```

### 2. Criar e activar o ambiente virtual

**Windows (PowerShell/CMD):**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux / macOS:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalar dependencias

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

> **Nota sobre MediaPipe:** se encontrares erro de API `mp.solutions`, instala a versao 0.10.14 que e compativel:
> ```bash
> pip install mediapipe==0.10.14
> ```

> **Nota sobre caminhos com acentos (Windows):** se o teu utilizador Windows tiver acentos no nome (ex: "Romão"), o MediaPipe pode falhar ao carregar modelos. Mais detalhes na seccao 8.3 do [DOCUMENTACAO_PROJETO.md](DOCUMENTACAO_PROJETO.md).

---

## Como Executar

Com o ambiente virtual activado:

```bash
python main.py
```

O sistema abrira uma janela com a imagem da webcam e o overlay de metricas. Aguarda **~2 segundos** para o warm-up inicial terminar antes de comecar a classificacao.

### Controlos

| Tecla | Accao |
|-------|-------|
| `ESC` | Encerrar o sistema |
| `S`   | Silenciar o alerta sonoro |
| `R`   | Reiniciar contadores (bocejos, PERCLOS, frames) |

---

## Treinar o Modelo (opcional)

O modelo ja vem treinado em [modelo/modelo_fadiga.pkl](modelo/modelo_fadiga.pkl). Para re-treinar com novos dados:

```bash
python modelo/treinar_modelo.py
```

Isto gera ficheiros actualizados de `modelo_fadiga.pkl` e `scaler.pkl`.

---

## Estados Detectados

| Estado | Indicador Visual | Alerta Sonoro |
|--------|------------------|---------------|
| NORMAL | Circulo verde | Nenhum |
| FADIGA MODERADA | Borda amarela + aviso | Beep suave (800 Hz) |
| SONOLENCIA CRITICA | Borda vermelha piscante | Alarme (1500 Hz) + audio MP3 |

---

## Documentacao Tecnica

Para detalhes sobre a arquitectura, formulas dos indicadores (EAR, MAR, PERCLOS, HPE), logica de classificacao e historico de correcoes, consulta [DOCUMENTACAO_PROJETO.md](DOCUMENTACAO_PROJETO.md).

---

## Referencias

- Soukupova, T. & Cech, J. (2016). *Real-time eye blink detection using facial landmarks.*
- Abbas, Q. et al. (2022). *A comprehensive approach for driver drowsiness detection.* IEEE Access.
- Wierwille, W.W. et al. (1994). *Research on vehicle-based driver status/performance monitoring.* NHTSA.
- Murphy-Chutorian, E. & Trivedi, M.M. (2009). *Head pose estimation in computer vision: A survey.* IEEE TPAMI.

---

## Licenca

Projecto academico desenvolvido no ambito do Trabalho de Fim de Curso em Engenharia Informatica da Universidade Oscar Ribas.
