"""
Pacote src - Sistema de Deteccao de Fadiga e Sonolencia
========================================================
Modulos:
- detector: Extracao de metricas faciais (EAR, MAR, PERCLOS, HPE)
- classificador: Classificacao CNN dos estados do condutor
- alerta: Sistema de alertas sonoros e visuais
- ui: Interface visual profissional
"""

from .detector import DetectorFadiga
from .classificador import ClassificadorFadiga
from .alerta import SistemaAlerta
from .ui import InterfaceVisual
