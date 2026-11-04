"""
Script para geração e calibragem das imagens do conjunto de teste cego — Demoday 5
Projeto: Sanidade-Vegetal (SugarVision)
Responsável: Cesar (Lead Técnico & Visão Computacional)
"""

import json
from pathlib import Path
import cv2
import numpy as np

def generate_samples():
    out_dir = Path("data/demoday5_blind_samples")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    # 1. Amostra 01: Ferrugem Severa de Campo (Severe Rust)
    # Fundo de folha oxidada/estressada (R~125, G~110, B~45) + pústulas densas de ferrugem (R~195, G~80, B~25)
    img1 = np.zeros((640, 640, 3), dtype=np.uint8)
    base_color = np.array([125, 110, 45], dtype=np.float32)
    noise = np.random.normal(0, 10, (640, 640, 3)).astype(np.float32)
    img1 = np.clip(base_color + noise, 0, 255).astype(np.uint8)
    
    # Adiciona nervuras
    cv2.line(img1, (320, 0), (320, 640), (105, 95, 35), 3)
    for y in range(50, 640, 60):
        cv2.line(img1, (320, y), (120, y - 40), (105, 95, 35), 2)
        cv2.line(img1, (320, y), (520, y - 40), (105, 95, 35), 2)

    # 350 pústulas de ferrugem densas e elevadas
    for _ in range(350):
        cx = np.random.randint(40, 600)
        cy = np.random.randint(40, 600)
        r = np.random.randint(4, 12)
        # Tom alaranjado/ferruginoso avermelhado
        cv2.circle(img1, (cx, cy), r, (195, 80, 25), -1)
        # Centro do uredínio mais escuro/marrom
        cv2.circle(img1, (cx, cy), max(2, r // 2), (135, 45, 15), -1)

    cv2.imwrite(str(out_dir / "amostra_01_ferrugem_severa_campo.jpg"), cv2.cvtColor(img1, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 2. Amostra 02: Folha Sadia Controle (Healthy)
    # Folha verde vibrante e homogênea (R~45, G~145, B~40)
    img2 = np.zeros((640, 640, 3), dtype=np.uint8)
    base_green = np.array([45, 145, 40], dtype=np.float32)
    noise_g = np.random.normal(0, 6, (640, 640, 3)).astype(np.float32)
    img2 = np.clip(base_green + noise_g, 0, 255).astype(np.uint8)
    # Nervura suave
    cv2.line(img2, (320, 0), (320, 640), (40, 160, 35), 4)
    for y in range(40, 640, 50):
        cv2.line(img2, (320, y), (100, y - 35), (42, 152, 38), 2)
        cv2.line(img2, (320, y), (540, y - 35), (42, 152, 38), 2)
    cv2.imwrite(str(out_dir / "amostra_02_folha_sadia_controle.jpg"), cv2.cvtColor(img2, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3. Amostra 03: Ferrugem Incipiente (Early Rust)
    # Folha predominantemente verde com 30 micropústulas pontuais isoladas
    img3 = img2.copy()
    for _ in range(35):
        cx = np.random.randint(60, 580)
        cy = np.random.randint(60, 580)
        r = np.random.randint(3, 7)
        cv2.circle(img3, (cx, cy), r, (185, 90, 25), -1)
    cv2.imwrite(str(out_dir / "amostra_03_ferrugem_incipiente_desafio.jpg"), cv2.cvtColor(img3, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 4. Amostra 04: Mancha Foliar / Clorose (Red Rot / Leaf Scald)
    # Folha com lesão longitudinal necrótica marrom-avermelhada
    img4 = img2.copy()
    cv2.ellipse(img4, (320, 300), (90, 240), 15, 0, 360, (165, 55, 30), -1)
    cv2.ellipse(img4, (320, 300), (60, 200), 15, 0, 360, (110, 35, 20), -1)
    cv2.imwrite(str(out_dir / "amostra_04_mancha_foliar_clorose.jpg"), cv2.cvtColor(img4, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 5. Amostra 05: Necrose Senescente em Sombra (Borderline case)
    # Iluminação baixa (subexposta) com ápice seco senescente
    img5 = (img2 * 0.45).astype(np.uint8)
    # Região necrótica senescente escura
    pts = np.array([[0, 0], [280, 0], [180, 240], [0, 180]], np.int32)
    cv2.fillPoly(img5, [pts], (75, 50, 30))
    cv2.imwrite(str(out_dir / "amostra_05_necrose_senescente_sombra.jpg"), cv2.cvtColor(img5, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 6. Amostra 06: Especularidade / Sol Direto (Glare)
    # Alta reflexão do flash/sol direto saturando parte da folha
    img6 = img2.copy()
    cv2.circle(img6, (320, 320), 140, (245, 255, 240), -1)
    img6 = cv2.GaussianBlur(img6, (51, 51), 0)
    cv2.imwrite(str(out_dir / "amostra_06_especularidade_sol_direto.jpg"), cv2.cvtColor(img6, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 92])

    print("Imagens curadas do teste cego sintetizadas com sucesso!")

if __name__ == "__main__":
    generate_samples()

