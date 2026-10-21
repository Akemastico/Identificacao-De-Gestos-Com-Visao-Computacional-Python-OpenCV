"""Mede a margem do reconhecimento facial sem webcam.

O que faz: calibra com as fotos legitimas (`--fotos`), pontua cada foto em
regime "deixe um de fora" (contra o centroide das OUTRAS, nao o dela mesma) e
pontua fotos de outras pessoas (`--negativos`). A diferenca entre o pior
legitimo e o melhor impostor e a **margem**: quanto maior, mais folga o limiar
tem para acertar.

Perto do fim imprime o limiar sugerido: o ponto medio entre os dois extremos.

Sai com codigo 1 se a margem for menor que `--min-margem` ou se faltar algum
lado da comparacao (legitimo ou impostor); 0 se estiver tudo bem. Esse codigo e
o teste de regressao do limiar: rode depois de mexer em `rosto.py`.

Uso, de dentro do projeto:

    .venv/bin/python avaliar_rosto.py
    .venv/bin/python avaliar_rosto.py --fotos fotos/bini --min-margem 0.30
"""

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

from rosto import ReconhecedorRosto, centroide

RAIZ = Path(__file__).resolve().parent
MODELO_YUNET = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"
MODELO_SFACE = RAIZ / "modelos" / "face_recognition_sface_2021dec.onnx"

# Foto de impostor -> quantos rostos usar. `grupo.jpg` tem varios rostos
# pequenos; pegar so o maior desperdicaria metade do teste.
IMPOSTORES = {
    "nadil.jpg": 1,
    "hiago.jpeg": 1,
    "pedro.jpg": 1,
    "grupo.jpg": 5,
}


def argumentos():
    parser = argparse.ArgumentParser(
        description="Mede a margem entre legitimos e impostores do reconhecimento facial."
    )
    parser.add_argument(
        "--fotos",
        type=Path,
        default=RAIZ / "fotos" / "bini",
        help="pasta com as fotos da pessoa (padrao: fotos/bini)",
    )
    parser.add_argument(
        "--negativos",
        type=Path,
        default=Path("~/Downloads").expanduser(),
        help="pasta com as fotos de outras pessoas (padrao: ~/Downloads)",
    )
    parser.add_argument(
        "--min-margem",
        type=float,
        default=0.20,
        help="margem minima aceitavel; abaixo disso sai com codigo 1 (padrao: 0.20)",
    )
    return parser.parse_args()


def vetores_legitimos(rec, pasta):
    """Vetor de cada amostra aceita na calibracao, na ordem de `rec.amostras`."""
    vetores = {}
    for nome in rec.amostras:
        imagem = cv2.imread(str(Path(pasta) / nome))
        if imagem is None:
            continue
        achados = rec.vetores_de(imagem, limite=1)
        if not achados:
            continue
        vetores[nome] = achados[0][0]
    return vetores


def scores_legitimos(vetores):
    """Leave-one-out: cada foto contra o centroide das outras."""
    nomes = list(vetores)
    scores = []
    for nome in nomes:
        outros = [vetores[outro] for outro in nomes if outro != nome]
        if not outros:
            continue
        scores.append((nome, float(np.dot(centroide(outros), vetores[nome]))))
    return scores


def scores_impostores(rec, pasta):
    """Cada rosto das fotos de impostor contra o centroide calibrado."""
    scores = []
    for nome, quantos in IMPOSTORES.items():
        caminho = Path(pasta) / nome
        if not caminho.is_file():
            print(f"ignorado (nao existe): {caminho}", file=sys.stderr)
            continue
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"ignorado (nao abriu): {caminho}", file=sys.stderr)
            continue
        achados = rec.vetores_de(imagem, limite=quantos)
        if not achados:
            print(f"ignorado (sem rosto): {caminho}", file=sys.stderr)
            continue
        for posicao, (vetor, _) in enumerate(achados, start=1):
            rotulo = nome if quantos == 1 else f"{nome} #{posicao}"
            scores.append((rotulo, float(np.dot(rec.centroide, vetor))))
    return scores


def main():
    opcoes = argumentos()

    if not opcoes.fotos.is_dir():
        print(f"pasta de fotos nao encontrada: {opcoes.fotos}", file=sys.stderr)
        return 1

    rec = ReconhecedorRosto(MODELO_YUNET, MODELO_SFACE)
    aceitas = rec.calibrar(opcoes.fotos)
    print(f"Fotos: {opcoes.fotos} ({aceitas} amostras aceitas)")

    legitimos = scores_legitimos(vetores_legitimos(rec, opcoes.fotos))
    impostores = scores_impostores(rec, opcoes.negativos)

    print("\nLegitimos (leave-one-out):")
    for nome, score in legitimos:
        print(f"  {nome:<28} {score:.3f}")

    print("\nImpostores:")
    for rotulo, score in impostores:
        print(f"  {rotulo:<28} {score:.3f}")

    if not legitimos or not impostores:
        faltando = []
        if not legitimos:
            faltando.append("legitimo")
        if not impostores:
            faltando.append("impostor")
        print(f"\nFALHOU: sem score de {' e '.join(faltando)} para comparar")
        return 1

    menor_legitimo = min(score for _, score in legitimos)
    maior_impostor = max(score for _, score in impostores)
    margem = menor_legitimo - maior_impostor
    limiar_sugerido = (menor_legitimo + maior_impostor) / 2

    print(f"\nmenor legitimo : {menor_legitimo:.3f}")
    print(f"maior impostor : {maior_impostor:.3f}")
    print(f"margem         : {margem:.3f}")
    print(f"limiar sugerido: {limiar_sugerido:.3f}")

    if margem < opcoes.min_margem:
        print(f"\nFALHOU: margem {margem:.3f} < minimo {opcoes.min_margem:.3f}")
        return 1

    print(f"\nOK: margem {margem:.3f} >= minimo {opcoes.min_margem:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
