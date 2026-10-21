"""Nucleo de reconhecimento facial: deteccao (YuNet) + assinatura (SFace).

A ideia em uma frase: cada rosto vira um vetor de 128 numeros (a "assinatura"
do SFace); a assinatura media das fotos boas e o `centroide`. Para saber se um
frame e da mesma pessoa, comparamos a assinatura do frame com o centroide por
similaridade de cosseno (0 = nada a ver, 1 = identico).

Quem decide se o score e suficiente e quem chama (`verificar` nao aplica
limiar). O `Estabilizador` evita piscar: exige alguns frames seguidos.
"""

from pathlib import Path

import cv2
import numpy as np

LIMIAR_PADRAO = 0.50
LIMIAR_CALIBRACAO = 0.60
FRAMES_CONFIRMACAO = 3

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png", ".webp")
LARGURA_MAXIMA = 1280
MAX_ITERACOES_SELECAO = 5
MINIMO_AMOSTRAS = 2


def _normalizar(vetor):
    norma = float(np.linalg.norm(vetor))
    if norma == 0.0:
        raise ValueError("vetor nulo nao pode ser normalizado")
    return np.asarray(vetor, dtype=np.float32) / norma


def centroide(vetores):
    """Media dos vetores, normalizada L2 (resultado tem norma 1)."""
    if len(vetores) == 0:
        raise ValueError("centroide precisa de pelo menos um vetor")
    return _normalizar(np.mean(np.asarray(vetores, dtype=np.float32), axis=0))


def _similaridades(vetores, referencia):
    """Cosseno de cada vetor contra `referencia` (que ja e unitario)."""
    normas = np.linalg.norm(vetores, axis=1)
    normas = np.where(normas == 0.0, 1.0, normas)
    return (vetores @ referencia) / normas


def selecionar_amostras(vetores, limiar=LIMIAR_CALIBRACAO):
    """Escolhe as amostras coerentes com a maioria.

    Iterativo: calcula o centroide com todas, descarta as que ficam abaixo de
    `limiar` de similaridade, recalcula e repete ate estabilizar (ou 5 voltas).

    Nunca devolve menos de 2 indices: se o limiar derrubar tudo, sobram as duas
    amostras mais parecidas com o centroide. Uma amostra so nao permite
    comparacao, entao lista com menos de 2 levanta `ValueError`.

    Devolve `(indices_aceitos, centroide_final)`; indices em ordem crescente.
    """
    matriz = np.asarray(vetores, dtype=np.float32)
    if matriz.ndim != 2 or len(matriz) < MINIMO_AMOSTRAS:
        raise ValueError(
            f"selecionar_amostras precisa de pelo menos {MINIMO_AMOSTRAS} vetores"
        )

    indices = list(range(len(matriz)))

    for _ in range(MAX_ITERACOES_SELECAO):
        referencia = centroide(matriz[indices])
        similaridades = _similaridades(matriz[indices], referencia)
        aceitos = [
            indice
            for indice, similaridade in zip(indices, similaridades)
            if similaridade >= limiar
        ]

        if len(aceitos) < MINIMO_AMOSTRAS:
            # Rede de seguranca: mantem os dois mais parecidos com o centroide.
            ordem = np.argsort(similaridades)[::-1][:MINIMO_AMOSTRAS]
            aceitos = [indices[posicao] for posicao in ordem]

        aceitos.sort()
        if aceitos == indices:
            break
        indices = aceitos

    return indices, centroide(matriz[indices])


class Estabilizador:
    """Transforma leituras ruidosas de score em um estado booleano estavel.

    O estado so muda depois de `frames` leituras seguidas do lado oposto, para
    um frame ruim isolado nao apagar o reconhecimento (e vice-versa).
    `score=None` (nenhum rosto) conta como "nao reconhecido".
    """

    def __init__(self, limiar=LIMIAR_PADRAO, frames=FRAMES_CONFIRMACAO):
        if frames < 1:
            raise ValueError("frames precisa ser >= 1")
        self.limiar = limiar
        self.frames = frames
        self.estado = False
        self._leituras_opostas = 0

    def atualizar(self, score):
        reconhecido = score is not None and score >= self.limiar

        if reconhecido == self.estado:
            self._leituras_opostas = 0
            return self.estado

        self._leituras_opostas += 1
        if self._leituras_opostas >= self.frames:
            self.estado = reconhecido
            self._leituras_opostas = 0
        return self.estado


def _redimensionar(imagem, largura_maxima=LARGURA_MAXIMA):
    """Limita a MAIOR dimensao a `largura_maxima` (nunca amplia).

    Duas escolhas medidas neste projeto (OpenCV 5):

    - Limitar a maior dimensao, nao a largura: em foto retrato com 1280 de
      largura (altura ~1707) o YuNet recorta o rosto diferente e a similaridade
      entre duas fotos da mesma pessoa cai de ~0.89 para ~0.61.
    - Interpolacao padrao (bilinear), nao INTER_AREA: o INTER_AREA suaviza a
      imagem e a similaridade da foto ruim (915542) sobe de 0.52 para 0.69,
      acima do LIMIAR_CALIBRACAO, e ela deixa de ser descartada.
    """
    altura, largura = imagem.shape[:2]
    maior = max(altura, largura)
    if maior <= largura_maxima:
        return imagem
    escala = largura_maxima / maior
    novo_tamanho = (max(1, round(largura * escala)), max(1, round(altura * escala)))
    return cv2.resize(imagem, novo_tamanho)


class ReconhecedorRosto:
    """Detecta o maior rosto do frame e compara a assinatura com o centroide."""

    def __init__(self, caminho_yunet, caminho_sface, limiar=LIMIAR_PADRAO):
        self.limiar = limiar
        self.centroide = None
        self.n_amostras = 0
        self.amostras = []
        self._detector = cv2.FaceDetectorYN.create(
            str(caminho_yunet), "", (320, 320), 0.6, 0.3, 5000
        )
        self._extrator = cv2.FaceRecognizerSF.create(str(caminho_sface), "")

    def _detectar_maior(self, imagem):
        """Devolve a linha de 15 floats do YuNet, ou None se nao houver rosto."""
        altura, largura = imagem.shape[:2]
        self._detector.setInputSize((largura, altura))
        _, rostos = self._detector.detect(imagem)
        if rostos is None or len(rostos) == 0:
            return None
        # A linha e (x, y, w, h, 5 pontos, score); a area decide o "maior".
        return max(rostos, key=lambda rosto: float(rosto[2]) * float(rosto[3]))

    def _assinatura(self, imagem, rosto):
        alinhado = self._extrator.alignCrop(imagem, rosto)
        vetor = self._extrator.feature(alinhado)
        return _normalizar(np.asarray(vetor, dtype=np.float32).flatten())

    def calibrar(self, pasta):
        """Aprende o rosto a partir das fotos da pasta; devolve quantas aceitou."""
        pasta = Path(pasta)
        caminhos = sorted(
            caminho
            for caminho in pasta.iterdir()
            if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM
        )

        vetores = []
        nomes = []
        for caminho in caminhos:
            imagem = cv2.imread(str(caminho))
            if imagem is None:
                continue
            imagem = _redimensionar(imagem)
            rosto = self._detectar_maior(imagem)
            if rosto is None:
                continue
            vetores.append(self._assinatura(imagem, rosto))
            nomes.append(caminho.name)

        if not vetores:
            raise ValueError(f"nenhum rosto encontrado em {pasta}")

        indices, centro = selecionar_amostras(vetores)
        self.centroide = centro
        self.n_amostras = len(indices)
        self.amostras = [nomes[indice] for indice in indices]
        return self.n_amostras

    def salvar(self, caminho):
        if self.centroide is None:
            raise ValueError("nada para salvar: calibre o reconhecedor antes")
        np.savez(
            str(caminho),
            centroide=self.centroide,
            limiar=float(self.limiar),
            n_amostras=int(self.n_amostras),
        )

    def carregar(self, caminho):
        with np.load(str(caminho)) as dados:
            self.centroide = np.asarray(dados["centroide"], dtype=np.float32)
            self.limiar = float(dados["limiar"])
            self.n_amostras = int(dados["n_amostras"])
        self.amostras = []

    def verificar(self, frame):
        """Devolve `(score, (x, y, w, h))` do maior rosto, ou None.

        O score e a similaridade de cosseno com o centroide (0 a 1). Nao aplica
        limiar: a decisao fica com quem chama.

        O frame e reduzido do mesmo jeito que na calibracao (para as features
        serem comparaveis) e a caixa volta para as coordenadas originais.
        """
        if self.centroide is None:
            raise RuntimeError("reconhecedor sem calibracao: chame calibrar() ou carregar()")

        imagem = _redimensionar(frame)
        rosto = self._detectar_maior(imagem)
        if rosto is None:
            return None

        score = float(np.dot(self.centroide, self._assinatura(imagem, rosto)))

        altura_original, largura_original = frame.shape[:2]
        escala_x = largura_original / imagem.shape[1]
        escala_y = altura_original / imagem.shape[0]
        x = int(round(float(rosto[0]) * escala_x))
        y = int(round(float(rosto[1]) * escala_y))
        largura = int(round(float(rosto[2]) * escala_x))
        altura = int(round(float(rosto[3]) * escala_y))

        # Arredondamento pode estourar 1 pixel para fora do frame.
        x = min(max(x, 0), largura_original)
        y = min(max(y, 0), altura_original)
        largura = min(largura, largura_original - x)
        altura = min(altura, altura_original - y)
        return score, (x, y, largura, altura)
