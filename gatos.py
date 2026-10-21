"""Programa de gestos com webcam e reacao ao rosto reconhecido.

Duas responsabilidades separadas:

- `gatos.py` (aqui): camera, gestos e janelas.
- `rosto.py`: deteccao + assinatura facial, sem saber de meme nenhum.

Regra de ouro deste arquivo: **importar nao pode abrir camera nem carregar
modelo**. Tudo que tem efeito colateral vive dentro de `main()`, senao o modulo
nao pode ser testado sem hardware.
"""

import subprocess
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from rosto import Estabilizador, ReconhecedorRosto

# Caminhos relativos ao arquivo, nunca ao diretorio de onde o script foi chamado.
RAIZ = Path(__file__).resolve().parent
MODELO_MAOS = RAIZ / "modelos" / "hand_landmarker.task"
MODELO_YUNET = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"
MODELO_SFACE = RAIZ / "modelos" / "face_recognition_sface_2021dec.onnx"
PASTA_FOTOS = RAIZ / "fotos" / "bini"
ARQUIVO_ROSTO = RAIZ / "rosto_bini.npz"

MEME_PUNHO = "gatos/punho.webp"
JANELA_CAMERA = "Camera"
JANELA_MEME = "Meme"

GESTOS = {
    frozenset(['dedao','dedoIndicador','dedoMeio','dedoAnelar','dedoMinimo']) : 'Aberta',
    frozenset([]) : 'Fechada',
    frozenset(['dedoMinimo','dedoIndicador','dedao']) : 'rock',
    frozenset(['dedao','dedoIndicador']) : 'L',
    frozenset(['dedao','dedoMinimo']) : 'deboa',
    frozenset(['dedoIndicador','dedoMeio']) : 'paz',
    frozenset(['dedoMeio']) : 'meio',
    frozenset(['dedao']) : 'dedao',
    frozenset(['dedoIndicador']) : 'indicador',
}


def criar_detector():
    """Cria o detector de maos (modelo carregado uma vez por execucao)."""
    modelAsset = python.BaseOptions(model_asset_path=str(MODELO_MAOS))
    modelConfig = vision.HandLandmarkerOptions(
        base_options=modelAsset,
        running_mode=vision.RunningMode.VIDEO,
        num_hands=2,
        min_hand_detection_confidence=0.6,
        min_hand_presence_confidence=0.6
    )
    return vision.HandLandmarker.create_from_options(modelConfig)


def verifyFingerIsUp(hand: list, point: int, holder: int, margin: float) -> bool:
    # Quanto Menor A Posicao Do Ponto Na Tela, Maior O Valor Y (eixo vertical)
    return hand[point].y < hand[holder].y - margin


def verifyThumbIsUp(hand: list, point: int, holder: int, margin: float) -> bool:
    return hand[point].x > hand[holder].x + margin 

def createMediapipeImage(rgbimage):
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbimage)


def verifyWhatFingerIsUp(hand) -> list:
    fingersDict = {
        'dedao': verifyThumbIsUp(hand, 4, 3, 0.02),
        'dedoIndicador': verifyFingerIsUp(hand, 8, 6, 0.03),
        'dedoMeio': verifyFingerIsUp(hand, 12, 10, 0.03),
        'dedoAnelar': verifyFingerIsUp(hand, 16, 14, 0.03),
        'dedoMinimo': verifyFingerIsUp(hand, 20, 18, 0.02),
    }
    fingersUp = [finger for finger, isUp in fingersDict.items() if isUp]
    return fingersUp

def verifyGesture(fingersUp: list) -> str:
    return GESTOS.get(frozenset(fingersUp), 'Não existe')


def GestureExecuter(gesture: str) -> None:
    # So dispara quando o gesto ainda nao tinha sido tratado nesse "estado"
    print(gesture)


def decidir_meme(reconhecido: bool) -> str | None:
    """Qual meme mostrar para este estado de reconhecimento (funcao pura)."""
    if reconhecido:
        return MEME_PUNHO
    return None


def mostrar_meme(caminho) -> None:
    """Mostra a imagem em tela cheia e volta ao apertar qualquer tecla.

    Tambem sai se a janela for fechada pelo gerenciador de janelas: sem essa
    checagem o `waitKey` ficaria esperando uma tecla que nunca chega.
    """
    caminho = Path(caminho)
    if not caminho.is_absolute():
        caminho = RAIZ / caminho

    imagem = cv2.imread(str(caminho))
    if imagem is None:
        print(f"Nao consegui abrir o meme: {caminho}")
        return

    cv2.namedWindow(JANELA_MEME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(JANELA_MEME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    cv2.imshow(JANELA_MEME, imagem)

    while True:
        if cv2.waitKey(30) & 0xFF != 255:  # 255 = nenhuma tecla
            break
        if cv2.getWindowProperty(JANELA_MEME, cv2.WND_PROP_VISIBLE) < 1:
            break

    try:
        cv2.destroyWindow(JANELA_MEME)
    except cv2.error:
        pass  # a janela ja foi fechada pelo gerenciador de janelas


def carregar_reconhecedor(caminho=ARQUIVO_ROSTO) -> ReconhecedorRosto:
    """Carrega o `.npz` salvo; se nao existir, calibra com `fotos/bini`.

    Calibrar leva alguns segundos (seis fotos passam pelo YuNet + SFace), por
    isso o resultado fica em disco.
    """
    reconhecedor = ReconhecedorRosto(MODELO_YUNET, MODELO_SFACE)
    caminho = Path(caminho)

    if caminho.is_file():
        reconhecedor.carregar(caminho)
        print(f"Rosto carregado de {caminho.name} ({reconhecedor.n_amostras} amostras)")
    else:
        aceitas = reconhecedor.calibrar(PASTA_FOTOS)
        reconhecedor.salvar(caminho)
        print(f"Rosto calibrado com {aceitas} fotos -> {caminho.name}")

    return reconhecedor


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nao consegui abrir a camera (indice 0)")
        return

    detector = criar_detector()

    # Rosto e um extra: se faltar modelo/foto, os gestos continuam funcionando.
    try:
        reconhecedor = carregar_reconhecedor()
    except (OSError, ValueError) as erro:
        print(f"Reconhecimento facial desligado: {erro}")
        reconhecedor = None

    estabilizador = Estabilizador()
    reconhecido_anterior = False
    counter = 0
    gesto_anterior = None

    try:
        while True:
            counter += 1
            ret, frame = cap.read()
            if not ret:
                print('Erro No RET(1 frame)')
                break

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            imageMp = createMediapipeImage(rgb)
            timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
            result = detector.detect_for_video(imageMp, timestamp_ms)

            for hand in result.hand_landmarks:
                ActualGesture = ''
                if counter % 5 == 0:
                    fingersUp = verifyWhatFingerIsUp(hand)
                    buffer = verifyGesture(fingersUp)
                    ActualGesture = buffer

                # debug pra ve se ta workando

                if ActualGesture != gesto_anterior:
                    GestureExecuter(ActualGesture)
                    gesto_anterior = ActualGesture

            # A cada 5 frames (mesmo ritmo dos gestos), pontua o rosto. O
            # estabilizador so vira True depois de 3 leituras seguidas boas.
            if reconhecedor is not None and counter % 5 == 0:
                leitura = reconhecedor.verificar(frame)
                score = leitura[0] if leitura is not None else None
                reconhecido = estabilizador.atualizar(score)

                # Borda de subida: mostra uma vez por entrada no estado True.
                if reconhecido and not reconhecido_anterior:
                    caminho_meme = decidir_meme(reconhecido)
                    if caminho_meme is not None:
                        mostrar_meme(caminho_meme)
                reconhecido_anterior = reconhecido

            cv2.imshow(JANELA_CAMERA, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
