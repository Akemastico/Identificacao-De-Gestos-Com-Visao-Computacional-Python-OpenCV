"""Programa de gestos com webcam e reacao ao rosto reconhecido.

Duas responsabilidades separadas:

- `gatos.py` (aqui): camera, gestos e janelas.
- `rosto.py`: deteccao + assinatura facial, sem saber de meme nenhum.

Regra de ouro deste arquivo: **importar nao pode abrir camera nem carregar
modelo**. Tudo que tem efeito colateral vive dentro de `main()`, senao o modulo
nao pode ser testado sem hardware.
"""

import subprocess
import zipfile
from pathlib import Path

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from rosto import LIMIAR_PADRAO, Estabilizador, ReconhecedorRosto

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

# Checagem do rosto a cada 30 frames: o reconhecimento facial e caro e nao
# precisa de 30 leituras por segundo. Com 2 leituras para confirmar, o gato
# aparece em ~2 s e some ~2 s depois que a pessoa sai do quadro.
INTERVALO_ROSTO = 30
LEITURAS_CONFIRMACAO_ROSTO = 2

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


class JanelaMeme:
    """Janela de tela cheia que fica visivel ENQUANTO o rosto estiver no quadro.

    Nao pode ser modal (nao pode parar o loop): o programa precisa continuar
    lendo a camera para perceber que a pessoa saiu. Por isso `mostrar()` so
    repinta a imagem e volta — quem decide quando esconder e `main()`.
    """

    def __init__(self, caminho, janela=JANELA_MEME):
        self.janela = janela
        caminho = Path(caminho)
        if not caminho.is_absolute():
            caminho = RAIZ / caminho

        self.imagem = cv2.imread(str(caminho))
        self.disponivel = self.imagem is not None
        self.visivel = False
        if not self.disponivel:
            print(f"Nao consegui abrir o meme: {caminho}")

    def mostrar(self) -> None:
        """Cria a janela na primeira vez e repinta a imagem nas seguintes."""
        if not self.disponivel:
            return

        try:
            if not self.visivel:
                cv2.namedWindow(self.janela, cv2.WINDOW_NORMAL)
                cv2.setWindowProperty(
                    self.janela, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN
                )
            cv2.imshow(self.janela, self.imagem)
            self.visivel = True
        except cv2.error as erro:
            # Wayland/Niri pode perder a janela: nao derruba o loop por isso.
            print(f"Janela do meme indisponivel: {erro}")
            self._esquecer()

    def esconder(self) -> None:
        """Volta para a camera; nao faz nada se a janela nunca apareceu."""
        if not self.visivel:
            return

        try:
            cv2.destroyWindow(self.janela)
        except cv2.error:
            pass  # o gerenciador de janelas ja fechou
        self._esquecer()

    def _esquecer(self) -> None:
        self.visivel = False


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


def criar_estabilizador(
    reconhecedor=None, leituras=LEITURAS_CONFIRMACAO_ROSTO
) -> Estabilizador:
    """Estabilizador com o MESMO limiar que o reconhecedor usa para aceitar.

    `ReconhecedorRosto.carregar` restaura o limiar do `.npz`, que pode diferir
    do `LIMIAR_PADRAO`. Se o estabilizador ficasse no padrao, aceitacao e
    estabilizacao discordariam nas bordas.
    """
    limiar = reconhecedor.limiar if reconhecedor is not None else LIMIAR_PADRAO
    return Estabilizador(limiar=limiar, frames=leituras)


def main() -> None:
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Nao consegui abrir a camera (indice 0)")
        return

    # Daqui para baixo a camera esta aberta: o `try/finally` garante que ela
    # seja liberada mesmo se o detector ou o rosto falharem na inicializacao.
    try:
        detector = criar_detector()

        # Rosto e um extra: se faltar modelo/foto, os gestos continuam. Os
        # erros reais: `.onnx` ausente -> cv2.error; `.npz` truncado ->
        # zipfile.BadZipFile/EOFError; pasta ou arquivo ausente -> OSError;
        # calibracao sem rosto -> ValueError.
        try:
            reconhecedor = carregar_reconhecedor()
        except (cv2.error, zipfile.BadZipFile, EOFError, OSError, ValueError) as erro:
            print(
                "Reconhecimento facial desligado "
                f"({type(erro).__name__}): {erro}"
            )
            reconhecedor = None

        estabilizador = criar_estabilizador(reconhecedor)
        # Carrega o meme uma unica vez; se o rosto for reconhecido, a janela
        # fica na tela por todos os frames seguintes (sem bloquear o loop).
        tela_meme = (
            JanelaMeme(decidir_meme(True)) if reconhecedor is not None else None
        )
        reconhecido = False
        counter = 0
        gesto_anterior = None

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

            # A cada INTERVALO_ROSTO frames pontua o rosto. O estabilizador so
            # muda de estado depois de LEITURAS_CONFIRMACAO_ROSTO leituras
            # seguidas do mesmo lado (~2 s para entrar e ~2 s para sair).
            if reconhecedor is not None and counter % INTERVALO_ROSTO == 0:
                leitura = reconhecedor.verificar(frame)
                score = leitura[0] if leitura is not None else None
                reconhecido = estabilizador.atualizar(score)

            # Enquanto o rosto reconhecido estiver no quadro, a tela mostra só
            # o meme; quando ele sai, volta para a camera.
            if reconhecido and tela_meme is not None:
                tela_meme.mostrar()
            else:
                if tela_meme is not None:
                    tela_meme.esconder()
                cv2.imshow(JANELA_CAMERA, frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
