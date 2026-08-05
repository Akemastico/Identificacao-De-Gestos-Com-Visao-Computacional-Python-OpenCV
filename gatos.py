import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import subprocess

# Variaveis Usadas Globalmente
cap = cv2.VideoCapture(0)
HandModel = '/home/akemistico/Documentos/Opencv/modelos/hand_landmarker.task'

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

# Configuracao Modelo Para Detector De Maos
modelAsset = python.BaseOptions(model_asset_path=HandModel)
modelConfig = vision.HandLandmarkerOptions(
    base_options=modelAsset,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(modelConfig)


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
    return GESTOS.get(frozenset(fingersUp), 'nao identificado')


def GestureExecuter(gesture: str) -> None:
    # So dispara quando o gesto ainda nao tinha sido tratado nesse "estado"
    print(gesture)


def main() -> None:
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
            fingersUp = verifyWhatFingerIsUp(hand)
            gesto_atual = verifyGesture(fingersUp)

            # debug pra ve se ta workando
            if counter % 15 == 0:
                print(fingersUp, "->", gesto_atual)

            if gesto_atual != gesto_anterior:
                GestureExecuter(gesto_atual)
                gesto_anterior = gesto_atual

        cv2.imshow('Camera', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


try:
    main()
finally:
    cap.release()
    cv2.destroyAllWindows()