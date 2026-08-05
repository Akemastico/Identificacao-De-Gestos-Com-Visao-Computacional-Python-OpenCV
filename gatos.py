import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import cv2
import os

#Variaveis Usadas Globalmente
cap = cv2.VideoCapture(0)
HandModel = '/home/akemistico/Documentos/Opencv/modelos/hand_landmarker.task'
contador = 0 #Usado Para Delay Em Outputs

#Configuracao Modelo Para Detector De Moes

modelAsset = python.BaseOptions(model_asset_path=HandModel)
modelConfig = vision.HandLandmarkerOptions(
    base_options=modelAsset,
    running_mode=vision.RunningMode.VIDEO,
    num_hands= 2,
    min_hand_detection_confidence=0.5,
    min_hand_presence_confidence=0.5
)
detector = vision.HandLandmarker.create_from_options(modelConfig)

def verifyFingerIsUp(hand: list,point :int,holder :int,margin:float) -> bool:
  # Quanto Menor A Posicao Do Ponto Na Tela, Maior O Valor Y
    return hand[point].y < hand[holder].y - margin

def createMediapipeImage(rgbimage) :
    return mp.Image(image_format=mp.ImageFormat.SRGB, data=rgbimage)

def verifyWhatFingerIsUp(hand):
 # Vou colocar em portugues pq ta de noite, to sem net, e nao sou obrigado a saber os dedos em ingles#
   fingersDict = {'dedao' : verifyFingerIsUp(hand,4,3,0.2),
   'dedoIndicador' : verifyFingerIsUp(hand,8,6,0.3),
   'dedoMeio' : verifyFingerIsUp(hand,12,10,0.3),
   'dedoAnelar' : verifyFingerIsUp(hand,16,14,0.3),
   "dedoMinimo" : verifyFingerIsUp(hand,16,14,0.2),
   }
   fingersUp = [finger for finger, isUp in fingersDict.items() if isUp]
   return fingersUp


def main() -> None:
    contador = 0
    while True:
        contador += 1
        ret, frame = cap.read()
        if not ret:
            print('Erro No RET(1 frame)')
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        imagemMp = createMediapipeImage(rgb)
        timestamp_ms = int(cv2.getTickCount() / cv2.getTickFrequency() * 1000)
        result = detector.detect_for_video(imagemMp,timestamp_ms)
        y,x = frame.shape[:2]
        for hand in result.hand_landmarks :
            fingersUp = verifyWhatFingerIsUp(hand)









