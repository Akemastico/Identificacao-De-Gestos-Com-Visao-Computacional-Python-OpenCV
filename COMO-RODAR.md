# Como rodar (e como testar sem o Bini)

Guia rápido para a máquina da feira e para o dia a dia.

## O que o programa faz

- Lê a webcam o tempo todo.
- **Reconhece o rosto do Bini**: quando ele aparece, a tela mostra só o gato do
  punho em tela cheia. Quando ele sai, volta a mostrar a câmera.
- Detecta gestos de mão (MediaPipe) e **imprime o nome do gesto no terminal**.
  Os memes de gesto ainda não aparecem na tela — só o gato do Bini aparece.

## Rodar

```bash
cd ~/Documentos/Estudo/Opencv
.venv/bin/python gatos.py
```

- `q` encerra o programa (precisa que a janela do OpenCV esteja em foco; se
  não responder, clique nela ou feche o terminal).
- Na primeira execução ele calibra o rosto a partir de `fotos/bini/` e salva
  `rosto_bini.npz` (uns segundos). Depois disso carrega o `.npz` direto.

## Testar SEM o Bini presente

```bash
.venv/bin/python simular.py              # fotos do Bini: tem que virar gato
.venv/bin/python simular.py --intrusos   # outros rostos: nao pode virar gato
.venv/bin/python simular.py --intrusos ~/Downloads   # de outra pasta
.venv/bin/python simular.py --sem-janela # so imprime os scores
```

O simulador usa os **mesmos** componentes do programa (mesmo reconhecedor,
mesmo estabilizador, mesma janela do meme) — só troca a câmera pelas fotos.
Saída esperada:

```
Fotos do Bini (esperado: GATO):
  AGC_20261021_123900504.jpg  score=+0.902 -> GATO
  ...
  AGC_20261021_123915542.jpg  score=+0.340 -> sem gato   (foto ruim, descartada)
Outras pessoas (esperado: sem gato):
  fulano.jpg                   score=+0.311 -> sem gato
```

Os intrusos sao os rostos da pasta indicada (por padrao `~/Downloads`) — nao
ha nome de ninguem no repo, porque ele e publico.

## Se a máquina da feira não for a sua

Três coisas **não estão no git** (de propósito: o repo é público e tem foto de
rosto de menor de idade, e os modelos são 37 MB):

| Item | Onde está | Como levar |
|---|---|---|
| `modelos/face_detection_yunet_2023mar.onnx` | [opencv_zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx) | `curl -L -o modelos/... <url>` |
| `modelos/face_recognition_sface_2021dec.onnx` | [opencv_zoo](https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx) | idem |
| `fotos/bini/` (6 fotos) | só na sua máquina | copiar a pasta |

Sem os dois `.onnx`, o programa **continua funcionando com os gestos** e
aparece no terminal: `Reconhecimento facial desligado (cv2.error): ...`.
Sem as fotos, ele tenta ler `rosto_bini.npz`; se não existir, calibra.

## Se o gato não aparecer na hora

1. Roda `.venv/bin/python avaliar_rosto.py` — imprime a margem atual.
2. Se a margem estiver boa (acima de 0.2), o problema é **luz**: a foto da
   webcam tem score menor que a foto do celular. Aí o jeito é gravar 3–5 fotos
   do Bini **na máquina da feira** (com a webcam) e recalibrar com elas.
3. Se a margem estiver ruim, o problema são as fotos de referência.

## Onde mexer nos números

| Constante | Arquivo | O que faz |
|---|---|---|
| `LIMIAR_PADRAO = 0.50` | `rosto.py` | similaridade mínima para aceitar o rosto |
| `LIMIAR_CALIBRACAO = 0.60` | `rosto.py` | descarta foto de referência ruim |
| `INTERVALO_ROSTO = 30` | `gatos.py` | checa o rosto a cada 30 frames (~1 s) |
| `LEITURAS_CONFIRMACAO_ROSTO = 2` | `gatos.py` | leituras seguidas para confirmar/soltar |
| `FALHAS_LEITURA_MAXIMAS = 10` | `gatos.py` | falhas de câmera seguidas antes de desistir |

## Testes

```bash
.venv/bin/python -m unittest discover -s testes -v
```

66 testes, sem webcam e sem internet. Os de integração com as fotos reais
pulam sozinhos se `fotos/bini/` não existir.
