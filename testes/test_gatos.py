"""Testes do programa de gestos (gatos.py).

O objetivo aqui nao e testar a webcam (nao da sem hardware), e sim garantir que:

- importar `gatos` nao tem efeito colateral (nao abre camera, nao entra em loop);
- `decidir_meme` e uma funcao pura que traduz o estado do reconhecimento no
  caminho do meme.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import gatos  # noqa: E402  (import depois do ajuste de sys.path)


class TestImportSemCamera(unittest.TestCase):
    """O import precisa ser seguro: nada de `cap` aberto no nivel do modulo."""

    def test_modulo_nao_expoe_camera_global(self):
        self.assertFalse(hasattr(gatos, "cap"))

    def test_modulo_tem_main(self):
        self.assertTrue(callable(gatos.main))


class TestDecidirMeme(unittest.TestCase):
    def test_reconhecido_devolve_o_gato_do_punho(self):
        self.assertEqual(gatos.decidir_meme(True), "gatos/punho.webp")

    def test_nao_reconhecido_nao_devolve_meme(self):
        self.assertIsNone(gatos.decidir_meme(False))

    def test_meme_aponta_para_arquivo_existente(self):
        caminho = RAIZ / gatos.decidir_meme(True)
        self.assertTrue(caminho.is_file(), f"meme ausente: {caminho}")


class TestMostrarMeme(unittest.TestCase):
    def test_getwindowproperty_com_erro_nao_derruba_o_programa(self):
        """Wayland/Niri pode matar a janela: o backend responde com cv2.error."""
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)

        with (
            mock.patch.object(gatos.cv2, "imread", return_value=imagem),
            mock.patch.object(gatos.cv2, "namedWindow"),
            mock.patch.object(gatos.cv2, "setWindowProperty"),
            mock.patch.object(gatos.cv2, "imshow"),
            mock.patch.object(gatos.cv2, "waitKey", return_value=255),
            mock.patch.object(
                gatos.cv2,
                "getWindowProperty",
                side_effect=cv2.error("janela ja destruida"),
            ),
            mock.patch.object(gatos.cv2, "destroyWindow") as destroyWindow,
        ):
            gatos.mostrar_meme("gatos/punho.webp")  # nao pode levantar

        destroyWindow.assert_called_once_with(gatos.JANELA_MEME)


class CameraFalsa:
    """Camera de mentira: so registra se `release` foi chamado."""

    def __init__(self):
        self.released = 0

    def isOpened(self):
        return True

    def read(self):
        return False, None

    def release(self):
        self.released += 1


class TestMainLiberaCamera(unittest.TestCase):
    def test_detector_sem_modelo_libera_camera(self):
        camera = CameraFalsa()

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(
                gatos, "criar_detector", side_effect=FileNotFoundError("sem .task")
            ),
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
        ):
            with self.assertRaises(FileNotFoundError):
                gatos.main()

        self.assertEqual(camera.released, 1)


if __name__ == "__main__":
    unittest.main()
