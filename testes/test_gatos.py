"""Testes do programa de gestos (gatos.py).

O objetivo aqui nao e testar a webcam (nao da sem hardware), e sim garantir que:

- importar `gatos` nao tem efeito colateral (nao abre camera, nao entra em loop);
- `decidir_meme` e uma funcao pura que traduz o estado do reconhecimento no
  caminho do meme.
"""

import contextlib
import io
import sys
import unittest
import zipfile
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import gatos  # noqa: E402  (import depois do ajuste de sys.path)
import rosto  # noqa: E402


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

    def test_waitkey_com_erro_nao_propaga(self):
        """`waitKey` pode falhar quando o backend highgui perde a janela."""
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)

        with (
            mock.patch.object(gatos.cv2, "imread", return_value=imagem),
            mock.patch.object(gatos.cv2, "namedWindow"),
            mock.patch.object(gatos.cv2, "setWindowProperty"),
            mock.patch.object(gatos.cv2, "imshow"),
            mock.patch.object(
                gatos.cv2, "waitKey", side_effect=cv2.error("sem janela")
            ),
            mock.patch.object(gatos.cv2, "getWindowProperty", return_value=1.0),
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


class TestMainInicializacaoResiliente(unittest.TestCase):
    """Modelo/npz faltando ou corrompido nao pode derrubar o programa."""

    def _rodar_main_com_erro_no_rosto(self, erro):
        camera = CameraFalsa()

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(gatos, "criar_detector", return_value=mock.Mock()),
            mock.patch.object(gatos, "carregar_reconhecedor", side_effect=erro),
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
        ):
            saida = io.StringIO()
            with contextlib.redirect_stdout(saida):
                gatos.main()  # nao pode levantar

        return camera, saida.getvalue()

    def test_onnx_ausente_nao_derruba_main_e_libera_camera(self):
        camera, saida = self._rodar_main_com_erro_no_rosto(
            cv2.error("modelo .onnx ausente")
        )

        self.assertEqual(camera.released, 1)
        self.assertIn("Reconhecimento facial desligado", saida)
        self.assertIn("modelo .onnx ausente", saida)

    def test_npz_truncado_nao_derruba_main_e_libera_camera(self):
        camera, saida = self._rodar_main_com_erro_no_rosto(
            zipfile.BadZipFile("npz truncado")
        )

        self.assertEqual(camera.released, 1)
        self.assertIn("Reconhecimento facial desligado", saida)
        self.assertIn("npz truncado", saida)


class TestLimiarDoEstabilizador(unittest.TestCase):
    """Aceitacao (reconhecedor) e estabilizacao precisam do MESMO limiar."""

    def test_usa_limiar_carregado_do_npz(self):
        reconhecedor = mock.Mock(limiar=0.73)

        estabilizador = gatos.criar_estabilizador(reconhecedor)

        self.assertAlmostEqual(estabilizador.limiar, 0.73)

    def test_sem_reconhecedor_usa_o_padrao(self):
        estabilizador = gatos.criar_estabilizador(None)

        self.assertAlmostEqual(estabilizador.limiar, rosto.LIMIAR_PADRAO)

    def test_main_constroi_estabilizador_com_o_limiar_do_npz(self):
        camera = CameraFalsa()
        reconhecedor = mock.Mock(limiar=0.73)

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(gatos, "criar_detector", return_value=mock.Mock()),
            mock.patch.object(
                gatos, "carregar_reconhecedor", return_value=reconhecedor
            ),
            mock.patch.object(gatos, "Estabilizador") as estabilizador,
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
        ):
            gatos.main()

        self.assertEqual(estabilizador.call_args.kwargs.get("limiar"), 0.73)


if __name__ == "__main__":
    unittest.main()
