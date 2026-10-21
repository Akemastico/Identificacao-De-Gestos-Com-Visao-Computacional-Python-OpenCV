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

    def test_npz_de_versao_antiga_keyerror_nao_derruba_main(self):
        """Arquivo sem a chave `limiar` levanta KeyError em versoes antigas."""
        camera, saida = self._rodar_main_com_erro_no_rosto(KeyError("limiar"))

        self.assertEqual(camera.released, 1)
        self.assertIn("Reconhecimento facial desligado", saida)


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


class TestIntervaloDoRosto(unittest.TestCase):
    def test_checagem_do_rosto_acontece_a_cada_30_frames(self):
        self.assertEqual(gatos.INTERVALO_ROSTO, 30)


class TestEstabilizadorDuasLeituras(unittest.TestCase):
    def test_estabilizador_confirma_com_duas_leituras(self):
        self.assertEqual(gatos.criar_estabilizador().frames, 2)


class TestJanelaMeme(unittest.TestCase):
    """O meme agora fica na tela ENQUANTO o rosto estiver no quadro.

    A janela nao pode ser modal (nao pode parar o loop): senao o programa
    nunca percebe que a pessoa saiu.
    """

    def test_carrega_a_imagem_do_meme(self):
        self.assertTrue(gatos.JanelaMeme(gatos.MEME_PUNHO).disponivel)

    def test_mostrar_cria_a_janela_uma_vez_so(self):
        with (
            mock.patch.object(gatos.cv2, "namedWindow") as named,
            mock.patch.object(gatos.cv2, "setWindowProperty"),
            mock.patch.object(gatos.cv2, "imshow") as imshow,
        ):
            tela = gatos.JanelaMeme(gatos.MEME_PUNHO)
            tela.mostrar()
            tela.mostrar()

        self.assertEqual(named.call_count, 1)
        self.assertEqual(imshow.call_count, 2)

    def test_esconder_so_destroi_a_janela_depois_de_mostrar(self):
        with mock.patch.object(gatos.cv2, "destroyWindow") as destroy:
            gatos.JanelaMeme(gatos.MEME_PUNHO).esconder()

        self.assertEqual(destroy.call_count, 0)

        with (
            mock.patch.object(gatos.cv2, "namedWindow"),
            mock.patch.object(gatos.cv2, "setWindowProperty"),
            mock.patch.object(gatos.cv2, "imshow"),
            mock.patch.object(gatos.cv2, "destroyWindow") as destroy,
        ):
            tela = gatos.JanelaMeme(gatos.MEME_PUNHO)
            tela.mostrar()
            tela.esconder()
            tela.esconder()  # segunda vez nao destroi de novo

        self.assertEqual(destroy.call_count, 1)

    def test_tela_cheia_falhando_ainda_mostra_e_nao_vaza_a_janela(self):
        """Se `setWindowProperty` falhar, a janela criada ainda existe.

        Sem isso o `except` deixava `visivel=False`, `esconder()` nunca
        destruia a janela e ela ficava presa na frente da camera.
        """
        with (
            mock.patch.object(gatos.cv2, "namedWindow"),
            mock.patch.object(
                gatos.cv2, "setWindowProperty", side_effect=cv2.error("sem fullscreen")
            ),
            mock.patch.object(gatos.cv2, "imshow") as imshow,
            mock.patch.object(gatos.cv2, "destroyWindow") as destroy,
        ):
            tela = gatos.JanelaMeme(gatos.MEME_PUNHO)
            tela.mostrar()

            self.assertTrue(tela.visivel)
            self.assertEqual(imshow.call_count, 1)

            tela.esconder()

        self.assertEqual(destroy.call_count, 1)

    def test_meme_ausente_nao_derruba_o_programa(self):
        tela = gatos.JanelaMeme("gatos/nao-existe.webp")

        self.assertFalse(tela.disponivel)
        tela.mostrar()  # nao deve levantar
        tela.esconder()

    def test_erro_de_janela_nao_propaga(self):
        with (
            mock.patch.object(
                gatos.cv2, "namedWindow", side_effect=cv2.error("sem janela")
            ),
            mock.patch.object(gatos.cv2, "destroyWindow"),
        ):
            tela = gatos.JanelaMeme(gatos.MEME_PUNHO)
            tela.mostrar()  # cv2.error tratado dentro da janela
            self.assertFalse(tela.visivel)


class CameraComFrames:
    """Camera de mentira que entrega um numero fixo de frames."""

    def __init__(self, frames=65, falhas=0):
        self.restantes = frames
        self.falhas_iniciais = falhas
        self.leituras = 0
        self.released = 0

    def isOpened(self):
        return True

    def read(self):
        self.leituras += 1
        if self.falhas_iniciais > 0:
            self.falhas_iniciais -= 1
            return False, None
        if self.restantes <= 0:
            return False, None
        self.restantes -= 1
        return True, np.zeros((10, 10, 3), dtype=np.uint8)

    def release(self):
        self.released += 1


class TestProximoTimestamp(unittest.TestCase):
    """MediaPipe exige timestamp estritamente crescente entre frames."""

    def test_avanca_quando_dois_frames_caem_no_mesmo_milissegundo(self):
        self.assertEqual(gatos.proximo_timestamp(100, 100), 101)

    def test_acompanha_o_relogio_quando_ele_avanca(self):
        self.assertEqual(gatos.proximo_timestamp(100, 150), 150)

    def test_primeiro_frame_nunca_e_zero(self):
        self.assertEqual(gatos.proximo_timestamp(0, 0), 1)


class TestFalhasDeLeitura(unittest.TestCase):
    """Um tropeco da webcam nao pode encerrar o programa na feira."""

    def _rodar_main(self, camera):
        detector = mock.Mock()
        detector.detect_for_video.return_value = mock.Mock(hand_landmarks=[])

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(gatos, "criar_detector", return_value=detector),
            mock.patch.object(gatos, "carregar_reconhecedor", side_effect=ValueError("off")),
            mock.patch.object(gatos.cv2, "imshow"),
            mock.patch.object(gatos.cv2, "waitKey", return_value=255),
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            gatos.main()

    def test_falhas_temporarias_nao_encerram_o_programa(self):
        camera = CameraComFrames(frames=3, falhas=3)

        self._rodar_main(camera)

        self.assertEqual(camera.restantes, 0)  # todos os frames bons foram usados
        self.assertEqual(camera.released, 1)

    def test_camera_morta_encerra_depois_do_limite(self):
        camera = CameraComFrames(frames=0)

        self._rodar_main(camera)

        self.assertEqual(camera.leituras, gatos.FALHAS_LEITURA_MAXIMAS)
        self.assertEqual(camera.released, 1)

    def test_tecla_q_e_processada_mesmo_com_camera_morta(self):
        """Sem bomba de eventos o `q` nao funciona enquanto a camera falha."""
        camera = CameraComFrames(frames=0)
        detector = mock.Mock()
        detector.detect_for_video.return_value = mock.Mock(hand_landmarks=[])

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(gatos, "criar_detector", return_value=detector),
            mock.patch.object(
                gatos, "carregar_reconhecedor", side_effect=ValueError("off")
            ),
            mock.patch.object(gatos.cv2, "imshow"),
            mock.patch.object(gatos.cv2, "waitKey", return_value=255) as waitKey,
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            gatos.main()

        self.assertGreater(waitKey.call_count, 0)


class TestMainExibeMeme(unittest.TestCase):
    """O meme fica na tela enquanto o rosto reconhecido estiver no quadro."""

    def _rodar_main(self, score):
        camera = CameraComFrames()
        reconhecedor = mock.Mock(limiar=0.5)
        reconhecedor.melhor_score.return_value = score
        detector = mock.Mock()
        detector.detect_for_video.return_value = mock.Mock(hand_landmarks=[])

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(gatos, "criar_detector", return_value=detector),
            mock.patch.object(
                gatos, "carregar_reconhecedor", return_value=reconhecedor
            ),
            mock.patch.object(gatos, "JanelaMeme") as janela,
            mock.patch.object(gatos.cv2, "imshow"),
            mock.patch.object(gatos.cv2, "waitKey", return_value=255),
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
        ):
            janela.return_value.disponivel = True
            gatos.main()

        self.assertEqual(camera.released, 1)
        return janela.return_value

    def test_meme_aparece_quando_o_rosto_e_reconhecido(self):
        tela = self._rodar_main(0.9)

        self.assertGreater(tela.mostrar.call_count, 1)

    def test_meme_nao_aparece_para_outra_pessoa(self):
        tela = self._rodar_main(0.1)

        self.assertEqual(tela.mostrar.call_count, 0)
        self.assertGreater(tela.esconder.call_count, 0)

    def _rodar_transicao(self):
        """Bini aparece, fica, e sai do quadro; a camera tem que voltar."""
        camera = CameraComFrames(frames=150)
        reconhecedor = mock.Mock(limiar=0.5)
        reconhecedor.melhor_score.side_effect = [0.9, 0.9, 0.1, 0.1, 0.0]
        detector = mock.Mock()
        detector.detect_for_video.return_value = mock.Mock(hand_landmarks=[])
        eventos = []

        def mostrar():
            eventos.append("meme")

        def esconder():
            eventos.append("esconder")

        def imshow(janela, _imagem):
            if janela == gatos.JANELA_CAMERA:
                eventos.append("camera")

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(gatos, "criar_detector", return_value=detector),
            mock.patch.object(
                gatos, "carregar_reconhecedor", return_value=reconhecedor
            ),
            mock.patch.object(gatos, "JanelaMeme") as janela,
            mock.patch.object(gatos.cv2, "imshow", side_effect=imshow),
            mock.patch.object(gatos.cv2, "waitKey", return_value=255),
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            janela.return_value.disponivel = True
            janela.return_value.mostrar.side_effect = mostrar
            janela.return_value.esconder.side_effect = esconder
            gatos.main()

        return eventos

    def test_volta_para_a_camera_quando_a_pessoa_sai_do_quadro(self):
        eventos = self._rodar_transicao()

        self.assertIn("meme", eventos, "o meme nunca apareceu")
        ultimo_meme = len(eventos) - 1 - eventos[::-1].index("meme")
        # `esconder` e um evento proprio: se o loop parar de chamar, este
        # teste tem que falhar (antes ele so olhava o imshow da camera).
        self.assertIn(
            "esconder", eventos[ultimo_meme:], "a janela do meme nao foi fechada"
        )
        self.assertIn(
            "camera", eventos[ultimo_meme:], "nao voltou para a camera"
        )
        self.assertGreater(eventos.count("camera"), 1)

    def test_meme_que_nao_carrega_nao_congela_a_camera(self):
        """Sem imagem, a camera continua aparecendo (nao fica tela parada)."""
        camera = CameraComFrames(frames=90)
        reconhecedor = mock.Mock(limiar=0.5)
        reconhecedor.melhor_score.return_value = 0.9
        detector = mock.Mock()
        detector.detect_for_video.return_value = mock.Mock(hand_landmarks=[])

        with (
            mock.patch.object(gatos.cv2, "VideoCapture", return_value=camera),
            mock.patch.object(gatos, "criar_detector", return_value=detector),
            mock.patch.object(
                gatos, "carregar_reconhecedor", return_value=reconhecedor
            ),
            mock.patch.object(gatos, "JanelaMeme") as janela,
            mock.patch.object(gatos.cv2, "imshow") as imshow,
            mock.patch.object(gatos.cv2, "waitKey", return_value=255),
            mock.patch.object(gatos.cv2, "destroyAllWindows"),
            contextlib.redirect_stdout(io.StringIO()),
        ):
            janela.return_value.disponivel = False
            gatos.main()

        janelas = [chamada.args[0] for chamada in imshow.call_args_list]
        # Reconhecido a partir do frame 60: sem a correcao, os ultimos 30
        # frames ficariam sem camera nenhuma na tela.
        self.assertGreater(janelas.count(gatos.JANELA_CAMERA), 80)


if __name__ == "__main__":
    unittest.main()
