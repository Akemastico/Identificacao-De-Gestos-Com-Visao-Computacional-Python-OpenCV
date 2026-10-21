"""Testes do simulador (simular.py).

O simulador existe para validar a pipeline SEM webcam: as fotos do Bini
têm que virar gato e os rostos de intruso não.
"""

import sys
import unittest
from pathlib import Path
from unittest import mock

RAIZ = Path(__file__).resolve().parent.parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import numpy as np  # noqa: E402

import gatos  # noqa: E402
import simular  # noqa: E402


class TestListarFotos(unittest.TestCase):
    def test_ignora_pasta_inexistente(self):
        self.assertEqual(simular.listar_fotos(RAIZ / "nao-existe"), [])

    def test_lista_somente_imagens(self):
        import tempfile

        with tempfile.TemporaryDirectory() as pasta:
            base = Path(pasta)
            (base / "foto.jpg").write_bytes(b"x")
            (base / "nota.txt").write_bytes(b"x")

            caminhos = simular.listar_fotos(base)

        self.assertEqual([c.name for c in caminhos], ["foto.jpg"])


class TestIntrusosDe(unittest.TestCase):
    """Os intrusos vem de uma pasta — nenhum nome de pessoa no repo."""

    def test_le_imagens_da_pasta_informada(self):
        import tempfile

        with tempfile.TemporaryDirectory() as pasta:
            base = Path(pasta)
            (base / "outro.jpg").write_bytes(b"x")
            (base / "leia-me.txt").write_bytes(b"x")

            caminhos = simular.intrusos_de(base)

        self.assertEqual([c.name for c in caminhos], ["outro.jpg"])

    def test_pasta_inexistente_nao_quebra(self):
        self.assertEqual(simular.intrusos_de("/tmp/pasta-que-nao-existe-1234"), [])


class TestSimular(unittest.TestCase):
    def _reconhecedor_falso(self, scores):
        reconhecedor = mock.Mock(limiar=0.5)
        reconhecedor.melhor_score.side_effect = scores
        return reconhecedor

    def test_foto_reconhecida_vira_gato(self):
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)
        caminho = Path("/tmp/foto_bini.jpg")
        caminho.write_bytes(b"nao-e-uma-jpeg-de-verdade")

        with (
            mock.patch.object(simular.cv2, "imread", return_value=imagem),
            mock.patch.object(simular, "gatos") as gatos_falso,
        ):
            gatos_falso.decidir_meme.side_effect = lambda estado: (
                "gatos/punho.webp" if estado else None
            )
            resultados = simular.simular(
                [caminho], self._reconhecedor_falso([0.9, 0.9]), mostrar_janela=False
            )

        self.assertEqual(len(resultados), 1)
        self.assertTrue(resultados[0][2], "a foto reconhecida deveria virar gato")

    def test_foto_nao_reconhecida_nao_vira_gato(self):
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)
        caminho = Path("/tmp/foto_estranha.jpg")
        caminho.write_bytes(b"nao-e-uma-jpeg-de-verdade")

        with (
            mock.patch.object(simular.cv2, "imread", return_value=imagem),
            mock.patch.object(simular, "gatos") as gatos_falso,
        ):
            gatos_falso.decidir_meme.side_effect = lambda estado: (
                "gatos/punho.webp" if estado else None
            )
            resultados = simular.simular(
                [caminho], self._reconhecedor_falso([0.1, 0.1]), mostrar_janela=False
            )

        self.assertFalse(resultados[0][2], "um intruso nao pode virar gato")

    def test_uma_leitura_boa_isolada_nao_vira_gato(self):
        """Uma foto boa seguida de uma ruim nao confirma ninguem."""
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)
        caminho = Path("/tmp/foto_unica.jpg")
        caminho.write_bytes(b"nao-e-uma-jpeg-de-verdade")

        with (
            mock.patch.object(simular.cv2, "imread", return_value=imagem),
            mock.patch.object(simular, "gatos") as gatos_falso,
        ):
            gatos_falso.decidir_meme.side_effect = lambda estado: (
                "gatos/punho.webp" if estado else None
            )
            resultados = simular.simular(
                [caminho],
                self._reconhecedor_falso([0.9, 0.1]),
                mostrar_janela=False,
            )

        self.assertFalse(resultados[0][2], "uma leitura boa só não pode confirmar")


    def test_foto_sem_rosto_nao_quebra_o_simulador(self):
        """Foto sem nenhum rosto (um gato, um desenho) nao pode estourar."""
        imagem = np.zeros((10, 10, 3), dtype=np.uint8)
        caminho = Path("/tmp/foto_sem_rosto.jpg")
        caminho.write_bytes(b"nao-e-uma-jpeg-de-verdade")

        with (
            mock.patch.object(simular.cv2, "imread", return_value=imagem),
            mock.patch.object(simular, "gatos") as gatos_falso,
        ):
            gatos_falso.decidir_meme.side_effect = lambda estado: (
                "gatos/punho.webp" if estado else None
            )
            resultados = simular.simular(
                [caminho],
                self._reconhecedor_falso([None, None]),
                mostrar_janela=False,
            )

        self.assertEqual(len(resultados), 1)
        self.assertFalse(resultados[0][2])


if __name__ == "__main__":
    unittest.main()
