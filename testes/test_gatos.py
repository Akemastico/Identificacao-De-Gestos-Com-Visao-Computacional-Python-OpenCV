"""Testes do programa de gestos (gatos.py).

O objetivo aqui nao e testar a webcam (nao da sem hardware), e sim garantir que:

- importar `gatos` nao tem efeito colateral (nao abre camera, nao entra em loop);
- `decidir_meme` e uma funcao pura que traduz o estado do reconhecimento no
  caminho do meme.
"""

import sys
import unittest
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
