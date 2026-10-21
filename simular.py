"""Simulador: roda a pipeline real do projeto SEM webcam.

Serve para validar tudo o que so dava para testar com o Bini presente:
calibracao, limiar, histerese e a abertura do meme em tela cheia. As fotos de
`fotos/bini` entram no lugar dos frames da camera.

    .venv/bin/python simular.py            # fotos do Bini (viram gato)
    .venv/bin/python simular.py --intrusos # rosto de outra pessoa (nao vira)
    .venv/bin/python simular.py --sem-janela
"""

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parent
if str(RAIZ) not in sys.path:
    sys.path.insert(0, str(RAIZ))

import cv2  # noqa: E402

import gatos  # noqa: E402
from rosto import LIMIAR_PADRAO, Estabilizador, ReconhecedorRosto  # noqa: E402

PASTA_FOTOS = RAIZ / "fotos" / "bini"
MODELO_YUNET = RAIZ / "modelos" / "face_detection_yunet_2023mar.onnx"
MODELO_SFACE = RAIZ / "modelos" / "face_recognition_sface_2021dec.onnx"

INTRUSOS_PADRAO = Path.home() / "Downloads"

EXTENSOES_IMAGEM = (".jpg", ".jpeg", ".png")


def intrusos_de(argumento=None):
    """Fotos de intrusos: a pasta informada ou, por padrao, ~/Downloads.

    Nenhum nome de pessoa fica no repo: o intruso e "qualquer rosto que nao
    seja o Bini", e a pasta e do usuario.
    """
    pasta = Path(argumento).expanduser() if argumento else INTRUSOS_PADRAO
    return listar_fotos(pasta)


def listar_fotos(pasta):
    """Fotos de uma pasta, em ordem de nome; lista vazia se a pasta nao existe."""
    pasta = Path(pasta)
    if not pasta.is_dir():
        return []
    return sorted(
        caminho
        for caminho in pasta.iterdir()
        if caminho.is_file() and caminho.suffix.lower() in EXTENSOES_IMAGEM
    )


def simular(caminhos, reconhecedor, mostrar_janela=True):
    """Passa cada foto pela pipeline; devolve [(caminho, score, virou_gato)].

    Cada foto alimenta DUAS leituras, como o loop real faria com duas leituras
    seguidas do mesmo lado: e a histerese do `Estabilizador` que exige isso.
    """
    estabilizador = Estabilizador(limiar=reconhecedor.limiar, frames=2)
    tela = gatos.JanelaMeme(gatos.decidir_meme(True)) if mostrar_janela else None
    resultados = []

    for caminho in caminhos:
        imagem = cv2.imread(str(caminho))
        if imagem is None:
            print(f"  ! nao consegui abrir {caminho.name}")
            continue

        score = None
        estado = False
        for _ in range(2):  # duas leituras: a histerese exige duas
            score = reconhecedor.melhor_score(imagem)
            estado = estabilizador.atualizar(score)

        meme = gatos.decidir_meme(estado)
        if score is None:
            print(f"  {caminho.name:<32} sem rosto detectado -> sem gato")
        else:
            print(
                f"  {caminho.name:<32} score={score:+.3f} "
                f"limiar={reconhecedor.limiar:.2f} -> "
                f"{'GATO' if meme else 'sem gato'}"
            )
        resultados.append((caminho, score, estado))

        if tela is not None and estado:
            tela.mostrar()
            print("    (meme aberto; pressione q ou Esc para fechar)")
            while True:
                tecla = cv2.waitKey(30) & 0xFF
                if tecla in (ord("q"), 27):
                    break
                try:
                    if (
                        cv2.getWindowProperty(gatos.JANELA_MEME, cv2.WND_PROP_VISIBLE)
                        < 1
                    ):
                        break
                except cv2.error:
                    break
            tela.esconder()

    if tela is not None:
        cv2.destroyAllWindows()
    return resultados


def main() -> int:
    argumentos = sys.argv[1:]
    mostrar_janela = "--sem-janela" not in argumentos
    intrusos = "--intrusos" in argumentos
    pasta_intrusos = None
    if intrusos:
        depois = argumentos[argumentos.index("--intrusos") + 1 :]
        if depois and not depois[0].startswith("--"):
            pasta_intrusos = depois[0]

    if not MODELO_YUNET.is_file() or not MODELO_SFACE.is_file():
        print("Modelos .onnx ausentes em modelos/ — veja COMO-RODAR.md")
        return 1

    reconhecedor = ReconhecedorRosto(MODELO_YUNET, MODELO_SFACE)
    aceitas = reconhecedor.calibrar(PASTA_FOTOS)
    total = len(listar_fotos(PASTA_FOTOS))
    print(f"Calibrado com {aceitas} fotos do Bini (limiar {reconhecedor.limiar:.2f})")
    print(f"Descartadas na calibracao: {total - aceitas} foto(s)\n")

    print("Fotos do Bini (esperado: GATO):")
    simular(listar_fotos(PASTA_FOTOS), reconhecedor, mostrar_janela)

    if intrusos:
        caminhos = intrusos_de(pasta_intrusos)
        if not caminhos:
            print("\nNenhum intruso encontrado para comparar.")
        else:
            print("\nOutras pessoas (esperado: sem gato):")
            simular(caminhos, reconhecedor, mostrar_janela)

    print(f"\nLimiar em uso: {LIMIAR_PADRAO:.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
