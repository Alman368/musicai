# MusicAI

Proyecto final de Deep Learning sobre generacion de musica de piano comparando dos enfoques:

- `musiclstm/`: implementacion tipo PerformanceRNN basada en LSTM
- `musictransformer/`: implementacion Music Transformer con Relative Position Representations

Esta version del repositorio esta preparada como portfolio en GitHub. Conserva el codigo, la memoria, las metricas y ahora tambien todos los ejemplos generados en un formato escuchable y ligero para web.

## Lo mas importante

- Memoria final: [docs/report/trabajo_final_deep_learning.pdf](docs/report/trabajo_final_deep_learning.pdf)
- Fuente LaTeX: [docs/report/trabajo.tex](docs/report/trabajo.tex)
- Propuesta inicial: [docs/report/propuesta-trabajo-music.pdf](docs/report/propuesta-trabajo-music.pdf)
- Resumen comparativo: [COMPARISON_SUMMARY.md](COMPARISON_SUMMARY.md)
- Galeria de escucha para GitHub Pages: [docs/index.html](docs/index.html)

## Resultados

| Metrica | LSTM | Transformer |
| --- | --- | --- |
| Val Loss | 2.2376 | 2.0426 |
| Val Accuracy | 33.43% | 40.67% |
| Val Perplexity | 9.53 | 7.71 |
| Parametros | 6.08M | ~25M |
| Tiempo de entrenamiento | 10h | 5h |

El Transformer fue el mejor modelo en calidad final, mientras que la LSTM resulto mas ligera y sencilla de entrenar y depurar.

## Ejemplos completos

Se han incluido todos los ejemplos generados:

- 45 archivos `MIDI` originales
- 45 previews `MP3` para escucha rapida en navegador

Los audios convertidos ocupan aproximadamente 26 MB en total. Los `WAV` originales se han dejado fuera para que el repositorio siga siendo razonable de clonar y navegar.

## GitHub Pages

La forma mas limpia de escuchar todo es activar GitHub Pages para publicar la galeria incluida en `docs/`.

Pasos:

1. En GitHub, abre `Settings > Pages`.
2. En `Build and deployment`, elige `Deploy from a branch`.
3. Selecciona la rama `main`.
4. Selecciona la carpeta `/docs`.
5. Guarda la configuracion.

GitHub permite publicar Pages desde la raiz del repositorio o desde `/docs`, asi que esta estructura esta preparada especificamente para usar `/docs` como fuente. Segun la documentacion oficial, los repositorios fuente de Pages tienen un limite recomendado de 1 GB y el sitio publicado no debe superar 1 GB, asi que esta galeria entra de sobra en un margen comodo.

Fuentes:

- https://docs.github.com/en/pages/getting-started-with-github-pages/configuring-a-publishing-source-for-your-github-pages-site
- https://docs.github.com/en/enterprise-cloud@latest/pages/getting-started-with-github-pages/github-pages-limits

## Estructura

```text
musicai/
├── docs/
│   ├── index.html               # Galeria reproducible para GitHub Pages
│   ├── showcase/assets/         # Todos los MIDIs y MP3s
│   └── report/                  # Memoria, propuesta y figuras
├── COMPARISON_SUMMARY.md
├── musiclstm/                   # Implementacion LSTM
└── musictransformer/            # Implementacion Transformer
```

## Lo que no se incluye

El entregable original superaba 1.3 GB e incluia:

- dataset MAESTRO completo
- checkpoints y pesos entrenados
- logs de TensorBoard
- `WAV` completos de trabajo

Esos artefactos no estan versionados aqui para evitar un repositorio pesado y poco practico. Las rutas siguen documentadas en cada subproyecto por si se quiere reconstruir el entorno original.

## Reproducir los modelos

### LSTM

```bash
cd musiclstm
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python train.py
```

El dataset esperado va en `musiclstm/data/`.

### Transformer

```bash
cd musictransformer
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python preprocess_midi.py dataset/maestro/maestro-v3.0.0 -output_dir dataset/e_piano
python train.py -input_dir dataset/e_piano -output_dir saved_models --rpr
```

El dataset bruto se espera en `musictransformer/dataset/maestro/` y el dataset procesado en `musictransformer/dataset/e_piano/`.
