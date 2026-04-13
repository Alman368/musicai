# PerformanceRNN (LSTM)

Implementacion en PyTorch Lightning de una version tipo PerformanceRNN para generacion de musica de piano.

## Que incluye este repo

- codigo fuente del modelo en `performance_net/`
- configuracion de entrenamiento en `config.yaml`
- resumen de resultados en `training_results.txt`
- grafica final en `final_training_metrics.png`

Todos los ejemplos escuchables del proyecto estan centralizados en `docs/showcase/assets/` y en la galeria publicada desde `docs/index.html`.

## Que no incluye

Para mantener el repositorio limpio en GitHub, esta carpeta no versiona:

- `data/` con el dataset MAESTRO
- `runs/` con checkpoints y logs
- lotes completos de `generated_pieces/`

## Estructura

```text
musiclstm/
├── performance_net/
│   ├── model.py
│   ├── data.py
│   ├── event_encoder.py
│   └── trainer.py
├── train.py
├── generate.py
├── visualize_training.py
├── extract_metrics.py
├── config.yaml
├── requirements.txt
└── final_training_metrics.png
```

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Entrenar

```bash
python train.py
```

El dataset esperado va en `data/`.

### Generar musica

```bash
python generate.py --checkpoint runs/20260105_010848/ \
    --output generated_pieces/piece.mid --num_steps 1536 --temperature 1.0
```

### Visualizar metricas

```bash
tensorboard --logdir runs/20260105_010848/tensorboard
```

## Resultados finales

| Metrica | Entrenamiento | Validacion |
| --- | --- | --- |
| Loss | 2.10 | 2.24 |
| Accuracy | 35.85% | 33.43% |
| Perplexity | 8.34 | 9.53 |

Entrenado durante 150 epochs en NVIDIA RTX 3070.
