# Music Transformer

Implementacion de Music Transformer con Relative Position Representations para generacion de musica de piano.

## Que incluye este repo

- arquitectura del modelo en `model/`
- dataset code en `dataset/e_piano.py`
- utilidades CLI y entrenamiento en `utilities/`
- scripts de entrenamiento, evaluacion, generacion y preprocesado
- metricas finales y trazas ligeras en `saved_models/results/`
- ejemplos ligeros en `examples/`

## Que no incluye

Para que el repositorio siga siendo ligero en GitHub, no se versionan:

- `dataset/maestro/` con los MIDIs brutos
- `dataset/e_piano/` con el dataset procesado
- `saved_models/weights/` con los pesos entrenados
- `saved_models/tensorboard/` con los logs
- lotes completos de `generated_pieces/`

## Estructura

```text
musictransformer/
├── model/
├── dataset/
│   └── e_piano.py
├── utilities/
├── third_party/midi_processor/
├── examples/
├── saved_models/
│   ├── model_params.txt
│   └── results/
├── train.py
├── evaluate.py
├── generate.py
├── preprocess_midi.py
└── final_training_metrics.png
```

## Instalacion

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

### Preprocesar dataset

```bash
python preprocess_midi.py dataset/maestro/maestro-v3.0.0 -output_dir dataset/e_piano
```

### Entrenar

```bash
python train.py -input_dir dataset/e_piano -output_dir saved_models --rpr
```

### Generar musica

```bash
python generate.py -output_dir generated_pieces/piece1 \
    -model_weights saved_models/weights/best_acc_weights.pickle --rpr
```

## Resultados finales

| Metrica | Entrenamiento | Validacion |
| --- | --- | --- |
| Loss | 1.88 | 2.04 |
| Accuracy | 43.74% | 40.67% |
| Perplexity | 6.57 | 7.71 |

Entrenado durante 100 epochs en NVIDIA RTX 3070.
