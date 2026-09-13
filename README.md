# RoadAI Road Extraction with U-Net

Memory-efficient TensorFlow/Keras road segmentation for the Massachusetts Roads Dataset. Images are paired with masks by exact filename stem and decoded lazily from TIFF/GeoTIFF files using Rasterio, with OpenCV fallback.

## Dataset layout

The configured DeepGlobe layout is `dataset/train`, `dataset/valid`, and `dataset/test`.
`valid`/`test` images may be unlabeled; in that case a deterministic 15% holdout from
the labelled training set is used for validation. The test set is never used for tuning.

## Run

```powershell
pip install -r requirements.txt
python train.py
python evaluate.py --model models/best_model_improved.keras --split validation
python predict.py
```

`train.py` verifies paired data before training, performs reproducible moderate paired
augmentation, and trains a candidate only. It never overwrites the production model:
the candidate is saved to `models/best_model_improved.keras` and its validation-selected
threshold is stored alongside it. Dataset and threshold reports are written to `results/`.

Compare checkpoints on a labelled split:

```powershell
python evaluate.py --compare --split validation
```

After reviewing `results/model_comparison.csv`, opt into the candidate for the Flask
backend (the default remains `models/best_model.keras`):

```powershell
$env:ROADAI_MODEL = "models/best_model_improved.keras"
python backend/app.py
```

The selected candidate threshold and optional conservative post-processing settings are
loaded automatically from its `.threshold.json` sidecar. Remove `ROADAI_MODEL` or open a
new terminal to return to the production checkpoint.
