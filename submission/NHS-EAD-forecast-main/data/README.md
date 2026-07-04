# Official data goes here

Place the two official contest CSVs in this folder (they are large / Git-LFS and
are not committed):

```
turingAI_forecasting_challenge_dataset.csv              (development data, to 30 Sep 2025)
turingAI_forecasting_challenge_validation_dataset.csv   (amended validation data, 1 Oct 2025 - 17 Feb 2026)
```

`src/data.py` resolves them from this folder. With both present, run the pipeline
from `../work/` as described in `../README.md`.
