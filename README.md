# Forecasting Gold Prices Using Random Forest Regressor and Long Short Term Memory Models

## Research Question
Which model performs best for forecasting daily gold prices in the medium-term (63 days ahead):
Naive Forecast (Benchmark), Random Forest Regressor (RF), or Long short term memory (LSTM)?

## Setup

# Create environment
conda env create -f environment.yml
conda activate gold_forecasting

## Usage

python main.py

Expected output: Accuracy comparison between three models.

## Project Structure
```
my-project/
├── main.py              # Main entry point
├── data/                # Data directory
│   ├── raw/             # Raw CSV files downloaded from data sources
├── src/                 # Source code
│   ├── data_loader.py   # Data loading and preprocessing
│   ├── models.py        # Model training and hyperparameter tuning
│   └── evaluation.py    # Evaluation metrics
├── results/             # Generated outputs
│   ├── metrics/         # Evaluation metrics table
│   ├── figures/         # Prediction plots
│   └── appendix/        # Correlation matrix and hyperparameter tuning top 5 results
└── environment.yml      # Software dependencies
```

## Results
- Naive Forecast (Benchmark):
      MSE : 12115.33
      RMSE: 110.07
      MAE : 91.04
- Random Forest Regressor (RF):
      MSE : 10809.31
      RMSE: 103.97
      MAE : 90.10
- Long short-term memory (LSTM):
      MSE : 8966.35
      RMSE: 94.69
      MAE : 78.28
- Winner: Long short-term memory (LSTM) for all evaluation metrics!

## Requirements
- Python 3.12
- scikit-learn, pandas, numpy, matplotlib, seaborn, tensorflow