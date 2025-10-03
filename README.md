# UFC Fight Prediction System

A comprehensive machine learning system for predicting UFC fight outcomes using historical fight data, advanced feature engineering, and XGBoost modeling.

## Features

- **Advanced ML Pipeline**: XGBoost-based prediction model with 60.6% accuracy
- **Recency Features**: Exponential decay weighting of recent performance (tau=180 days, last 7 fights)
- **Web Interface**: Flask-based web application for interactive predictions
- **Data Scraping**: Automated UFC stats collection from ufcstats.com
- **Feature Engineering**: 140+ engineered features including fighter stats, opponent analysis, and historical performance

## Quick Start

### Prerequisites

- Python 3.8+
- Conda (recommended) or pip
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/ufc-shamen.git
   cd ufc-shamen
   ```

2. **Create conda environment**
   ```bash
   conda create -n ufc-shamen python=3.11
   conda activate ufc-shamen
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the web application**
   ```bash
   python webapp/app.py
   ```

5. **Open your browser** to `http://127.0.0.1:5000`

## Model Performance

- **Test Accuracy**: 60.6%
- **F1 Score**: 70.4%
- **Features**: 140 engineered features
- **Training Data**: 8,172 historical fights (1994-2025)

### Top Features by Importance
1. Previous distance significant strikes landed (Fighter B)
2. Age difference
3. Reach (Fighter B)
4. **Recent win rate (Fighter A)** *Recency feature*
5. Previous significant strikes attempted (Fighter B)

## Architecture

```
ufc-shamen/
├── modeling/
│   ├── train_win_model.py      # Main training pipeline
│   ├── tune_recency.py         # Hyperparameter tuning
│   └── artifacts/
│       └── win_model.joblib    # Trained model
├── scrape_ufc_stats/           # Data collection
│   ├── scrape_ufc_stats_library.py
│   └── *.csv                   # UFC data files
├── webapp/
│   ├── app.py                  # Flask web application
│   ├── templates/
│   └── static/
└── requirements.txt
```

## Advanced Features

### Recency Weighting
The model uses exponential decay to weight recent fights more heavily:
- **Tau**: 180 days (optimal decay rate)
- **Memory Window**: Last 7 fights
- **Formula**: `weight = exp(-days_since_fight / tau)`

### Feature Engineering
- **Fighter Stats**: Physical attributes, historical performance
- **Opponent Analysis**: Head-to-head patterns, style matchups
- **Recency Features**: Recent performance trends
- **Fight Context**: Weight class, venue, etc.

## Usage

### Training a New Model

```bash
# Train with default parameters
python modeling/train_win_model.py

# Custom test split
python modeling/train_win_model.py --test-fraction 0.3

# Export dataset for analysis
python modeling/train_win_model.py --export-dataset data/matchups.csv
```

### Making Predictions

```bash
# Command line prediction
python modeling/train_win_model.py --predict-only --fighter-a "Jon Jones" --fighter-b "Stipe Miocic"

# Or use the web interface
python webapp/app.py
```

### Hyperparameter Tuning

```bash
# Tune recency parameters
python modeling/tune_recency.py
```

## Performance Optimization

The system is optimized for production use:
- **Fast Startup**: Basic data loading without expensive recency computation
- **On-Demand Features**: Recency features computed only when needed
- **Pre-trained Model**: Recency features embedded in trained XGBoost model
- **Efficient Caching**: Fighter data cached for quick name suggestions

## Development

### Data Collection
The scraping module collects data from ufcstats.com:
- Fight results and statistics
- Fighter profiles and career totals
- Event details and rankings

### Model Development
1. **Data Preparation**: Clean and engineer features
2. **Training**: XGBoost with cross-validation
3. **Evaluation**: Hold-out test set (chronological split)
4. **Optimization**: Hyperparameter tuning with Optuna

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## Disclaimer

This system is for educational and research purposes only. Sports betting involves risk, and past performance does not guarantee future results.

## Acknowledgments

- UFC Stats (ufcstats.com) for providing comprehensive fight data
- XGBoost team for the excellent gradient boosting framework
- Flask team for the lightweight web framework

---

**Built by Jerry using Python, XGBoost, and Flask**