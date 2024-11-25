# Stock Classifier and Analytics

[![CI](https://github.com/francisco3511/stocksense/actions/workflows/ci.yml/badge.svg)](https://github.com/francisco3511/stocksense/actions/workflows/ci.yml)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)


This project implements an intelligent dynamic stock selection system using a **Genetic Algorithm-optimized XGBoost** (GA-XGBoost) classifier to identify stocks with potential market outperformance. The model analyzes quarterly financial statements, market data, insider trading patterns and other external data to predict whether a stock will outperform the S&P 500 index over a one-year horizon over a large margin. The project includes a **Streamlit-based analytics dashboard** that provides comprehensive stock analysis tools, including technical indicators, financial metrics visualization, and model-driven insights.


## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Data Management](#data-management)
   - [Model Training](#model-training)
   - [Stocksense App](#streamlit-app)
5. [Contributing](#contributing)
6. [Acknowledgments & References](#acknowledgments-references)
7. [License](#license)

## Project Overview

The stock classifier is built using GA-XGBoost and trained on:
- Quarterly financial statements
- Market data and technical indicators
- Insider trading information
- Growth and valuation metrics

The model predicts whether a stock will outperform the S&P 500 over a one-year horizon. Key features include:
- Automated data collection and preprocessing pipeline
- SQLite database for efficient data storage
- Interactive Streamlit dashboard for analysis
- Genetic Algorithm optimization for feature selection


## Features

- **Model Training**
  - GA-XGBoost classifier with optimized hyperparameters
  - Feature engineering including growth ratios, financial metrics, price momentum, and volatility
  - Cross-validation and performance metrics

- **Streamlit App**
  - Market overview dashboard
  - Individual stock analysis with technical indicators
  - Financial ratio visualization
  - Insider trading patterns
  - Model predictions and insights

- **Data Management**
  - SQLite database with market, financial, and insider trading data
  - Automated data updates and validation
  - Historical S&P 500 constituent tracking

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-user/stocksense.git
   cd stocksense
   ```

2. Install dependencies using pyproject.toml:
   ```bash
   pip install .
   ```

### Development Setup

1. Install development dependencies:
   ```bash
   pip install -e ".[dev]"
   ```

2. Install pre-commit hooks:
   ```bash
   chmod +x scripts/install-hooks.sh
   ./scripts/install-hooks.sh
   ```

## Usage

### Data Management

The project uses a trading date observation window, which sets 4 portfolio rebalancing dates per year. The last trading date is used for model training and stock scoring by default.

First, update the stock database:
   ```bash
   stocksense --update
   ```

### Model Training

Train the model for a given trade date:
   ```bash
   stocksense --train --trade-date YYYY-MM-DD
   ```

Score stocks for a given trade date:
   ```bash
   stocksense --score --trade-date YYYY-MM-DD
   ```

### Streamlit App

To open the Streamlit app:
   ```bash
   stocksense --app
   ```


## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request

## Acknowledgments & References

This project's methodology was inspired by the following research:

Yang, H., Liu, X. Y., & Wu, Q. (2020). A Practical Machine Learning Approach for Dynamic Stock Recommendation. *Columbia University*. [[paper]](add_link_if_available)

Ye, Z. J., & Schuller, B. W. (2023). Capturing dynamics of post-earnings-announcement drift using a genetic algorithm-optimized XGBoost. *Imperial College London*. [[paper]](add_link_if_available)

Liu, X. Y., Yang, H., & Chen, Q. (2019). A Sustainable Quantitative Stock Selection Strategy Based on Dynamic Factor Adjustment. *Columbia University*. [[paper]](add_link_if_available)

```bibtex
@article{yang2020practical,
  title={A Practical Machine Learning Approach for Dynamic Stock Recommendation},
  author={Yang, Hongyang and Liu, Xiao-Yang and Wu, Qingwei},
  institution={Columbia University},
  year={2020}
}

@article{ye2023capturing,
  title={Capturing dynamics of post-earnings-announcement drift using a genetic algorithm-optimized XGBoost},
  author={Ye, Zhengxin Joseph and Schuller, Bj{\"o}rn W.},
  institution={Imperial College London},
  year={2023}
}

@article{liu2019sustainable,
  title={A Sustainable Quantitative Stock Selection Strategy Based on Dynamic Factor Adjustment},
  author={Liu, Xiao-Yang and Yang, Hongyang and Chen, Qingwei},
  institution={Columbia University},
  year={2019}
}
```


## License

This project is licensed under the MIT License.
