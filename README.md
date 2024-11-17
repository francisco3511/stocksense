# Stock Classifier and Analytics

[![CI](https://github.com/francisco3511/stocksense/actions/workflows/ci.yml/badge.svg)](https://github.com/francisco3511/stocksense/actions/workflows/ci.yml)

This project is a **machine learning stock classifier** that selects stocks based on quarterly financial and market data. The model aims to predict whether a stock will outperform the S&P 500 index over a one-year horizon. The project also includes a **Streamlit app** for stock analytics, where users can visualize stock metrics, growth ratios, and model predictions.

## Table of Contents

1. [Project Overview](#project-overview)
2. [Features](#features)
3. [Installation](#installation)
4. [Usage](#usage)
   - [Data Management](#data-management)
   - [Stocksense App](#streamlit-app)
   - [Model Training](#model-training)
5. [Data Source](#data-source)
6. [Directory Structure](#directory-structure)
7. [License](#license)

## Project Overview

The stock classifier is trained using financial ratios and growth features derived from **quarterly financial statements** and **market data**. The goal is to predict whether a stock will beat the **S&P 500** over the next year. The app also includes interactive analytics tools to explore stock metrics and model results.

## Features

- **Model Training**: A classifier using GA-XGBoost with features including growth ratios, financial metrics, price momentum, and volatility.
- **Streamlit App**: A web-based interface for exploring stock metrics, visualizing growth ratios, and viewing model predictions.
- **SQLite Database**: Locally stored market, financials, insider trading and status data for historical and current S&P500 members.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-user/stocksense.git
   cd stocksense
   ```

2.	Install dependencies using pyproject.toml:
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

To manually run all pre-commit hooks on all files:
```bash
pre-commit run --all-files
```
