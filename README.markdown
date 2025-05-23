# Development of LSTM Based Model for Air Pollutants-Related Public Health Consequence Assessment

## Overview

This repository contains the code, data processing scripts, and documentation for the conference paper *"Development of LSTM Based Model for Air Pollutants-Related Public Health Consequence Assessment"* by Huawei Han and Wesley S. Burr, presented at the 6th International Conference on Statistics: Theory and Applications (ICSTA'24). The paper proposes a Long Short-Term Memory (LSTM) neural network model to assess the public health impacts of short-term exposure to multiple air pollutants, addressing limitations of traditional Generalized Additive Models (GAMs) in handling distributed lags and multi-pollutant interactions. The repository includes Python scripts for data preprocessing, LSTM modeling, and performance evaluation, using datasets from the National Morbidity, Mortality and Air Pollution Study (NMMAPS).

## Paper Description

The paper focuses on evaluating the adverse effects of short-term exposure to ambient air pollutants (e.g., PM₁₀, O₃) on public health outcomes, such as daily non-accidental mortality, in environmental epidemiology. It introduces an LSTM-based model to capture temporal dependencies and joint effects of multiple pollutants with distributed lags, offering improved flexibility over GAMs.

### Key Objectives

- **Data Preparation**: Preprocess NMMAPS datasets (Chicago, 1987–2000) for air pollutants (PM₁₀, O₃), temperature, and daily mortality, handling missing values and outliers.
- **LSTM Modeling**: Develop an LSTM model with weighted evaluation to assess health outcomes from multi-pollutant exposure sequences with distributed lags (1–8 days).
- **Performance Evaluation**: Compare model performance across different lag lengths using RMSE and MAE, assessing its ability to capture mortality trends.

### Methodology

1. **Data Preparation**:
   - **Source**: NMMAPS dataset (Chicago, Jan 1, 1987–Dec 31, 2000), including daily mean concentrations of PM₁₀, O₃, temperature, and non-accidental mortality.
   - **Preprocessing**:
     - Interpolated missing daily mean pollutant values.
     - Replaced outliers (e.g., mortality spike on Jul 15, 1995) with averages of adjacent days.
     - Standardized data to zero mean and unit variance (Eqn. 1).
     - Reorganized data into sequences of length \( m \) (1–8 days) for distributed lag analysis (Eqn. 2).
   - **Data Split**: 70% training, 30% testing.

2. **LSTM Model**:
   - Designed an LSTM network with 5 recurrent layers, 13 hidden features per layer, and a variable learning rate (initial 0.05, decaying 0.95 every 100 epochs).
   - Processed input sequences of pollutant exposures (PM₁₀, O₃, temperature) over \( m \) days using forget, input, and output gates.
   - Applied a softmax layer for weighted evaluation of LSTM outputs and a linear output layer to predict health outcomes.
   - Trained for 8000 epochs.

3. **Evaluation**:
   - Used RMSE and MAE to assess prediction accuracy on standardized datasets.
   - Compared performance for lag lengths \( m = 1 \) to \( m = 8 \).

4. **Tools and Libraries**:
   - **Python**: NumPy, Pandas, TensorFlow/Keras (v2.1.1+cu121), Matplotlib.
   - **R**: NMMAPSdata package for data access, renv for dependency management.
   - **Data Source**: NMMAPS database.

### Key Results

- **Model Performance** :
  - The LSTM model captured mortality fluctuation trends on both training and testing sets across all lag lengths.
  - Training set: RMSE and MAE decreased with increasing \( m \), with lowest errors at \( m = 5 \) (RMSE: 0.0000, MAE: 0.0000).
  - Testing set: Best performance at \( m = 5 \) (RMSE: 1.3949, MAE: 1.1171), but no clear downward trend due to noise and data distribution differences.
  - Longer lags (\( m \)) provided more information but risked overfitting due to noise.

- **Insights**:
  - The model effectively handled distributed lags and multi-pollutant effects, showing potential as an alternative to GAMs.
  - Limitations include sensitivity to noise, potential overfitting, and the need for hyperparameter tuning and denoising.

- **Conclusions**:
  - The LSTM model successfully captured accumulated impacts of air pollutant exposure, addressing GAMs' challenges with distributed lags and cross-pollutant effects.
  - Future improvements include denoising, feature selection, and finer hyperparameter tuning.


## Citation

If you use this code or findings, please cite:

> Han, H., Burr, W. S. (2024). "Development of LSTM Based Model for Air Pollutants-Related Public Health Consequence Assessment." *6th International Conference on Statistics: Theory and Applications (ICSTA'24)*.

## Funding

This research was partly funded by Health Canada.

## Contact

- **Huawei Han**: rebeccahan@trentu.ca
- **GitHub**: [github.com/RebeccaHuaweiHan](https://github.com/RebeccaHuaweiHan)

