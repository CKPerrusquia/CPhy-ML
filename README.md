# Uncovering Drone Intention using Control Physics Informed Machine Learning
This repository provides the data and code of the paper "Uncovering Drone Intentions using Control Physics Informed Machine Learning" 

![imagen](https://github.com/CKPerrusquia/CPhy-ML/assets/100733638/646939d2-0703-4ca4-85a6-da5c02116f37)

## Requirements
- python 3.9.0
- numpy 1.24.3
- matplotlib 3.7.1
- pandas 1.4.0
- jupyter 1.0.0 (optional: to run jupyter notebooks)
- tensorflow 2.10.1
- stonesoup 0.1b12
- scikit-learn 1.2.2

## Framework description
- DataGeneration
  - `DataPreprocessing.py` Methods to preprocess the input trajectories and split the data.
  - `Initiator.py` Basic Initiator for simulating radar measurements.
  - `MyDeleter.py` Method to delete tracks that are not useful.
  - `OffsetCoordinates.py` To move the ground truth trajectories with respect to the location of the radar.
  - `OutputStandardiser.py` Standardize the outputs for regression purposes.
  - `Simulator.py` Simulate flights in accordance with the location of the radar and noise intensity.
  - `Standardiser.py` Basic class to transform the input trajectories into adequate input data.
    
-  HybridClassifier
    - `Attention_Layer.py` Custom Attention Layer for the neural classifiers
    - `CLSTM_Attention_Classifier.py` Custom Convolutional Bidirectional LSTM with Attention Classifier
    - `CLSTM_Autoencoder.py` Custom Convolutional Bidirectional LSTM Autoencoder (Hybrid Classifier & Novelty Detection)
    - `ConvLSTM_Classifier.py` Custom Convolutional LSTM Classifier
    - `Convolutional_Autoencoder.py` Custom Convolutional Autoencoder for Novelty Detection
    - `Convolutional_Classifier.py` Convolutional Neural Network with/without Attention Classifier
    - `Convolutional_Classifier_Autoencoder.py` Custom Convolutional Classifier with Autoencoder (Hybrid Classifier & Novelty Detection)
    - `GRU_Classifier.py` Custom GRU Classifier
    - `RandomForest_Classifier.py` Custom Random Forest Classifier (Baseline)
    - `Transformer_Classifier.py` Custom Transformer Classifier (Not reported)
    - `Visualization.py` Basic class to visualise the results of the classifier and novelty detector

-  TrajectoryRegression
    - `CLSTM_Regressor.py` Custom Multi Input Convolutional Bidirectional LSTM Regressor
    - `CNN_Regressor.py` Custom Multi Input Convolutional Neural Network Regressor 
    - `LSTM_Regressor.py` Custom Multi Input LSTM Regressor
    - `Linear_Regressor.py` Custom Multi Input Linear Regressor
    - `MoE.py` Custom Mixture of Experts Network based on `CNN_Regressor.py` 
    - `Visualization.py` Basic class to visualise the results of the regression models and trajectory bounds

## Data
The `ResearchData` folder contains the data collected from open access sources that are relevant for the research. The complete datasets can be downloaded from
> Jason Whelan, Thanigajan Sangarapillai, Omar Minawi, Abdulaziz Almehmadi, Khalil El-Khatib, February 26, 2020, "UAV Attack Dataset", IEEE Dataport, doi: https://dx.doi.org/10.21227/00dg-0d12.

> Keipour, Azarakhsh; Mousaei, Mohammadreza; Scherer, Sebastian, 2020, "ALFA: A Dataset for UAV Fault and Anomaly Detection". Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/12707963.v1

> M. Street, 2021, “Drone identification and tracking,” 2021. Dataset. https://kaggle.com/competitions/icmcis-drone-tracking

> Rodrigues, Thiago A.; Patrikar, Jay; Choudhry, Arnav; Feldgoise, Jacob; Arcot, Vaibhav; Gahlaut, Aradhana; et al., 2020, "Data Collected with Package Delivery Quadcopter Drone". Carnegie Mellon University. Dataset. https://doi.org/10.1184/R1/12683453.v1

The `Matlab_ResearchData`contains custom flight data using a personal use drone. 

## Usage
1. Run the notebook `Synthetic_Data_Generation.ipynb` to generate synthetic data from `ResearchData`.
2. Use the notebook `Classifiers.ipynb` to train the trajectory intention classifiers. Follow the comments to obtain the results reported in the research.
3. Use the notebook `RegressionModels.ipynb` to train the trajectory intention regression models. Follow the comments to obtain the results reported in the research.
4. Run `Reservoir.m` to obtain the trajectory prediction results.
5. Run `DMD.m` to obtain linear representations of the drone dynamics and trajectory tracking.
6. Run `Model_based_Objective.m` to infer the reward function of drone's linear model. 
