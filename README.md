# Drone_Intention
This repository provides the data and code of the paper "Uncovering Drone Intentions using Control Physics Informed Machine Learning" 

![imagen](https://github.com/CKPerrusquia/Drone_Intention/assets/100733638/c0297c59-0b2e-4b33-bc2d-ee6be1d8a65d)

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
  - `Attention_Layer.py`
  - `CLSTM_Attention_Classifier.py`
  - `CLSTM_Autoencoder.py`
  - `ConvLSTM_Classifier.py`
  - `Convolutional_Autoencoder.py`
  - `Convolutional_Classifier.py`
  - `Convolutional_Classifier_Autoencoder.py`
  - `GRU_Classifier.py`
  - `RandomForest_Classifier.py`
  - `Transformer_Classifier.py`
  - `Visualization.py`
-  TrajectoryRegression
  - `CLSTM_Regressor.py`
  - `CNN_Regressor.py`
  - `LSTM_Regressor.py`
  - `Linear_Regressor.py`
  - `MoE.py`
  - `Visualization.py`
