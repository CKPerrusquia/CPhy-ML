from Framework.HybridClassifier.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
import numpy as np
import pandas as pd
from time import time
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

class RandomForest:
    
    def Random_Forest(self, Xtrain, ytrain,
                      Xtest, ytest, Xval, yval,
                      WINDOW_SIZE, n_estimators = 20):
        
        # Initialize the plotter
        Plots = Results()
        rf_clf_start_time = time()
        
        # Instantiate the Random Forest Classifier
        rf_clf = RandomForestClassifier(n_estimators = n_estimators)
        
        # Train the classifier
        rf_clf.fit(Xtrain, ytrain)
        
        # Compute the training time
        rf_clf_trg_time = time()-rf_clf_start_time
        print(f"RF Clf training time: {rf_clf_trg_time}")
        
        # Predict under the Xval and Xtest datasets
        val_preds = rf_clf.predict(Xval)
        test_preds = rf_clf.predict(Xtest)
        
        print(f"{'-'*20} Validation Data {'-'*20}")
        print(classification_report(val_preds, yval, digits=4))
        print(f"\n{'-'*20} Test Data {'-'*20}")
        print(classification_report(test_preds, ytest, digits=4))
        
        # time how long it takes to make predictions on test set
        start_pred_time = time()
        test_preds = rf_clf.predict(Xtest)
        test_inference_time = time() - start_pred_time
        avg_pred_time = test_inference_time / Xtest.shape[0]

        print(f"Test Inference Time: {test_inference_time}")
        print(f"Individual Pred time: {avg_pred_time}")
        
        # Model results for saving
        model_results = Plots.get_model_results(yval, val_preds, ytest, test_preds, "RF Classifier")
        model_results['Training Time'] = rf_clf_trg_time
        model_results['Avg Pred Time'] = avg_pred_time
        model_results['Window Size'] = WINDOW_SIZE
        
        return rf_clf, model_results

    
    

