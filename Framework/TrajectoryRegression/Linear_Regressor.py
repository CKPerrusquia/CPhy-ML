from Framework.TrajectoryRegression.Visualization import Results
from Framework.DataGeneration.Standardiser import TrajectoryStandardiser
import numpy as np
import pandas as pd
from time import time
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

class Linear_Regression:
    
    def LinearRegressor(self, Xtrain, ytrain, Xval, yval, Xtest, ytest, WINDOW_SIZE):
        
        Plots = Results()
        lin_reg = LinearRegression()
        lr_start_time = time()
        
        # train model on simplified trajectory input features
        lin_reg.fit(Xtrain, ytrain)

        lr_trg_time = time() - lr_start_time
        print(f"Linear Regression training time: {lr_trg_time}")
        
        val_preds = lin_reg.predict(Xval)
        
        start_pred_time = time()
        test_preds = lin_reg.predict(Xtest)
        test_inference_time = time() - start_pred_time
        avg_pred_time = test_inference_time / Xtest.shape[0]

        print(f"Test Inference Time: {test_inference_time}")
        print(f"Individual Pred time: {avg_pred_time}")
        
        model_results = Plots.get_model_results(yval, val_preds, 
                                                ytest, test_preds, 
                                                "Linear Regression")
        model_results['Training Time'] = lr_trg_time
        model_results['Avg Pred Time'] = avg_pred_time
        model_results['Window Size'] = WINDOW_SIZE
        
        # Computation of the metrics
        val_rmse = mean_squared_error(yval, val_preds, squared = False)
        test_rmse = mean_squared_error(ytest, test_preds, squared = False)

        val_mae = mean_absolute_error(yval, val_preds)
        test_mae = mean_absolute_error(ytest, test_preds)

        val_r2 = r2_score(yval, val_preds)
        test_r2 = r2_score(ytest, test_preds)
        
        print("Validation Data:")
        print(f"    - RMSE: {val_rmse:.4f}")
        print(f"    - MAE: {val_mae:.4f}")
        print(f"    - R^2: {val_r2:.4f}")

        print("\nTest Data:")
        print(f"    - RMSE: {test_rmse:.4f}")
        print(f"    - MAE: {test_mae:.4f}")
        print(f"    - R^2: {test_r2:.4f}")
        
        return model_results, val_preds, test_preds
        
        
        
    

