import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class Results:
    
    def get_model_results(self, y_val, val_preds, y_test, 
                          test_preds, model_name):
        """ Generic helper function to get DNN Model Regression
            Results in dictionary format """
    
        # create dataframe to store all results from spoof predictions
        results_df = pd.DataFrame()

        # calculate rmse metrics
        val_rmse = mean_squared_error(y_val, val_preds, squared=False)
        test_rmse = mean_squared_error(y_test, test_preds, squared=False)
    
        # calculate mae metrics
        val_mae = mean_absolute_error(y_val, val_preds)
        test_mae = mean_absolute_error(y_test, test_preds)
    
        # calculate mae metrics
        val_r2 = r2_score(y_val, val_preds)
        test_r2 = r2_score(y_test, test_preds)
    
        # produce dictionary of final metrics
        metrics = {"Model Name" : model_name, 
                   "Val RMSE" : val_rmse, 
                   "Val MAE" : val_mae,
                   "Val R2" : val_r2,
                   "Test RMSE" : test_rmse, 
                   "Test MAE" : test_mae,
                   "Test R2" : test_r2}
    
        return metrics
    
    def RMSE_plots(self, model_history):
        # Plot the RMSE results
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        
        model_history_df = pd.DataFrame(model_history.history)
        fig, ax = plt.subplots(1,2, figsize = (14,5))
        model_history_df['root_mean_squared_error'].plot(ax = ax[0], 
                                   color=['tab:red'])
        ax[0].grid(0.5)
        ax[0].set_title("Training Root Mean Squared Error (RMSE)")
        ax[0].set_xlabel("Epochs", weight = "bold")
        ax[0].set_ylabel("RMSE", weight = "bold")


        model_history_df['val_root_mean_squared_error'].plot(ax = ax[1],
                                    color = ['tab:blue'])
        ax[1].grid(0.5)
        ax[1].set_title("Validation Root Mean Squared Error (RMSE)")
        ax[1].set_xlabel("Epochs", weight = "bold")
        ax[1].set_ylabel("RMSE", weight = "bold")
        plt.show()
        return
    
    def RMSE_DMoE(self, moe_model):
        """ RMSE plot for the DMoE"""
        
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        
        fig, ax = plt.subplots(2,2, figsize=(12,10))
        ax = ax.flatten()

        best_mins = []

        for i in range(moe_model.n_experts):
            temp_df = pd.DataFrame(moe_model.trg_histories[i])
            expert_name = moe_model.expert_names[i]
            temp_df[['root_mean_squared_error', 'val_root_mean_squared_error']].plot(ax=ax[i], 
                                                                                     color=['tab:blue', 'tab:orange'])
    
            ax[i].grid(0.5)
            ax[i].set_title(f"Expert: {expert_name}", weight="bold")
            ax[i].set_xlabel("Epochs", weight="bold")
            ax[i].set_ylabel("RMSE", weight="bold")
    
            best_val = temp_df['root_mean_squared_error'].min()
            best_mins.append(best_val)
            
        plt.savefig('RMSEDMoE.eps', format='eps', bbox_inches = 'tight', dpi=1200)
        plt.show()
        return
    
    def RMS_pred(self, moe_model):
        """ Helper to visualize the RMS for predictions"""
        
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        fig, ax = plt.subplots(4,2, figsize=(8,7))
        ax = ax.flatten()

        best_mins = []

        for i in range(moe_model.n_experts):
            temp_df = pd.DataFrame(moe_model.trg_histories[i])
            expert_name = moe_model.expert_names[i]
    
            trg_ax = int(2*i)
            val_ax = int((2*i) + 1)
    
            temp_df['root_mean_squared_error'].plot(ax=ax[trg_ax], 
                                       color='tab:blue')
    
            temp_df['val_root_mean_squared_error'].plot(ax=ax[val_ax], 
                                    color='tab:orange')
    
            ax[trg_ax].grid(0.5)
            ax[val_ax].grid(0.5)
            ax[trg_ax].set_title(f"Expert: {expert_name} (Training)", 
                            weight="bold")
            ax[val_ax].set_title(f"Expert: {expert_name} (Validation)", 
                              weight="bold")
            ax[trg_ax].set_xlabel("Epochs", weight="bold")
            ax[trg_ax].set_ylabel("RMSE", weight="bold")
            ax[val_ax].set_xlabel("Epochs", weight="bold")
            ax[val_ax].set_ylabel("RMSE", weight="bold")
    
            best_val = temp_df['val_root_mean_squared_error'].min()
            best_mins.append(best_val)
        plt.tight_layout()
        plt.savefig('RMSDMoE.eps', format='eps', bbox_inches = 'tight', dpi=1200)
        plt.show()
        return
    
    #self, idx, index, X, y, y_traj, preds, bound_idx=0,
    #                        UAV_TYPE_IDX, UAV_INTENT_IDX, timestep=None,
    #                        figsize=(7,5), legend=True, anomaly=False
    def plot_bounds_results(self, idx, index, X, y, y_traj, preds,
                            UAV_TYPE_IDX, UAV_INTENT_IDX, bound_idx = 0,
                            timestep = None, figsize = (7,5), legend = True,
                            anomaly = False):
        
        """ Helper function for visualising bound results """
        
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        
        uav_type = X[idx, UAV_TYPE_IDX, 0]
        intent_type = X[idx, UAV_INTENT_IDX, 0]
    
        # only use up to specified timestep if given
        if timestep is not None:
            y_traj_ = y_traj[idx, :, :timestep].copy()
            t_bounds = timestep
        else:
            y_traj_ = y_traj[idx, :, :].copy()
            t_bounds = y_traj.shape[2]
        
        # get associated preds for bound number chosen
        bound_preds = preds[idx, 6*bound_idx : 6*(bound_idx + 1)]
        true_bounds = y[idx, 6*bound_idx : 6*(bound_idx + 1)]
    
        plt.figure(figsize=figsize)
        plt.scatter(X[idx, 0, :], X[idx, 1, :], 
                    color="tab:blue", label="Current Track")
        plt.plot(X[idx, 0, :], X[idx, 1, :], 
                 alpha=0.3, color="tab:blue")

        plt.scatter(y_traj_[0, :], y_traj_[1, :], 
                    color="tab:orange", label="Future Trajectory")
        plt.plot(y_traj_[0, :], y_traj_[1, :], 
                 alpha=0.3, color="tab:orange")

        plt.scatter(X[idx, 0, 0], X[idx, 1, 0],
                    label='Start', marker='*', color='tab:green', s=200)

        plt.scatter(y_traj_[0, -1], y_traj_[1, -1],
                    label='End', marker='*', color='tab:red', s=200)

        # plot maximum and minimum limits of lookahead trajectory
        min_x = true_bounds[0] + X[idx, 0, -1]
        min_y = true_bounds[1] + X[idx, 1, -1]
        max_x = true_bounds[3] + X[idx, 0, -1]
        max_y = true_bounds[4] + X[idx, 1, -1]

        #print(f"\nMinimum and maximum bound labels: \n{y[idx]}")
        #print(f"\nLast input timestep co-ords: {X[idx, 0, -1]}, {X[idx, 1, 0]}")
        #print(f"\nx_min, y_min, x_max, y_max:\n{min_x}, {min_y}, {max_x}, {max_y}")

        # plot ground-truth bounds
        plt.hlines(min_y, min_x, max_x, color="tab:green", alpha=0.5)
        plt.hlines(max_y, min_x, max_x, color="tab:green", alpha=0.5)
        plt.vlines(min_x, min_y, max_y, color="tab:green", alpha=0.5)
        plt.vlines(max_x, min_y, max_y, color="tab:green", alpha=0.5, 
                   label="True Bounds")
    
        # if prediction is anomalous, add safety factor
        if anomaly:
            safety_factor = 1.5
        else:
            safety_factor = 1.0
    
        # extract predictions and correct co-ords
        pred_x_min = (safety_factor*bound_preds[0]) + X[idx, 0, -1]
        pred_y_min = (safety_factor*bound_preds[1]) + X[idx, 1, -1]
        pred_x_max = (safety_factor*bound_preds[3]) + X[idx, 0, -1]
        pred_y_max = (safety_factor*bound_preds[4]) + X[idx, 1, -1]

        # plot bounds for predictions
        plt.hlines(pred_y_min, pred_x_min, pred_x_max, color="tab:red", alpha=0.5)
        plt.hlines(pred_y_max, pred_x_min, pred_x_max, color="tab:red", alpha=0.5)
        plt.vlines(pred_x_min, pred_y_min, pred_y_max, color="tab:red", alpha=0.5)
        plt.vlines(pred_x_max, pred_y_min, pred_y_max, color="tab:red", alpha=0.5, 
                   label="Predicted Bounds")

        plt.title(f"Flight Intent: {intent_type}\nUAV Type: {uav_type}\nBounds Time: {t_bounds} s", 
                  weight="bold")
        plt.xlabel('$X$ (m)', weight='bold')
        plt.ylabel('$Y$ (m)', weight='bold')
        plt.grid(0.5)
        if legend:
            plt.legend(loc="best")
            
        model_name = f"g{index}.eps"
        plt.savefig(model_name, format='eps', bbox_inches = 'tight', dpi=1200)
        plt.show()
        return
    
    
    def plot_bounds_results_3d(self, idx, index, X, y, y_traj, preds, 
                               UAV_TYPE_IDX, UAV_INTENT_IDX, bound_idx=0,
                               timestep=None, figsize=(7,5), 
                               legend=True, anomaly=False, init_alt=50):
        
        """ Helper function for visualising bound results """
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        
        uav_type = X[idx, UAV_TYPE_IDX, 0]
        intent_type = X[idx, UAV_INTENT_IDX, 0]
    
        # only use up to specified timestep if given
        if timestep is not None:
            y_traj_ = y_traj[idx, :, :timestep].copy()
            t_bounds = timestep
        else:
            y_traj_ = y_traj[idx, :, :].copy()
            t_bounds = y_traj.shape[2]
        
        # get associated preds for bound number chosen
        bound_preds = preds[idx, 6*bound_idx : 6*(bound_idx + 1)]
        true_bounds = y[idx, 6*bound_idx : 6*(bound_idx + 1)]
    
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(projection='3d')
    
        ax.scatter(X[idx, 0, :], X[idx, 1, :], X[idx, 2, :] + init_alt,
                    color="tab:blue", label="Current Track")
        ax.plot(X[idx, 0, :], X[idx, 1, :], X[idx, 2, :] + init_alt,
                 alpha=0.3, color="tab:blue")

        ax.scatter(y_traj_[0, :], y_traj_[1, :], y_traj_[2, :] + init_alt,
                    color="tab:orange", label="Future Trajectory")
        ax.plot(y_traj_[0, :], y_traj_[1, :], y_traj_[2, :] + init_alt,
                 alpha=0.3, color="tab:orange")

        ax.scatter(X[idx, 0, 0], X[idx, 1, 0], X[idx, 2, 0] + init_alt,
                    label='Start', marker='*', color='tab:green', s=200)

        ax.scatter(y_traj_[0, -1], y_traj_[1, -1], y_traj_[2, -1] + init_alt,
                    label='End', marker='*', color='tab:red', s=200)

        # plot maximum and minimum limits of lookahead trajectory
        min_x = true_bounds[0] + X[idx, 0, -1]
        min_y = true_bounds[1] + X[idx, 1, -1]
        min_z = true_bounds[2] + X[idx, 2, -1] + init_alt
    
        max_x = true_bounds[3] + X[idx, 0, -1]
        max_y = true_bounds[4] + X[idx, 1, -1]
        max_z = true_bounds[5] + X[idx, 2, -1] + init_alt
    
        # create co-ords to plot 3d bounding box of true bounds
        true_box_xs = [min_x, max_x, max_x, min_x, min_x, min_x,
                       min_x, min_x, min_x,
                       max_x, max_x, max_x,
                       max_x, max_x, max_x,
                       min_x, min_x, min_x,
                       min_x, min_x]
    
        true_box_ys = [min_y, min_y, max_y, max_y, min_y, min_y,
                       min_y, min_y, min_y,
                       min_y, min_y, min_y,
                       max_y, max_y, max_y,
                       max_y, max_y, max_y,
                       min_y, min_y]
        
        true_box_zs = [min_z, min_z, min_z, min_z, min_z, max_z,
                       max_z, min_z, max_z,
                       max_z, min_z, max_z,
                       max_z, min_z, max_z,
                       max_z, min_z, max_z,
                       max_z, max_z]
        
        ax.plot(true_box_xs, true_box_ys, true_box_zs, color="tab:green", 
                alpha=0.4, label="True Bounds")
        
        # if prediction is anomalous, add safety factor
        if anomaly:
            safety_factor = 1.5
        else:
            safety_factor = 1.0
    
        pred_min_x = (safety_factor*bound_preds[0]) + X[idx, 0, -1]
        pred_min_y = (safety_factor*bound_preds[1]) + X[idx, 1, -1]
        pred_min_z = (safety_factor*bound_preds[2]) + X[idx, 2, -1] + init_alt
    
        pred_max_x = (safety_factor*bound_preds[3]) + X[idx, 0, -1]
        pred_max_y = (safety_factor*bound_preds[4]) + X[idx, 1, -1]
        pred_max_z = (safety_factor*bound_preds[5]) + X[idx, 2, -1] + init_alt
    
        # create co-ords to plot 3d bounding box of true bounds
        pred_box_xs = [pred_min_x, pred_max_x, pred_max_x, pred_min_x, pred_min_x, pred_min_x,
                       pred_min_x, pred_min_x, pred_min_x,
                       pred_max_x, pred_max_x, pred_max_x,
                       pred_max_x, pred_max_x, pred_max_x,
                       pred_min_x, pred_min_x, pred_min_x,
                       pred_min_x, pred_min_x]
        
        pred_box_ys = [pred_min_y, pred_min_y, pred_max_y, pred_max_y, pred_min_y, pred_min_y,
                       pred_min_y, pred_min_y, pred_min_y,
                       pred_min_y, pred_min_y, pred_min_y,
                       pred_max_y, pred_max_y, pred_max_y,
                       pred_max_y, pred_max_y, pred_max_y,
                       pred_min_y, pred_min_y]
    
        pred_box_zs = [pred_min_z, pred_min_z, pred_min_z, pred_min_z, pred_min_z, pred_max_z,
                       pred_max_z, pred_min_z, pred_max_z,
                       pred_max_z, pred_min_z, pred_max_z,
                       pred_max_z, pred_min_z, pred_max_z,
                       pred_max_z, pred_min_z, pred_max_z,
                       pred_max_z, pred_max_z]
    
        ax.plot(pred_box_xs, pred_box_ys, pred_box_zs, color="tab:red", 
                alpha=0.4, label="Pred Bounds")

        ax.set_title(f"Flight Intent: {intent_type}\nUAV Type: {uav_type}\nBounds Time: {t_bounds} s", 
                  weight="bold")
        ax.set_xlabel(' $X$ (m)')
        ax.set_ylabel(' $Y$ (m)')
        ax.set_zlabel(' $Z$ m)')
        
        # also set z limits to better represent scale of plots
        pred_box_zs = np.array(pred_box_zs)
        avg_pred_box_z = pred_box_zs.mean()
        min_pred_box_z = min(pred_box_zs.min(), 0.0)
        max_pred_box_z = pred_box_zs.max()
        diff_pred_box_z = max_pred_box_z - min_pred_box_z
        ax.set_zlim(min_pred_box_z, max_pred_box_z + diff_pred_box_z / 4.0)
    
        ax.grid(0.5)
        if legend:
            plt.legend(loc="best")

        model_name = f"g{index}.eps"
        ax.figure.savefig(model_name, format='eps', bbox_inches = 'tight', dpi=1200)
        plt.show()
        return

