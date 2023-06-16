import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, \
                             classification_report, precision_score, recall_score

class Results:
     
    def plot_confusion_matrix(self, true_y, pred_y, 
                          title='Confusion Matrix', figsize=(7,5)):
        """ Custom function for plotting a confusion matrix for predicted results 
    
        Args: 
            true_y (np.array) : true output class labels. 
            pred_y (np.array) : Output class predictions.
            title (str) : title of the plot.
            figsize (tuple) : figsize of the plot.
        """
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        
        conf_matrix = confusion_matrix(true_y, pred_y)
        conf_df = pd.DataFrame(conf_matrix, columns=np.unique(true_y), 
                               index = np.unique(true_y))
        conf_df.index.name = 'Actual Outputs'
        conf_df.columns.name = 'Predicted Outputs'
        plt.figure(figsize = figsize)
        plt.title(title, weight="bold")
        sns.set(font_scale=1.4)
        sns.heatmap(conf_df, cmap="Blues", annot=True, 
                    annot_kws={"size": 16}, fmt='g')
        plt.xlabel("Predicted", weight="bold")
        plt.ylabel("Actual", weight="bold")
        plt.grid()
        plt.show()
        return
    
    def classification_results(self, predictions, true_labels, model_name, show_results=True):
        """ Display confusion matrix, and return accuracy, precision, recall
            and F1 scores for the given predictions and labels """
    
        # calculate accuracy, f1, precision & recall
        accuracy = accuracy_score(predictions, true_labels)
        precision = precision_score(true_labels, predictions, average='macro')
        recall = recall_score(true_labels, predictions, average='macro')
        f1 = f1_score(true_labels, predictions, average='macro')
    
        # show confusion matrix
        if show_results:
            plot_confusion_matrix(predictions, true_labels)
    
            # show main statistics
            print(f"{model_name} Accuracy: {accuracy*100.0:.4f}%")
            print(f"{model_name} Precision Score: {precision:.4f}")
            print(f"{model_name} Recall: {recall:.4f}")
            print(f"{model_name} F1 Score: {f1:.4f}")
    
        return accuracy, precision, recall, f1
    
    def get_model_results(self, y_val, val_preds, y_test, test_preds, model_name):
        """ Generic helper function to get DNN Model Results 
            in dataframe format """
    
        # create dataframe to store all results from spoof predictions
        results_df = pd.DataFrame()

        # calculate accuracy metrics
        val_acc = accuracy_score(y_val, val_preds)
        test_acc = accuracy_score(y_test, test_preds)
    
        # calculate metric scores (using macro average)
        val_prec = precision_score(y_val, val_preds, average='macro')
        val_rec = recall_score(y_val, val_preds, average='macro')
        val_f1 = f1_score(y_val, val_preds, average='macro')
    
        # calculate metric scores (using macro average)
        test_prec = precision_score(y_test, test_preds, average='macro')
        test_rec = recall_score(y_test, test_preds, average='macro')
        test_f1 = f1_score(y_test, test_preds, average='macro')
    
        # produce dictionary of final metrics
        metrics = {"Model Name" : model_name, 
                   "Val Accuracy" : val_acc, 
                   "Val Precision" : val_prec,
                   "Val Recall" : val_rec,
                   "Val F1" : val_f1,
                   "Test Accuracy" : test_acc, 
                   "Test Precision" : test_prec,
                   "Test Recall" : test_rec,
                   "Test F1" : test_f1}
    
        return metrics
    
    def AccLoss(self,model_history):
        """ Generic function to plot Accuracy and Loss curves """
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
        fig, ax = plt.subplots(1,2, figsize=(14,5))
        model_history_df[['accuracy', 'val_accuracy']].plot(ax=ax[0], 
                                    color=['tab:blue', 'tab:orange'])
        ax[0].grid(0.5)
        ax[0].set_title("Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        model_history_df[['loss', 'val_loss']].plot(ax=ax[1],
                                    color=['tab:blue', 'tab:orange'])
        ax[1].grid(0.5)
        ax[1].set_title("Loss (Categorical Cross-Entropy)")
        ax[1].set_xlabel("Epochs", weight="bold")
        ax[1].set_ylabel("Loss (Categorical Cross-Entropy)")
        plt.show()
        return
    
    def Custom_AccLoss(self, model_history):
        """ Custom function to plot the accuracy and loss curves"""
        conv_ae_df = pd.DataFrame(model_history.history)
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)

        fig, ax = plt.subplots(2,2, figsize=(14,10))
        ax = ax.flatten()

        conv_ae_df[['clf_out_accuracy', 'val_clf_out_accuracy']].plot(ax=ax[0], 
                                 color=['tab:blue', 'tab:orange'])
        ax[0].grid(0.5)
        ax[0].set_title("Classification Accuracy")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("Accuracy")
        ax[0].legend(['Training', 'Validation'])
        ax[0].set_ylim([0.90, 0.985])

        conv_ae_df[['clf_out_loss', 'val_clf_out_loss']].plot(ax=ax[1],
                                    color=['tab:blue', 'tab:orange'])
        ax[1].grid(0.5)
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss (Categorical Cross-Entropy)")
        ax[1].legend(['Training', 'Validation'])

        conv_ae_df[['ae_out_mse', 'val_ae_out_mse']].plot(ax=ax[2], 
                color=['tab:blue', 'tab:orange'], label=['Training', 'Validation'])
        ax[2].grid(0.5)
        ax[2].set_title("Autoencoder Loss")
        ax[2].set_xlabel("Epochs")
        ax[2].set_ylabel("Loss (MSE)")
        ax[2].legend(['Training', 'Validation'])
        conv_ae_df['val_ae_out_mse'].plot(ax=ax[3], 
                color=['tab:orange'], label='Validation')
        ax[3].grid(0.5)
        ax[3].set_title("Autoencoder Validation Loss")
        ax[3].set_xlabel("Epochs")
        ax[3].set_ylabel("Loss (MSE)")
        plt.tight_layout()
        plt.show()
        return
    
    def RMSE(self, model):
        """ Computes the RMSE of the reconstructions"""
        model_history_df = pd.DataFrame(model.history)
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)

        fig, ax = plt.subplots(1,2, figsize=(14,5))

        model_history_df[['val_root_mean_squared_error']].plot(ax=ax[0], 
                                    color=['tab:blue', 'tab:orange'])
        ax[0].grid(0.5)
        ax[0].set_title("Validation Mean Squared Error")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("RMSE")

        model_history_df[['loss', 'val_loss']].plot(ax=ax[1],
                                    color=['tab:blue', 'tab:orange'])
        ax[1].grid(0.5)
        ax[1].set_title("Training Losses (MSE)")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss (MSE)")
        plt.show()
        return
    
    def MSE(self, model):
        """ Shows the MSE results of the network"""
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        
        model_history_df = pd.DataFrame(model.history)

        fig, ax = plt.subplots(1,2, figsize=(14,5))

        model_history_df[['val_mse']].plot(ax=ax[0], 
                                    color=['tab:blue', 'tab:orange'])
        ax[0].grid(0.5)
        ax[0].set_title("Mean Squared Error")
        ax[0].set_xlabel("Epochs")
        ax[0].set_ylabel("MSE")

        model_history_df[['loss', 'val_loss']].plot(ax=ax[1],
                                    color=['tab:blue', 'tab:orange'])
        ax[1].grid(0.5)
        ax[1].set_title("Loss")
        ax[1].set_xlabel("Epochs")
        ax[1].set_ylabel("Loss")
        plt.show()
        return
    
    def plot_recon_results(self, idx, X, y, X_recons, trg_recon_mse, 
                           val_recon_mse, figsize=(9,5)):
        """ Helper function for visualising novelty detection results """
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 16) 
        plt.rc('ytick', labelsize = 16)
        plt.rcParams['font.size'] = 16 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 16, labelsize = 16)
        # get true intent label and predicted intent labels
        intent_type = str(y[idx])
    
        fig, ax = plt.subplots(1,2, figsize=figsize)
        ax = ax.flatten()
    
        # plot input trajectory scatterplot
        ax[0].scatter(X[idx, 0, :], X[idx, 1, :], 
                    color="tab:blue", label="Input Track")
        ax[0].plot(X[idx, 0, :], X[idx, 1, :], 
                 alpha=0.3, color="tab:blue")

        ax[0].scatter(X[idx, 0, 0], X[idx, 1, 0],
                    label='Start', marker='*', color='tab:green', s=200)

        ax[0].scatter(X[idx, 0, -1], X[idx, 1, -1],
                    label='End', marker='*', color='tab:red', s=200)

        ax[0].set_title(f"Intent: {intent_type}",
                        weight="bold")
        ax[0].set_xlabel('$X$ (m)')
        ax[0].set_ylabel('$Y$ (m)')
        ax[0].grid(0.5)
        ax[0].legend(loc='best')
    
        # plot reconstruction errors
        pred_recon_mse = np.mean(np.power(X[idx] - X_recons[idx], 2), 
                                 axis=(0,1))
        xlabels = ["Train MSE", "Val MSE", "Pred MSE"]
        ax[1].bar(xlabels, height=[trg_recon_mse, val_recon_mse, pred_recon_mse])
        ax[1].set_ylabel('Reconstruction MSE', weight='bold')
        ax[1].set_title(f"Novelty Detection Assessment", weight="bold")
        plt.show()
        return
    
    def get_avg_recon_results(self, preds, true):
        """ Get MSE for each feature in reconstructions, and average """
        recon_mses = np.mean(np.power(true - preds, 2), axis=(1,2)).mean()
        recon_stds = np.mean(np.power(true - preds, 2), axis=(1,2)).std()
        return recon_mses, recon_stds
    
    def get_avg_mse(self, preds, true):
        """ Get MSE for each feature in reconstructions, and average """
        return np.mean(np.power(true - preds, 2), axis=(0,1)).mean()
        
    def plot_classification_results(self, idx, index, X_non_std, X, y, preds, X_recons, trg_recon_mse, 
                                    val_recon_mse, ID_TO_INTENT_MAP,
                                    UAV_INTENT_MAP, figsize=(21,4)):
        """ Helper function for visualising classification results """
        # get true intent label and predicted intent labels
        sns.set_style('white')
        plt.rcParams["font.family"] = "Times New Roman"
        plt.rcParams['text.usetex'] = True
        plt.rc('xtick', labelsize = 20) 
        plt.rc('ytick', labelsize = 20)
        plt.rcParams['font.size'] = 20 
        plt.rcParams['lines.linewidth'] = 2
        plt.rc('savefig', dpi=300)
        plt.rc('axes', titlesize = 22, labelsize = 20)
        
        intent_type = str(y[idx])
        pred_probs = preds[idx].copy()
        pred_intent = ID_TO_INTENT_MAP[np.argmax(pred_probs)]
    
        fig, ax = plt.subplots(1,3, figsize=figsize)
        ax = ax.flatten()
    
        # plot input trajectory scatterplot
        ax[0].scatter(X_non_std[idx, 0, :], X_non_std[idx, 1, :], 
                    color="tab:blue", label="Input Track")
        ax[0].plot(X_non_std[idx, 0, :], X_non_std[idx, 1, :], 
                 alpha=0.3, color="tab:blue")

        ax[0].scatter(X_non_std[idx, 0, 0], X_non_std[idx, 1, 0],
                    label='Start', marker='*', color='tab:green', s=200)

        ax[0].scatter(X_non_std[idx, 0, -1], X_non_std[idx, 1, -1],
                    label='End', marker='*', color='tab:red', s=200)

        ax[0].set_title(f"True Intent: {intent_type}\nPredicted Intent: {pred_intent}")
        ax[0].set_xlabel('$X$ (m)')
        ax[0].set_ylabel('$Y$ (m)')
        ax[0].grid(0.5)
        ax[0].legend(loc='best')
    
        # plot barplot of softmax prob scores for intent predictions
        ax[1].bar(UAV_INTENT_MAP.keys(), height=pred_probs,
                 color=['tab:blue', 'tab:green', 'tab:orange', 'tab:red'])
        ax[1].xaxis.set_ticks(list(UAV_INTENT_MAP.values()))
        ax[1].set_xticklabels(UAV_INTENT_MAP.keys(), rotation = 45)
        ax[1].set_ylabel('Class Probability')
        ax[1].set_title(f"Output class probability distribution")
        ax[1].set_ylim(0, 1.0)
        ax[1].grid(0.5)
        
        # plot reconstruction errors
        pred_recon_mse = np.mean(np.power(X[idx] - X_recons[idx], 2), 
                                 axis=(0,1))
        xlabels = ["Train MSE", "Val MSE", "Pred MSE"]
        ax[2].bar(xlabels, height=[trg_recon_mse, val_recon_mse, pred_recon_mse], 
                  color=['tab:green', 'tab:orange', 'tab:red'])
        ax[2].set_ylabel('Reconstruction MSE')
        ax[2].set_title(f"Novelty Detection Assessment")
        ax[2].grid(0.5)
        model_name = f"f{index}.eps"
        ax[2].figure.savefig(model_name, format='eps', bbox_inches = 'tight', dpi=1200)
        plt.show()
        return
   