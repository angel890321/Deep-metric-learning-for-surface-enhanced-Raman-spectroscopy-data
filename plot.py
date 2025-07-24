import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.manifold import TSNE
from boxsers._boxsers_utils import _lightdark_switch
from sklearn.metrics import classification_report, confusion_matrix
import torch
from sklearn.metrics import (
    auc,
    roc_curve,
)
from config import ORDER, STRAINS
import pandas as pd
import plotly.express as px

def plot_10_genus_ROC_curve(save_name, y_true, y_test, y_pred_prob):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['Acinetobacter', 'Enterobacter','Enterococcus','Escherichia','Klebsiella','Staphylococcus','Streptococcus','Salmonella','Pseudomonas','CoNS']

    for i in range(np.unique(y_true).shape[0]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"ROC_curve_{save_name}.png")
    plt.close()

def plot_MRSA_MSSA_ROC_curve(save_name, y_true, y_test, y_pred_prob):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['MRSA','MSSA']

    for i in range(np.unique(y_true).shape[0]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"ROC_curve_{save_name}.png")
    plt.close()

def plot_VRE_VSE_ROC_curve(save_name, y_true, y_test, y_pred_prob):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['VRE','VSE']

    for i in range(np.unique(y_true).shape[0]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"ROC_curve_{save_name}.png")
    plt.close()

def plot_CR_Ecoli_CS_Ecoli_ROC_curve(save_name, y_true, y_test, y_pred_prob):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['CR Ecoli','CS Ecoli']

    for i in range(np.unique(y_true).shape[0]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"ROC_curve_{save_name}.png")
    plt.close()
    
def plot_CR_K_pneumonia_CS_K_pneumonia_ROC_curve(save_name, y_true, y_test, y_pred_prob):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['CR K_pneumonia','CS K_pneumonia']

    for i in range(np.unique(y_true).shape[0]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"ROC_curve_{save_name}.png")
    plt.close()

def plot_E_cloacae_CS_E_cloacae_ROC_curve(save_name, y_true, y_test, y_pred_prob):
    plt.figure(figsize=(20, 12))
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(np.unique(y_true).shape[0]):
        fpr[i], tpr[i], _ = roc_curve(y_test == i, y_pred_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    label = ['CR E_cloacae','CS E_cloacae']

    for i in range(np.unique(y_true).shape[0]):
        plt.plot(
            fpr[i],
            tpr[i],
            label="ROC curve of class {} (area = {:.2f})"
            "".format(label[i], roc_auc[i]),
        )

    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Receiver Operating Characteristic")

    plt.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(f"ROC_curve_{save_name}.png")
    plt.close()

    
def plot_heatmap(save_name,conf_matrix , normalize='true', class_names=None, title=None,
              xlabel='Predicted label', ylabel='True label', darktheme=False,
              color_map='Blues', fmt='.0f', fontsize=16, fig_width=15, fig_height=12,
              save_path=None):

    

    #conf_matrix = confusion_matrix(y_true, y_pred, normalize=normalize)  # return a sklearn conf. matrix

    # update theme related parameters
    frame_color, bg_color, alpha_value = _lightdark_switch(darktheme)
    # creates a figure object
    fig = plt.figure(figsize=(fig_width, fig_height))
    # add an axes object
    ax = fig.add_subplot(1, 1, 1)  # nrows, ncols, index
    # plot a Seaborn heatmap with the confusion matrix
    sns.heatmap(conf_matrix, annot=True, cmap=color_map, fmt=fmt, cbar=False, annot_kws={"fontsize": fontsize},
                square=True, )

    # titles settings
    ax.set_title(title, fontsize=fontsize + 1.2, color=frame_color)  # sets the plot title, 1.2 points larger font size
    ax.set_xlabel(xlabel, fontsize=fontsize, color=frame_color)  # sets the x-axis title
    ax.set_ylabel(ylabel, fontsize=fontsize, color=frame_color)  # sets the y-axis title

    # tick settings
    ax.tick_params(axis='both', which='major',
                   labelsize=fontsize - 2,  # 2.0 points smaller font size
                   color=frame_color)
    ax.tick_params(axis='x', colors=frame_color)  # setting up X-axis values color
    ax.tick_params(axis='y', colors=frame_color)  # setting up Y-axis values color

    for _, spine in ax.spines.items():
        # adds a black outline to the confusion matrix and setting up spines color
        spine.set_visible(True)
        spine.set_color(frame_color)

    if class_names is not None:
        # sets the xticks labels at an angle of 45°
        ax.set_xticklabels(class_names, rotation=45, ha="right", rotation_mode="anchor",
                           fontsize=fontsize-1.3,  # 1.2 points smaller font size
                           color=frame_color)
        # sets the yticks labels vertically
        ax.set_yticklabels(class_names, rotation=0, fontsize=fontsize-2, color=frame_color)

    # set figure and axes facecolor
    fig.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    # adjusts subplot params so that the subplot(s) fits in to the figure area
    fig.tight_layout()
    plt.savefig(f"heatmap_{save_name}.png")
    plt.close()


from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# 固定的顏色與樣式字典
COLOR_PALETTE = sns.color_palette("tab20", 20)  # Tab20 提供 20 種顏色
STYLE_PALETTE = ["o", "s", "P", "X", "D", "*", "^", "v", "h","<"]  # 每個類別固定的樣式
CUSTOM_COLOR_MAP = {
    "Acinetobacter": COLOR_PALETTE[0],  # 確保不同的顏色
    "Enterobacter": COLOR_PALETTE[1],
    "Enterococcus": COLOR_PALETTE[2],
    "Escherichia": COLOR_PALETTE[10],
    "Klebsiella": COLOR_PALETTE[4],
    "Staphylococcus": COLOR_PALETTE[5],
    "Streptococcus": COLOR_PALETTE[6],
    "Salmonella": COLOR_PALETTE[7],
    "Pseudomonas": COLOR_PALETTE[8],
    'CoNS': COLOR_PALETTE[9],
}
classnames=['Acinetobacter', 'Enterobacter','Enterococcus','Escherichia','Klebsiella','Staphylococcus','Streptococcus','Salmonella','Pseudomonas']

def plot_tsne(train_save_name, test_save_name, train_feature, test_feature, train_targets, test_targets, classnames):
    import matplotlib.pyplot as plt
    from sklearn.manifold import TSNE
    import pandas as pd
    import seaborn as sns

    
    tsne = TSNE(n_components=2, init='random', learning_rate='auto')

    train_feature = train_feature.cpu().numpy() if isinstance(train_feature, torch.Tensor) else train_feature
    test_feature = test_feature.cpu().numpy() if isinstance(test_feature, torch.Tensor) else test_feature

    data_combined = np.vstack((train_feature, test_feature))

    # Use the same t-SNE parameters to fit all data
    embedding_combined = tsne.fit_transform(data_combined)

    # Split the projected results
    train_embedding = embedding_combined[:len(train_feature)]
    test_embedding = embedding_combined[len(train_feature):]

    # Results
    print("Train Embedding Shape:", train_embedding.shape)
    print("Test Embedding Shape:", test_embedding.shape)

    # Map labels to class names
    train_labels = [classnames[int(label)] for label in train_targets]
    test_labels = [classnames[int(label)] for label in test_targets]

    # Create DataFrames for plotting
    df_train = pd.DataFrame()
    df_train["y"] = train_labels
    df_train["comp1"] = train_embedding[:, 0]
    df_train["comp2"] = train_embedding[:, 1]

    df_test = pd.DataFrame()
    df_test["y"] = test_labels
    df_test["comp1"] = test_embedding[:, 0]
    df_test["comp2"] = test_embedding[:, 1]

    unique_classes = classnames
    color_map = {cls: f"C{i}" for i, cls in enumerate(unique_classes)}  # Default colors
    style_map = {cls: 'o' for cls in unique_classes}  # Default markers
    
    plt.figure(figsize=(10, 8))
    # Plot train t-SNE
    for cls in unique_classes:
        subset = df_train[df_train["y"] == cls]
        sns.scatterplot(
            x="comp1",
            y="comp2",
            data=subset,
            label=cls,
            color=color_map[cls],
            marker=style_map[cls],
        )

    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc="upper left", 
        borderaxespad=0, 
        fontsize=14
    )
    plt.title("T-SNE Projection (Train)", fontsize=18)
    plt.xlabel("Component 1", fontsize=16)
    plt.ylabel("Component 2", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{train_save_name}.png")
    plt.close()

    plt.figure(figsize=(10, 8))
    # Plot test t-SNE
    for cls in unique_classes:
        subset = df_test[df_test["y"] == cls]
        sns.scatterplot(
            x="comp1",
            y="comp2",
            data=subset,
            label=cls,
            color=color_map[cls],
            marker=style_map[cls],
        )

    plt.legend(
        bbox_to_anchor=(1.05, 1), 
        loc="upper left", 
        borderaxespad=0, 
        fontsize=14
    )
    plt.title("T-SNE Projection (Test)", fontsize=18)
    plt.xlabel("Component 1", fontsize=16)
    plt.ylabel("Component 2", fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{test_save_name}.png")
    plt.close()

def plot_tsne_interactive_html(train_save_name, test_save_name, 
                                train_feature, test_feature, 
                                train_targets, test_targets, 
                                train_patient_ids, test_patient_ids, 
                                classnames):
    tsne = TSNE(n_components=2, init='pca', random_state=0)

    train_feature = train_feature.cpu().numpy() if isinstance(train_feature, torch.Tensor) else train_feature
    test_feature = test_feature.cpu().numpy() if isinstance(test_feature, torch.Tensor) else test_feature
    data_combined = np.vstack((train_feature, test_feature))

    embedding_combined = tsne.fit_transform(data_combined)
    train_embedding = embedding_combined[:len(train_feature)]
    test_embedding = embedding_combined[len(train_feature):]

    train_labels = [classnames[int(label)] for label in train_targets]
    test_labels = [classnames[int(label)] for label in test_targets]

    df_train = pd.DataFrame({
        "y": train_labels,
        "comp1": train_embedding[:, 0],
        "comp2": train_embedding[:, 1],
        "patient_id": train_patient_ids
    })

    df_test = pd.DataFrame({
        "y": test_labels,
        "comp1": test_embedding[:, 0],
        "comp2": test_embedding[:, 1],
        "patient_id": test_patient_ids
    })

    # 🎨 固定每個類別顏色
    color_palette = px.colors.qualitative.Plotly  # 你可以換別的色盤
    color_discrete_map = {cls: color_palette[i % len(color_palette)] for i, cls in enumerate(classnames)}

    # --- 畫 Train ---
    fig_train = px.scatter(
        df_train,
        x="comp1",
        y="comp2",
        color="y",
        hover_data=["patient_id", "y"],
        title="T-SNE Projection (Train)",
        width=1000,
        height=800,
        color_discrete_map=color_discrete_map   # <<== 固定顏色
    )
    fig_train.update_traces(marker=dict(size=5))

    # 存成 HTML 檔
    fig_train.write_html(f"{train_save_name}.html")
    print(f"Train t-SNE interactive HTML saved to {train_save_name}.html")

    # --- 畫 Test ---
    fig_test = px.scatter(
        df_test,
        x="comp1",
        y="comp2",
        color="y",
        hover_data=["patient_id", "y"],
        title="T-SNE Projection (Test)",
        width=1000,
        height=800,
        color_discrete_map=color_discrete_map   # <<== 固定顏色
    )
    fig_test.update_traces(marker=dict(size=5))

    fig_test.write_html(f"{test_save_name}.html")
    print(f"Test t-SNE interactive HTML saved to {test_save_name}.html")

import umap
def plot_umap_interactive_html(train_save_name, test_save_name, 
                               train_feature, test_feature, 
                               train_targets, test_targets, 
                               train_patient_ids, test_patient_ids, 
                               classnames):
    reducer = umap.UMAP(n_components=2, random_state=42)

    # 將 tensor 轉為 numpy
    train_feature = train_feature.cpu().numpy() if isinstance(train_feature, torch.Tensor) else train_feature
    test_feature = test_feature.cpu().numpy() if isinstance(test_feature, torch.Tensor) else test_feature
    data_combined = np.vstack((train_feature, test_feature))

    # UMAP fitting on combined data
    train_embedding = reducer.fit_transform(train_feature)
    test_embedding = reducer.transform(test_feature)

    # 轉換 label 為 class 名稱
    train_labels = [classnames[int(label)] for label in train_targets]
    test_labels = [classnames[int(label)] for label in test_targets]

    # 建立 DataFrame
    df_train = pd.DataFrame({
        "y": train_labels,
        "comp1": train_embedding[:, 0],
        "comp2": train_embedding[:, 1],
        "patient_id": train_patient_ids
    })

    df_test = pd.DataFrame({
        "y": test_labels,
        "comp1": test_embedding[:, 0],
        "comp2": test_embedding[:, 1],
        "patient_id": test_patient_ids
    })

    # 🎨 固定類別顏色
    color_palette = px.colors.qualitative.Plotly
    color_discrete_map = {cls: color_palette[i % len(color_palette)] for i, cls in enumerate(classnames)}

    # --- 畫 Train ---
    fig_train = px.scatter(
        df_train,
        x="comp1",
        y="comp2",
        color="y",
        hover_data=["patient_id", "y"],
        title="UMAP Projection (Train)",
        width=1000,
        height=800,
        color_discrete_map=color_discrete_map
    )
    fig_train.update_traces(marker=dict(size=5))
    fig_train.write_html(f"{train_save_name}.html")
    print(f"Train UMAP interactive HTML saved to {train_save_name}.html")

    # --- 畫 Test ---
    fig_test = px.scatter(
        df_test,
        x="comp1",
        y="comp2",
        color="y",
        hover_data=["patient_id", "y"],
        title="UMAP Projection (Test)",
        width=1000,
        height=800,
        color_discrete_map=color_discrete_map
    )
    fig_test.update_traces(marker=dict(size=5))
    fig_test.write_html(f"{test_save_name}.html")
    print(f"Test UMAP interactive HTML saved to {test_save_name}.html")
