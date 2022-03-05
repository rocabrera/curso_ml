import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.neighbors import KNeighborsClassifier

import seaborn as sns

def prepareData(sklearnDataSet):
    bunch = sklearnDataSet()
    X, y = bunch["data"], bunch["target"]
    data  = np.hstack([X,y.reshape(-1,1)])
    cols_name_lst = [f"feature_{i+1}" for i in range(X.shape[1])] + ["target"]
    return pd.DataFrame(data, columns = cols_name_lst), bunch["DESCR"]

def plot2Dimensions(df, x, y, test_examples, figsize=((20,6))):
    
    _ = plt.figure(figsize=figsize)
    test_label = "Exemplos Teste"
    g1 = sns.scatterplot(data=df, x=x, y=y, hue="target", palette="bright",edgecolor="black", linewidth=1.5)
    g2 = plt.scatter(test_examples[:,0], test_examples[:,1], s=150, c="k", marker = "X", label = test_label)
    
    handles, labels  =  g1.get_legend_handles_labels()
    labels = [f"Classe {col}" for col in range(len(labels)-1)] + [test_label]
    g1.legend(handles, labels, shadow=True)
    plt.title(f"Localização dos {len(test_examples)} pontos teste.")
    plt.show()
    
    
def plot_Kneighbors(df_distance, x, y, test_examples, figsize=((20,6)), k_neighbors=5):
    
    _ = plt.figure(figsize=figsize)
    test_label = "Exemplos Teste"
    g1 = sns.scatterplot(data=df_distance, x=x, y=y, hue="target", palette="bright",edgecolor="black", linewidth=1.5)
    g2 = plt.scatter(test_examples[:,0], test_examples[:,1], s=150, c="k", marker = "X", label = test_label)
    
    for test_example, col in zip(test_examples, df_distance.filter(regex = "distance")):
        
        aux=df_distance.nsmallest(k_neighbors, col).filter(regex="feature")
        for _, val in aux.iterrows():
            _ = plt.plot([test_example[0], val[x]], [test_example[1], val[y]], 'go--', c='k', linewidth=1.5, marker='+')
    
    handles, labels = g1.get_legend_handles_labels()
    labels = [f"Classe {col}" for col in range(len(labels)-1)] + [test_label]
    g1.legend(handles, labels, shadow=True)
    plt.title(f"Vizinhos mais próximos dos {len(test_examples)} pontos teste.")
    plt.show()
    
def my_final_plot_comparison(df, fst_dim, snd_dim):
    
    fig, ax = plt.subplots(figsize=(20,6))

    g1 = sns.scatterplot(data=df, 
                         x=fst_dim, 
                         y=snd_dim, 
                         hue="target", 
                         marker="o",
                         style="tipo",
                         palette="bright",
                         alpha=0.8,
                         s=100,
                         linewidth=1.5,
                         edgecolor="black",
                         ax=ax)

    legend1 = ax.legend(shadow=True, loc="upper left")

    plt.show()
    
def plot_decision_boundary_k(X_df, y_df, n_neighbors_lst):
    
    X = X_df.values
    y = y_df.target
    rows = len(n_neighbors_lst)//3
    fig, axs = plt.subplots(rows, 3, figsize=(20,5*rows))
    h = .02  # step size in the mesh
    columns = X_df.columns
    
    # Create color maps
    classes_qtd = len(y.unique())
    if classes_qtd == 3:
        cmap_light = ListedColormap(['orange', 'cyan', 'cornflowerblue'])
        cmap_bold  = ['darkorange', 'c', 'darkblue']
        
    elif classes_qtd == 2:
        cmap_light = ListedColormap(['orange', 'cyan'])
        cmap_bold = ['darkorange', 'c']
    else:
        return "Muitas classes. Por enquanto somente 1 ou 2 classes."

    for ax, k_neighbors in zip(axs.flatten(),n_neighbors_lst):
        # we create an instance of Neighbours Classifier and fit the data.
        clf = KNeighborsClassifier(k_neighbors, weights="uniform")
        clf.fit(X, y)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
        y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cmap_light)

        # Plot also the training points
        sns.scatterplot(x=X[:, 0], 
                        y=X[:, 1], 
                        hue=y,
                        palette=cmap_bold, 
                        alpha=1.0, 
                        edgecolor="black", 
                        linewidth=1.5, 
                        ax=ax)
        
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        ax.set_title(f"3-Class classification ($k$ = {k_neighbors})")
        #plt.xlabel(columns[0])
        #plt.ylabel(columns[1])

    plt.tight_layout()
    plt.show()