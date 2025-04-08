import colorlover as cl
import pandas as pd
import plotly.graph_objs as go
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix


def serve_prediction_plot(
        model, X_train, X_test, y_train, y_test, Z, xx, yy, mesh_step, threshold
):
    # Get train and test score from model
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    train_score = metrics.accuracy_score(y_true=y_train, y_pred=y_pred_train)
    test_score = metrics.accuracy_score(y_true=y_test, y_pred=y_pred_test)

    # Compute threshold
    scaled_threshold = threshold * (Z.max() - Z.min()) + Z.min()
    range = max(abs(scaled_threshold - Z.min()), abs(scaled_threshold - Z.max()))

    # Colorscale
    bright_cscale = [[0, "#ff3700"], [1, "#0b8bff"]]
    cscale = [
        [0.0000000, "#ff744c"],
        [0.1428571, "#ff916d"],
        [0.2857143, "#ffc0a8"],
        [0.4285714, "#ffe7dc"],
        [0.5714286, "#e5fcff"],
        [0.7142857, "#c8feff"],
        [0.8571429, "#9af8ff"],
        [1.0000000, "#20e6ff"],
    ]

    # Create the plot
    # Plot the prediction contour of the SVM
    trace0 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        zmin=scaled_threshold - range,
        zmax=scaled_threshold + range,
        hoverinfo="none",
        showscale=False,
        contours=dict(showlines=False),
        colorscale=cscale,
        opacity=0.9,
    )

    # Plot the threshold
    trace1 = go.Contour(
        x=np.arange(xx.min(), xx.max(), mesh_step),
        y=np.arange(yy.min(), yy.max(), mesh_step),
        z=Z.reshape(xx.shape),
        showscale=False,
        hoverinfo="none",
        contours=dict(
            showlines=False, type="constraint", operation="=", value=scaled_threshold
        ),
        name=f"Threshold ({scaled_threshold:.3f})",
        line=dict(color="#708090"),
    )

    # Plot Training Data
    trace2 = go.Scatter(
        x=X_train[:, 0],
        y=X_train[:, 1],
        mode="markers",
        name=f"Training Data (accuracy={train_score:.3f})",
        marker=dict(size=10, color=y_train, colorscale=bright_cscale),
    )

    # Plot Test Data
    trace3 = go.Scatter(
        x=X_test[:, 0],
        y=X_test[:, 1],
        mode="markers",
        name=f"Test Data (accuracy={test_score:.3f})",
        marker=dict(
            size=10, symbol="triangle-up", color=y_test, colorscale=bright_cscale
        ),
    )

    layout = go.Layout(
        xaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        yaxis=dict(ticks="", showticklabels=False, showgrid=False, zeroline=False),
        hovermode="closest",
        legend=dict(x=0, y=-0.01, orientation="h"),
        margin=dict(l=0, r=0, t=0, b=0),
        plot_bgcolor="#282b38",
        paper_bgcolor="#282b38",
        font={"color": "#a5b1cd"},
    )

    data = [trace0, trace1, trace2, trace3]
    figure = go.Figure(data=data, layout=layout)

    return figure


def serve_roc_curve(model, X_test, y_test):
    # Binarize the output labels for multiclass classification
    y_test_bin = label_binarize(y_test, classes=[0, 1, 2])  # assuming the 3 classes for the species
    y_score = model.predict_proba(X_test)  # probability predictions for the classes

    species = {0: 'Adelie', 1: 'Chinstrap', 2: 'Gentoo'}
    # Prepare the ROC curve for each class
    fpr, tpr, roc_auc = dict(), dict(), dict()
    for i in range(3):
        fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Create the ROC curve plot using Plotly
    fig = go.Figure()

    for i in range(3):
        species_mapping = species[i]
        fig.add_trace(go.Scatter(
            x=fpr[i], y=tpr[i],
            mode='lines',
            name=f'{species_mapping} (AUC = {roc_auc[i]:.2f})',
            line=dict(width=2)
        ))

    # Format the layout
    fig.update_layout(
        title="Receiver Operating Characteristic (ROC) Curve for each class",
        xaxis_title="False Positive Rate",
        yaxis_title="True Positive Rate",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1]),
        showlegend=True
    )

    return fig


def serve_confusion_matrix_table(model, X_test, y_test, Z, threshold):
    # Predict the classes
    y_pred_test = model.predict(X_test)
    if isinstance(y_test, pd.Series):
        y_test = y_test.values
    matrix = metrics.confusion_matrix(y_true=y_test, y_pred=y_pred_test)
    matrix_df = pd.DataFrame(matrix, index=[f'Class {i}' for i in range(len(matrix))],
                             columns=[f'Pred {i}' for i in range(len(matrix[0]))])

    return matrix_df
