
import time

import dash
import dash.dash_table as dt
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.svm import SVC
import palmerpenguins
from sklearn.model_selection import train_test_split
from Penguins_Dataset_Dashboard.utils import dash_reusable_components as drc
from Penguins_Dataset_Dashboard.utils import figures as figs
from data import penguins_df_cleaning
from layout import layout
from Update_svm_Callback import svm_callback

app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"},
    ],
)
app.title = "Support Vector Machine"
server = app.server

app.layout = layout

@app.callback(
    Output("slider-svm-parameter-gamma-coef", "marks"),
    [Input("slider-svm-parameter-gamma-power", "value")],
)
def update_slider_svm_parameter_gamma_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


@app.callback(
    Output("slider-svm-parameter-C-coef", "marks"),
    [Input("slider-svm-parameter-C-power", "value")],
)
def update_slider_svm_parameter_C_coef(power):
    scale = 10 ** power
    return {i: str(round(i * scale, 8)) for i in range(1, 10, 2)}


# Callback for the Default Button


@app.callback(
    Output("dropdown-svm-parameter-kernel", "value", allow_duplicate=True),
    Output("slider-svm-parameter-degree", "value", allow_duplicate=True),
    Output("slider-svm-parameter-C-coef", "value", allow_duplicate=True),
    Output("slider-svm-parameter-C-power", "value", allow_duplicate=True),
    Output("slider-svm-parameter-gamma-coef", "value", allow_duplicate=True),
    Output("slider-svm-parameter-gamma-power", "value", allow_duplicate=True),
    Output("slider-dataset-noise-level", "value", allow_duplicate=True),
    Output("radio-svm-parameter-shrinking", "value", allow_duplicate=True),
    Output("slider-threshold", "value", allow_duplicate=True),
    Output("slider-dataset-sample-size", "value", allow_duplicate=True),
    [Input('button-default', 'n_clicks')],
    prevent_initial_call=True
)
def reset_default(n_clicks):
    if n_clicks > 0:
        return 'rbf', 3, 1, 0, 5, -1, 0.2, 'True', 0.5, 300
    return dash.no_update


@app.callback(
    Output("slider-threshold", "value"),
    [Input("button-zero-threshold", "n_clicks")],
    [State("graph-prediction", "figure")],
)
def reset_threshold_center(n_clicks, figure):
    if n_clicks:
        Z = np.array(figure["data"][0]["z"])
        value = -Z.min() / (Z.max() - Z.min())
    else:
        value = 0.4959986285375595
    return value


# Disable Sliders if kernel not in the given list
@app.callback(
    Output("slider-svm-parameter-degree", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_degree(kernel):
    return kernel != "poly"


@app.callback(
    Output("slider-svm-parameter-gamma-coef", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_coef(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


@app.callback(
    Output("slider-svm-parameter-gamma-power", "disabled"),
    [Input("dropdown-svm-parameter-kernel", "value")],
)
def disable_slider_param_gamma_power(kernel):
    return kernel not in ["rbf", "poly", "sigmoid"]


svm_callback(app)


# Running the server
if __name__ == "__main__":
    app.run(debug=True)
