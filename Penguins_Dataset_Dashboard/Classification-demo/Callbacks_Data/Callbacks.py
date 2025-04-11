import time
import importlib
import dash
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from data import penguins_df_cleaning

import utils.dash_reusable_components as drc
import utils.figures as figs
import dash.dash_table as dt


def callbacks(app):
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

    @app.callback(
        Output("slider-threshold", "value"),
        [Input("button-zero-threshold", "n_clicks")],
        [State("graph-sklearn-svm", "figure")],
    )
    def reset_threshold_center(n_clicks, figure):
        if n_clicks:
            try:
                z_values = figure["data"][0].get("z", None)
                if z_values is not None:
                    Z = np.array(z_values)
                    if Z.ndim > 0:
                        return -Z.min() / (Z.max() - Z.min())
            except Exception as e:
                print("Error in threshold reset:", e)
            return 0.5  # fallback
        return dash.no_update

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
            return 'rbf', 3, 1, 0, 5, -1, 0.2, 'True', 0.5, 150
        return dash.no_update

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

    @app.callback(
        Output("div-graphs", "children"),
        [
            #  Input('x_column_species', 'value'),
            # Input('y_column_species', 'value'),
            Input("dropdown-svm-parameter-kernel", "value"),
            Input("slider-svm-parameter-degree", "value"),
            Input("slider-svm-parameter-C-coef", "value"),
            Input("slider-svm-parameter-C-power", "value"),
            Input("slider-svm-parameter-gamma-coef", "value"),
            Input("slider-svm-parameter-gamma-power", "value"),
            Input("slider-dataset-noise-level", "value"),
            Input("radio-svm-parameter-shrinking", "value"),
            Input("slider-threshold", "value"),
            Input("slider-dataset-sample-size", "value"),
        ], )
    def update_svm_graph(
            kernel,
            degree,
            C_coef,
            C_power,
            gamma_coef,
            gamma_power,
            noise,
            shrinking,
            threshold,
            sample_size,
    ):
        if threshold is None:
            threshold = 0.5
        t_start = time.time()
        h = 0.3  # step size in the mesh
        if shrinking == "True":
            flag = True
        else:
            flag = False
        sampled_data = penguins_df_cleaning.sample(n=sample_size, random_state=42)
        # Data Pre-processing
        X = sampled_data[['bill_length_mm', 'flipper_length_mm']]
        y = sampled_data['species']
        # Apply encoding
        encoder = LabelEncoder()
        for col in X.columns:
            if X[col].dtype == 'object':
                X.loc[:, col] = encoder.fit_transform(X[col])
        y_encoded = encoder.fit_transform(y)
        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

        # Train the logistic regression model, because the flipper length and some other factors might be in different
        # lengths or sizes
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # Train the model
        C = C_coef * 10 ** C_power
        gamma = gamma_coef * 10 ** gamma_power
        clf = SVC(C=C, kernel=kernel, degree=degree, gamma=gamma, shrinking=flag, probability=True)
        clf.fit(X_train_scaled, y_train)

        # Create the mesh grid for the decision boundary
        x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
        y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        z = z.reshape(xx.shape)

        prediction_figure = figs.serve_prediction_plot(
            model=clf,
            X_train=X_train_scaled,
            X_test=X_test_scaled,
            y_train=y_train,
            y_test=y_test,
            Z=z,
            xx=xx,
            yy=yy,
            mesh_step=h,
            threshold=threshold,
        )
        prediction_figure.update_layout(
            xaxis_title='Bill Length (mm)',
            yaxis_title='Flipper Length (mm)',
            xaxis=dict(title=dict(standoff=50)),
            margin=dict(l=50, r=50, t=50, b=150),
        )

        roc_figure = figs.serve_roc_curve(model=clf, X_test=X_test_scaled, y_test=y_test)

        confusion_figure = figs.serve_confusion_matrix_table(
            model=clf, X_test=X_test_scaled, y_test=y_test, Z=z, threshold=threshold
        )
        confusion_table_data = confusion_figure.reset_index().to_dict('records')
        confusion_table_columns = [{'name': col, 'id': col} for col in confusion_figure.columns]
        confusion_table_columns.insert(0, {'name': 'Prediction/Actual', 'id': 'index'})

        return [
            html.Div(
                id="svm-graph-container",
                children=dcc.Loading(
                    className="graph-wrapper",
                    children=dcc.Graph(id="graph-sklearn-svm", figure=prediction_figure),
                    style={"display": "none"},
                ),
            ),
            html.Div(
                id="graphs-container",
                children=[
                    dcc.Loading(
                        className="graph-wrapper",
                        children=dcc.Graph(id="graph-line-roc-curve", figure=roc_figure),
                    ),
                    dcc.Loading(
                        className='graph-wrapper',
                        children=dt.DataTable(
                            id='table-confusion-matrix',
                            columns=confusion_table_columns,
                            data=confusion_table_data,
                            style_table={'margin': '20px', 'overflowX': 'auto'},
                            style_cell={'textAlign': 'center'},
                            style_header={'fontWeight': 'bold'},
                        ),
                    ),
                ],
            ),
        ]

