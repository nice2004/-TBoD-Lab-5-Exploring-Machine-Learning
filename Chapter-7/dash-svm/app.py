import time
import importlib
import dash
from dash import Dash, dcc, html, Input, Output, State
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import datasets
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder

import utils.dash_reusable_components as drc
import utils.figures as figs
import dash.dash_table as dt

app = Dash(
    __name__,
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1.0"}
    ],
)
app.title = "Support Vector Machine"
server = app.server

import palmerpenguins

penguins_df = palmerpenguins.load_penguins()
columns_to_exclude = ['species', 'year', 'sex']
remaining_columns = penguins_df.columns.difference(columns_to_exclude)
penguins_df_cleaning = penguins_df.dropna()
print(penguins_df_cleaning.head())
print(penguins_df_cleaning.columns)

app.layout = html.Div(
    children=[
        # .container class is fixed, .container.scalable is scalable
        html.Div(
            className="banner",
            children=[
                # Change App Name here
                html.Div(
                    className="container scalable",
                    children=[
                        # Change App Name here
                        html.H2(
                            id="banner-title",
                            children=[
                                html.A(
                                    "Support Vector Machine (SVM) Explorer for Palmer Penguins",
                                    style={
                                        "text-decoration": "none",
                                        "color": "inherit",
                                    },
                                )
                            ],
                        ),
                    ],
                )
            ],
        ),
        html.Div(
            id="body",
            className="container scalable",
            children=[
                html.Div(
                    id="app-container",
                    # className="row",
                    children=[
                        html.Div(
                            # className="three columns",
                            id="left-column",
                            children=[
                                drc.Card(
                                    id="first-card",
                                    children=[
                                        # drc.NamedDropdown(
                                        #  name="Select Dataset",
                                        # id="dropdown-select-dataset",
                                        #options=[
                                        #   {"label": "Moons", "value": "moons"},
                                        #  {
                                        #     "label": "Linearly Separable",
                                        #    "value": "linear",
                                        #},
                                        #{
                                        #   "label": "Circles",
                                        #  "value": "circles",
                                        # },
                                        #],
                                        #clearable=False,
                                        #searchable=False,
                                        #value="moons",
                                        #),
                                        drc.NamedSlider(
                                            name="Sample Size",
                                            id="slider-dataset-sample-size",
                                            min=100,
                                            max=333,
                                            step=50,
                                            marks={
                                                str(i): str(i)
                                                for i in [50, 100, 150, 200, 250, 300]
                                            },
                                            value=150,
                                        ),
                                        drc.NamedSlider(
                                            name="Noise Level",
                                            id="slider-dataset-noise-level",
                                            min=0,
                                            max=1,
                                            marks={
                                                i / 10: str(i / 10)
                                                for i in range(0, 11, 2)
                                            },
                                            step=0.1,
                                            value=0.2,
                                        ),
                                        html.Button(
                                            "Default Button",
                                            id="button-default",
                                            n_clicks=0,
                                            style={

                                                'color': 'white',

                                            }
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="button-card",
                                    children=[
                                        drc.NamedSlider(
                                            name="Threshold",
                                            id="slider-threshold",
                                            min=0,
                                            max=1,
                                            value=0.5,
                                            step=0.01,
                                        ),
                                        html.Button(
                                            "Reset Threshold",
                                            id="button-zero-threshold",
                                        ),
                                    ],
                                ),
                                drc.Card(
                                    id="last-card",
                                    children=[
                                        drc.NamedDropdown(
                                            name="Kernel",
                                            id="dropdown-svm-parameter-kernel",
                                            options=[
                                                {
                                                    "label": "Radial basis function (RBF)",
                                                    "value": "rbf",
                                                },
                                                {"label": "Linear", "value": "linear"},
                                                {
                                                    "label": "Polynomial",
                                                    "value": "poly",
                                                },
                                                {
                                                    "label": "Sigmoid",
                                                    "value": "sigmoid",
                                                },
                                            ],
                                            value="rbf",
                                            clearable=False,
                                            searchable=False,
                                        ),
                                        drc.NamedSlider(
                                            name="Cost (C)",
                                            id="slider-svm-parameter-C-power",
                                            min=-2,
                                            max=4,
                                            value=0,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-2, 5)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-C-coef",
                                            min=1,
                                            max=9,
                                            value=1,
                                        ),
                                        drc.NamedSlider(
                                            name="Degree",
                                            id="slider-svm-parameter-degree",
                                            min=2,
                                            max=10,
                                            value=3,
                                            step=1,
                                            marks={
                                                str(i): str(i) for i in range(2, 11, 2)
                                            },
                                        ),
                                        drc.NamedSlider(
                                            name="Gamma",
                                            id="slider-svm-parameter-gamma-power",
                                            min=-5,
                                            max=0,
                                            value=-1,
                                            marks={
                                                i: "{}".format(10 ** i)
                                                for i in range(-5, 1)
                                            },
                                        ),
                                        drc.FormattedSlider(
                                            id="slider-svm-parameter-gamma-coef",
                                            min=1,
                                            max=9,
                                            value=5,
                                        ),
                                        html.Div(
                                            id="shrinking-container",
                                            children=[
                                                html.P(children="Shrinking"),
                                                dcc.RadioItems(
                                                    id="radio-svm-parameter-shrinking",
                                                    labelStyle={
                                                        "margin-right": "7px",
                                                        "display": "inline-block",
                                                    },
                                                    options=[
                                                        {
                                                            "label": " Enabled",
                                                            "value": "True",
                                                        },
                                                        {
                                                            "label": " Disabled",
                                                            "value": "False",
                                                        },
                                                    ],
                                                    value="True",
                                                ),
                                            ],
                                        ),
                                    ],
                                ),
                            ],
                        ),
                        html.Div(
                            id="div-graphs",
                            children=dcc.Graph(
                                id="graph-sklearn-svm",
                                figure=dict(
                                    layout=dict(
                                        plot_bgcolor="#282b38", paper_bgcolor="#282b38"
                                    )
                                ),
                            ),
                        ),
                    ],
                )
            ],
        ),
    ]
)


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


# Running the server
if __name__ == "__main__":
    app.run(debug=True)
