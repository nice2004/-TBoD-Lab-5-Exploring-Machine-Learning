import pandas as py
import palmerpenguins


penguins_df = palmerpenguins.load_penguins()
columns_to_exclude = ['species', 'year', 'sex']
remaining_columns = penguins_df.columns.difference(columns_to_exclude)
penguins_df_cleaning = penguins_df.dropna()
print(penguins_df_cleaning.head())
print(penguins_df_cleaning.columns)









# Just Comments
# x_axis_dropdown = html.Div([
#     html.Label('Select X-Axis Column'),
#     dcc.Dropdown(
#         id='x_column_species',
#         options=[{'label': str(i), 'value': i} for i in sorted(penguins_df[remaining_columns])],
#         value=sorted(penguins_df.columns)[0],  # Default to first value
#         clearable=False,
#         style={'width': '100%'}
#     ), ])

# y_axis_dropdown = html.Div([
#     html.Label('Select Y-Axis Column'),
#     dcc.Dropdown(
#         id='y_column_species',
#         options=[{'label': str(i), 'value': i} for i in sorted(penguins_df[remaining_columns])],
#         value=sorted(penguins_df.columns)[1],  # Default to first value
#         clearable=False,
#         style={'width': '100%'}
#     ), ])
# print(penguins_df_cleaning[remaining_columns])
# STARTING WORKING ON MACHINE LEARNING (YAYY)
row_count = len(penguins_df_cleaning)
print(row_count)
# label and encode the target species into numeric value

# model = LogisticRegression(max_iter=200)
# model.fit(x_train_scaled, y_train)
# The species are Adelie, Gentoo, Chinstrap
# print(penguins_df['species'].values)

# [Find a place where to add this in the layout and remove what is being displayed ]