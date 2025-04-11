import palmerpenguins

penguins_df = palmerpenguins.load_penguins()
columns_to_exclude = ['species', 'year', 'sex']
remaining_columns = penguins_df.columns.difference(columns_to_exclude)
penguins_df_cleaning = penguins_df.dropna()
print(penguins_df_cleaning.head())
print(penguins_df_cleaning.columns)