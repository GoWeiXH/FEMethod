import pandas as pd


def __new_col_data(col_data: pd.Series, func_name: str) -> pd.DataFrame:
    col_name = getattr(col_data, 'name')
    new_data = pd.DataFrame(col_data)
    new_data.columns = [f'{col_name}_{func_name}']
    return new_data
