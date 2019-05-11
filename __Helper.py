import pandas as pd


def __new_col_data(col_data: pd.DataFrame, func_name: str) -> pd.DataFrame:
    col_name = getattr(col_data, 'name')
    max_min_data = pd.DataFrame(col_data)
    max_min_data.columns = [f'{col_name}_{func_name}']
    return max_min_data
