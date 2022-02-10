import numpy as np
import pandas as pd
from IPython.core.display import display, HTML


def prepare_data(sklearnDataSet):
    
    bunch = sklearnDataSet()
    X, y = bunch["data"], bunch["target"]
    data  = np.hstack([X,y.reshape(-1,1)])
    cols_name_lst = bunch["feature_names"].tolist() + ["target"]
    return pd.DataFrame(data, columns = cols_name_lst), bunch["DESCR"]


def display_side_by_side(dfs:list, captions:list,space:tuple):
    """Display tables side by side to save vertical space
    Input:
        dfs: list of pandas.DataFrame
        captions: list of table captions
    """
    init_spacing, final_spacing = space
    output = "\xa0"*init_spacing
    combined = dict(zip(captions, dfs))
    for caption, df in combined.items():
        output += df.style.set_table_attributes("style='display:inline'").set_caption(caption)._repr_html_()
        output += "\xa0"*final_spacing
    display(HTML(output))