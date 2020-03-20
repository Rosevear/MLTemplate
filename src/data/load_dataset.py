from pathlib import Path
from pandas import read_csv

def load_csv(path):
    """
    Loads the data stored in the csv file present on the provided path Loads the raw,
    and returns the corresponding Pandas dataframe object (or series if the data is one-dimensional)
    """
    
    data_frame = read_csv(path, header=0, squeeze=True)

    return data_frame
