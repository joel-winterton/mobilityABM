"""
This exposes a function that will give a numpy commuter matrix to avoid doing this many times in each analysis.
"""
import pandas as pd


def get_matrix(resolution='LAD'):
    """"
    Returns commuter matrix as numpy ndarray. Currently only has LAD resolution.
    Read more about data: `2011_census_data/README.md`
    """
    if resolution == 'LAD':
        data = pd.read_csv('2011_census_data/WF02EW LAD2011.csv', index_col=0, skiprows=9)
        data.drop(data.index[range(346, len(data.index))], inplace=True)
        return data.values


def get_population_sizes(resolution='LAD'):
    """"
    Returns population size numpy array. Currently only has LAD resolution.
    Read more about data: `2011_census_data/README.md`
    """
    if resolution == 'LAD':
        data = pd.read_csv('2011_census_data/QS102EW LAD2011.csv', index_col=0, skiprows=8)
        data.drop(data.index[range(346, len(data.index))], inplace=True)
        # drop footer text
        return data.values.flatten()