"""Handling of climate index data files.

Climate timeseries in the correct format can be found on
https://psl.noaa.gov/data/timeseries/month/ (checked on 01 April 2025).

Written by Erick Madrigal-Solis. Adapted from M. Reinert.
(https://github.com/markusReinert/ExtremeSurgeAnalysis/tree/main).
"""

import numpy as np
import pandas as pd
import re, requests
from enum import Enum

class ClimateIndex(Enum):
    """Selector for the available climate indices.

    The name of each ClimateIndex is a common abbreviation for it; the 
    value of each ClimateIndex is a short description of it. The file
    is expected to be in the PSL format, see ‘read_NOAA_PSL_file’ for 
    more information.
    """
    NINO1870 = ("Nino3.4 HadISST (1870-2025)", 
                "https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data")
    NINO1950 = ("Nino3.4 ERSSTv5 (1950-2025)", 
               "https://psl.noaa.gov/data/correlation/nina34.anom.data")

    def description(self):
        return self.value[0]

    def url(self):
        return self.value[1]


def load_NOAA_data(dataset: ClimateIndex) -> dict:
    """Load the timeseries of a climate index from NOAA.

    See the documentation of the class ‘ClimateIndex’ for more
    information on the argument ‘dataset’.

    The returned object is a dictionary with the following keys:
     - name (string, abbreviation or short description of the climate index)
     - full_name (string, full name or description of the climate index)
     - year_start (int, first year of the timeseries)
     - year_end (int, last year of the timeseries)
     - t_years (array, time values in years)
     - index (array, values of the climate index)
    """
    #url = 'https://psl.noaa.gov/data/timeseries/month/data/nino34.long.anom.data'
    url = dataset.url()
    print(f"Reading NOAA data for {dataset.description()}")

    year_start, year_end, index = read_NOAA_PSL(url)
    #print("{:9_d} records".format(np.count_nonzero(~np.isnan(index))))
    return {
        "name": dataset.name,
        "full_name": dataset.description(),
        "year_start": year_start,
        "year_end": year_end,
        "t_years": np.arange(year_start, year_end + 1, 1 / 12),
        "index": index,
    }


def read_NOAA_PSL(url): 
    """Read data from a file in the Monthly PSL Standard Format by NOAA.

    The method returns three objects in the following order:
     1. the first year of the timeseries,
     2. the last year of the timeseris,
     3. the monthly data.
    The first two values are ints, the data is a NumPy array of floats
    with length 12 times the number of years in the timeseries,
    including the first and last year.
    """
    response = requests.get(url)
    lines = response.text.splitlines()

    match = re.search(r"(\d{4})\D+(\d{4})", lines[0])

    if match:
        year_start, year_end = map(int, match.groups())
        print(f"Start Year: {year_start}, End Year: {year_end}")
    else:
        raise ValueError("Could not extract year range from first line.")

    n_years = year_end - year_start + 1
    print(f"Number of years: {n_years}")

    values = pd.read_csv(url, sep='\s+', skiprows=1, header=None)
    #data = values.iloc[:, 1:].values.flatten()
    data = np.zeros(n_years * 12)
    for i in range(n_years):
        #values = pd.read_csv(url, delim_whitespace=True, skiprows=1, header=None)
        data[i*12 : (i+1)*12] = values.iloc[i, 1:]

    return year_start, year_end, data
