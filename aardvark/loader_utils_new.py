import numpy as np
import torch
import pandas as pd

LATLON_SCALE_FACTOR = 360
DAYS_IN_YEAR = 366

DAILY_SCALE_FACTOR = {
    "ERA5": 4,
    "HADISD": 4,
    "IR": 4,
    "SOUNDER": 1,
    "ICOADS": 1,
}

date_list = [
    "1979-01-01",
    "1999-01-01",
    "1999-01-02",
    "2002-01-02",
    "2007-01-01",
    "2007-01-02",
    "2007-01-03",
    "2013-01-02",
    "2014-01-01",
    "2017-01-02",
    "2018-01-01",
    "2019-01-01",
    "2020-01-02",
    "2020-01-01",
    "2021-01-02",
    "2021-01-01",
]


def generate_offsets(date_list, dates):
    offsets = {}
    for d in date_list:
        try:
            offsets[d] = np.where(dates == d)[0][0]
        except:
            offsets[d] = -1
    return offsets


def build_offsets(freq):
    if freq == "1D":
        daily_start = "2007-01-02"
        daily_end = "2019-12-30"
        return {
            "ic": generate_offsets(
                date_list, pd.date_range("1999-01-02", "2021-12-31", freq=freq)
            ),
            "amsua": generate_offsets(
                date_list, pd.date_range(daily_start, daily_end, freq=freq)
            ),
            "amsub": generate_offsets(
                date_list, pd.date_range(daily_start, daily_end, freq=freq)
            ),
            "ascat": generate_offsets(
                date_list, pd.date_range(daily_start, daily_end, freq=freq)
            ),
            "atms": generate_offsets(
                date_list, pd.date_range("2013-01-02", "2021-12-31", freq=freq)
            ),
            "icoads": generate_offsets(
                date_list, pd.date_range(daily_start, daily_end, freq=freq)
            ),
            "igra": generate_offsets(
                date_list, pd.date_range(daily_start, daily_end, freq=freq)
            ),
            "sat": generate_offsets(
                date_list, pd.date_range(daily_start, daily_end, freq=freq)
            ),
            "hadisd": generate_offsets(
                date_list, pd.date_range(daily_start, daily_end, freq=freq)
            ),
        }
    return {
        "ic": generate_offsets(
            date_list, pd.date_range("1999-01-02", "2021-12-31 18:00", freq=freq)
        ),
        "amsua": generate_offsets(
            date_list, pd.date_range("2007-01-01", "2021-12-31 18:00", freq=freq)
        ),
        "amsub": generate_offsets(
            date_list, pd.date_range("2007-01-01", "2021-12-31 18:00", freq=freq)
        ),
        "ascat": generate_offsets(
            date_list, pd.date_range("2007-01-01", "2021-12-31", freq=freq)
        ),
        "atms": generate_offsets(
            date_list, pd.date_range("2013-01-02", "2021-12-31", freq="1D")
        ),
        "icoads": generate_offsets(
            date_list, pd.date_range("1999-01-01 06:00", "2021-12-31", freq=freq)
        ),
        "igra": generate_offsets(
            date_list, pd.date_range("1999-01-01 00:00", "2021-12-31 18:00", freq=freq)
        ),
        "sat": generate_offsets(
            date_list, pd.date_range("1990-01-01 00:00", "2021-12-31 18:00", freq=freq)
        ),
        "hadisd": generate_offsets(
            date_list, pd.date_range("1950-01-01 00:00", "2021-12-31 18:00", freq=freq)
        ),
    }


IC_OFFSETS = build_offsets("6H")["ic"]
AMSUA_OFFSETS = build_offsets("6H")["amsua"]
AMSUB_OFFSETS = build_offsets("6H")["amsub"]
ASCAT_OFFSETS = build_offsets("6H")["ascat"]
ATMS_OFFSETS = build_offsets("6H")["atms"]
ICOADS_OFFSETS = build_offsets("6H")["icoads"]
IGRA_OFFSETS = build_offsets("6H")["igra"]
SAT_OFFSETS = build_offsets("6H")["sat"]
HADISD_OFFSETS = build_offsets("6H")["hadisd"]


def lon_to_0_360(x):
    return (x + 360) % 360


def lat_to_m90_90(x):
    return torch.flip(x, [-1])
