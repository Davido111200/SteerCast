#!/usr/bin/env python
"""Convert a Monash forecasting archive ``.tsf`` file into a benchmark CSV.

The output has a ``date`` column followed by one column per series (named after
``series_name``), which is the layout expected by the benchmark loaders:

    python tools/convert_monash_tsf.py dataset/us_births_dataset.tsf
    # -> dataset/us_births_dataset.csv

The ``.tsf`` files are available from the Monash Time Series Forecasting
Archive (https://forecastingdata.org/).
"""
import argparse
from datetime import datetime

import pandas as pd

PANDAS_FREQ = {
    "yearly": "Y",
    "quarterly": "Q",
    "monthly": "M",
    "weekly": "W",
    "daily": "D",
    "hourly": "H",
}


def read_tsf(path):
    """Return (list of {'series_name', 'start_timestamp', 'values'}, frequency)."""
    attributes, series, frequency, in_data = [], [], None, False
    with open(path, "r", encoding="cp1252") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if not in_data:
                if line.startswith("@attribute"):
                    _, name, kind = line.split(" ")
                    attributes.append((name, kind))
                elif line.startswith("@frequency"):
                    frequency = line.split(" ")[1]
                elif line.startswith("@data"):
                    in_data = True
                continue
            fields = line.split(":")
            record = {}
            for (name, kind), value in zip(attributes, fields[:-1]):
                if kind == "date":
                    value = datetime.strptime(value, "%Y-%m-%d %H-%M-%S")
                elif kind == "numeric":
                    value = int(value)
                record[name] = value
            record["values"] = [float("nan") if v == "?" else float(v) for v in fields[-1].split(",")]
            series.append(record)
    return series, frequency


def tsf_to_frame(path):
    series, frequency = read_tsf(path)
    freq = PANDAS_FREQ.get(str(frequency).lower(), "D")
    frames = []
    for record in series:
        dates = pd.date_range(start=record["start_timestamp"], periods=len(record["values"]), freq=freq)
        frames.append(pd.DataFrame({"date": dates, record["series_name"]: record["values"]}))
    df = frames[0]
    for other in frames[1:]:
        df = df.merge(other, on="date", how="outer")
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = df["date"].dt.strftime("%Y-%m-%d %H:%M:%S")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("tsf", help="Input .tsf file")
    parser.add_argument("--out", default=None, help="Output CSV (default: same name with .csv)")
    args = parser.parse_args()
    out = args.out or args.tsf[:-4] + ".csv"
    tsf_to_frame(args.tsf).to_csv(out, index=False)
    print(f"Wrote {out}")
