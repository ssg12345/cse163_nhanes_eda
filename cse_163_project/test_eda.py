# Sahana Ganesan
# CSE 163 AB - Autumn 2026
# Tests for the NHANES EDA project
# This file checks that the data-cleaning function removes
# missing and invalid responses correctly.

import pandas as pd
from eda import clean_data


def test_clean_data():
    """
    Tests whether cleaning removes younger participants,
    missing scores, and invalid sleep responses.
    """
    data = pd.DataFrame({
        "RIDAGEYR": [65, 72, 55, 68, 75],
        "SLD010H": [7, 99, 8, 6, 5],
        "PAQ650": [1, 2, 1, 2, 1],
        "CFDDS": [50, 40, 60, None, 55]
    })

    result = clean_data(data)

    assert len(result) == 2
    assert result["RIDAGEYR"].min() >= 60
    assert result["SLD010H"].isna().sum() == 0
    assert result["CFDDS"].isna().sum() == 0
    assert result["PAQ650"].isna().sum() == 0
    assert 99 not in result["SLD010H"].values
    assert result["CFDDS"].tolist() == [50, 55]

    print("All tests passed!")


def main():
    """
    Runs the data-cleaning test.
    """
    test_clean_data()


if __name__ == "__main__":
    main()
