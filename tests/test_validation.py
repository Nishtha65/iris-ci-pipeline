import pandas as pd

def test_data_not_empty():
    df = pd.read_csv("data/iris.csv")
    assert not df.empty, "Dataset is empty!"

def test_no_null_values():
    df = pd.read_csv("data/iris.csv")
    assert df.isnull().sum().sum() == 0, "Dataset has missing values!"
