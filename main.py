import pandas as pd

try:
    df = pd.read_csv("data/raw/creditcard.csv")

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nShape:")
    print(df.shape)

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nClass distribution:")
    print(df["Class"].value_counts())

except Exception as e:
    print("Error:", e)