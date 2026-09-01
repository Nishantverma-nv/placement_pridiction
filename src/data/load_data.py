import pandas as pd

def load_data():
    df = pd.read_csv(r"C:\Users\prath\PycharmProjects\PythonPlacementpredictionsystem\data\placement_data.csv")
    return df

def get_summary(df):
    return {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "target": "placement status"
    }

if __name__ == "__main__":
    df = load_data()
    summary = get_summary(df)
    print("Summary:", summary)


    print("\nFirst 5 rows:")
    print(df.head())
    print(df.tail())

