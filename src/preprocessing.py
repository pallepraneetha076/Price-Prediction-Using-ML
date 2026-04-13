import pandas as pd

def preprocess_data(path):
    data = pd.read_csv(path)

    # Handle missing values
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna(data[col].mode()[0])
        else:
            data[col] = data[col].fillna(data[col].median())

    # Split target
    y = data["SalePrice"]
    X = data.drop(["SalePrice", "Id"], axis=1)

    # Encoding
    X = pd.get_dummies(X, drop_first=True)

    return X, y