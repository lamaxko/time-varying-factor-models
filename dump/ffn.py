import torch
import pandas as pd
from fred import FredApi
from lstm import state_rnn
from calc_factors import FactorCalc

def prepare_x_ti():
    # Get I_t (macro vars) → h_t
    I_t = FredApi().data_stationary
    h_t = state_rnn(I_t)  # DataFrame: index = date, cols = h0, h1, ...

    # Format h_t
    h_t = h_t.reset_index()
    h_t["date"] = pd.to_datetime(h_t["sasdate"])
    h_t = h_t.drop(columns=["sasdate"])

    # Get I_ti (firm panel)
    I_ti = FactorCalc().panel # contains 'date', 'id_stock', firm features, and ExcessRet

    # Merge h_t into I_ti to form x_ti
    x_ti_df = I_ti.merge(h_t, on="date", how="inner")

    # Separate inputs and target
    input_cols = [col for col in x_ti_df.columns if col not in ["date", "id_stock", "ExcessRet"]]
    X_ti = torch.tensor(x_ti_df[input_cols].values, dtype=torch.float32)
    R_ti = torch.tensor(x_ti_df["ExcessRet"].values, dtype=torch.float32).unsqueeze(1)

    return X_ti, R_ti, x_ti_df  # returns design matrix, returns, full merged df

def train_test_split(X_ti, R_ti, full_df, train_frac=0.7):
    """
    Splits the data into train/test sets based on time.
    
    Parameters:
    - X_ti: tensor of inputs
    - R_ti: tensor of excess returns
    - full_df: original DataFrame with 'date' column
    - train_frac: fraction of time used for training (default: 70%)

    Returns:
    - X_train, R_train, X_test, R_test
    """
    # Ensure datetime and sort
    full_df["date"] = pd.to_datetime(full_df["date"])
    full_df = full_df.sort_values("date")

    # Compute time-based split
    split_date = full_df["date"].quantile(train_frac)

    # Boolean masks
    train_mask = full_df["date"] <= split_date
    test_mask = full_df["date"] > split_date

    # Apply masks
    X_train = X_ti[train_mask.values]
    R_train = R_ti[train_mask.values]
    X_test = X_ti[test_mask.values]
    R_test = R_ti[test_mask.values]

    print(f"Train split: {train_mask.sum()} samples")
    print(f"Test split: {test_mask.sum()} samples")
    print(f"Split date: {split_date.strftime('%Y-%m-%d')}")

    return X_train, R_train, X_test, R_test

if __name__ == "__main__":
    X_ti, R_ti, full_df = prepare_x_ti()
    print("x_ti shape:", X_ti.shape)
    print("r_ti shape:", R_ti.shape)
    print("Features:", full_df.columns.tolist())

    X_train, R_train, X_test, R_test = train_test_split(X_ti, R_ti, full_df)

