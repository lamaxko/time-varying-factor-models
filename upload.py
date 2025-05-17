import os
import pandas as pd
from helpers.db_manager import DB

def read_and_merge_last_omega_columns(path_dir):
    merged_df = None

    for filename in sorted(os.listdir(path_dir)):
        if "_trained_weights.csv" in filename:
            file_path = os.path.join(path_dir, filename)

            # Read required columns only
            df = pd.read_csv(file_path, usecols=["date", "id_stock", "omega"])

            # Pivot: id_stock = index, date = columns, omega = values
            df_pivot = df.pivot(index="id_stock", columns="date", values="omega")

            # Get the last column (max date)
            last_date = df_pivot.columns.max()
            df_last_col = df_pivot[[last_date]].copy()

            # Rename column to match the year (extracted from filename)
            year_key = filename.split("_")[0].split("-")[1]  # e.g. "2006"
            df_last_col.columns = [f"omega_{year_key}"]

            # Merge with overall dataframe
            if merged_df is None:
                merged_df = df_last_col
            else:
                merged_df = merged_df.join(df_last_col, how="outer")

    return merged_df

def read_and_merge_avg_omega_columns(path_dir):
    merged_df = None

    for filename in sorted(os.listdir(path_dir)):
        if "_trained_weights.csv" in filename:
            file_path = os.path.join(path_dir, filename)

            # Read required columns
            df = pd.read_csv(file_path, usecols=["date", "id_stock", "omega"])

            # Compute mean omega per id_stock
            df_yearly_avg = df.groupby("id_stock")["omega"].mean().to_frame()

            # Rename column to match the year (extracted from filename)
            year_key = filename.split("_")[0].split("-")[1]  # e.g. "2006"
            df_yearly_avg.columns = [f"omega_{year_key}"]

            # Merge with overall dataframe
            if merged_df is None:
                merged_df = df_yearly_avg
            else:
                merged_df = merged_df.join(df_yearly_avg, how="outer")

    return merged_df

def read_and_merge_avg_omega_fullfile(path_dir):
    merged_df = None

    for filename in sorted(os.listdir(path_dir)):
        if "_trained_weights.csv" in filename:
            file_path = os.path.join(path_dir, filename)

            # Read required columns
            df = pd.read_csv(file_path, usecols=["id_stock", "omega"])

            # Compute full average omega per id_stock (across all dates in the file)
            df_avg = df.groupby("id_stock")["omega"].mean().to_frame()

            # Use the last year from the filename for the column name
            year_key = filename.split("_")[0].split("-")[1]
            df_avg.columns = [f"omega_{year_key}"]

            # Merge
            if merged_df is None:
                merged_df = df_avg
            else:
                merged_df = merged_df.join(df_avg, how="outer")

    return merged_df

def insert_factor_if_not_exists(db, factor_name):
    query = "SELECT id_factor FROM factor_mapper WHERE factor_name = ?"
    existing_id = db.fetch(query, params=(factor_name,), output="one")

    if existing_id:
        return existing_id

    max_id = db.fetch("SELECT MAX(id_factor) FROM factor_mapper", output="one") or 0
    new_id = max_id + 1

    df_new_factor = pd.DataFrame({
        "id_factor": [new_id],
        "factor_name": [factor_name]
    })
    db.upload_df(df_new_factor, "factor_mapper")

    return new_id


def reshape_omega_dataframe(merged_df, id_factor):
    long_format = []

    for col in merged_df.columns:
        if col.startswith("omega_"):
            year = int(col.split("_")[1])
            df_year = merged_df[[col]].copy()
            df_year = df_year.rename(columns={col: "value"})
            df_year["year"] = year
            df_year["id_stock"] = df_year.index
            df_year["id_factor"] = id_factor
            long_format.append(df_year[["year", "id_factor", "id_stock", "value"]])

    final_df = pd.concat(long_format, ignore_index=True)
    return final_df


if __name__ == "__main__":
    path_dir = r"C:\Users\lasse.kock\Desktop\ms_thesis\code\gan_weights"

    # Step 1: Read & merge omega columns
    final_df = read_and_merge_avg_omega_fullfile(path_dir)
    print("Merged DataFrame shape:", final_df.shape)
    print(final_df.head())

    # Step 2: Insert 'omega_i' into `factors` table
    factor_id = insert_factor_if_not_exists(DB, "omega_lag_one_window_avrg")
    print(f"Using id_factor = {factor_id}")
 
    # Step 3: Reshape to long format for DB upload
    omega_long_df = reshape_omega_dataframe(final_df, factor_id)
    print("Long-format DataFrame preview:")
    print(omega_long_df.head())
 
    # # Step 4: Upload to database
    DB.upload_df(omega_long_df, "factors")
    print("Upload complete.")
