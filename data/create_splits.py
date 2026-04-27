"""
Create train/validation/test splits from converted_data.csv
using the provided NACC ID files.
"""
import pandas as pd
import os

# paths
ROOT = os.path.dirname(os.path.abspath(__file__))
converted_path = os.path.join(ROOT, "datasets", "example_conversion_scripts", "converted_data.csv")
train_ids_path = os.path.join(ROOT, "datasets", "NACC_train_ids.csv")
test_ids_path = os.path.join(ROOT, "datasets", "NACC_test_ids.csv")

# output directories
train_cohort_dir = os.path.join(ROOT, "training_cohorts")
split_dir = os.path.join(ROOT, "train_vld_test_split_updated")
os.makedirs(train_cohort_dir, exist_ok=True)
os.makedirs(split_dir, exist_ok=True)

# load data
print("Loading converted_data.csv ...")
df = pd.read_csv(converted_path)
print(f"  Total rows: {len(df)}")

train_ids = pd.read_csv(train_ids_path)
test_ids = pd.read_csv(test_ids_path)
print(f"  Train IDs: {len(train_ids)}, Test IDs: {len(test_ids)}")

# rename NACCID -> ID if needed (CSVDataset expects 'ID')
if "NACCID" in df.columns and "ID" not in df.columns:
    df.rename(columns={"NACCID": "ID"}, inplace=True)

# match on (ID, NACCVNUM)
train_ids = train_ids.rename(columns={"NACCID": "ID"})
test_ids = test_ids.rename(columns={"NACCID": "ID"})

train_keys = set(zip(train_ids["ID"], train_ids["NACCVNUM"]))
test_keys = set(zip(test_ids["ID"], test_ids["NACCVNUM"]))

df["_key"] = list(zip(df["ID"], df["NACCVNUM"]))
df_train_all = df[df["_key"].isin(train_keys)].drop(columns=["_key"])
df_test = df[df["_key"].isin(test_keys)].drop(columns=["_key"])
df.drop(columns=["_key"], inplace=True)

print(f"  Matched train rows: {len(df_train_all)}")
print(f"  Matched test rows: {len(df_test)}")

# split train into train/validation (90/10)
df_train_all = df_train_all.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(len(df_train_all) * 0.9)
df_train = df_train_all.iloc[:split_idx]
df_vld = df_train_all.iloc[split_idx:]

print(f"  Train: {len(df_train)}, Validation: {len(df_vld)}")

# save
# 1. full training cohort (train + val combined)
out1 = os.path.join(train_cohort_dir, "new_nacc_revised_selection.csv")
df_train_all.to_csv(out1, index=False)
print(f"Saved: {out1}")

# 2. train split
out2 = os.path.join(split_dir, "demo_train.csv")
df_train.to_csv(out2, index=False)
print(f"Saved: {out2}")

# 3. validation split
out3 = os.path.join(split_dir, "demo_vld.csv")
df_vld.to_csv(out3, index=False)
print(f"Saved: {out3}")

# 4. test split
out4 = os.path.join(split_dir, "nacc_test_with_np_cli.csv")
df_test.to_csv(out4, index=False)
print(f"Saved: {out4}")

print("\nDone!")
