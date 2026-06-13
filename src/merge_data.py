import pandas as pd

phi = pd.read_csv("data/processed/phi_results.csv")
clinical = pd.read_excel("data/raw/subject_description.xlsx")

# Clean keys
phi["file_id"] = phi["file_id"].astype(str).str.strip()

clinical["subject_id"] = (
    clinical["Annonymized_Name"]
    .astype(str)
    .str.replace(".edf", "", regex=False)
    .str.strip()
)

print("PHI rows:", len(phi))
print("Clinical rows:", len(clinical))

merged = pd.merge(
    phi,
    clinical,
    left_on="file_id",
    right_on="subject_id",
    how="inner"
)

print("Merged rows:", len(merged))

merged.to_csv(
    "data/processed/merged_dataset.csv",
    index=False
)

print("Saved merged_dataset.csv")