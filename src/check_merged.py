import pandas as pd

df = pd.read_csv("data/processed/merged_dataset.csv")

print(df.shape)

print(
    df[
        [
            "file_id",
            "brain_health_score",
            "total_cognition_score",
            "fluid_cognition_score",
            "crystallized_cognition_score",
            "AHI_1_B"
        ]
    ].head()
)