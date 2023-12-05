import pandas as pd


def testing():
    fastandfurious_df = pd.read_parquet(
        "/opt/viva/analysis/results/results_variable,fastandfurious,clc@0.1,crashes-otherstuff"
    )
    fastandfurious_df = fastandfurious_df.drop_duplicates(subset=["frameuri"])
    print(len(fastandfurious_df))


testing()
