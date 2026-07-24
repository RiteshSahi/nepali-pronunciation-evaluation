import pandas as pd
df = pd.read_csv("../dataset/user2_pronunciation_dataset.csv")

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", None)

print("=== Mean / Min / Max per label ===")
print(df.groupby("label")[["dtw","duration","wer","cer"]].agg(["mean","min","max"]))

print("\n=== Row counts per label ===")
print(df["label"].value_counts())