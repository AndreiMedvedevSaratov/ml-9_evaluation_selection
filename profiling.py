import pandas as pd
import pandas_profiling

df = pd.read_csv("data/train.csv")

results = df.profile_report(title="Profiling Report", progress_bar=False)

results.to_file("profiling.html")
