'''
none_data_clean.py

Whole twitter dataset from: https://www.kaggle.com/datasets/prkhrawsthi/twitter-sentiment-dataset-3-million-labelled-rows

Download as twitter_dataset.csv and run this script to get the clean none dataset.
'''

import pandas as pd

# ---- files ----
INPUT_FILE = "twitter_dataset.csv"
OUTPUT_FILE = "none_sentiment.csv"

# ---- how many random tweets you want ----
SAMPLE_SIZE = 100000   # change this to whatever you want

# ---- read csv ----
df = pd.read_csv(INPUT_FILE)

# ---- randomly sample tweets ----
random_df = df.sample(n=SAMPLE_SIZE, random_state=42)

# ---- save result ----
random_df.to_csv(OUTPUT_FILE, index=False)

# ---- print summary ----
print(f"Original rows: {len(df):,}")
print(f"Random rows selected: {len(random_df):,}")
print(f"Saved to: {OUTPUT_FILE}")