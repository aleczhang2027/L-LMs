import pandas as pd
from sklearn.model_selection import train_test_split

INPUT_PATH = "data/twitter_training.csv"
TRAIN_OUTPUT = "data/videogames_train.csv"
TEST_OUTPUT  = "data/videogames_test.csv"

TEST_SIZE   = 0.15
RANDOM_SEED = 42

VIDEOGAME_TOPICS = [
    "ApexLegends",
    "AssassinsCreed",
    "Battlefield",
    "Borderlands",
    "CS-GO",
    "CallOfDuty",
    "CallOfDutyBlackopsColdWar",
    "Cyberpunk2077",
    "Dota2",
    "FIFA",
    "Fortnite",
    "GrandTheftAuto(GTA)",
    "Hearthstone",
    "LeagueOfLegends",
    "MaddenNFL",
    "NBA2K",
    "Overwatch",
    "PlayStation5(PS5)",
    "PlayerUnknownsBattlegrounds(PUBG)",
    "RedDeadRedemption(RDR)",
    "TomClancysGhostRecon",
    "TomClancysRainbowSix",
    "WorldOfCraft",
    "Xbox(Xseries)",
]

df = pd.read_csv(INPUT_PATH, header=None, names=["id", "topic", "sentiment", "text"])
print(f"Loaded {len(df):,} rows total")

df = df[df["topic"].isin(VIDEOGAME_TOPICS)].copy()
print(f"After filtering to video game topics: {len(df):,} rows")

df = df[df["sentiment"] != "Irrelevant"].copy()
print(f"After dropping Irrelevant sentiment: {len(df):,} rows")

df["domain"] = "video_games"

df = df.dropna(subset=["text"]).copy()
df["text"] = df["text"].astype(str).str.strip()
df = df[df["text"] != ""].copy()
print(f"After dropping empty text rows: {len(df):,} rows")

train_df, test_df = train_test_split(
    df,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=df["sentiment"],
)

print(f"\nTrain size: {len(train_df):,}")
print(f"Test size:  {len(test_df):,}")
print(f"\nSentiment distribution in train:\n{train_df['sentiment'].value_counts()}")
print(f"\nSentiment distribution in test:\n{test_df['sentiment'].value_counts()}")

train_df.to_csv(TRAIN_OUTPUT, index=False)
test_df.to_csv(TEST_OUTPUT, index=False)

print(f"\nSaved train -> {TRAIN_OUTPUT}")
print(f"Saved test  -> {TEST_OUTPUT}")