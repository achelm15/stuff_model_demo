# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Materialize MLB GUMBO Pitch Data
# MAGIC
# MAGIC This notebook follows the MLB GUMBO Slim pattern from
# MAGIC `ABoothInTheWild/databricks_demos/MLB GUMBO E2E/Slim`:
# MAGIC
# MAGIC 1. Call the MLB Stats API schedule endpoint to collect `gamePk` values.
# MAGIC 2. Pull each game's GUMBO live-feed JSON from `api/v1.1/game/{gamePk}/feed/live`.
# MAGIC 3. Land raw JSON files in a Unity Catalog Volume.
# MAGIC 4. Incrementally ingest the raw files into Delta with Auto Loader.
# MAGIC 5. Explode `liveData.plays.allPlays[*].playEvents[*]` into pitch rows.
# MAGIC 6. Write the `silver_pitches` contract consumed by the MLflow notebooks.
# MAGIC
# MAGIC The target `pitch_rv` is a deterministic outcome proxy from actual GUMBO pitch results.
# MAGIC It is not random data and can be swapped for a richer run-expectancy target later.

# COMMAND ----------

# MAGIC %run ./_config

# COMMAND ----------

dbutils.widgets.text("seasons", "2024,2025")
dbutils.widgets.text("team_id", "")  # blank = all MLB teams
dbutils.widgets.text("max_games_per_season", "0")  # 0 = all scheduled games
dbutils.widgets.dropdown("refresh_raw", "false", ["false", "true"])

SEASONS = [int(s.strip()) for s in dbutils.widgets.get("seasons").split(",") if s.strip()]
TEAM_ID_RAW = dbutils.widgets.get("team_id").strip()
TEAM_ID = int(TEAM_ID_RAW) if TEAM_ID_RAW else None
MAX_GAMES_PER_SEASON = int(dbutils.widgets.get("max_games_per_season"))
REFRESH_RAW = dbutils.widgets.get("refresh_raw").lower() == "true"

VOLUME = "gumbo_raw"
RAW_DIR = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/mlb_gumbo"
DATA_LOCATION = f"{RAW_DIR}/season=*"
CHECKPOINT_LOCATION_BRONZE = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/_checkpoints/bronze_gumbo_games"
CHECKPOINT_LOCATION_SILVER = f"/Volumes/{CATALOG}/{SCHEMA}/{VOLUME}/_checkpoints/silver_pitches"
SILVER_TABLE = f"{CATALOG}.{SCHEMA}.silver_pitches"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA}.bronze_gumbo_games"

# COMMAND ----------

import json
import os
import time
from datetime import datetime

import requests
from pyspark.sql import functions as F

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")
spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.{VOLUME}")
os.makedirs(RAW_DIR, exist_ok=True)

# COMMAND ----------

def request_json(url, retries=3, sleep_seconds=0.4):
    last_error = None
    for attempt in range(retries):
        try:
            response = requests.get(url, timeout=30)
            response.raise_for_status()
            return response.json()
        except Exception as err:
            last_error = err
            if attempt < retries - 1:
                time.sleep(sleep_seconds * (attempt + 1))
    raise RuntimeError(f"Failed request after {retries} attempts: {url}") from last_error


def schedule_game_pks(season, team_id, max_games):
    url = (
        "https://statsapi.mlb.com/api/v1/schedule"
        f"?sportId=1&gameType=R,D,F,L,W&startDate={season}-03-01&endDate={season}-11-05"
    )
    if team_id is not None:
        url += f"&teamId={team_id}"
    data = request_json(url)
    games = []
    for date in data.get("dates", []):
        for game in date.get("games", []):
            if game.get("gamePk"):
                games.append(int(game["gamePk"]))
    return games if max_games <= 0 else games[:max_games]


downloaded = []
for season in SEASONS:
    season_dir = f"{RAW_DIR}/season={season}"
    os.makedirs(season_dir, exist_ok=True)
    game_pks = schedule_game_pks(season, TEAM_ID, MAX_GAMES_PER_SEASON)
    scope = f"team_id={TEAM_ID}" if TEAM_ID is not None else "all MLB teams"
    print(f"season={season} {scope} games={len(game_pks)}")
    for game_pk in game_pks:
        path = f"{season_dir}/game_data_{game_pk}.json"
        if os.path.exists(path) and not REFRESH_RAW:
            downloaded.append(path)
            continue
        url = f"https://statsapi.mlb.com/api/v1.1/game/{game_pk}/feed/live?hydrate=credits,alignment,flags,officials,preState"
        payload = request_json(url)
        with open(path, "w") as handle:
            json.dump(payload, handle)
        downloaded.append(path)
        time.sleep(0.05)

print(f"Raw GUMBO files available: {len(downloaded)} under {RAW_DIR}")

# COMMAND ----------

# Define bronze table
current_run = datetime.now()

# Ingest
query = (
    spark.readStream
    .format("cloudFiles")
    .option("cloudFiles.format", "json")
    .option("singleVariantColumn", "data")
    .load(f"{DATA_LOCATION}/*.json")
    .withColumn("file_path", F.col("_metadata.file_path"))
    .withColumn("file_name", F.col("_metadata.file_name"))
    .withColumn("file_size", F.col("_metadata.file_size"))
    .withColumn("file_modification_time", F.col("_metadata.file_modification_time"))
    .withColumn("file_batch_time", F.lit(current_run))
    .withColumn("last_update_time", F.current_timestamp())
    .writeStream
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_LOCATION_BRONZE)
    .trigger(availableNow=True)
    .toTable(BRONZE_TABLE)
)

query.awaitTermination()

# COMMAND ----------

pitch_events = (
    spark.readStream
    .table(BRONZE_TABLE)
    .withColumn("play", F.explode(F.expr("data:liveData.plays.allPlays::array<variant>")))
    .withColumn("event", F.explode(F.expr("play:playEvents::array<variant>")))
    .where(F.expr("event:isPitch::boolean = true"))
    .selectExpr(
        "data:gameData.game.season::int AS season",
        "data:gameData.datetime.officialDate::date AS game_date",
        "data:gamePk::int AS game_pk",
        "data:gameData.teams.away.id::int AS away_team_id",
        "data:gameData.teams.away.name::string AS away_team_name",
        "data:gameData.teams.home.id::int AS home_team_id",
        "data:gameData.teams.home.name::string AS home_team_name",
        "play:about.atBatIndex::int AS at_bat_index",
        "play:about.inning::int AS inning",
        "event:index::int AS pitch_index",
        "event:pitchNumber::int AS pitch_number",
        "event:isPitch::boolean AS is_pitch",
        "event:details.call.code::string AS call_code",
        "event:details.call.description::string AS call_description",
        "event:details.description::string AS pitch_description",
        "event:details.code::string AS pitch_code",
        "event:details.type.code::string AS pitch_type",
        "event:details.type.description::string AS pitch_type_description",
        "event:details.isInPlay::boolean AS is_in_play",
        "event:details.isStrike::boolean AS is_strike",
        "event:details.isBall::boolean AS is_ball",
        "event:count.balls::int AS balls_count",
        "event:count.strikes::int AS strikes_count",
        "event:preCount.balls::int AS pre_balls_count",
        "event:preCount.strikes::int AS pre_strikes_count",
        "event:pitchData.startSpeed::double AS release_speed",
        "event:pitchData.endSpeed::double AS end_speed",
        "event:pitchData.coordinates.pfxX::double AS pfx_x",
        "event:pitchData.coordinates.pfxZ::double AS pfx_z",
        "event:pitchData.coordinates.pX::double AS plate_x",
        "event:pitchData.coordinates.pZ::double AS plate_z",
        "event:pitchData.coordinates.vX0::double AS vx0",
        "event:pitchData.coordinates.vY0::double AS vy0",
        "event:pitchData.coordinates.vZ0::double AS vz0",
        "event:pitchData.coordinates.aX::double AS ax",
        "event:pitchData.coordinates.aY::double AS ay",
        "event:pitchData.coordinates.aZ::double AS az",
        "event:pitchData.coordinates.x0::double AS release_x",
        "event:pitchData.coordinates.z0::double AS release_z",
        "event:pitchData.breaks.spinRate::double AS spin_rate",
        "event:pitchData.breaks.spinDirection::double AS spin_direction",
        "event:pitchData.breaks.breakVerticalInduced::double AS induced_vertical_break",
        "event:pitchData.breaks.breakHorizontal::double AS horizontal_break",
        "event:pitchData.extension::double AS extension",
        "play:result.event::string AS play_event",
        "play:result.eventType::string AS play_event_type",
        "play:matchup.pitcher.id::int AS pitcher_id",
        "play:matchup.pitcher.fullName::string AS pitcher_name",
        "play:matchup.pitchHand.code::string AS p_throws",
        "play:matchup.batter.id::int AS batter_id",
        "play:matchup.batter.fullName::string AS batter_name",
        "play:matchup.batSide.code::string AS stand",
        "file_path",
        "file_batch_time",
    )
)

# COMMAND ----------

silver = (
    pitch_events
    .withColumn("pfx_x_hnorm", F.expr("CASE WHEN p_throws = 'L' THEN -pfx_x ELSE pfx_x END"))
    .withColumn("release_x_hnorm", F.expr("CASE WHEN p_throws = 'L' THEN -release_x ELSE release_x END"))
    .withColumn("call_text", F.lower(F.coalesce("call_description", "pitch_description", F.lit(""))))
    .withColumn("play_text", F.lower(F.coalesce("play_event_type", "play_event", F.lit(""))))
    .withColumn("vaa", F.expr("""
      CASE
      WHEN vy0 IS NOT NULL AND ay IS NOT NULL AND ay != 0
           AND (pow(vy0, 2) - 2 * ay * (50.0 - 17.0 / 12.0)) > 0
      THEN degrees(-atan((vz0 + az * ((-sqrt(pow(vy0, 2) - 2 * ay * (50.0 - 17.0 / 12.0)) - vy0) / ay)) /
                         (-sqrt(pow(vy0, 2) - 2 * ay * (50.0 - 17.0 / 12.0)))))
      ELSE NULL
      END
    """))
    .withColumn("is_whiff", F.expr("""
      CASE
      WHEN call_text LIKE '%swinging strike%' THEN 1
      ELSE 0
      END
    """))
    .withColumn("pitch_rv", F.expr("""
      CASE
      WHEN play_text LIKE '%home_run%' OR play_text LIKE '%home run%' THEN 0.140
      WHEN play_text LIKE '%triple%' THEN 0.100
      WHEN play_text LIKE '%double%' THEN 0.075
      WHEN play_text LIKE '%single%' THEN 0.050
      WHEN call_text LIKE '%hit by pitch%' THEN 0.045
      WHEN call_text LIKE '%ball%' THEN 0.025
      WHEN call_text LIKE '%foul%' THEN -0.005
      WHEN call_text LIKE '%called strike%' THEN -0.025
      WHEN call_text LIKE '%swinging strike%' THEN -0.040
      WHEN is_in_play THEN 0.015
      WHEN is_strike THEN -0.018
      ELSE 0.000
      END
    """))
    .where("""
      pitch_type IS NOT NULL
      AND release_speed IS NOT NULL
      AND spin_rate IS NOT NULL
      AND pfx_x IS NOT NULL
      AND pfx_z IS NOT NULL
      AND extension IS NOT NULL
      AND p_throws IN ('R', 'L')
      AND stand IN ('R', 'L')
    """)
    .select(
        "game_pk", "at_bat_index", "pitch_number", "season", "game_date",
        "pitcher_id", "pitcher_name", "pitch_type", "p_throws", "stand",
        "release_speed", "spin_rate", "spin_direction", "pfx_x", "pfx_z",
        "pfx_x_hnorm", "release_x", "release_x_hnorm", "release_z", "extension",
        "plate_x", "plate_z", "vx0", "vy0", "vz0", "ax", "ay", "az", "vaa",
        "pitch_rv", "is_whiff", "call_code", "call_description", "pitch_description",
        "play_event", "play_event_type", "pre_balls_count", "pre_strikes_count",
        "file_path", "file_batch_time",
    )
)

silver_query = (
    silver.writeStream
    .outputMode("append")
    .option("checkpointLocation", CHECKPOINT_LOCATION_SILVER)
    .trigger(availableNow=True)
    .toTable(SILVER_TABLE)
)

silver_query.awaitTermination()

spark.sql(f"ALTER TABLE {SILVER_TABLE} SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
spark.sql(f"""
  COMMENT ON TABLE {SILVER_TABLE}
  IS 'Pitch-level MLB GUMBO data from MLB Stats API live feeds, transformed for the MLflow workshop.'
""")

display(spark.sql(f"""
  SELECT season, pitch_type, count(*) AS pitches, round(avg(pitch_rv), 4) AS avg_pitch_rv
  FROM {SILVER_TABLE}
  GROUP BY season, pitch_type
  ORDER BY season, pitch_type
"""))

dbutils.notebook.exit(json.dumps({
    "bronze_table": BRONZE_TABLE,
    "silver_table": SILVER_TABLE,
    "raw_dir": RAW_DIR,
    "files": len(downloaded),
    "rows": spark.table(SILVER_TABLE).count(),
}))
