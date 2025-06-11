import pandas as pd
import psycopg2
import logging
from tqdm import tqdm
import json
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('precompute_percentages.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Database connection config
DATABASE_URL = os.environ.get("DATABASE_URL")

# Checkpoint file
CHECKPOINT_FILE = 'checkpoint.json'

# Connect to local PostgreSQL
try:
    conn = psycopg2.connect(DATABASE_URL)
    cursor = conn.cursor()
    logger.info("Connected to local PostgreSQL")
except Exception as e:
    logger.error(f"Failed to connect to database: {str(e)}")
    raise

# Drop and recreate table to ensure fresh start
try:
    logger.info("Dropping historical_percentages table if exists")
    cursor.execute("DROP TABLE IF EXISTS historical_percentages")
    cursor.execute("""
        CREATE TABLE historical_percentages (
            branch VARCHAR,
            move_type VARCHAR,
            month INT,
            day INT,
            avg_percentage FLOAT,
            PRIMARY KEY (branch, move_type, month, day)
        )
    """)
    conn.commit()
    logger.info("Created fresh historical_percentages table")
except Exception as e:
    logger.error(f"Error creating table: {str(e)}")
    raise

# Clear checkpoint
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
    logger.info("Cleared checkpoint file")

# Load checkpoint (should be empty)
def load_checkpoint():
    if os.path.exists(CHECKPOINT_FILE):
        with open(CHECKPOINT_FILE, 'r') as f:
            return json.load(f)
    return {'branch': None, 'move_type': None, 'month': 1, 'day': 1}

# Save checkpoint
def save_checkpoint(branch, move_type, month, day):
    with open(CHECKPOINT_FILE, 'w') as f:
        json.dump({'branch': branch, 'move_type': move_type, 'month': month, 'day': day}, f)

# Load data for 2021–2024
try:
    logger.info("Loading data from PostgreSQL for 2021–2024")
    historical_df = pd.read_sql_query('SELECT "Date", "Branch", "Count" FROM historical_df WHERE EXTRACT(YEAR FROM "Date") IN (2021, 2022, 2023, 2024)', conn)
    move_df = pd.read_sql_query('SELECT "Date", "Branch", "MoveType", "Count" FROM move_df WHERE EXTRACT(YEAR FROM "Date") IN (2021, 2022, 2023, 2024)', conn)
    logger.info(f"Loaded {len(historical_df)} rows from historical_df, {len(move_df)} rows from move_df")
except Exception as e:
    logger.error(f"Error loading data: {str(e)}")
    raise

# Validate data
try:
    if historical_df.empty or move_df.empty:
        raise ValueError("One or both DataFrames are empty")
    if historical_df['Count'].isnull().any() or move_df['Count'].isnull().any():
        logger.warning("Null values found in Count column")
except Exception as e:
    logger.error(f"Data validation error: {str(e)}")
    raise

# Convert dates and extract month/day
try:
    historical_df['Date'] = pd.to_datetime(historical_df['Date'])
    move_df['Date'] = pd.to_datetime(move_df['Date'])
    move_df['Month'] = move_df['Date'].dt.month
    move_df['Day'] = move_df['Date'].dt.day
    historical_df['Month'] = historical_df['Date'].dt.month
    historical_df['Day'] = historical_df['Date'].dt.day
except Exception as e:
    logger.error(f"Error processing dates: {str(e)}")
    raise

# Pre-group data
try:
    logger.info("Pre-grouping data")
    move_grouped = move_df.groupby(['Branch', 'MoveType', 'Month', 'Day'])['Count'].sum().reset_index()
    historical_grouped = historical_df.groupby(['Branch', 'Month', 'Day'])['Count'].sum().reset_index()
    logger.info(f"Grouped data: {len(move_grouped)} move records, {len(historical_grouped)} historical records")
except Exception as e:
    logger.error(f"Error grouping data: {str(e)}")
    raise

# Prepare batch inserts
batch_size = 1000  # Reduced for stability
batch = []
total_inserts = 0

# Load checkpoint
checkpoint = load_checkpoint()
start_branch = checkpoint['branch']
start_move_type = checkpoint['move_type']
start_month = checkpoint['month']
start_day = checkpoint['day']
skip_until = False if start_branch is None else True

# Iterate over combinations
branches = sorted(move_df['Branch'].unique())  # Sorted for consistency
move_types = sorted(move_df['MoveType'].unique())
total_iterations = len(branches) * len(move_types) * 12 * 31
progress_bar = tqdm(total=total_iterations, desc="Processing combinations", initial=0)

for branch in branches:
    for move_type in move_types:
        for month in range(1, 13):
            for day in range(1, 32):
                # Skip until checkpoint
                if skip_until:
                    if (branch == start_branch and move_type == start_move_type and month == start_month and day == start_day):
                        skip_until = False
                    progress_bar.update(1)
                    continue

                try:
                    # Skip invalid dates
                    if pd.isna(pd.to_datetime(f'2021-{month:02d}-{day:02d}', errors='coerce')):
                        progress_bar.update(1)
                        continue

                    # Get counts
                    move_count = move_grouped[
                        (move_grouped['Branch'] == branch) &
                        (move_grouped['MoveType'] == move_type) &
                        (move_grouped['Month'] == month) &
                        (move_grouped['Day'] == day)
                    ]['Count'].sum()

                    total_count = historical_grouped[
                        (historical_grouped['Branch'] == branch) &
                        (historical_grouped['Month'] == month) &
                        (historical_grouped['Day'] == day)
                    ]['Count'].sum()

                    if total_count > 0:
                        avg_percentage = (move_count / total_count) * 100
                        batch.append((branch, move_type, month, day, avg_percentage))
                    else:
                        logger.warning(f"Skipping {branch}, {move_type}, {month}, {day}: total_count is 0")
                        continue

                    # Batch insert
                    if len(batch) >= batch_size:
                        try:
                            cursor.executemany("""
                                INSERT INTO historical_percentages (branch, move_type, month, day, avg_percentage)
                                VALUES (%s, %s, %s, %s, %s)
                                ON CONFLICT (branch, move_type, month, day) DO UPDATE
                                SET avg_percentage = EXCLUDED.avg_percentage
                            """, batch)
                            conn.commit()
                            total_inserts += len(batch)
                            logger.info(f"Inserted {total_inserts} records")
                            batch = []
                        except Exception as e:
                            logger.error(f"Error inserting batch: {str(e)}")
                            conn.rollback()

                    # Save checkpoint
                    save_checkpoint(branch, move_type, month, day + 1)

                except Exception as e:
                    logger.error(f"Error processing {branch}, {move_type}, {month}, {day}: {str(e)}")
                    save_checkpoint(branch, move_type, month, day + 1)
                finally:
                    progress_bar.update(1)

# Insert remaining batch
if batch:
    try:
        cursor.executemany("""
            INSERT INTO historical_percentages (branch, move_type, month, day, avg_percentage)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (branch, move_type, month, day) DO UPDATE
            SET avg_percentage = EXCLUDED.avg_percentage
        """, batch)
        conn.commit()
        total_inserts += len(batch)
        logger.info(f"Inserted {total_inserts} records (final batch)")
    except Exception as e:
        logger.error(f"Error inserting final batch: {str(e)}")
        conn.rollback()

# Clean up
cursor.close()
conn.close()
progress_bar.close()
logger.info("Completed precomputation")
if os.path.exists(CHECKPOINT_FILE):
    os.remove(CHECKPOINT_FILE)
