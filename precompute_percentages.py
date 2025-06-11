import pandas as pd
import psycopg2
import os
import logging
from tqdm import tqdm
 
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
 
DATABASE_URL = os.environ.get("DATABASE_URL")
conn = psycopg2.connect(DATABASE_URL)
cursor = conn.cursor()
 
cursor.execute("""
    CREATE TABLE IF NOT EXISTS historical_percentages (
        branch VARCHAR,
        move_type VARCHAR,
        month INT,
        day INT,
        avg_percentage FLOAT,
        PRIMARY KEY (branch, move_type, month, day)
    )
""")
conn.commit()
logger.info("Created historical_percentages table")
 
logger.info("Loading data from PostgreSQL for 2023–2024")
historical_df = pd.read_sql_query('SELECT "Date", "Branch", "Count" FROM historical_df WHERE EXTRACT(YEAR FROM "Date") IN (2023, 2024)', conn)
move_df = pd.read_sql_query('SELECT "Date", "Branch", "MoveType", "Count" FROM move_df WHERE EXTRACT(YEAR FROM "Date") IN (2023, 2024)', conn)
 
historical_df['Date'] = pd.to_datetime(historical_df['Date'])
move_df['Date'] = pd.to_datetime(move_df['Date'])
move_df['Month'] = move_df['Date'].dt.month
move_df['Day'] = move_df['Date'].dt.day
historical_df['Month'] = historical_df['Date'].dt.month
historical_df['Day'] = historical_df['Date'].dt.day
 
logger.info("Pre-grouping data")
move_grouped = move_df.groupby(['Branch', 'MoveType', 'Month', 'Day'])['Count'].sum().reset_index()
historical_grouped = historical_df.groupby(['Branch', 'Month', 'Day'])['Count'].sum().reset_index()
 
batch_size = 10000
batch = []
total_inserts = 0
 
branches = move_df['Branch'].unique()
move_types = move_df['MoveType'].unique()
total_iterations = len(branches) * len(move_types) * 12 * 31
progress_bar = tqdm(total=total_iterations, desc="Processing combinations")
 
for branch in branches:
    for move_type in move_types:
        for month in range(1, 13):
            for day in range(1, 32):
                try:
                    if pd.isna(pd.to_datetime(f'2023-{month:02d}-{day:02d}', errors='coerce')):
                        progress_bar.update(1)
                        continue
 
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
 
                        if len(batch) >= batch_size:
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
                    logger.error(f"Error processing {branch}, {move_type}, {month}, {day}: {str(e)}")
                finally:
                    progress_bar.update(1)
 
if batch:
    cursor.executemany("""
        INSERT INTO historical_percentages (branch, move_type, month, day, avg_percentage)
        VALUES (%s, %s, %s, %s, %s)
        ON CONFLICT (branch, move_type, month, day) DO UPDATE
        SET avg_percentage = EXCLUDED.avg_percentage
    """, batch)
    conn.commit()
    total_inserts += len(batch)
    logger.info(f"Inserted {total_inserts} records (final batch)")
 
cursor.close()
conn.close()
progress_bar.close()
logger.info("Completed precomputation")
