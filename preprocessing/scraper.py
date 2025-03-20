"""
This script fetches song lyrics from Genius API for the songs in the SiTunes dataset.
The lyrics data is saved in both JSON and Parquet formats for flexibility in downstream processing.
"""

import pandas as pd
import time
import lyricsgenius
import json
import pyarrow.parquet as pq
import pyarrow as pa
from tqdm import tqdm
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Set up file paths
root_dir = os.path.dirname(os.path.dirname(__file__))
data_path = os.path.join(root_dir, 'data', 'music_metadata.csv')

# Load the music metadata
data = pd.read_csv(data_path)

# Initialize Genius API client
genius_access_token = os.getenv('GENIUS_ACCESS_TOKEN')
if not genius_access_token:
    raise ValueError("GENIUS_ACCESS_TOKEN not found in environment variables")
genius = lyricsgenius.Genius(genius_access_token)

def get_lyrics(song, singer):
    """
    Fetch lyrics for a specific song from Genius API.
    
    Args:
        song (str): Name of the song
        singer (str): Name of the artist/singer
    
    Returns:
        str or None: Song lyrics if found, None otherwise
    """
    try:
        song_obj = genius.search_song(song, singer)
        if song_obj and song_obj.lyrics:
            return song_obj.lyrics
    except Exception as e:
        print(f"Error fetching lyrics for {song} by {singer}: {e}")
    return None

def fetch_and_save_lyrics(data):
    """
    Fetch lyrics for all songs in the dataset and save results.
    
    Args:
        data (pd.DataFrame): DataFrame containing music metadata with 'music' and 'singer' columns
    
    Saves two files:
        - JSON format for human readability
        - Parquet format for efficient storage and querying
    """
    results = []
    for idx, row in tqdm(data.iterrows(), total=len(data), desc="Fetching lyrics"):
        try:
            song_lyrics = get_lyrics(row['music'], row['singer'])
            results.append({
                "music": row['music'],
                "singer": row['singer'],
                "lyrics": song_lyrics
            })
            
            # Rate limiting to avoid API throttling
            time.sleep(2)
        except Exception as e:
            print(f"Error fetching lyrics for {row['music']} by {row['singer']}: {e}")
            continue
    
    # Save results in JSON format
    json_path = os.path.join(root_dir, 'data', 'lyrics_data.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    # Save results in Parquet format
    parquet_path = os.path.join(root_dir, 'data', 'lyrics_data.parquet')
    df = pd.DataFrame(results)
    table = pa.Table.from_pandas(df)
    pq.write_table(table, parquet_path)

    print(f"\nProcessed {len(results)} songs")
    print("Lyrics saved in both JSON and Parquet formats")

if __name__ == "__main__":
    fetch_and_save_lyrics(data)