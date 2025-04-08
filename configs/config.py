from dotenv import load_dotenv
import os
load_dotenv()

#data lake base path
DATA_LAKE_BASE_PATH = "/Users/jjcosta/Desktop/git repo/timescale_v_duckdb_testing/data"

#mySQL DB connection url function using lambda
DB_URL = lambda database: f"mysql+pymysql://{os.getenv('DB_USER')}:{os.getenv('DB_PASSWORD')}@{os.getenv('DB_HOST')}:{os.getenv('DB_PORT')}/{database}"
