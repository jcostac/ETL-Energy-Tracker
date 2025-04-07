from dotenv import load_dotenv
import os
load_dotenv()

#data lake base path
DATA_LAKE_BASE_PATH = "/Users/jjcosta/Desktop/git repo/timescale_v_duckdb_testing/data"

#mySQL DB connection url string
DB_URL = (lambda user, password, host, port, database: f"mysql+pymysql://{user}:{password}@{host}:{port}/{database}")(
    os.getenv('DB_USER'),
    os.getenv('DB_PASSWORD'),
    os.getenv('DB_HOST'),
    os.getenv('DB_PORT'),
    os.getenv('DB_DATABASE')
)




