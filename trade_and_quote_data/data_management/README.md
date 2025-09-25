# Data Management

This directory contains tools for downloading and managing SPY options data using Spark.

## SPY Trades Downloader

Downloads SPY options trades data directly from Polygon.io's S3 bucket using Spark for efficient processing.

### Features

- **Direct S3 Access**: Reads from Polygon.io's S3 bucket (`s3a://flatfiles/us_options_opra/trades_v1/`)
- **Spark Processing**: Uses PySpark for distributed processing of large datasets
- **SPY Filtering**: Automatically filters for SPY options only
- **No Aggregations**: Downloads raw trades data without any processing
- **Year-based**: Downloads data for a specific year

### Usage

```bash
# Download SPY trades for 2020
python data_management/spy_trades_downloader.py --year 2020

# Download SPY trades for 2021 with custom output directory
python data_management/spy_trades_downloader.py --year 2021 --output-dir data/spy_2021

# Download SPY trades for 2022
python data_management/spy_trades_downloader.py --year 2022
```

### Requirements

- Java 11 or 17 (for Spark compatibility)
- PySpark
- PostgreSQL (local instance)
- S3 credentials for Polygon.io
- Python packages: `psycopg2-binary`, `sqlalchemy`, `pandas`

### Output

- **Parquet Format**: `data/spy_trades_{year}/` (default)
- **PostgreSQL Table**: `spy_trades` table in local database
- **Content**: Raw SPY options trades data without aggregations

### PostgreSQL Setup

1. **Install PostgreSQL** (if not already installed):
   ```bash
   # macOS with Homebrew
   brew install postgresql
   brew services start postgresql
   
   # Or use Docker
   docker run --name postgres-spy -e POSTGRES_PASSWORD=postgres -p 5432:5432 -d postgres
   ```

2. **Create Database**:
   ```sql
   CREATE DATABASE spy_trades;
   ```

3. **Install Python Dependencies**:
   ```bash
   pip install -r data_management/requirements.txt
   ```

### Examples

```bash
# Download 2020 data (both Parquet and PostgreSQL)
python data_management/spy_trades_downloader.py --year 2020

# Download 2021 data with custom PostgreSQL settings
python data_management/spy_trades_downloader.py --year 2021 \
  --postgres-host localhost \
  --postgres-db spy_trades \
  --postgres-user postgres \
  --postgres-password your_password

# Download 2022 data (Parquet only, skip PostgreSQL)
python data_management/spy_trades_downloader.py --year 2022 --skip-postgres

# Download 2023 data to custom location
python data_management/spy_trades_downloader.py --year 2023 --output-dir data/raw/spy_2023
```

### Data Schema

The downloaded data contains raw SPY options trades with columns like:
- `ticker`: Options symbol (e.g., O:SPY240102C00402000)
- `price`: Trade price
- `size`: Trade size
- `sip_timestamp`: Timestamp
- And other raw trade fields

### Performance

- **Memory**: 4GB driver/executor memory
- **Partitions**: Optimized for ~256MB partitions
- **Processing**: Distributed Spark processing
- **Storage**: Efficient Parquet format
