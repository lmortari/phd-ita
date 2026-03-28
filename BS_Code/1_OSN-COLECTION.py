# ==============================================================================
# INTEGRATED OPENSKY COLLECTION SYSTEM 
# ==============================================================================
# Description: Unified system for collecting flight trajectories via OpenSky,
# with automatic failure recovery and optimized saving in IPC format.
#
# Features:
# 1. Trajectory collection by period (year, month or week)
# 2. Incremental saving in blocks (.ipc format)
# 3. Lazy processing with Polars for memory optimization
# 4. Failure logging and automatic recovery
# 5. Final report of successes and definitive failures
# ==============================================================================

import pandas as pd
import polars as pl
import pyarrow as pa
import pyarrow.feather as feather
import time
import os
import sys
from datetime import datetime, timedelta, timezone
from pyopensky.trino import Trino

# ==============================================================================
# GLOBAL CONFIGURATIONS
# ==============================================================================

# --- CREDENTIALS (Must be in environment variables) ---
# export OPENSKY_USERNAME="leon.mortari.102254@ga.ita.br
# export OPENSKY_PASSWORD="Leon1234"

# --- OUTPUT DIRECTORY ---
OUTPUT_DIR = "/home/leonmortari/code/phd-ita/BS_Code/Analise/Voos"

# --- COLLECTION CONFIGURATIONS ---
BLOCO_SIZE = 10  # Number of flights per block
MAX_RETRIES = 3  # Attempts per flight
RETRY_DELAY = 5  # Seconds between attempts

# --- PERIOD CONFIGURATIONS ---
# Options: 'year', 'month', 'week'
PERIOD_TYPE = 'year'  # <-- CHANGE HERE
YEAR = 2013          # <-- CHANGE HERE
MONTH = None          # <-- Define if PERIOD_TYPE = 'month' (1-12)
WEEK = None           # <-- Define if PERIOD_TYPE = 'week' (1-52)

# --- DATA TYPE OPTIMIZATION (POLARS SCHEMA) ---
POLARS_SCHEMA = {
    'icao24': pl.Categorical,
    'lat': pl.Float32,
    'lon': pl.Float32,
    'velocity': pl.Float32,
    'heading': pl.Float32,
    'vertrate': pl.Float32,
    'baroaltitude': pl.Float32,
    'geoaltitude': pl.Float32,
    'time': pl.Datetime('us', 'UTC'),  # Datetime with UTC timezone
    'callsign': pl.Categorical,
    'indicat': pl.Categorical,
    'onground': pl.Boolean,
    'alert': pl.Boolean,
    'spi': pl.Boolean,
    'squawk': pl.Utf8,
    'lastposupdate': pl.Float64,
    'lastcontact': pl.Float64
}

# --- ESSENTIAL COLUMNS (for reading filter) ---
ESSENTIAL_COLS = [
    'time', 'icao24', 'lat', 'lon', 'velocity', 'heading', 
    'vertrate', 'baroaltitude', 'geoaltitude', 'callsign', 
    'indicat', 'onground'
]

# ==============================================================================
# CONNECTION FUNCTIONS
# ==============================================================================

def conectar_trino(max_tentativas=5, delay=10):
    """
    Connects to Trino using pyopensky.
    Reads credentials from environment variables.
    
    Returns:
        Trino: Authenticated Trino client
    """
    if not os.environ.get("OPENSKY_USERNAME") or not os.environ.get("OPENSKY_PASSWORD"):
        print("❌ ERROR: Environment variables OPENSKY_USERNAME and OPENSKY_PASSWORD not defined.")
        print("Use 'export OPENSKY_USERNAME=...' in terminal before running the script.")
        sys.exit(1)
    
    for tentativa in range(1, max_tentativas + 1):
        try:
            trino_client = Trino()
            print(f"✅ Trino client initialized (attempt {tentativa}).")
            return trino_client
        except Exception as e:
            print(f"⚠️ Failure in attempt {tentativa}: {e}")
            if tentativa < max_tentativas:
                print(f"⏳ Retrying in {delay} seconds...")
                time.sleep(delay)
            else:
                print("❌ Critical failure connecting to Trino.")
                sys.exit(1)

# ==============================================================================
# PERIOD FUNCTIONS
# ==============================================================================

def calcular_periodo(period_type, year, month=None, week=None):
    """
    Calculates start and end date based on period type.
    
    Args:
        period_type (str): 'year', 'month' or 'week'
        year (int): Year
        month (int, optional): Month (1-12)
        week (int, optional): Week (1-52)
    
    Returns:
        tuple: (start_date, end_date, period_label)
    """
    if period_type == 'year':
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, 12, 31).date()
        label = f"{year}"
        
    elif period_type == 'month':
        if month is None or month < 1 or month > 12:
            raise ValueError("Invalid month. Use values between 1-12.")
        start_date = datetime(year, month, 1).date()
        if month == 12:
            end_date = datetime(year, 12, 31).date()
        else:
            end_date = (datetime(year, month + 1, 1) - timedelta(days=1)).date()
        label = f"{year}_{month:02d}"
        
    elif period_type == 'week':
        if week is None or week < 1 or week > 52:
            raise ValueError("Invalid week. Use values between 1-52.")
        jan_first = datetime(year, 1, 1)
        start_date = (jan_first + timedelta(weeks=week-1)).date()
        end_date = (start_date + timedelta(days=6))
        label = f"{year}_W{week:02d}"
        
    else:
        raise ValueError("Invalid period type. Use 'year', 'month' or 'week'.")
    
    return start_date, end_date, label

# ==============================================================================
# COLLECTION FUNCTIONS
# ==============================================================================

def coletar_trajetoria_voo(trino_client, callsign, incident_date):
    """
    Collects the complete trajectory of a specific flight.
    
    Args:
        trino_client: Connected Trino client
        callsign (str): Flight callsign
        incident_date: Incident date
    
    Returns:
        tuple: (Trajectory DataFrame, error if any)
    """
    try:
        # Search trajectory for the entire day
        start_of_day = pd.to_datetime(incident_date).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=timezone.utc
        )
        stop_time = start_of_day + timedelta(days=1)
        
        trajectory = pd.DataFrame(trino_client.history(
            start=start_of_day,
            stop=stop_time,
            callsign=callsign,
            cached=False,
            compress=True
        ))
        
        if trajectory.empty:
            return pd.DataFrame(), "Empty trajectory"
        
        # Add identifier
        trajectory['indicat'] = callsign
        
        # Convert to Polars for type optimization
        df_polars = pl.from_pandas(trajectory)
        
        # Apply optimized schema (only for existing columns)
        cast_exprs = []
        for col, dtype in POLARS_SCHEMA.items():
            if col in df_polars.columns:
                try:
                    if dtype == pl.Datetime('us', 'UTC'):
                        # Special conversion for datetime with timezone
                        cast_exprs.append(
                            pl.col(col).cast(pl.Datetime('us')).dt.replace_time_zone('UTC').alias(col)
                        )
                    else:
                        cast_exprs.append(pl.col(col).cast(dtype, strict=False))
                except:
                    pass  # Keep original type if conversion fails
        
        if cast_exprs:
            df_polars = df_polars.with_columns(cast_exprs)
        
        # Remove rows with null lat/lon and convert back to pandas
        df_polars = df_polars.filter(
            pl.col('lat').is_not_null() & pl.col('lon').is_not_null()
        )
        
        trajectory = df_polars.to_pandas()
        
        return trajectory, None
        
    except Exception as e:
        return pd.DataFrame(), str(e)

def coletar_bloco_voos(trino_client, df_voos, bloco_num, total_blocos):
    """
    Collects trajectories of a block of flights with automatic retry.
    
    Args:
        trino_client: Trino client
        df_voos (DataFrame): DataFrame with flights to process
        bloco_num (int): Current block number
        total_blocos (int): Total number of blocks
    
    Returns:
        tuple: (Collected trajectories, failure list, updated client)
    """
    all_trajectories = pd.DataFrame()
    log_falhas = []
    total_voos = len(df_voos)
    
    print(f"\n{'='*60}")
    print(f"🧩 BLOCK {bloco_num}/{total_blocos} - {total_voos} flights")
    print(f"{'='*60}")
    
    for i, row in df_voos.iterrows():
        callsign = str(row['CALLSIGN']).strip().replace(" ", "").upper()
        incident_date = row['INCIDENT_DATE']
        idx_local = i - df_voos.index[0] + 1
        
        print(f"[{idx_local}/{total_voos}] 🎯 Processing {callsign} on {incident_date}...")
        
        success = False
        erro_final = None
        
        # Attempts with retry
        for attempt in range(MAX_RETRIES):
            trajectory, erro = coletar_trajetoria_voo(trino_client, callsign, incident_date)
            
            if erro is None and not trajectory.empty:
                all_trajectories = pd.concat([all_trajectories, trajectory], ignore_index=True)
                print(f"  ✅ {len(trajectory)} points collected")
                success = True
                break
            elif erro == "Empty trajectory":
                print(f"  ⚠️ Empty trajectory (OK)")
                success = True
                break
            else:
                erro_final = erro
                print(f"  ⚠️ Attempt {attempt + 1} failed: {str(erro)[:80]}")
                
                # Reconnect if authentication error
                if any(x in str(erro).lower() for x in ['token', 'auth', '401']):
                    print(f"  🔄 Reconnecting to Trino...")
                    trino_client = conectar_trino()
                    time.sleep(RETRY_DELAY)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        # Log definitive failure
        if not success:
            log_falhas.append({
                'INCIDENT_DATE': incident_date,
                'CALLSIGN': callsign,
                'ERRO_FINAL': erro_final
            })
            print(f"  ❌ Definitive failure after {MAX_RETRIES} attempts")
    
    return all_trajectories, log_falhas, trino_client

# ==============================================================================
# SAVING AND OPTIMIZATION FUNCTIONS (IPC FORMAT + POLARS)
# ==============================================================================

def salvar_ipc_incremental(df, filepath, modo='append'):
    """
    Saves DataFrame in Apache Arrow IPC (Feather v2) format.
    Uses Polars for memory optimization before saving.
    
    Args:
        df (DataFrame): Data to save (Pandas)
        filepath (str): File path
        modo (str): 'append' or 'write'
    """
    if df.empty:
        return
    
    try:
        # Convert to Polars for optimization
        df_polars = pl.from_pandas(df)
        
        # Apply optimized schema
        cast_exprs = []
        for col, dtype in POLARS_SCHEMA.items():
            if col in df_polars.columns:
                try:
                    if dtype == pl.Datetime('us', 'UTC'):
                        cast_exprs.append(
                            pl.col(col).cast(pl.Datetime('us')).dt.replace_time_zone('UTC').alias(col)
                        )
                    else:
                        cast_exprs.append(pl.col(col).cast(dtype, strict=False))
                except:
                    pass
        
        if cast_exprs:
            df_polars = df_polars.with_columns(cast_exprs)
        
        # Convert to Arrow Table via Polars (more efficient)
        table = df_polars.to_arrow()
        
        if modo == 'append' and os.path.exists(filepath):
            # Read existing file and concatenate
            existing_table = feather.read_table(filepath)
            combined_table = pa.concat_tables([existing_table, table])
            feather.write_feather(combined_table, filepath, compression='lz4')
        else:
            # Create new file
            feather.write_feather(table, filepath, compression='lz4')
        
        print(f"💾 {len(df)} records saved in {os.path.basename(filepath)}")
        
    except Exception as e:
        print(f"❌ Error saving IPC: {e}")

def otimizar_arquivo_final(filepath):
    """
    Optimizes the final file using Polars lazy processing.
    Removes duplicates, sorts and consolidates the data.
    
    Args:
        filepath (str): Path to IPC file to optimize
    
    Returns:
        dict: Optimization statistics
    """
    if not os.path.exists(filepath):
        print(f"⚠️ File {filepath} not found")
        return None
    
    print(f"\n{'='*60}")
    print("🔧 OPTIMIZING FINAL FILE WITH POLARS")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Lazy reading of the file
        print("📖 Loading file in lazy mode...")
        lazy_df = pl.scan_ipc(filepath)
        
        # Check available columns
        available_cols = lazy_df.columns
        cols_to_read = [c for c in available_cols if c in ESSENTIAL_COLS or c in POLARS_SCHEMA.keys()]
        
        print(f"📊 Found columns: {len(available_cols)}")
        print(f"📋 Essential columns: {len(cols_to_read)}")
        
        # Lazy processing
        lazy_df = (
            lazy_df
            .select(cols_to_read)
            # Remove duplicates based on time, icao24, lat, lon
            .unique(subset=['time', 'icao24', 'lat', 'lon'], maintain_order=True)
            # Sort by callsign and time
            .sort(['callsign', 'time'])
        )
        
        # Statistics before collecting
        print("📈 Calculating statistics...")
        stats_before = {
            'total_rows': pl.scan_ipc(filepath).select(pl.count()).collect()[0, 0]
        }
        
        # Collect with streaming to save memory
        print("⚙️ Processing data with streaming engine...")
        df_optimized = lazy_df.collect(streaming=True)
        
        stats_after = {
            'total_rows': len(df_optimized),
            'unique_flights': df_optimized['callsign'].n_unique(),
            'date_range': (
                df_optimized['time'].min(),
                df_optimized['time'].max()
            ),
            'memory_usage_mb': df_optimized.estimated_size('mb')
        }
        
        # Save optimized version (overwrites original)
        print("💾 Saving optimized file...")
        df_optimized.write_ipc(
            filepath,
            compression='lz4'
        )
        
        elapsed = time.time() - start_time
        
        # Final statistics
        print(f"\n{'='*60}")
        print("✅ OPTIMIZATION COMPLETED")
        print(f"{'='*60}")
        print(f"Rows before: {stats_before['total_rows']:,}")
        print(f"Rows after: {stats_after['total_rows']:,}")
        print(f"Duplicates removed: {stats_before['total_rows'] - stats_after['total_rows']:,}")
        print(f"Unique flights: {stats_after['unique_flights']:,}")
        print(f"Period: {stats_after['date_range'][0]} to {stats_after['date_range'][1]}")
        print(f"Memory usage: {stats_after['memory_usage_mb']:.2f} MB")
        print(f"Execution time: {elapsed:.2f}s")
        print(f"{'='*60}\n")
        
        return stats_after
        
    except Exception as e:
        print(f"❌ Error in optimization: {e}")
        return None

def salvar_log_csv(log_list, filepath, modo='append'):
    """
    Saves failure log in CSV.
    
    Args:
        log_list (list): List of dictionaries with failures
        filepath (str): File path
        modo (str): 'append' or 'write'
    """
    if not log_list:
        return
    
    df_log = pd.DataFrame(log_list)
    
    if modo == 'append' and os.path.exists(filepath):
        df_log.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_log.to_csv(filepath, mode='w', header=True, index=False, encoding='utf-8')
    
    print(f"📋 {len(log_list)} failures logged")

# ==============================================================================
# MAIN COLLECTION FUNCTION
# ==============================================================================

def executar_coleta_principal(trino_client, df_voos, data_path, log_path):
    """
    Executes main collection in blocks with incremental saving.
    
    Args:
        trino_client: Trino client
        df_voos (DataFrame): Flights to process
        data_path (str): Data file path
        log_path (str): Log file path
    
    Returns:
        int: Number of flights successfully processed
    """
    total_voos = len(df_voos)
    total_blocos = (total_voos // BLOCO_SIZE) + (1 if total_voos % BLOCO_SIZE else 0)
    
    print(f"\n{'='*60}")
    print(f"🚀 STARTING MAIN COLLECTION")
    print(f"{'='*60}")
    print(f"Total flights: {total_voos}")
    print(f"Total blocks: {total_blocos}")
    print(f"Block size: {BLOCO_SIZE}")
    
    voos_sucesso = 0
    
    for i in range(0, total_voos, BLOCO_SIZE):
        bloco = df_voos.iloc[i:i + BLOCO_SIZE]
        bloco_num = (i // BLOCO_SIZE) + 1
        
        try:
            trajectories, falhas, trino_client = coletar_bloco_voos(
                trino_client, bloco, bloco_num, total_blocos
            )
            
            # Save trajectories in IPC
            if not trajectories.empty:
                salvar_ipc_incremental(trajectories, data_path, modo='append')
                voos_sucesso += trajectories['callsign'].nunique()
            
            # Save failure log
            if falhas:
                salvar_log_csv(falhas, log_path, modo='append')
            
            print(f"✅ Block {bloco_num} completed\n")
            
        except Exception as e:
            print(f"❌ Error in block {bloco_num}: {e}")
            print("⏳ Waiting 10s before reconnecting...")
            time.sleep(10)
            trino_client = conectar_trino()
    
    return voos_sucesso

# ==============================================================================
# FAILURE RECOVERY FUNCTION
# ==============================================================================

def executar_recuperacao_falhas(trino_client, log_path, data_path, log_final_path):
    """
    Attempts to recover flights that failed in main collection.
    
    Args:
        trino_client: Trino client
        log_path (str): Failure log path
        data_path (str): Data file path
        log_final_path (str): Definitive failure log path
    
    Returns:
        tuple: (Recovered flights, definitive failures)
    """
    if not os.path.exists(log_path):
        print("✅ No failures logged. Skipping recovery.")
        return 0, 0
    
    print(f"\n{'='*60}")
    print(f"🔄 STARTING FAILURE RECOVERY")
    print(f"{'='*60}")
    
    # Read failure log
    df_falhas = pd.read_csv(log_path, encoding='utf-8')
    df_falhas.columns = df_falhas.columns.str.strip().str.upper()
    
    if 'INCIDENT_DATE' not in df_falhas.columns or 'CALLSIGN' not in df_falhas.columns:
        print("❌ Failure log with invalid format")
        return 0, 0
    
    df_falhas['INCIDENT_DATE'] = pd.to_datetime(df_falhas['INCIDENT_DATE'], errors='coerce')
    df_falhas = df_falhas.dropna(subset=['INCIDENT_DATE', 'CALLSIGN'])
    
    total_falhas = len(df_falhas)
    print(f"📚 {total_falhas} flights to recover")
    
    # Execute recovery
    trajectories_recuperadas = pd.DataFrame()
    falhas_definitivas = []
    
    for i, row in df_falhas.iterrows():
        callsign = str(row['CALLSIGN']).strip().replace(" ", "").upper()
        incident_date = row['INCIDENT_DATE']
        
        print(f"[{i+1}/{total_falhas}] 🔍 Recovering {callsign}...")
        
        trajectory, erro = coletar_trajetoria_voo(trino_client, callsign, incident_date)
        
        if erro is None and not trajectory.empty:
            trajectories_recuperadas = pd.concat([trajectories_recuperadas, trajectory], ignore_index=True)
            print(f"  ✅ Recovered: {len(trajectory)} points")
        else:
            falhas_definitivas.append({
                'INCIDENT_DATE': incident_date,
                'CALLSIGN': callsign,
                'ERRO_FINAL': erro or "Empty trajectory"
            })
            print(f"  ❌ Definitive failure")
    
    # Save recovered data
    if not trajectories_recuperadas.empty:
        salvar_ipc_incremental(trajectories_recuperadas, data_path, modo='append')
    
    # Save definitive failure log
    if falhas_definitivas:
        salvar_log_csv(falhas_definitivas, log_final_path, modo='write')
    
    # Remove temporary log
    if os.path.exists(log_path):
        os.remove(log_path)
    
    voos_recuperados = trajectories_recuperadas['callsign'].nunique() if not trajectories_recuperadas.empty else 0
    
    return voos_recuperados, len(falhas_definitivas)

# ==============================================================================
# FINAL REPORT FUNCTION
# ==============================================================================

def gerar_relatorio_final(period_label, voos_total, voos_sucesso, voos_recuperados, falhas_definitivas):
    """
    Generates consolidated collection report.
    """
    print(f"\n{'='*60}")
    print("📊 FINAL COLLECTION REPORT")
    print(f"{'='*60}")
    print(f"Period: {period_label}")
    print(f"Total flights processed: {voos_total}")
    print(f"Flights collected (1st attempt): {voos_sucesso}")
    print(f"Flights recovered: {voos_recuperados}")
    print(f"Definitive failures: {falhas_definitivas}")
    print(f"Success rate: {((voos_sucesso + voos_recuperados) / voos_total * 100):.2f}%")
    print(f"{'='*60}\n")


# ==============================================================================
# MAIN - SYSTEM EXECUTION
# ==============================================================================

def main():
    """
    Main function that orchestrates the entire collection process.
    """
    print("\n" + "="*60)
    print("🛰️  INTEGRATED OPENSKY COLLECTION SYSTEM - IPC FORMAT")
    print("="*60 + "\n")
    
    # 1. Calculate period
    try:
        start_date, end_date, period_label = calcular_periodo(PERIOD_TYPE, YEAR, MONTH, WEEK)
        print(f"📅 Period: {period_label}")
        print(f"   Start: {start_date}")
        print(f"   End: {end_date}\n")
    except Exception as e:
        print(f"❌ Error calculating period: {e}")
        sys.exit(1)
    
    # 2. Define file paths
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_path = os.path.join(OUTPUT_DIR, f"{period_label}_trajetorias.ipc")
    log_path = os.path.join(OUTPUT_DIR, f"{period_label}_log_falhas.csv")
    log_final_path = os.path.join(OUTPUT_DIR, f"{period_label}_falhas_definitivas.csv")
    
    # 3. Load bird-strike data from IPC file using Polars lazy reading
    IPC_FILENAME = "/home/leonmortari/code/phd-ita/BS_Code/Analise/data_strike_z_20260328_060328.ipc"
    
    # Read IPC file in lazy mode (streaming without loading full data into memory)
    print("📖 Reading bird-strike data in lazy mode...")
    lazy_df = pl.scan_ipc(IPC_FILENAME)
    
    # Filter by period directly in lazy evaluation
    lazy_df = lazy_df.filter(
        (pl.col("INCIDENT_DATE") >= start_date) & 
        (pl.col("INCIDENT_DATE") <= end_date)
    )
    
    # Collect results to pandas for compatibility with existing code
    print("⚙️ Processing and collecting filtered data...")
    data_strike = lazy_df.collect().to_pandas()
    
    # Sort by incident date
    data_strike = data_strike.sort_values(by="INCIDENT_DATE", ascending=True).reset_index(drop=True)
    
    # Use filtered data directly (period filtering already done in lazy evaluation)
    df_periodo = data_strike.copy()
    
    if df_periodo.empty:
        print(f"⚠️ No incidents found in period {period_label}")
        sys.exit(0)
    
    print(f"✅ {len(df_periodo)} incidents found in period\n")
    
    # 4. Connect to Trino
    trino_client = conectar_trino()
    
    # 5. Execute main collection
    voos_sucesso = executar_coleta_principal(trino_client, df_periodo, data_path, log_path)
    
    # 6. Execute failure recovery
    voos_recuperados, falhas_definitivas = executar_recuperacao_falhas(
        trino_client, log_path, data_path, log_final_path
    )
    
    # 7. Generate final report
    gerar_relatorio_final(
        period_label,
        len(df_periodo),
        voos_sucesso,
        voos_recuperados,
        falhas_definitivas
    )
    
    # 8. Final optimization with Polars (OPTIONAL - may consume memory)
    print("\n" + "="*60)
    resposta = input("🔧 Do you want to optimize the final file with Polars? (y/n): ").strip().lower()
    if resposta == 's':
        otimizar_arquivo_final(data_path)
    else:
        print("⏩ Optimization skipped. File saved in original format.")
    
    # 9. Close connection
    try:
        trino_client.conn.close()
        print("🔌 Trino connection closed successfully")
    except:
        pass
    
    print("\n✅ Execution completed!\n")

# ==============================================================================
# ENTRY POINT
# ==============================================================================

if __name__ == "__main__":
    main()