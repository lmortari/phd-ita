# ============================================================
# merge_strike_flight
# Geospatial Match Bird Strike ↔ Trajectories (Polars Lazy + Streaming)
# Uses DATETIME_UTC and temporal merge with HH:MM:SS precision
# ============================================================

import polars as pl
from datetime import timezone
import time
import sys
import gc
from tqdm import tqdm
from pathlib import Path
import os


# --- Paths ---
STRIKE_DATA_PATH = Path("/home/labgeo/code/bird-strike/data_strike_z_20251111_174649.ipc")
IPC_PATH = Path("/home/labgeo/code/bird-strike/dados_trajetorias/2025_trajetorias_opt_compact.ipc")
OUTPUT_FILENAME = "2025_point_strike.csv"

# --- Parameters ---
TIME_BUFFER_SECONDS = 30
ANO_ALVO = 2025


def run_matching_process():
    start_time = time.time()
    print("=" * 70)
    print("🚀 STRIKE ↔ TRAJECTORY MAPPING (Polars Lazy + Streaming)")
    print("🕒 Temporal merge (hour:minute:second precision)")
    print("💾 Optimized output in CSV for QGIS use")
    print("=" * 70)

    # -----------------------------
    # LAZY READING OF .IPC BASES
    # -----------------------------
    def ler_arquivos_ipc_lazy():
        try:
            lf_strikes = pl.scan_ipc(str(STRIKE_DATA_PATH))
            lf_trajetorias = pl.scan_ipc(str(IPC_PATH))
            print("✅ Bases loaded (lazy mode).")
            return lf_strikes, lf_trajetorias
        except Exception as e:
            print(f"❌ Error opening IPC files: {e}")
            return None, None

    lf_strike_data, lf_trajectory_data = ler_arquivos_ipc_lazy()
    if lf_strike_data is None or lf_trajectory_data is None:
        print("❌ Failed to read IPC files.")
        return

    # ------------------------------------------------------------
    # 1️⃣ Adds unique incremental ID for each strike
    # ------------------------------------------------------------
    lf_strike_data = lf_strike_data.with_columns(
        pl.arange(0, pl.len()).alias("STRIKE_ID")
    )

    # ------------------------------------------------------------
    # 2️⃣ Prepares key columns and temporal merge
    # ------------------------------------------------------------
    # Standardizes join column names (CALLSIGN ↔ callsign)
    lf_strike_data = lf_strike_data.rename({"CALLSIGN": "CALLSIGN_STRIKE"})
    lf_trajectory_data = lf_trajectory_data.rename({"callsign": "CALLSIGN_TRAJ"})

    # ------------------------------------------------------------
    # 3️⃣ Temporal merge by CALLSIGN + DATETIME_UTC (± 30s)
    # ------------------------------------------------------------
    print("⚙️ Starting temporal streaming merge...")

    merged_lazy = (
        lf_trajectory_data.join(
            lf_strike_data,
            left_on="CALLSIGN_TRAJ",
            right_on="CALLSIGN_STRIKE",
            how="inner"
        )
        .with_columns([
            (pl.col("time") - pl.col("DATETIME_UTC")).dt.total_seconds().alias("TIME_DIFF_SEC")
        ])
        .filter(pl.col("TIME_DIFF_SEC").abs() <= TIME_BUFFER_SECONDS)
    )

    # ------------------------------------------------------------
    # 4️⃣ Collects in streaming mode with progress bar
    # ------------------------------------------------------------
    print("💾 Processing merge (streaming mode)...")
    with tqdm(total=100, desc="🔄 Merge in progress", ncols=80) as pbar:
        result = merged_lazy.collect(engine="streaming")
        pbar.update(100)

    # ------------------------------------------------------------
    # 5️⃣ Safe conversion of nested columns (List/Struct)
    # ------------------------------------------------------------
    print("🧹 Checking nested columns...")
    nested_cols = [c for c, dt in zip(result.columns, result.dtypes)
                   if isinstance(dt, (pl.List, pl.Struct))]

    if nested_cols:
        print(f"⚠️ Nested columns detected: {nested_cols}")
        for col in nested_cols:
            dtype = result.schema[col]
            if isinstance(dtype, pl.List):
                result = result.with_columns(
                    pl.col(col)
                    .list.eval(pl.element().cast(pl.Utf8))
                    .list.join(",")
                    .alias(col)
                )
            elif isinstance(dtype, pl.Struct):
                result = result.with_columns(pl.col(col).struct.json_encode().alias(col))
    else:
        print("✅ No nested columns detected.")

    # ------------------------------------------------------------
    # 6️⃣ Exports all integrated columns
    # ------------------------------------------------------------
    print("🧾 Saving final result to CSV...")
    result.write_csv(OUTPUT_FILENAME)
    print(f"✅ File saved: {OUTPUT_FILENAME}")

    # ------------------------------------------------------------
    # 7️⃣ Finalization
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"🎉 Completed! {result.height:,} matches saved in '{OUTPUT_FILENAME}'")
    print(f"⏱️ Total time: {time.time() - start_time:.1f} seconds")
    print("=" * 70)

    del result, lf_strike_data, lf_trajectory_data
    gc.collect()


# ------------------------------------------------------------
# DIRECT EXECUTION
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_matching_process()
    except KeyboardInterrupt:
        print("🛑 Execution manually interrupted.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)
