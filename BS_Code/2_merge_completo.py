# ============================================================
# merge_strike_flight_QGIS_FINAL_v7.py
# Match geoespacial Bird Strike ↔ Trajetórias (Polars Lazy + Streaming)
# Usa DATETIME_UTC e merge temporal com precisão HH:MM:SS
# ============================================================

import polars as pl
from datetime import timezone
import time
import sys
import gc
from tqdm import tqdm
from pathlib import Path
import os


# --- Caminhos ---
STRIKE_DATA_PATH = Path("/home/labgeo/code/bird-strike/data_strike_z_20251111_174649.ipc")
IPC_PATH = Path("/home/labgeo/code/bird-strike/dados_trajetorias/2025_trajetorias_opt_compact.ipc")
OUTPUT_FILENAME = "2025_point_strike.csv"

# --- Parâmetros ---
TIME_BUFFER_SECONDS = 30
ANO_ALVO = 2025


def run_matching_process():
    start_time = time.time()
    print("=" * 70)
    print("🚀 MAPEAMENTO STRIKE ↔ TRAJETÓRIA (Polars Lazy + Streaming)")
    print("🕒 Merge temporal (precisão hora:minuto:segundo)")
    print("💾 Saída otimizada em CSV para uso no QGIS")
    print("=" * 70)

    # -----------------------------
    # LEITURA LAZY DAS BASES .IPC
    # -----------------------------
    def ler_arquivos_ipc_lazy():
        try:
            lf_strikes = pl.scan_ipc(str(STRIKE_DATA_PATH))
            lf_trajetorias = pl.scan_ipc(str(IPC_PATH))
            print("✅ Bases carregadas (modo lazy).")
            return lf_strikes, lf_trajetorias
        except Exception as e:
            print(f"❌ Erro ao abrir arquivos IPC: {e}")
            return None, None

    lf_strike_data, lf_trajectory_data = ler_arquivos_ipc_lazy()
    if lf_strike_data is None or lf_trajectory_data is None:
        print("❌ Falha na leitura dos arquivos IPC.")
        return

    # ------------------------------------------------------------
    # 1️⃣ Adiciona ID único incremental para cada strike
    # ------------------------------------------------------------
    lf_strike_data = lf_strike_data.with_columns(
        pl.arange(0, pl.len()).alias("STRIKE_ID")
    )

    # ------------------------------------------------------------
    # 2️⃣ Prepara colunas de chave e merge temporal
    # ------------------------------------------------------------
    # Uniformiza nomes de colunas de join (CALLSIGN ↔ callsign)
    lf_strike_data = lf_strike_data.rename({"CALLSIGN": "CALLSIGN_STRIKE"})
    lf_trajectory_data = lf_trajectory_data.rename({"callsign": "CALLSIGN_TRAJ"})

    # ------------------------------------------------------------
    # 3️⃣ Merge temporal por CALLSIGN + DATETIME_UTC (± 30s)
    # ------------------------------------------------------------
    print("⚙️ Iniciando merge temporal streaming...")

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
    # 4️⃣ Coleta em modo streaming com barra de progresso
    # ------------------------------------------------------------
    print("💾 Processando merge (modo streaming)...")
    with tqdm(total=100, desc="🔄 Merge em andamento", ncols=80) as pbar:
        result = merged_lazy.collect(engine="streaming")
        pbar.update(100)

    # ------------------------------------------------------------
    # 5️⃣ Conversão segura de colunas aninhadas (List/Struct)
    # ------------------------------------------------------------
    print("🧹 Verificando colunas aninhadas...")
    nested_cols = [c for c, dt in zip(result.columns, result.dtypes)
                   if isinstance(dt, (pl.List, pl.Struct))]

    if nested_cols:
        print(f"⚠️ Colunas aninhadas detectadas: {nested_cols}")
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
        print("✅ Nenhuma coluna aninhada detectada.")

    # ------------------------------------------------------------
    # 6️⃣ Exporta todas as colunas integradas
    # ------------------------------------------------------------
    print("🧾 Salvando resultado final em CSV...")
    result.write_csv(OUTPUT_FILENAME)
    print(f"✅ Arquivo salvo: {OUTPUT_FILENAME}")

    # ------------------------------------------------------------
    # 7️⃣ Finalização
    # ------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"🎉 Concluído! {result.height:,} correspondências salvas em '{OUTPUT_FILENAME}'")
    print(f"⏱️ Tempo total: {time.time() - start_time:.1f} segundos")
    print("=" * 70)

    del result, lf_strike_data, lf_trajectory_data
    gc.collect()


# ------------------------------------------------------------
# EXECUÇÃO DIRETA
# ------------------------------------------------------------
if __name__ == "__main__":
    try:
        run_matching_process()
    except KeyboardInterrupt:
        print("🛑 Execução interrompida manualmente.")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Erro inesperado: {e}")
        sys.exit(1)
