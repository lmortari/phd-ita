# ==============================================================================
# SISTEMA INTEGRADO DE COLETA OPENSKY - FORMATO IPC + POLARS
# ==============================================================================
# Descrição: Sistema unificado para coleta de trajetórias de voos via OpenSky,
# com recuperação automática de falhas e salvamento otimizado em formato IPC.
#
# Funcionalidades:
# 1. Coleta de trajetórias por período (ano, mês ou semana)
# 2. Salvamento incremental em blocos (formato .ipc)
# 3. Processamento lazy com Polars para otimização de memória
# 4. Log de falhas e recuperação automática
# 5. Relatório final de sucessos e falhas definitivas
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
# CONFIGURAÇÕES GLOBAIS
# ==============================================================================

# --- CREDENCIAIS (Devem estar nas variáveis de ambiente) ---
# export OPENSKY_USERNAME="leon.mortari.102254@ga.ita.br
# export OPENSKY_PASSWORD="Leon1234"

# --- DIRETÓRIO DE SAÍDA ---
OUTPUT_DIR = "/home/labgeo/code/bird-strike/dados_trajetorias"

# --- CONFIGURAÇÕES DE COLETA ---
BLOCO_SIZE = 10  # Número de voos por bloco
MAX_RETRIES = 3  # Tentativas por voo
RETRY_DELAY = 5  # Segundos entre tentativas

# --- CONFIGURAÇÕES DE PERÍODO ---
# Opções: 'year', 'month', 'week'
PERIOD_TYPE = 'year'  # <-- ALTERAR AQUI
YEAR = 2024           # <-- ALTERAR AQUI
MONTH = None          # <-- Definir se PERIOD_TYPE = 'month' (1-12)
WEEK = None           # <-- Definir se PERIOD_TYPE = 'week' (1-52)

# --- OTIMIZAÇÃO DE TIPOS DE DADOS (POLARS SCHEMA) ---
POLARS_SCHEMA = {
    'icao24': pl.Categorical,
    'lat': pl.Float32,
    'lon': pl.Float32,
    'velocity': pl.Float32,
    'heading': pl.Float32,
    'vertrate': pl.Float32,
    'baroaltitude': pl.Float32,
    'geoaltitude': pl.Float32,
    'time': pl.Datetime('us', 'UTC'),  # Datetime com timezone UTC
    'callsign': pl.Categorical,
    'indicat': pl.Categorical,
    'onground': pl.Boolean,
    'alert': pl.Boolean,
    'spi': pl.Boolean,
    'squawk': pl.Utf8,
    'lastposupdate': pl.Float64,
    'lastcontact': pl.Float64
}

# --- COLUNAS ESSENCIAIS (para filtrar leitura) ---
ESSENTIAL_COLS = [
    'time', 'icao24', 'lat', 'lon', 'velocity', 'heading', 
    'vertrate', 'baroaltitude', 'geoaltitude', 'callsign', 
    'indicat', 'onground'
]

# ==============================================================================
# FUNÇÕES DE CONEXÃO
# ==============================================================================

def conectar_trino(max_tentativas=5, delay=10):
    """
    Conecta ao Trino usando pyopensky.
    Lê credenciais das variáveis de ambiente.
    
    Returns:
        Trino: Cliente Trino autenticado
    """
    if not os.environ.get("OPENSKY_USERNAME") or not os.environ.get("OPENSKY_PASSWORD"):
        print("❌ ERRO: Variáveis de ambiente OPENSKY_USERNAME e OPENSKY_PASSWORD não definidas.")
        print("Use 'export OPENSKY_USERNAME=...' no terminal antes de rodar o script.")
        sys.exit(1)
    
    for tentativa in range(1, max_tentativas + 1):
        try:
            trino_client = Trino()
            print(f"✅ Cliente Trino inicializado (tentativa {tentativa}).")
            return trino_client
        except Exception as e:
            print(f"⚠️ Falha na tentativa {tentativa}: {e}")
            if tentativa < max_tentativas:
                print(f"⏳ Repetindo em {delay} segundos...")
                time.sleep(delay)
            else:
                print("❌ Falha crítica ao conectar ao Trino.")
                sys.exit(1)

# ==============================================================================
# FUNÇÕES DE PERÍODO
# ==============================================================================

def calcular_periodo(period_type, year, month=None, week=None):
    """
    Calcula data de início e fim baseado no tipo de período.
    
    Args:
        period_type (str): 'year', 'month' ou 'week'
        year (int): Ano
        month (int, optional): Mês (1-12)
        week (int, optional): Semana (1-52)
    
    Returns:
        tuple: (start_date, end_date, period_label)
    """
    if period_type == 'year':
        start_date = datetime(year, 1, 1).date()
        end_date = datetime(year, 12, 31).date()
        label = f"{year}"
        
    elif period_type == 'month':
        if month is None or month < 1 or month > 12:
            raise ValueError("Mês inválido. Use valores entre 1-12.")
        start_date = datetime(year, month, 1).date()
        if month == 12:
            end_date = datetime(year, 12, 31).date()
        else:
            end_date = (datetime(year, month + 1, 1) - timedelta(days=1)).date()
        label = f"{year}_{month:02d}"
        
    elif period_type == 'week':
        if week is None or week < 1 or week > 52:
            raise ValueError("Semana inválida. Use valores entre 1-52.")
        jan_first = datetime(year, 1, 1)
        start_date = (jan_first + timedelta(weeks=week-1)).date()
        end_date = (start_date + timedelta(days=6))
        label = f"{year}_W{week:02d}"
        
    else:
        raise ValueError("Tipo de período inválido. Use 'year', 'month' ou 'week'.")
    
    return start_date, end_date, label

# ==============================================================================
# FUNÇÕES DE COLETA
# ==============================================================================

def coletar_trajetoria_voo(trino_client, callsign, incident_date):
    """
    Coleta a trajetória completa de um voo específico.
    
    Args:
        trino_client: Cliente Trino conectado
        callsign (str): Callsign do voo
        incident_date: Data do incidente
    
    Returns:
        tuple: (DataFrame da trajetória, erro se houver)
    """
    try:
        # Busca trajetória do dia inteiro
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
            return pd.DataFrame(), "Trajetória vazia"
        
        # Adiciona identificador
        trajectory['indicat'] = callsign
        
        # Conversão para Polars para otimização de tipos
        df_polars = pl.from_pandas(trajectory)
        
        # Aplica schema otimizado (apenas colunas que existem)
        cast_exprs = []
        for col, dtype in POLARS_SCHEMA.items():
            if col in df_polars.columns:
                try:
                    if dtype == pl.Datetime('us', 'UTC'):
                        # Conversão especial para datetime com timezone
                        cast_exprs.append(
                            pl.col(col).cast(pl.Datetime('us')).dt.replace_time_zone('UTC').alias(col)
                        )
                    else:
                        cast_exprs.append(pl.col(col).cast(dtype, strict=False))
                except:
                    pass  # Mantém tipo original se conversão falhar
        
        if cast_exprs:
            df_polars = df_polars.with_columns(cast_exprs)
        
        # Remove linhas com lat/lon nulos e converte de volta para pandas
        df_polars = df_polars.filter(
            pl.col('lat').is_not_null() & pl.col('lon').is_not_null()
        )
        
        trajectory = df_polars.to_pandas()
        
        return trajectory, None
        
    except Exception as e:
        return pd.DataFrame(), str(e)

def coletar_bloco_voos(trino_client, df_voos, bloco_num, total_blocos):
    """
    Coleta trajetórias de um bloco de voos com retry automático.
    
    Args:
        trino_client: Cliente Trino
        df_voos (DataFrame): DataFrame com voos a processar
        bloco_num (int): Número do bloco atual
        total_blocos (int): Total de blocos
    
    Returns:
        tuple: (trajetórias coletadas, lista de falhas, cliente atualizado)
    """
    all_trajectories = pd.DataFrame()
    log_falhas = []
    total_voos = len(df_voos)
    
    print(f"\n{'='*60}")
    print(f"🧩 BLOCO {bloco_num}/{total_blocos} - {total_voos} voos")
    print(f"{'='*60}")
    
    for i, row in df_voos.iterrows():
        callsign = str(row['CALLSIGN']).strip().replace(" ", "").upper()
        incident_date = row['INCIDENT_DATE']
        idx_local = i - df_voos.index[0] + 1
        
        print(f"[{idx_local}/{total_voos}] 🎯 Processando {callsign} em {incident_date}...")
        
        success = False
        erro_final = None
        
        # Tentativas com retry
        for attempt in range(MAX_RETRIES):
            trajectory, erro = coletar_trajetoria_voo(trino_client, callsign, incident_date)
            
            if erro is None and not trajectory.empty:
                all_trajectories = pd.concat([all_trajectories, trajectory], ignore_index=True)
                print(f"  ✅ {len(trajectory)} pontos coletados")
                success = True
                break
            elif erro == "Trajetória vazia":
                print(f"  ⚠️ Trajetória vazia (OK)")
                success = True
                break
            else:
                erro_final = erro
                print(f"  ⚠️ Tentativa {attempt + 1} falhou: {str(erro)[:80]}")
                
                # Reconecta se erro de autenticação
                if any(x in str(erro).lower() for x in ['token', 'auth', '401']):
                    print(f"  🔄 Reconectando ao Trino...")
                    trino_client = conectar_trino()
                    time.sleep(RETRY_DELAY)
                elif attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_DELAY)
        
        # Log de falha definitiva
        if not success:
            log_falhas.append({
                'INCIDENT_DATE': incident_date,
                'CALLSIGN': callsign,
                'ERRO_FINAL': erro_final
            })
            print(f"  ❌ Falha definitiva após {MAX_RETRIES} tentativas")
    
    return all_trajectories, log_falhas, trino_client

# ==============================================================================
# FUNÇÕES DE SALVAMENTO E OTIMIZAÇÃO (IPC FORMAT + POLARS)
# ==============================================================================

def salvar_ipc_incremental(df, filepath, modo='append'):
    """
    Salva DataFrame em formato Apache Arrow IPC (Feather v2).
    Usa Polars para otimização de memória antes de salvar.
    
    Args:
        df (DataFrame): Dados a salvar (Pandas)
        filepath (str): Caminho do arquivo
        modo (str): 'append' ou 'write'
    """
    if df.empty:
        return
    
    try:
        # Converte para Polars para otimização
        df_polars = pl.from_pandas(df)
        
        # Aplica schema otimizado
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
        
        # Converte para Arrow Table via Polars (mais eficiente)
        table = df_polars.to_arrow()
        
        if modo == 'append' and os.path.exists(filepath):
            # Lê arquivo existente e concatena
            existing_table = feather.read_table(filepath)
            combined_table = pa.concat_tables([existing_table, table])
            feather.write_feather(combined_table, filepath, compression='lz4')
        else:
            # Cria novo arquivo
            feather.write_feather(table, filepath, compression='lz4')
        
        print(f"💾 {len(df)} registros salvos em {os.path.basename(filepath)}")
        
    except Exception as e:
        print(f"❌ Erro ao salvar IPC: {e}")

def otimizar_arquivo_final(filepath):
    """
    Otimiza o arquivo final usando Polars lazy processing.
    Remove duplicatas, ordena e consolida os dados.
    
    Args:
        filepath (str): Caminho do arquivo IPC a otimizar
    
    Returns:
        dict: Estatísticas da otimização
    """
    if not os.path.exists(filepath):
        print(f"⚠️ Arquivo {filepath} não encontrado")
        return None
    
    print(f"\n{'='*60}")
    print("🔧 OTIMIZANDO ARQUIVO FINAL COM POLARS")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Leitura lazy do arquivo
        print("📖 Carregando arquivo em modo lazy...")
        lazy_df = pl.scan_ipc(filepath)
        
        # Verifica colunas disponíveis
        available_cols = lazy_df.columns
        cols_to_read = [c for c in available_cols if c in ESSENTIAL_COLS or c in POLARS_SCHEMA.keys()]
        
        print(f"📊 Colunas encontradas: {len(available_cols)}")
        print(f"📋 Colunas essenciais: {len(cols_to_read)}")
        
        # Processamento lazy
        lazy_df = (
            lazy_df
            .select(cols_to_read)
            # Remove duplicatas baseado em time, icao24, lat, lon
            .unique(subset=['time', 'icao24', 'lat', 'lon'], maintain_order=True)
            # Ordena por callsign e time
            .sort(['callsign', 'time'])
        )
        
        # Estatísticas antes de coletar
        print("📈 Calculando estatísticas...")
        stats_before = {
            'total_rows': pl.scan_ipc(filepath).select(pl.count()).collect()[0, 0]
        }
        
        # Coleta com streaming para economizar memória
        print("⚙️ Processando dados com streaming engine...")
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
        
        # Salva versão otimizada (sobrescreve o original)
        print("💾 Salvando arquivo otimizado...")
        df_optimized.write_ipc(
            filepath,
            compression='lz4'
        )
        
        elapsed = time.time() - start_time
        
        # Estatísticas finais
        print(f"\n{'='*60}")
        print("✅ OTIMIZAÇÃO CONCLUÍDA")
        print(f"{'='*60}")
        print(f"Linhas antes: {stats_before['total_rows']:,}")
        print(f"Linhas depois: {stats_after['total_rows']:,}")
        print(f"Duplicatas removidas: {stats_before['total_rows'] - stats_after['total_rows']:,}")
        print(f"Voos únicos: {stats_after['unique_flights']:,}")
        print(f"Período: {stats_after['date_range'][0]} a {stats_after['date_range'][1]}")
        print(f"Uso de memória: {stats_after['memory_usage_mb']:.2f} MB")
        print(f"Tempo de execução: {elapsed:.2f}s")
        print(f"{'='*60}\n")
        
        return stats_after
        
    except Exception as e:
        print(f"❌ Erro na otimização: {e}")
        return None

def salvar_log_csv(log_list, filepath, modo='append'):
    """
    Salva log de falhas em CSV.
    
    Args:
        log_list (list): Lista de dicionários com falhas
        filepath (str): Caminho do arquivo
        modo (str): 'append' ou 'write'
    """
    if not log_list:
        return
    
    df_log = pd.DataFrame(log_list)
    
    if modo == 'append' and os.path.exists(filepath):
        df_log.to_csv(filepath, mode='a', header=False, index=False, encoding='utf-8')
    else:
        df_log.to_csv(filepath, mode='w', header=True, index=False, encoding='utf-8')
    
    print(f"📋 {len(log_list)} falhas registradas em log")

# ==============================================================================
# FUNÇÃO PRINCIPAL DE COLETA
# ==============================================================================

def executar_coleta_principal(trino_client, df_voos, data_path, log_path):
    """
    Executa coleta principal em blocos com salvamento incremental.
    
    Args:
        trino_client: Cliente Trino
        df_voos (DataFrame): Voos a processar
        data_path (str): Caminho do arquivo de dados
        log_path (str): Caminho do arquivo de log
    
    Returns:
        int: Número de voos processados com sucesso
    """
    total_voos = len(df_voos)
    total_blocos = (total_voos // BLOCO_SIZE) + (1 if total_voos % BLOCO_SIZE else 0)
    
    print(f"\n{'='*60}")
    print(f"🚀 INICIANDO COLETA PRINCIPAL")
    print(f"{'='*60}")
    print(f"Total de voos: {total_voos}")
    print(f"Total de blocos: {total_blocos}")
    print(f"Tamanho do bloco: {BLOCO_SIZE}")
    
    voos_sucesso = 0
    
    for i in range(0, total_voos, BLOCO_SIZE):
        bloco = df_voos.iloc[i:i + BLOCO_SIZE]
        bloco_num = (i // BLOCO_SIZE) + 1
        
        try:
            trajectories, falhas, trino_client = coletar_bloco_voos(
                trino_client, bloco, bloco_num, total_blocos
            )
            
            # Salva trajetórias em IPC
            if not trajectories.empty:
                salvar_ipc_incremental(trajectories, data_path, modo='append')
                voos_sucesso += trajectories['callsign'].nunique()
            
            # Salva log de falhas
            if falhas:
                salvar_log_csv(falhas, log_path, modo='append')
            
            print(f"✅ Bloco {bloco_num} concluído\n")
            
        except Exception as e:
            print(f"❌ Erro no bloco {bloco_num}: {e}")
            print("⏳ Aguardando 10s antes de reconectar...")
            time.sleep(10)
            trino_client = conectar_trino()
    
    return voos_sucesso

# ==============================================================================
# FUNÇÃO DE RECUPERAÇÃO DE FALHAS
# ==============================================================================

def executar_recuperacao_falhas(trino_client, log_path, data_path, log_final_path):
    """
    Tenta recuperar voos que falharam na coleta principal.
    
    Args:
        trino_client: Cliente Trino
        log_path (str): Caminho do log de falhas
        data_path (str): Caminho do arquivo de dados
        log_final_path (str): Caminho do log de falhas definitivas
    
    Returns:
        tuple: (voos_recuperados, falhas_definitivas)
    """
    if not os.path.exists(log_path):
        print("✅ Nenhuma falha registrada. Pulando recuperação.")
        return 0, 0
    
    print(f"\n{'='*60}")
    print(f"🔄 INICIANDO RECUPERAÇÃO DE FALHAS")
    print(f"{'='*60}")
    
    # Lê log de falhas
    df_falhas = pd.read_csv(log_path, encoding='utf-8')
    df_falhas.columns = df_falhas.columns.str.strip().str.upper()
    
    if 'INCIDENT_DATE' not in df_falhas.columns or 'CALLSIGN' not in df_falhas.columns:
        print("❌ Log de falhas com formato inválido")
        return 0, 0
    
    df_falhas['INCIDENT_DATE'] = pd.to_datetime(df_falhas['INCIDENT_DATE'], errors='coerce')
    df_falhas = df_falhas.dropna(subset=['INCIDENT_DATE', 'CALLSIGN'])
    
    total_falhas = len(df_falhas)
    print(f"📚 {total_falhas} voos a recuperar")
    
    # Executa recuperação
    trajectories_recuperadas = pd.DataFrame()
    falhas_definitivas = []
    
    for i, row in df_falhas.iterrows():
        callsign = str(row['CALLSIGN']).strip().replace(" ", "").upper()
        incident_date = row['INCIDENT_DATE']
        
        print(f"[{i+1}/{total_falhas}] 🔍 Recuperando {callsign}...")
        
        trajectory, erro = coletar_trajetoria_voo(trino_client, callsign, incident_date)
        
        if erro is None and not trajectory.empty:
            trajectories_recuperadas = pd.concat([trajectories_recuperadas, trajectory], ignore_index=True)
            print(f"  ✅ Recuperado: {len(trajectory)} pontos")
        else:
            falhas_definitivas.append({
                'INCIDENT_DATE': incident_date,
                'CALLSIGN': callsign,
                'ERRO_FINAL': erro or "Trajetória vazia"
            })
            print(f"  ❌ Falha definitiva")
    
    # Salva dados recuperados
    if not trajectories_recuperadas.empty:
        salvar_ipc_incremental(trajectories_recuperadas, data_path, modo='append')
    
    # Salva log de falhas definitivas
    if falhas_definitivas:
        salvar_log_csv(falhas_definitivas, log_final_path, modo='write')
    
    # Remove log temporário
    if os.path.exists(log_path):
        os.remove(log_path)
    
    voos_recuperados = trajectories_recuperadas['callsign'].nunique() if not trajectories_recuperadas.empty else 0
    
    return voos_recuperados, len(falhas_definitivas)

# ==============================================================================
# FUNÇÃO DE RELATÓRIO FINAL
# ==============================================================================

def gerar_relatorio_final(period_label, voos_total, voos_sucesso, voos_recuperados, falhas_definitivas):
    """
    Gera relatório consolidado da coleta.
    """
    print(f"\n{'='*60}")
    print("📊 RELATÓRIO FINAL DA COLETA")
    print(f"{'='*60}")
    print(f"Período: {period_label}")
    print(f"Total de voos processados: {voos_total}")
    print(f"Voos coletados (1ª tentativa): {voos_sucesso}")
    print(f"Voos recuperados: {voos_recuperados}")
    print(f"Falhas definitivas: {falhas_definitivas}")
    print(f"Taxa de sucesso: {((voos_sucesso + voos_recuperados) / voos_total * 100):.2f}%")
    print(f"{'='*60}\n")

# ==============================================================================
# MAIN - EXECUÇÃO DO SISTEMA
# ==============================================================================

def main():
    """
    Função principal que orquestra todo o processo de coleta.
    """
    print("\n" + "="*60)
    print("🛰️  SISTEMA INTEGRADO DE COLETA OPENSKY - FORMATO IPC")
    print("="*60 + "\n")
    
    # 1. Calcula período
    try:
        start_date, end_date, period_label = calcular_periodo(PERIOD_TYPE, YEAR, MONTH, WEEK)
        print(f"📅 Período: {period_label}")
        print(f"   Início: {start_date}")
        print(f"   Fim: {end_date}\n")
    except Exception as e:
        print(f"❌ Erro ao calcular período: {e}")
        sys.exit(1)
    
    # 2. Define caminhos de arquivos
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    data_path = os.path.join(OUTPUT_DIR, f"{period_label}_trajetorias.ipc")
    log_path = os.path.join(OUTPUT_DIR, f"{period_label}_log_falhas.csv")
    log_final_path = os.path.join(OUTPUT_DIR, f"{period_label}_falhas_definitivas.csv")
    
    # 3. Carrega dados de bird-strike (ADAPTAR PARA SUA FONTE DE DADOS)
    CSV_FILENAME = "/home/labgeo/code/bird-strike/data_strike_x20251021_160805.csv"
    
    data_strike = pd.read_csv(CSV_FILENAME, sep=";")
    
    data_strike["INCIDENT_DATE"] = pd.to_datetime(data_strike["INCIDENT_DATE"], errors="coerce").dt.date
    data_strike = data_strike.sort_values(by="INCIDENT_DATE", ascending=True).reset_index(drop=True)
    
    # Filtra por período
    df_periodo = data_strike[
        (data_strike["INCIDENT_DATE"] >= start_date) &
        (data_strike["INCIDENT_DATE"] <= end_date)
    ].copy()
    
    if df_periodo.empty:
        print(f"⚠️ Nenhum incidente encontrado no período {period_label}")
        sys.exit(0)
    
    print(f"✅ {len(df_periodo)} incidentes encontrados no período\n")
    
    # 4. Conecta ao Trino
    trino_client = conectar_trino()
    
    # 5. Executa coleta principal
    voos_sucesso = executar_coleta_principal(trino_client, df_periodo, data_path, log_path)
    
    # 6. Executa recuperação de falhas
    voos_recuperados, falhas_definitivas = executar_recuperacao_falhas(
        trino_client, log_path, data_path, log_final_path
    )
    
    # 7. Gera relatório final
    gerar_relatorio_final(
        period_label,
        len(df_periodo),
        voos_sucesso,
        voos_recuperados,
        falhas_definitivas
    )
    
    # 8. Otimização final com Polars (OPCIONAL - pode consumir memória)
    print("\n" + "="*60)
    resposta = input("🔧 Deseja otimizar o arquivo final com Polars? (s/n): ").strip().lower()
    if resposta == 's':
        otimizar_arquivo_final(data_path)
    else:
        print("⏩ Otimização pulada. Arquivo salvo no formato original.")
    
    # 9. Fecha conexão
    try:
        trino_client.conn.close()
        print("🔌 Conexão Trino fechada com sucesso")
    except:
        pass
    
    print("\n✅ Execução finalizada!\n")

# ==============================================================================
# PONTO DE ENTRADA
# ==============================================================================

if __name__ == "__main__":
    main()