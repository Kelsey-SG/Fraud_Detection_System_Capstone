"""
db_scoring.py
=============
Importable Python module mirroring the cells of db_scoring.ipynb.
deploy_model imports this file as a regular module.
"""

import os

"""
db_scoring.ipynb
================
Python integration layer for the PostgreSQL (Neon) relational-database
anomaly scoring component of the hybrid fraud detector.

This notebook is also exported to db_scoring.py so deploy_model can import
its functions as a regular module.
"""

import uuid
import time
import logging
import numpy as np
import pandas as pd
from contextlib import contextmanager
from typing import List, Optional

import psycopg2
import psycopg2.pool
import psycopg2.extras

# Logger — keep timestamps short so batch progress is easy to scan.
log = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)

# ── Database configuration ──────────────────────────────────────────────
# Credentials are loaded from environment variables so they are never
# committed to source control. Set these in your shell before running:
#
#   export DB_HOST="your-host.example.com"
#   export DB_NAME="your_database"
#   export DB_USER="your_user"
#   export DB_PASSWORD="your_password"
#
# The defaults below are placeholders only and will not connect to anything.
DB_CONFIG = {
    "host":     os.environ.get("DB_HOST",     "<DB_HOST_PLACEHOLDER>"),
    "port":     int(os.environ.get("DB_PORT", 5432)),
    "dbname":   os.environ.get("DB_NAME",     "<DB_NAME_PLACEHOLDER>"),
    "user":     os.environ.get("DB_USER",     "<DB_USER_PLACEHOLDER>"),
    "password": os.environ.get("DB_PASSWORD", "<DB_PASSWORD_PLACEHOLDER>"),
    "sslmode":  os.environ.get("DB_SSLMODE",  "require"),
}

# Connection-pool sizing. The pool keeps a small set of warm connections
# open so that scoring calls don't pay the SSL/handshake cost each time.
POOL_MIN_CONN      = 2
POOL_MAX_CONN      = 10

# Default cut-off for marking a transaction as high-risk.
DEFAULT_THRESHOLD  = 0.5

# Batch sizes — tuned to balance memory use against round-trip count.
BULK_INSERT_BATCH  = 5_000
SCORING_BATCH_SIZE = 500

# ── Connection pool ─────────────────────────────────────────────────────
# We keep a single module-level pool so callers can fan out without each
# of them paying the cost of opening a fresh TCP/SSL connection.
_pool = None


def init_pool(config: dict = DB_CONFIG) -> None:
    """Create the threaded connection pool. Safe to call multiple times."""
    global _pool
    _pool = psycopg2.pool.ThreadedConnectionPool(
        POOL_MIN_CONN, POOL_MAX_CONN, **config)
    log.info("Connection pool initialised (min=%d, max=%d).",
             POOL_MIN_CONN, POOL_MAX_CONN)


def close_pool() -> None:
    """Tear down the pool. Call this at the end of a deployment run."""
    global _pool
    if _pool:
        _pool.closeall()
        _pool = None
        log.info("Connection pool closed.")


@contextmanager
def get_conn():
    """Borrow a connection from the pool.

    Commits on clean exit, rolls back if anything raises, and always
    returns the connection to the pool — even on error.
    """
    if _pool is None:
        init_pool()
    conn = _pool.getconn()
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        _pool.putconn(conn)

def load_paysim_csv(filepath: str,
                    limit: Optional[int] = None) -> pd.DataFrame:
    """Read the PaySim CSV and keep only the transaction types we model.

    Only TRANSFER and CASH_OUT contain the fraud cases in PaySim, so the
    other types are dropped here to keep downstream work focused.
    """
    log.info("Reading PaySim CSV: %s", filepath)
    df = pd.read_csv(filepath, nrows=limit)
    df = df[df['type'].isin(['TRANSFER', 'CASH_OUT'])].copy()
    # Order by user, then by step — required so per-user behaviour
    # features (velocity, time-since-last-txn) come out right.
    df = df.sort_values(by=['nameOrig', 'step']).reset_index(drop=True)
    log.info("Loaded %d transactions after filtering.", len(df))
    return df


def insert_transactions(df: pd.DataFrame,
                        batch_size: int = BULK_INSERT_BATCH) -> List[str]:
    """Bulk-insert a dataframe of transactions into the `transactions` table.

    Each row gets a fresh UUID; duplicates are silently skipped via the
    ON CONFLICT clause so re-runs are idempotent.

    Returns the list of UUIDs in the same order as the input rows so
    callers can attach them back to the dataframe.
    """
    uuids = [str(uuid.uuid4()) for _ in range(len(df))]

    insert_sql = """
        INSERT INTO transactions (
            transaction_id, step, type, amount,
            name_orig, old_balance_orig, new_balance_orig,
            name_dest, old_balance_dest, new_balance_dest,
            is_fraud
        )
        VALUES %s
        ON CONFLICT (transaction_id) DO NOTHING
    """

    log.info("Inserting %d transactions in batches of %d...", len(df), batch_size)
    start = time.perf_counter()

    with get_conn() as conn:
        with conn.cursor() as cur:
            for batch_start in range(0, len(df), batch_size):
                batch   = df.iloc[batch_start: batch_start + batch_size]
                batch_u = uuids[batch_start: batch_start + batch_size]
                rows = [
                    (
                        batch_u[i],
                        int(row.step), str(row.type), float(row.amount),
                        str(row.nameOrig),
                        float(row.oldbalanceOrg), float(row.newbalanceOrig),
                        str(row.nameDest),
                        float(row.oldbalanceDest), float(row.newbalanceDest),
                        int(row.isFraud) if hasattr(row, 'isFraud') else 0,
                    )
                    for i, (_, row) in enumerate(batch.iterrows())
                ]
                psycopg2.extras.execute_values(
                    cur, insert_sql, rows, page_size=batch_size)
                log.info("  Inserted batch %d/%d",
                         batch_start // batch_size + 1,
                         (len(df) - 1) // batch_size + 1)

    elapsed = time.perf_counter() - start
    log.info("Insert complete: %.2fs (%.0f rows/s)", elapsed, len(df) / elapsed)
    return uuids

def score_single_transaction(transaction_id: str) -> dict:
    """Score one transaction by UUID and return its full rule breakdown.

    Used for the real-time path. The DB function `compute_sql_anomaly_score`
    persists the score; we then read the individual rule contributions back.
    """
    start = time.perf_counter()

    with get_conn() as conn:
        with conn.cursor() as cur:
            # Step 1 — compute and persist the score for this txn.
            cur.execute(
                "SELECT compute_sql_anomaly_score(%s::uuid)",
                (transaction_id,)
            )
            cur.fetchone()

            # Step 2 — fetch the most recent score row, including each rule.
            cur.execute(
                """
                SELECT score_balance_mismatch, score_zero_drain,
                       score_large_amount,     score_high_velocity,
                       score_dest_jump,        sql_anomaly_score
                FROM   anomaly_scores
                WHERE  transaction_id = %s::uuid
                ORDER  BY scored_at DESC
                LIMIT  1
                """,
                (transaction_id,)
            )
            row = cur.fetchone()

    latency_ms = (time.perf_counter() - start) * 1000

    if row is None:
        raise ValueError(f"No score found for transaction {transaction_id}")

    return {
        "transaction_id":         transaction_id,
        "score_balance_mismatch": float(row[0]),
        "score_zero_drain":       float(row[1]),
        "score_large_amount":     float(row[2]),
        "score_high_velocity":    float(row[3]),
        "score_dest_jump":        float(row[4]),
        "sql_anomaly_score":      float(row[5]),
        "latency_ms":             round(latency_ms, 2),
    }


def score_batch(transaction_ids: List[str],
                batch_size: int = SCORING_BATCH_SIZE) -> pd.DataFrame:
    """Score many transactions in one round-trip per chunk.

    The DB function `score_transaction_batch` accepts a UUID array and
    returns one score per id. Doing it this way is much faster than
    calling `score_single_transaction` in a loop.
    """
    log.info("Scoring %d transactions in batches of %d...",
             len(transaction_ids), batch_size)
    start    = time.perf_counter()
    all_rows = []
    query    = ("SELECT transaction_id::text, sql_anomaly_score "
                "FROM score_transaction_batch(%s::uuid[])")

    with get_conn() as conn:
        with conn.cursor() as cur:
            for i in range(0, len(transaction_ids), batch_size):
                chunk    = transaction_ids[i: i + batch_size]
                pg_array = "{" + ",".join(chunk) + "}"
                cur.execute(query, (pg_array,))
                all_rows.extend(cur.fetchall())

    log.info("Batch scoring complete: %.2fs", time.perf_counter() - start)
    scores_df = pd.DataFrame(all_rows,
                             columns=['transaction_id', 'sql_anomaly_score'])
    scores_df['sql_anomaly_score'] = scores_df['sql_anomaly_score'].astype(float)

    # Restore the caller's ordering — the DB doesn't guarantee it.
    order_map          = {tid: idx for idx, tid in enumerate(transaction_ids)}
    scores_df['_ord']  = scores_df['transaction_id'].map(order_map)
    scores_df          = scores_df.sort_values('_ord').drop(columns=['_ord'])
    return scores_df.reset_index(drop=True)


def fetch_existing_scores(transaction_ids: List[str]) -> pd.DataFrame:
    """Read previously-computed scores without recomputing them."""
    query    = ("SELECT transaction_id::text, sql_anomaly_score "
                "FROM anomaly_scores WHERE transaction_id = ANY(%s::uuid[])")
    pg_array = "{" + ",".join(transaction_ids) + "}"
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (pg_array,))
            rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['transaction_id', 'sql_anomaly_score'])
    df['sql_anomaly_score'] = df['sql_anomaly_score'].astype(float)
    return df


def flag_high_risk(transaction_ids, scores, threshold=DEFAULT_THRESHOLD):
    """Persist the IDs of transactions whose score exceeds `threshold`.

    Returns the count of newly-flagged rows. Re-flagging is a no-op
    thanks to ON CONFLICT DO NOTHING.
    """
    threshold = float(threshold)   # cast numpy.float64 -> Python float
    risky = [(tid, float(s)) for tid, s in zip(transaction_ids, scores)
             if s > threshold]
    if not risky:
        return 0
    insert_sql = """
        INSERT INTO flagged_transactions
               (transaction_id, sql_anomaly_score, threshold_used)
        VALUES %s ON CONFLICT DO NOTHING
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            psycopg2.extras.execute_values(
                cur, insert_sql,
                [(r[0], r[1], threshold) for r in risky])
    log.info("Flagged %d high-risk transactions (threshold=%.4f).",
             len(risky), threshold)
    return len(risky)


def align_sql_scores_with_sequences(
        test_df: pd.DataFrame, scores_df: pd.DataFrame,
        context_length: int, prediction_length: int) -> np.ndarray:
    """Line up SQL scores with the Transformer's sliding-window outputs.

    The Transformer emits one prediction per window; this picks the
    matching SQL score (the one for the last txn in each window) so the
    two streams can be fused element-wise.
    """
    test_df   = test_df.reset_index(drop=True)
    score_map = dict(zip(scores_df['transaction_id'],
                         scores_df['sql_anomaly_score']))
    window  = context_length + prediction_length
    per_seq = []
    for user, group in test_df.groupby('nameOrig'):
        uuids = group['transaction_id'].tolist()
        for start in range(0, len(uuids) - window + 1):
            per_seq.append(score_map.get(uuids[start + window - 1], 0.0))
    return np.array(per_seq)


def score_new_transaction_realtime(transaction_row: pd.Series,
                                   upsert: bool = True) -> dict:
    """End-to-end real-time path: insert one txn, score it, return the result."""
    tid = str(uuid.uuid4())
    if upsert:
        insert_sql = """
            INSERT INTO transactions (
                transaction_id, step, type, amount,
                name_orig, old_balance_orig, new_balance_orig,
                name_dest, old_balance_dest, new_balance_dest, is_fraud
            ) VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s)
            ON CONFLICT (transaction_id) DO NOTHING
        """
        with get_conn() as conn:
            with conn.cursor() as cur:
                cur.execute(insert_sql, (
                    tid,
                    int(transaction_row['step']), str(transaction_row['type']),
                    float(transaction_row['amount']),
                    str(transaction_row['nameOrig']),
                    float(transaction_row['oldbalanceOrg']),
                    float(transaction_row['newbalanceOrig']),
                    str(transaction_row.get('nameDest', 'unknown')),
                    float(transaction_row['oldbalanceDest']),
                    float(transaction_row['newbalanceDest']),
                    int(transaction_row.get('isFraud', 0)),
                ))
    return score_single_transaction(tid)


# ── Reporting helpers ───────────────────────────────────────────────────
# These read pre-defined views so the dashboard / demo can pull summary
# stats without touching the underlying tables directly.
def get_score_summary() -> dict:
    """Return aggregate stats from the v_rule_score_summary view."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM v_rule_score_summary;")
            row = cur.fetchone()
    return dict(row) if row else {}


def get_rule_hit_rates() -> dict:
    """Return how often each rule fires (from v_rule_hit_rates)."""
    with get_conn() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("SELECT * FROM v_rule_hit_rates;")
            row = cur.fetchone()
    return dict(row) if row else {}


def get_high_risk_sample(limit: int = 20) -> pd.DataFrame:
    """Pull a small sample of the riskiest transactions for inspection."""
    query = """
        SELECT transaction_id, step, type, amount, name_orig,
               score_balance_mismatch, score_zero_drain, score_large_amount,
               score_high_velocity, score_dest_jump, sql_anomaly_score, is_fraud
        FROM   v_high_risk_transactions LIMIT %s
    """
    with get_conn() as conn:
        with conn.cursor() as cur:
            cur.execute(query, (limit,))
            rows = cur.fetchall()
            cols = [d[0] for d in cur.description]
    return pd.DataFrame(rows, columns=cols)

