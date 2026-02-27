#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#使用方法：
# pip install requests pandas pyarrow
# python .\download_binance_klines.py --symbol BTCUSDT --interval 1h --start 2017-01-01 --end 2026-01-01 --outdir .\data --format csv
# data range: 2019-09-23 08:00:00+00:00 to 2026-01-01

"""

说明：
Binance 5m BTC/USDT downloader with local cache + resume.

Features:
- Downloads spot klines (candles) from Binance public API (no key needed)
- Interval: default 5m
- Saves locally as monthly Parquet/CSV shards: data/BTCUSDT/5m/klines_YYYY_MM.parquet
- Resume-safe: will append missing data only
- Basic rate limiting + retry
- Outputs a consolidated index CSV for convenience

Dependencies:
- requests
- pandas
- (optional but recommended) pyarrow for parquet

Install:
  pip install requests pandas pyarrow

Example:
  python download_binance_klines.py --symbol BTCUSDT --interval 5m --start 2020-01-01 --outdir ./data
  python download_binance_klines.py --symbol BTCUSDT --interval 1h --start 2023-02-03 --end 2026-02-02 --format csv
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import time
from dataclasses import dataclass
from typing import Iterable, Optional, Tuple, List

import pandas as pd
import requests

print(">>> script started <<<")
BINANCE_SPOT_BASE = "https://api.binance.us"
KLINES_ENDPOINT = "/api/v3/klines"


@dataclass
class DownloadConfig:
    symbol: str
    interval: str
    start_ms: int
    end_ms: int
    outdir: str
    fmt: str  # "parquet" or "csv"
    limit: int = 1000
    sleep_s: float = 0.25
    max_retries: int = 6
    timeout_s: int = 20


def to_utc_ms(date_str: str) -> int:
    """
    Parse 'YYYY-MM-DD' or ISO datetime string to UTC milliseconds.
    Examples:
      2020-01-01
      2020-01-01T00:00:00
      2020-01-01 00:00:00
    """
    s = date_str.strip()
    # Allow date-only
    if len(s) == 10:
        d = dt.datetime.strptime(s, "%Y-%m-%d")
    else:
        # Try a couple common formats
        for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M:%S"):
            try:
                d = dt.datetime.strptime(s, fmt)
                break
            except ValueError:
                d = None
        if d is None:
            raise ValueError(f"Cannot parse date/time: {date_str}")
    # Treat as UTC
    d = d.replace(tzinfo=dt.timezone.utc)
    return int(d.timestamp() * 1000)


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def request_klines(
    session: requests.Session,
    symbol: str,
    interval: str,
    start_ms: int,
    end_ms: int,
    limit: int,
    timeout_s: int,
    max_retries: int,
    sleep_s: float,
) -> List[list]:
    """
    Fetch up to `limit` klines starting from `start_ms` (inclusive).
    Binance returns klines with openTime >= startTime and openTime < endTime (effectively).
    """
    url = BINANCE_SPOT_BASE + KLINES_ENDPOINT
    params = {
        "symbol": symbol,
        "interval": interval,
        "startTime": start_ms,
        "endTime": end_ms,
        "limit": limit,
    }

    backoff = 1.0
    for attempt in range(1, max_retries + 1):
        try:
            resp = session.get(url, params=params, timeout=timeout_s)
            if resp.status_code == 429 or resp.status_code == 418:
                # rate limit / IP ban protection
                time.sleep(backoff)
                backoff = min(backoff * 2, 60)
                continue
            resp.raise_for_status()
            data = resp.json()
            if not isinstance(data, list):
                raise RuntimeError(f"Unexpected response: {data}")
            time.sleep(sleep_s)  # gentle throttle
            return data
        except Exception as e:
            if attempt == max_retries:
                raise
            time.sleep(backoff)
            backoff = min(backoff * 2, 30)

    return []


def klines_to_df(raw: List[list]) -> pd.DataFrame:
    """
    Convert Binance kline array to a typed DataFrame.
    Binance kline schema:
      [
        0 open time (ms),
        1 open,
        2 high,
        3 low,
        4 close,
        5 volume,
        6 close time (ms),
        7 quote asset volume,
        8 number of trades,
        9 taker buy base asset volume,
        10 taker buy quote asset volume,
        11 ignore
      ]
    """
    if not raw:
        return pd.DataFrame()

    cols = [
        "open_time",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_volume",
        "num_trades",
        "taker_buy_volume",
        "taker_buy_quote_volume",
        "ignore",
    ]
    df = pd.DataFrame(raw, columns=cols)

    # Types
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
    df["close_time"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

    float_cols = ["open", "high", "low", "close", "volume", "quote_volume", "taker_buy_volume", "taker_buy_quote_volume"]
    for c in float_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["num_trades"] = pd.to_numeric(df["num_trades"], errors="coerce").astype("Int64")

    df = df.drop(columns=["ignore"])
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return df


def month_key(ts: pd.Timestamp) -> Tuple[int, int]:
    return ts.year, ts.month


def month_start_end_utc(year: int, month: int) -> Tuple[pd.Timestamp, pd.Timestamp]:
    start = pd.Timestamp(dt.datetime(year, month, 1, 0, 0, 0, tzinfo=dt.timezone.utc))
    if month == 12:
        end = pd.Timestamp(dt.datetime(year + 1, 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc))
    else:
        end = pd.Timestamp(dt.datetime(year, month + 1, 1, 0, 0, 0, tzinfo=dt.timezone.utc))
    return start, end


def iter_months(start_utc: pd.Timestamp, end_utc: pd.Timestamp) -> Iterable[Tuple[int, int]]:
    """
    Yield (year, month) covering [start_utc, end_utc)
    """
    cur = pd.Timestamp(dt.datetime(start_utc.year, start_utc.month, 1, tzinfo=dt.timezone.utc))
    endm = pd.Timestamp(dt.datetime(end_utc.year, end_utc.month, 1, tzinfo=dt.timezone.utc))
    while cur <= endm:
        yield cur.year, cur.month
        # next month
        if cur.month == 12:
            cur = pd.Timestamp(dt.datetime(cur.year + 1, 1, 1, tzinfo=dt.timezone.utc))
        else:
            cur = pd.Timestamp(dt.datetime(cur.year, cur.month + 1, 1, tzinfo=dt.timezone.utc))


def shard_path(outdir: str, symbol: str, interval: str, year: int, month: int, fmt: str) -> str:
    folder = os.path.join(outdir, symbol.upper(), interval)
    ensure_dir(folder)
    ext = "parquet" if fmt == "parquet" else "csv"
    return os.path.join(folder, f"klines_{year:04d}_{month:02d}.{ext}")


def load_shard(path: str, fmt: str) -> pd.DataFrame:
    if not os.path.exists(path):
        return pd.DataFrame()
    if fmt == "parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path, parse_dates=["open_time", "close_time"])


def save_shard(df: pd.DataFrame, path: str, fmt: str) -> None:
    df = df.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    if fmt == "parquet":
        df.to_parquet(path, index=False)
    else:
        df.to_csv(path, index=False)


def get_last_open_time_ms(existing: pd.DataFrame) -> Optional[int]:
    if existing.empty:
        return None
    last_ts = pd.to_datetime(existing["open_time"], utc=True).max()
    return int(last_ts.value // 10**6)  # ns -> ms


def download_range_to_df(cfg: DownloadConfig) -> pd.DataFrame:
    """
    Download [cfg.start_ms, cfg.end_ms] klines into a DataFrame (may be large).
    Prefer shard-by-month via main(), but this is useful for small ranges.
    """
    all_parts = []
    with requests.Session() as session:
        cursor = cfg.start_ms
        while cursor <= cfg.end_ms:
            raw = request_klines(
                session=session,
                symbol=cfg.symbol,
                interval=cfg.interval,
                start_ms=cursor,
                end_ms=cfg.end_ms,
                limit=cfg.limit,
                timeout_s=cfg.timeout_s,
                max_retries=cfg.max_retries,
                sleep_s=cfg.sleep_s,
            )
            if not raw:
                break
            df_part = klines_to_df(raw)
            if df_part.empty:
                break
            all_parts.append(df_part)

            last_open = df_part["open_time"].max()
            # Move cursor forward by 1 ms beyond last open_time to avoid duplicates
            cursor = int((last_open.value // 10**6) + 1)

            # If Binance returns less than limit, likely reached end
            if len(raw) < cfg.limit:
                break

    if not all_parts:
        return pd.DataFrame()
    df_all = pd.concat(all_parts, ignore_index=True)
    df_all = df_all.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
    return df_all


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol", type=str, default="BTCUSDT")
    parser.add_argument("--interval", type=str, default="5m")
    parser.add_argument("--start", type=str, required=True, help="Start date/time (UTC), e.g. 2020-01-01 or 2020-01-01T00:00:00")
    parser.add_argument("--end", type=str, default=None, help="End date/time (UTC). Default: now (UTC).")
    parser.add_argument("--outdir", type=str, default="./data")
    parser.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--sleep", type=float, default=0.25, help="Sleep seconds between requests.")
    parser.add_argument("--limit", type=int, default=1000, help="Max klines per request (Binance max=1000).")
    args = parser.parse_args()

    start_ms = to_utc_ms(args.start)
    if args.end is None:
        end_ms = int(dt.datetime.now(dt.timezone.utc).timestamp() * 1000)
    else:
        end_ms = to_utc_ms(args.end)

    if end_ms <= start_ms:
        raise ValueError("End must be after start.")

    cfg = DownloadConfig(
        symbol=args.symbol.upper(),
        interval=args.interval,
        start_ms=start_ms,
        end_ms=end_ms,
        outdir=args.outdir,
        fmt=args.format,
        limit=min(args.limit, 1000),
        sleep_s=max(args.sleep, 0.0),
    )

    base_folder = os.path.join(cfg.outdir, cfg.symbol, cfg.interval)
    ensure_dir(base_folder)

    start_utc = pd.to_datetime(cfg.start_ms, unit="ms", utc=True)
    end_utc = pd.to_datetime(cfg.end_ms, unit="ms", utc=True)

    manifest_rows = []

    print(f"[INFO] Downloading {cfg.symbol} {cfg.interval} from {start_utc} to {end_utc}")
    print(f"[INFO] Output folder: {base_folder} (format={cfg.fmt})")

    with requests.Session() as session:
        for (y, m) in iter_months(start_utc, end_utc):
            m_start, m_end = month_start_end_utc(y, m)

            # intersect month with requested [start_utc, end_utc]
            seg_start = max(m_start, start_utc)
            seg_end = min(m_end, end_utc)
            if seg_end <= seg_start:
                continue

            path = shard_path(cfg.outdir, cfg.symbol, cfg.interval, y, m, cfg.fmt)
            existing = load_shard(path, cfg.fmt)

            # Resume: start from last open_time + 1ms
            resume_ms = None
            last_ms = get_last_open_time_ms(existing)
            seg_start_ms = int(seg_start.value // 10**6)
            seg_end_ms = int(seg_end.value // 10**6)

            if last_ms is not None and last_ms >= seg_start_ms:
                resume_ms = last_ms + 1
            else:
                resume_ms = seg_start_ms

            if resume_ms > seg_end_ms:
                # Nothing to do for this shard
                if not existing.empty:
                    manifest_rows.append([y, m, path, len(existing), existing["open_time"].min(), existing["open_time"].max()])
                print(f"[SKIP] {y:04d}-{m:02d}: already complete ({os.path.basename(path)})")
                continue

            print(f"[MONTH] {y:04d}-{m:02d}: fetching from {pd.to_datetime(resume_ms, unit='ms', utc=True)} to {seg_end}")

            cursor = resume_ms
            parts = []
            while cursor <= seg_end_ms:
                raw = request_klines(
                    session=session,
                    symbol=cfg.symbol,
                    interval=cfg.interval,
                    start_ms=cursor,
                    end_ms=seg_end_ms,
                    limit=cfg.limit,
                    timeout_s=cfg.timeout_s,
                    max_retries=cfg.max_retries,
                    sleep_s=cfg.sleep_s,
                )
                if not raw:
                    break

                df_part = klines_to_df(raw)
                if df_part.empty:
                    break

                parts.append(df_part)

                last_open = df_part["open_time"].max()
                cursor = int((last_open.value // 10**6) + 1)

                if len(raw) < cfg.limit:
                    break

            if parts:
                new_df = pd.concat(parts, ignore_index=True)
                if existing.empty:
                    merged = new_df
                else:
                    # Ensure consistent dtypes by concatenating after parsing
                    merged = pd.concat([existing, new_df], ignore_index=True)

                merged = merged.sort_values("open_time").drop_duplicates(subset=["open_time"], keep="last").reset_index(drop=True)
                save_shard(merged, path, cfg.fmt)
                print(f"[SAVE] {os.path.basename(path)} rows={len(merged)}")
                manifest_rows.append([y, m, path, len(merged), merged["open_time"].min(), merged["open_time"].max()])
            else:
                # No new data returned; still record existing if present
                if not existing.empty:
                    manifest_rows.append([y, m, path, len(existing), existing["open_time"].min(), existing["open_time"].max()])
                print(f"[WARN] {y:04d}-{m:02d}: no data returned (maybe before listing or API issue).")

    # Write manifest for easy loading
    if manifest_rows:
        manifest = pd.DataFrame(
            manifest_rows,
            columns=["year", "month", "path", "rows", "min_open_time", "max_open_time"],
        )
        manifest = manifest.sort_values(["year", "month"]).reset_index(drop=True)
        manifest_path = os.path.join(base_folder, "manifest.csv")
        manifest.to_csv(manifest_path, index=False)
        print(f"[INFO] Wrote manifest: {manifest_path}")

    # Merge all CSV shards into one consolidated file
    if manifest_rows and cfg.fmt == "csv":
        print(f"[INFO] Merging all CSV shards into consolidated file...")
        all_dfs = []
        for row in manifest_rows:
            path = row[2]  # path column
            if os.path.exists(path):
                df_shard = pd.read_csv(path, parse_dates=["open_time", "close_time"])
                if not df_shard.empty:
                    all_dfs.append(df_shard)
                    print(f"[MERGE] Added {os.path.basename(path)}: {len(df_shard)} rows")
        
        if all_dfs:
            consolidated = pd.concat(all_dfs, ignore_index=True)
            consolidated = consolidated.sort_values("open_time").drop_duplicates(
                subset=["open_time"], keep="last"
            ).reset_index(drop=True)
            
            # Generate consolidated filename
            start_date = consolidated["open_time"].min().strftime("%Y%m%d")
            end_date = consolidated["open_time"].max().strftime("%Y%m%d")
            consolidated_path = os.path.join(
                base_folder, f"{cfg.symbol}_{cfg.interval}_{start_date}_{end_date}.csv"
            )
            
            consolidated.to_csv(consolidated_path, index=False)
            print(f"[INFO] Consolidated CSV saved: {consolidated_path}")
            print(f"[INFO] Total rows: {len(consolidated)}")
            print(f"[INFO] Date range: {consolidated['open_time'].min()} to {consolidated['open_time'].max()}")
        else:
            print(f"[WARN] No CSV shards found to merge.")

    print("[DONE]")


if __name__ == "__main__":
    main()
