#!/usr/bin/env python3
"""建立 uid→user name 對照檔（outputs/uid-map.json），給 usage_stats.py 顯示用。

用法：
    venv/bin/python scripts/build_uid_map.py            # 產生／更新對照檔
    venv/bin/python scripts/build_uid_map.py --show     # 看目前對照檔內容

原理：
    統計的 uid = SHA256(客戶端送來的 key 字串 + STATS_SALT) 前 16 碼，反推不回、
    但正向可算。這支腳本唯讀連上網關的正式 PostgreSQL（Levi 2026-08-23 確認），
    撈 tokens×users，把每把 key 在**記憶體裡**算成 uid，只落地 uid→使用者名稱。

安全邊界（跟統計「外流也無害」的設計配套）：
    · 明文 key 不落地——連線、計算、丟棄，全程只在記憶體。
    · 產出的對照檔是**去匿名化資料**，跟報表同級保管：在 outputs/（gitignore、
      不進 Docker image），過期就刪、要更新重跑。
    · 對 DB 只有 SELECT，不建立任何長連線。

實作細節：
    · 網關（new-api）tokens.key 存的是**不含 sk- 前綴**的字串，客戶端實際送的是
      "sk-"+key——兩種形都算 uid（保險起見），對到同一個使用者。
    · STATS_SALT 正式環境為空字串（Levi 先前裁示），要改 salt 兩邊要一起改。
    · DB 憑證讀共用 .env 的 [production DB postgresql] 區塊；那份檔有重複鍵，
      必須按區塊解析，不能整檔當 key=value 讀。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ENV_PATH = Path("/Users/levi/nen_ai_project/.env")
OUT = ROOT / "outputs" / "uid-map.json"
STATS_SALT = ""   # 與正式環境一致（未設 = 空字串）


def _db_creds() -> dict:
    """從共用 .env 的 [production DB postgresql] 區塊組出連線參數。

    ⚠️ 該區塊的 ip/user/passwd 是那台 VM 的 SSH 憑證，**不是** Postgres 帳密——
    真正的資料庫憑證在同區塊的 `postgresql://…` URI 裡（實測踩過：拿 SSH 的
    root/密碼去連 DB 會 password authentication failed）。組合方式：
    帳密與 db 名取自 URI（密碼是 URL 編碼要解掉；URI 內的 host 是內網 IP，
    從外面連不到）、host 用區塊的 ip=（公網）。pg_hba 拒絕明文連線，要帶 SSL。
    """
    from urllib.parse import unquote, urlparse
    cur, ip, uri = None, "", ""
    for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("["):
            cur = line
            continue
        if cur == "[production DB postgresql]":
            if line.startswith("ip="):
                ip = line.split("=", 1)[1].strip()
            elif line.startswith("postgresql://"):
                uri = line
    if not ip or not uri:
        sys.exit("共用 .env 的 [production DB postgresql] 區塊缺 ip= 或 postgresql:// URI")
    u = urlparse(uri)
    return {"host": ip, "user": u.username, "password": unquote(u.password or ""),
            "dbname": (u.path or "/newapi").lstrip("/")}


def _uid(key: str) -> str:
    return hashlib.sha256((key + STATS_SALT).encode()).hexdigest()[:16]


def build() -> None:
    import psycopg2  # 僅本地腳本用，刻意不進 requirements.txt

    c = _db_creds()
    conn = psycopg2.connect(host=c["host"], port=5432, user=c["user"],
                            password=c["password"], dbname=c["dbname"],
                            sslmode="require", connect_timeout=10)
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT t.key, t.name, u.id, u.username, u.display_name
                FROM tokens t JOIN users u ON u.id = t.user_id
            """)
            rows = cur.fetchall()
    finally:
        conn.close()

    mapping: dict[str, dict] = {}
    for key, token_name, user_id, username, display_name in rows:
        name = (display_name or username or f"user#{user_id}").strip()
        info = {"user": name, "user_id": user_id, "token_name": token_name or ""}
        # 兩種 key 形式都對：客戶端送 sk-<key>（正常情況）與裸 key（保險）
        mapping[_uid(f"sk-{key}")] = info
        mapping[_uid(key)] = info

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"{len(rows)} 把 key → {len(mapping)} 個 uid 對照 → {OUT}")
    print("⚠️ 這是去匿名化對照檔，過期請刪；更新對照重跑本腳本即可。")


def show() -> None:
    if not OUT.exists():
        sys.exit(f"{OUT} 不存在，先跑一次 build（不帶 --show）")
    data = json.loads(OUT.read_text(encoding="utf-8"))
    seen = set()
    for uid, info in data.items():
        k = (info["user_id"], info["token_name"])
        tag = "" if k in seen else f'  {info["user"]}（user#{info["user_id"]}，token "{info["token_name"]}"）'
        seen.add(k)
        print(f"{uid}{tag}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="建立 uid→user 對照檔")
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    show() if args.show else build()
