#!/usr/bin/env python3
"""使用者統計報表——把 playground 記下的呼叫紀錄整理成一份本機 HTML。

用法：
    venv/bin/python scripts/usage_stats.py                  # 最近 7 天，產生並開啟 HTML
    venv/bin/python scripts/usage_stats.py --days 30
    venv/bin/python scripts/usage_stats.py --source local   # 只讀本機 outputs/stats/
    venv/bin/python scripts/usage_stats.py --no-open        # 只產生檔案不開瀏覽器

資料從哪來：
    app.py 的統計 middleware 每 50 筆或 60 秒把紀錄寫成 jsonl，優先寫雲端物件儲存
    （GCS/S3/OSS，路徑 stats/YYYY-MM-DD/HH_{instance}_{ts}.jsonl），沒有雲端憑證時
    退回本機 outputs/stats/。這支腳本兩邊都讀，合併後去重。

為什麼是「產生本機 HTML」而不是做一個網頁後台：
    平台本身完全沒有查詢入口——沒有網址、沒有管理密碼、沒有可以被猜到的路徑。
    報表是你在自己機器上產生的檔案，看完可以刪。這樣「後台外洩」這件事在設計上
    就不存在，而不是靠密碼去防。

⚠️ 紀錄裡的 uid 是 SHA256(api_key + STATS_SALT) 的前 16 碼，**不可反推回 key**。
    所以這份報表能回答「有幾個人在用、誰用得最多」，但不能回答「這個人是誰」——
    要對應到真實客戶得由網關端提供 key→user 的對照，那是另一件事。
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = ROOT / "outputs" / "stats"
OUT_HTML = ROOT / "outputs" / "usage-report.html"
PREFIX = os.environ.get("STATS_PREFIX", "stats")


def _load_local(since: datetime) -> list[dict]:
    rows: list[dict] = []
    if not LOCAL_DIR.exists():
        return rows
    for fp in LOCAL_DIR.glob("*.jsonl"):
        for line in fp.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if _ts(r) >= since:
                rows.append(r)
    return rows


def _load_gcs(since: datetime) -> list[dict]:
    """從 GCS 讀。憑證沿用 app.py 那組環境變數，沒設就直接跳過。"""
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    if not bucket_name:
        return []
    try:
        from google.cloud import storage as gcs_storage
    except ImportError:
        print("[warn] 未安裝 google-cloud-storage，略過雲端來源", file=sys.stderr)
        return []
    try:
        creds_json = os.environ.get("GCS_CREDENTIALS_JSON", "")
        if creds_json:
            from google.oauth2 import service_account
            info = json.loads(creds_json)
            client = gcs_storage.Client(
                credentials=service_account.Credentials.from_service_account_info(info),
                project=info.get("project_id"))
        else:
            client = gcs_storage.Client()
        rows: list[dict] = []
        for blob in client.list_blobs(bucket_name, prefix=f"{PREFIX}/"):
            # 檔名帶日期，先用它粗篩，省下不必要的下載
            day = blob.name.split("/")[1] if "/" in blob.name else ""
            try:
                if day and datetime.strptime(day, "%Y-%m-%d") < since - timedelta(days=1):
                    continue
            except ValueError:
                pass
            for line in blob.download_as_text().splitlines():
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if _ts(r) >= since:
                    rows.append(r)
        return rows
    except Exception as e:
        print(f"[warn] 讀取 GCS 失敗（{type(e).__name__}: {e}），只用本機資料", file=sys.stderr)
        return []


def _ts(r: dict) -> datetime:
    try:
        return datetime.fromisoformat(r.get("ts", ""))
    except ValueError:
        return datetime.min


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _bar(pct: float, color: str = "#01A0C7") -> str:
    return (f'<div class="bar"><i style="width:{pct:.1f}%;background:{color}"></i></div>')


def build_html(rows: list[dict], days: int) -> str:
    total = len(rows)
    users = Counter(r.get("uid", "?") for r in rows)
    endpoints = Counter(r.get("endpoint", "?") for r in rows)
    failures = [r for r in rows if not r.get("ok")]
    by_day: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_day[_ts(r).strftime("%Y-%m-%d")][r.get("uid", "?")] += 1

    first_seen: dict[str, datetime] = {}
    last_seen: dict[str, datetime] = {}
    for r in rows:
        uid, t = r.get("uid", "?"), _ts(r)
        if uid not in first_seen or t < first_seen[uid]:
            first_seen[uid] = t
        if uid not in last_seen or t > last_seen[uid]:
            last_seen[uid] = t

    fail_by_uid = Counter(r.get("uid", "?") for r in failures)
    top_user = users.most_common(1)[0][1] if users else 1

    user_rows = "".join(
        f"<tr><td><code>{_esc(uid)}</code></td><td>{n}</td>"
        f"<td>{fail_by_uid.get(uid, 0)}</td>"
        f"<td>{first_seen[uid]:%Y-%m-%d %H:%M}</td>"
        f"<td>{last_seen[uid]:%Y-%m-%d %H:%M}</td>"
        f"<td>{_bar(n / top_user * 100)}</td></tr>"
        for uid, n in users.most_common())

    top_ep = endpoints.most_common(1)[0][1] if endpoints else 1
    ep_rows = "".join(
        f"<tr><td><code>{_esc(ep)}</code></td><td>{n}</td>"
        f"<td>{_bar(n / top_ep * 100, '#0064C8')}</td></tr>"
        for ep, n in endpoints.most_common(20))

    day_rows = "".join(
        f"<tr><td>{d}</td><td>{sum(c.values())}</td><td>{len(c)}</td></tr>"
        for d, c in sorted(by_day.items(), reverse=True))

    fail_codes = Counter(f"{r.get('status')} {r.get('endpoint')}" for r in failures)
    fail_rows = "".join(
        f"<tr><td><code>{_esc(k)}</code></td><td>{n}</td></tr>"
        for k, n in fail_codes.most_common(15)) or "<tr><td colspan=2>（沒有失敗紀錄）</td></tr>"

    ok_rate = (1 - len(failures) / total) * 100 if total else 0
    return f"""<!doctype html><html lang="zh-Hant"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>playground 使用統計（最近 {days} 天）</title><style>
*{{box-sizing:border-box}}
body{{margin:0;background:#F5F6F8;color:#3D4151;
font:14px/1.7 -apple-system,"PingFang TC","Noto Sans TC",sans-serif}}
.wrap{{max-width:960px;margin:0 auto;padding:40px 24px 80px}}
h1{{font-size:24px;margin:0 0 4px;color:#1A1A2E}}
.sub{{color:#6B7280;font-size:13px;margin:0 0 28px}}
h2{{font-size:16px;margin:36px 0 10px;color:#1A1A2E}}
.kpi{{display:grid;grid-template-columns:repeat(auto-fit,minmax(140px,1fr));gap:12px;margin:20px 0}}
.k{{background:#fff;border:1px solid #E0E3E8;border-radius:10px;padding:14px 16px}}
.k b{{display:block;font-size:24px;color:#1A1A2E}}
.k span{{font-size:12px;color:#6B7280}}
table{{width:100%;border-collapse:collapse;background:#fff;border:1px solid #E0E3E8;border-radius:8px;overflow:hidden}}
th,td{{padding:8px 12px;text-align:left;border-bottom:1px solid #EEF0F3;font-size:13px}}
th{{background:#FAFBFC;color:#6B7280;font-weight:600}}
td:nth-child(2),td:nth-child(3){{font-variant-numeric:tabular-nums}}
code{{font:12px ui-monospace,Menlo,monospace;background:#F0F1F3;padding:1px 5px;border-radius:4px}}
.bar{{background:#EEF0F3;border-radius:3px;height:8px;width:100%;min-width:80px}}
.bar i{{display:block;height:100%;border-radius:3px}}
.note{{background:#FFF8EC;border:1px solid #F0DCB4;border-left:4px solid #D9A441;
border-radius:8px;padding:12px 16px;font-size:13px;margin:24px 0}}
footer{{margin-top:40px;color:#6B7280;font-size:12px}}
</style></head><body><div class="wrap">
<h1>playground 使用統計</h1>
<p class="sub">最近 {days} 天 · 產生於 {datetime.now():%Y-%m-%d %H:%M}</p>

<div class="kpi">
<div class="k"><b>{len(users)}</b><span>不同使用者</span></div>
<div class="k"><b>{total}</b><span>總呼叫次數</span></div>
<div class="k"><b>{ok_rate:.1f}%</b><span>成功率</span></div>
<div class="k"><b>{len(failures)}</b><span>失敗次數</span></div>
<div class="k"><b>{len(by_day)}</b><span>有活動的天數</span></div>
</div>

<div class="note"><b>uid 是不可反推的雜湊。</b>
它能回答「有幾個人在用、誰用得最多、什麼時候開始用的」，但<b>不能</b>回答「這個人是誰」——
識別碼是 <code>SHA256(api_key + STATS_SALT)</code> 的前 16 碼，明文金鑰從未落地。
要對應到真實客戶，需要網關端提供 key→user 的對照。</div>

<h2>使用者（依呼叫次數）</h2>
<table><thead><tr><th>uid</th><th>呼叫</th><th>失敗</th><th>首次</th><th>最後</th><th></th></tr></thead>
<tbody>{user_rows or '<tr><td colspan=6>（沒有資料）</td></tr>'}</tbody></table>

<h2>端點使用分佈</h2>
<table><thead><tr><th>端點</th><th>次數</th><th></th></tr></thead>
<tbody>{ep_rows or '<tr><td colspan=3>（沒有資料）</td></tr>'}</tbody></table>

<h2>每日活動</h2>
<table><thead><tr><th>日期</th><th>呼叫次數</th><th>不同使用者</th></tr></thead>
<tbody>{day_rows or '<tr><td colspan=3>（沒有資料）</td></tr>'}</tbody></table>

<h2>失敗分佈</h2>
<table><thead><tr><th>狀態碼 + 端點</th><th>次數</th></tr></thead>
<tbody>{fail_rows}</tbody></table>

<footer>資料來源：雲端物件儲存 <code>{PREFIX}/</code> 與本機 <code>outputs/stats/</code>，
兩邊合併。統計不含 prompt、生成結果與 IP。</footer>
</div></body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="產生 playground 使用統計的本機 HTML 報表")
    ap.add_argument("--days", type=int, default=7, help="統計最近幾天（預設 7）")
    ap.add_argument("--source", choices=["both", "local", "cloud"], default="both")
    ap.add_argument("--no-open", action="store_true", help="只產生檔案，不開瀏覽器")
    args = ap.parse_args()

    since = datetime.now() - timedelta(days=args.days)
    rows: list[dict] = []
    if args.source in ("both", "local"):
        rows += _load_local(since)
    if args.source in ("both", "cloud"):
        rows += _load_gcs(since)

    # 同一筆可能同時存在雲端與本機（極少見，但 flush 失敗重試就會發生）
    seen: set[tuple] = set()
    uniq: list[dict] = []
    for r in rows:
        k = (r.get("ts"), r.get("uid"), r.get("endpoint"), r.get("ms"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(uniq, args.days), encoding="utf-8")
    print(f"讀到 {len(uniq)} 筆紀錄（去重前 {len(rows)}）→ {OUT_HTML}")
    if not args.no_open:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
