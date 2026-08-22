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

來源 IP（--no-ip 可關）：
    統計檔**刻意不存 IP**（外流也無害的設計前提），但 Cloud Run 平台層的請求日誌
    本來就記錄每個請求的來源 IP（Cloud Logging，預設保留 30 天）。這支腳本在產生
    報表的當下用本機的 gcloud 憑證即時查詢、以「端點＋狀態碼＋最接近的時間」對回
    每筆統計紀錄——IP 只出現在你本機產生的這份 HTML 裡，從頭到尾不落入統計檔。
    需要本機 `gcloud` 已登入且能讀該專案的 Logging；查不到就自動略過，報表照出。

時區：統計紀錄與 Cloud Logging 都是 UTC（Cloud Run 容器沒設 TZ），報表顯示一律
    轉成台北時間（UTC+8）。若混入本機開發時寫的紀錄（本機時間戳），該幾筆會偏移。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import webbrowser
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parent.parent
LOCAL_DIR = ROOT / "outputs" / "stats"
OUT_HTML = ROOT / "outputs" / "usage-report.html"
PREFIX = os.environ.get("STATS_PREFIX", "stats")

# Cloud Logging 查詢用；跟部署（.github/workflows/deploy-cloud-run.yml）一致
GCLOUD_PROJECT = os.environ.get("GCLOUD_PROJECT", "ai-model-hub-newapi")
RUN_SERVICE = os.environ.get("CLOUD_RUN_SERVICE", "nenai-testing-platform")
TPE = timedelta(hours=8)   # 台北時間顯示位移（台灣無夏令時間，固定 +8 即可）


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


def _disp(t: datetime) -> str:
    """UTC 時間戳轉台北時間顯示。"""
    return f"{t + TPE:%Y-%m-%d %H:%M}"


def _load_uid_names() -> dict[str, str]:
    """uid → 使用者名稱。對照檔由 scripts/build_uid_map.py 產生（唯讀查網關 DB，
    key 不落地）；檔案不存在就退回只顯示 uid。⚠️ 對照檔是去匿名化資料，同報表級保管。"""
    fp = ROOT / "outputs" / "uid-map.json"
    try:
        data = json.loads(fp.read_text(encoding="utf-8"))
        return {uid: info.get("user", "") for uid, info in data.items() if info.get("user")}
    except FileNotFoundError:
        print("[hint] 沒有 uid 對照檔，報表只顯示 uid；要顯示使用者名稱先跑 "
              "scripts/build_uid_map.py", file=sys.stderr)
        return {}
    except Exception as e:
        print(f"[warn] uid 對照檔讀取失敗（{type(e).__name__}）", file=sys.stderr)
        return {}


def _load_model_names() -> dict[str, str]:
    """model id → 顯示名稱。直接 import app 拿 MODELS（單一真實來源，不另抄一份）。"""
    try:
        sys.path.insert(0, str(ROOT))
        import app as _app
        names: dict[str, str] = {}
        for v in _app.MODELS.values():
            for lst in (v.values() if isinstance(v, dict) else [v]):
                for m in lst:
                    names.setdefault(m["id"], m.get("name") or m["id"])
        return names
    except Exception as e:
        print(f"[warn] 讀不到 app.MODELS（{type(e).__name__}），模型將顯示原始 id", file=sys.stderr)
        return {}


def _load_cloud_logging(since_utc: datetime) -> list[dict]:
    """從 Cloud Logging 撈這段期間所有 /api/* 與 /login 請求的來源 IP。

    只在報表產生的當下查詢，結果不落地——這是「統計不存 IP、要看時查平台日誌」
    這個設計的配套。gcloud 不存在或無權限就回空清單，報表退化成沒有 IP 的版本。
    """
    fil = (
        'resource.type="cloud_run_revision" '
        f'AND resource.labels.service_name="{RUN_SERVICE}" '
        'AND (httpRequest.requestUrl:"/api/" OR httpRequest.requestUrl:"/login") '
        f'AND timestamp>="{since_utc:%Y-%m-%dT%H:%M:%S}Z"'
    )
    try:
        out = subprocess.run(
            ["gcloud", "logging", "read", fil, f"--project={GCLOUD_PROJECT}",
             "--format=json", "--limit=8000"],
            capture_output=True, text=True, timeout=120)
        if out.returncode != 0:
            print(f"[warn] Cloud Logging 查詢失敗：{out.stderr.strip()[:200]}", file=sys.stderr)
            return []
        entries = json.loads(out.stdout or "[]")
    except FileNotFoundError:
        print("[warn] 找不到 gcloud，報表將不含 IP（--no-ip 可關掉這個警告）", file=sys.stderr)
        return []
    except Exception as e:
        print(f"[warn] Cloud Logging 查詢失敗（{type(e).__name__}: {e}）", file=sys.stderr)
        return []
    logs: list[dict] = []
    for e in entries:
        hr = e.get("httpRequest") or {}
        url = hr.get("requestUrl", "")
        if not url:
            continue
        try:
            t = datetime.fromisoformat(e.get("timestamp", "").replace("Z", "+00:00"))
            t = t.replace(tzinfo=None)   # 轉成 naive UTC，跟統計紀錄同一基準
        except ValueError:
            continue
        logs.append({
            "t": t,
            "path": urlparse(url).path,
            "status": int(hr.get("status", 0) or 0),
            "ip": hr.get("remoteIp", ""),
            "ua": hr.get("userAgent", ""),
        })
    return logs


def _attach_ips(rows: list[dict], logs: list[dict]) -> int:
    """把 Cloud Logging 的 IP 對回統計紀錄（就地寫進 r['_ip']/r['_ua']）。

    對應鍵是「路徑＋狀態碼＋最接近的時間」：統計的 ts 記在請求結束、日誌的
    timestamp 記在請求開始，長任務（影片生成可達數分鐘）兩者會差一段，容忍
    窗開到 10 分鐘、取最接近且未被用掉的一筆。同秒同端點的並發請求可能互相
    張冠李戴，但來源分析要的是「這個人從哪些 IP 來」，個別筆序錯置無妨。
    """
    buckets: dict[tuple, list[dict]] = defaultdict(list)
    for lg in logs:
        buckets[(lg["path"], lg["status"])].append(lg)
    for b in buckets.values():
        b.sort(key=lambda x: x["t"])
    matched = 0
    for r in sorted(rows, key=_ts):
        cands = buckets.get((r.get("endpoint"), r.get("status")), [])
        t = _ts(r)
        best, best_dt = None, timedelta(minutes=10)
        for lg in cands:
            if lg.get("_used"):
                continue
            dt = abs(lg["t"] - t)
            if dt < best_dt:
                best, best_dt = lg, dt
        if best is not None:
            best["_used"] = True
            r["_ip"], r["_ua"] = best["ip"], best["ua"]
            matched += 1
    return matched


def _esc(s: str) -> str:
    return (str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))


def _bar(pct: float, color: str = "#01A0C7") -> str:
    return (f'<div class="bar"><i style="width:{pct:.1f}%;background:{color}"></i></div>')


def build_html(rows: list[dict], days: int,
               uid_names: dict[str, str] | None = None,
               model_names: dict[str, str] | None = None) -> str:
    uid_names = uid_names or {}
    model_names = model_names or {}

    def _who(uid: str, short: bool = False) -> str:
        """uid 的顯示形：查得到名稱就「名稱＋小字 uid」，查不到就原樣。"""
        name = uid_names.get(uid)
        if name:
            return (f"<b>{_esc(name)}</b>" if short
                    else f"<b>{_esc(name)}</b><br><code>{_esc(uid)}</code>")
        return f"<code>{_esc(uid if not short else uid[:9])}</code>"

    def _mname(mid: str) -> str:
        return model_names.get(mid, mid)

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

    models_by_uid: dict[str, Counter] = defaultdict(Counter)
    ips_by_uid: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        uid = r.get("uid", "?")
        if r.get("model"):
            models_by_uid[uid][_mname(r["model"])] += 1
        if r.get("_ip"):
            ips_by_uid[uid][r["_ip"]] += 1

    def _top_list(c: Counter, cap: int = 3, code: bool = True) -> str:
        wrap = (lambda s: f"<code>{_esc(s)}</code>") if code else _esc
        items = [wrap(k) for k, _ in c.most_common(cap)]
        extra = len(c) - cap
        return ("、".join(items) + (f" +{extra}" if extra > 0 else "")) if items else "—"

    user_rows = "".join(
        f"<tr><td>{_who(uid)}</td><td>{n}</td>"
        f"<td>{fail_by_uid.get(uid, 0)}</td>"
        f"<td>{_top_list(models_by_uid.get(uid, Counter()), code=False)}</td>"
        f"<td>{_top_list(ips_by_uid.get(uid, Counter()))}</td>"
        f"<td>{_disp(first_seen[uid])}</td>"
        f"<td>{_disp(last_seen[uid])}</td>"
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

    # ── 來源 IP 彙總（只彙整有對到 IP 的紀錄）─────────────────────────
    by_ip: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("_ip"):
            by_ip[r["_ip"]].append(r)
    ip_rows = "".join(
        f"<tr><td><code>{_esc(ip)}</code></td><td>{len(rs)}</td>"
        f"<td>{_top_list(Counter(uid_names.get(x.get('uid', '?'), x.get('uid', '?')) for x in rs), code=False)}</td>"
        f"<td>{_top_list(Counter(x.get('endpoint', '?') for x in rs))}</td>"
        f"<td class='ua'>{_esc(next((x.get('_ua', '') for x in rs if x.get('_ua')), '')[:80])}</td></tr>"
        for ip, rs in sorted(by_ip.items(), key=lambda kv: -len(kv[1])))

    # ── 請求明細（最近 1000 筆，前端分頁）────────────────────────────
    def _detail_row(r: dict) -> str:
        model = _esc(_mname(r["model"])) if r.get("model") else "—"
        ip = f"<code>{_esc(r['_ip'])}</code>" if r.get("_ip") else "—"
        return (f"<tr><td>{_disp(_ts(r))}</td>"
                f"<td>{_who(str(r.get('uid', '?')), short=True)}</td>"
                f"<td>{ip}</td><td>{model}</td>"
                f"<td><code>{_esc(r.get('endpoint', '?'))}</code></td>"
                f"<td>{r.get('status', '—')}</td><td>{r.get('ms', '—')}</td></tr>")

    detail_rows = "".join(_detail_row(r) for r in sorted(rows, key=_ts, reverse=True)[:1000])

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
.ua{{font-size:11px;color:#9CA3AF;max-width:260px;overflow:hidden;text-overflow:ellipsis;white-space:nowrap}}
.pager{{display:flex;align-items:center;gap:12px;justify-content:flex-end;margin:8px 0 0;font-size:12px;color:#6B7280}}
.pager button{{border:1px solid #E0E3E8;background:#fff;border-radius:6px;padding:4px 10px;cursor:pointer;font-size:12px}}
.pager button:disabled{{opacity:.4;cursor:default}}
td b{{font-weight:600;color:#1A1A2E}}
footer{{margin-top:40px;color:#6B7280;font-size:12px}}
</style></head><body><div class="wrap">
<h1>playground 使用統計</h1>
<p class="sub">最近 {days} 天 · 產生於 {datetime.now():%Y-%m-%d %H:%M} · 表內時間均為台北時間（UTC+8）</p>

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
要對應到真實客戶，需要網關端提供 key→user 的對照。<br>
<b>IP 是產生報表當下即時查自 Cloud Logging 的</b>（平台請求日誌，保留 30 天），
統計檔本身不存 IP；<code>anonymous</code> ＝ 沒帶 API key 的請求（多為外部掃描器）。</div>

<h2>使用者（依呼叫次數）</h2>
<table class="paged" data-pp="15"><thead><tr><th>使用者</th><th>呼叫</th><th>失敗</th><th>模型</th><th>來源 IP</th><th>首次</th><th>最後</th><th></th></tr></thead>
<tbody>{user_rows or '<tr><td colspan=8>（沒有資料）</td></tr>'}</tbody></table>

<h2>來源 IP（來自 Cloud Logging）</h2>
<table class="paged" data-pp="15"><thead><tr><th>IP</th><th>次數</th><th>使用者</th><th>主要端點</th><th>User-Agent</th></tr></thead>
<tbody>{ip_rows or '<tr><td colspan=5>（無 IP 資料——gcloud 未設定、超出 30 天保留期，或加了 --no-ip）</td></tr>'}</tbody></table>

<h2>請求明細（最近 1000 筆）</h2>
<table class="paged" data-pp="25"><thead><tr><th>時間</th><th>使用者</th><th>IP</th><th>模型</th><th>端點</th><th>狀態</th><th>ms</th></tr></thead>
<tbody>{detail_rows or '<tr><td colspan=7>（沒有資料）</td></tr>'}</tbody></table>

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
兩邊合併；IP 另由 Cloud Logging 即時查詢對照，不落地。統計檔不含 prompt、生成結果與 IP。
模型欄位自 2026-08-23 起記錄，之前的紀錄沒有這個維度。
使用者名稱來自本機 uid 對照檔（scripts/build_uid_map.py 產生），沒有對照檔就顯示 uid。</footer>
</div>
<script>
// 前端分頁：table.paged 的 tbody 依 data-pp 筆數分頁，資料仍全在檔案裡（搜尋可用 Ctrl+F 前先切到該頁）
document.querySelectorAll("table.paged").forEach(function (t) {{
  var pp = parseInt(t.dataset.pp || "25", 10);
  var rows = Array.prototype.slice.call(t.tBodies[0].rows);
  if (rows.length <= pp) return;
  var page = 0, pages = Math.ceil(rows.length / pp);
  var nav = document.createElement("div");
  nav.className = "pager";
  t.insertAdjacentElement("afterend", nav);
  function render() {{
    rows.forEach(function (r, i) {{
      r.style.display = (i >= page * pp && i < (page + 1) * pp) ? "" : "none";
    }});
    nav.innerHTML = "";
    var prev = document.createElement("button");
    prev.textContent = "‹ 上一頁"; prev.disabled = page === 0;
    prev.onclick = function () {{ page--; render(); }};
    var info = document.createElement("span");
    info.textContent = (page + 1) + " / " + pages + " 頁（共 " + rows.length + " 筆）";
    var next = document.createElement("button");
    next.textContent = "下一頁 ›"; next.disabled = page === pages - 1;
    next.onclick = function () {{ page++; render(); }};
    nav.append(prev, info, next);
  }}
  render();
}});
</script>
</body></html>"""


def main() -> None:
    ap = argparse.ArgumentParser(description="產生 playground 使用統計的本機 HTML 報表")
    ap.add_argument("--days", type=int, default=7, help="統計最近幾天（預設 7）")
    ap.add_argument("--source", choices=["both", "local", "cloud"], default="both")
    ap.add_argument("--no-open", action="store_true", help="只產生檔案，不開瀏覽器")
    ap.add_argument("--no-ip", action="store_true", help="不查 Cloud Logging，報表不含來源 IP")
    args = ap.parse_args()

    # 統計紀錄的 ts 是容器裡的 datetime.now()＝UTC（Cloud Run 沒設 TZ），
    # 篩選基準也要用 UTC，不然在 UTC+8 的機器上會少算 8 小時
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=args.days)
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

    if not args.no_ip:
        logs = _load_cloud_logging(since)
        if logs:
            n = _attach_ips(uniq, logs)
            print(f"Cloud Logging 撈到 {len(logs)} 筆請求日誌，對上 {n}/{len(uniq)} 筆統計紀錄")

    OUT_HTML.parent.mkdir(parents=True, exist_ok=True)
    OUT_HTML.write_text(build_html(uniq, args.days,
                                   uid_names=_load_uid_names(),
                                   model_names=_load_model_names()),
                        encoding="utf-8")
    print(f"讀到 {len(uniq)} 筆紀錄（去重前 {len(rows)}）→ {OUT_HTML}")
    if not args.no_open:
        webbrowser.open(OUT_HTML.as_uri())


if __name__ == "__main__":
    main()
