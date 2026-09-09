"""管理後台（/admin）：Google 登入後查看 playground 的使用紀錄。

參考官網（website repo）的後台門設計，同樣三種模式、依環境變數自動選，fail closed：

  1. Google OAuth（正式環境）—— 設了 ADMIN_EMAILS 與 GOOGLE_OAUTH_CLIENT_ID/SECRET 就啟用。
     官網是 nginx + oauth2-proxy 容器；playground 跑在 Cloud Run 沒有 sidecar，所以這裡自己走
     OpenID Connect：/admin/login 轉去 Google → /admin/callback 換 token → 用 Google 的
     tokeninfo 端點驗 id_token（簽章與效期由 Google 驗，我們只看 aud／email_verified／網域）
     → email 在 ADMIN_EMAILS 名單才發 session cookie。網域內但不在名單的人：登得進 Google，
     回來被拒（403 頁面明講）。
       GOOGLE_OAUTH_CLIENT_ID / GOOGLE_OAUTH_CLIENT_SECRET   Google Cloud 的 OAuth Client（Web application）
       ADMIN_EMAILS            允許的 email，逗號分隔
       OAUTH_EMAIL_DOMAINS     允許的 Workspace 網域，預設 highercloud.com.tw
       OAUTH_REDIRECT_URL      選填；不設就用請求的 host 組 https://<host>/admin/callback
                               （Console 登記的 redirect URI 要一字不差）
       ADMIN_SESSION_SECRET    簽 session cookie 的密鑰；不設就每次啟動隨機產生，
                               多實例／重啟後 session 會失效（能用，但會常被登出）
  2. Basic Auth（本機／測試）—— 沒設 ADMIN_EMAILS 但設了 ADMIN_USER / ADMIN_PASS。
  3. fail closed —— 兩組都沒設，/admin 底下一律 404（不透露路徑存在）。

資料來源沿用 scripts/usage_stats.py 的讀取與報表函式（GCS 的 stats/*.jsonl，沒雲端就讀本機
outputs/stats/），所以後台看到的跟本機報表一模一樣。另外兩項是後台專屬（2026-09-09 Levi 要求
「知道是哪個 key 的 user 登入使用、來源 IP」）：

  · **uid → 使用者**：`scripts/build_uid_map.py --upload` 產生對照並上傳到同一個 bucket 的
    `stats-meta/uid-map.json`（明文金鑰全程只在本機記憶體，落地的只有 uid→名稱）。後台讀它來顯示
    「誰在用」。⚠️ 這是去匿名化資料，與統計同級保管；bucket 是私有的，只有服務帳戶讀得到。
  · **來源 IP**：統計檔仍然刻意不存 IP，改成查詢時即時問 Cloud Logging（Cloud Run 的請求日誌本來
    就記 IP，預設留 30 天）。本機腳本用 gcloud CLI，容器裡沒有 CLI，所以這裡用執行身分（Cloud Run
    的服務帳戶）直接打 Logging REST API。查不到就退化成沒有 IP 的版本，不影響其他欄位。

uid 仍是 SHA256 後的識別碼，不可反推金鑰。

⚠️ usage_stats.py 檔頭原本寫「刻意不做網頁後台，避免後台外洩」——2026-09-09 Levi 裁示改做
後台但用 Google SSO 守門，等於把「沒有入口」換成「入口只認 Workspace 帳號」。
"""
from __future__ import annotations

import asyncio
import base64
import hashlib
import importlib.util
import os
import secrets
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional
from urllib.parse import urlencode

import httpx
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse, Response
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

ROOT = Path(__file__).resolve().parent
router = APIRouter(prefix="/admin")

_SESSION_COOKIE = "nenai_admin"
_STATE_COOKIE = "nenai_admin_oauth"
_SESSION_MAX_AGE = 8 * 3600
_GOOGLE_AUTH_URL = "https://accounts.google.com/o/oauth2/v2/auth"
_GOOGLE_TOKEN_URL = "https://oauth2.googleapis.com/token"
_GOOGLE_TOKENINFO_URL = "https://oauth2.googleapis.com/tokeninfo"


# ── 設定（都從環境變數讀，讀的時候才讀，測試可以改 os.environ） ────────────────
def _env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def _split_list(s: str) -> list[str]:
    return [x.strip().lower() for x in s.split(",") if x.strip()]


def admin_mode() -> str:
    """'oauth' | 'basic' | 'off'。跟官網 admin-gate 同一套判斷順序。"""
    if _env("ADMIN_EMAILS") and _env("GOOGLE_OAUTH_CLIENT_ID") and _env("GOOGLE_OAUTH_CLIENT_SECRET"):
        return "oauth"
    if _env("ADMIN_USER") and _env("ADMIN_PASS"):
        return "basic"
    return "off"


def email_allowed(email: str, hd: Optional[str] = None) -> bool:
    """email 必須同時：在允許的網域內、且在 ADMIN_EMAILS 名單上。"""
    email = (email or "").strip().lower()
    if not email or "@" not in email:
        return False
    domains = _split_list(_env("OAUTH_EMAIL_DOMAINS", "highercloud.com.tw"))
    domain = email.rsplit("@", 1)[1]
    if domain not in domains:
        return False
    if hd and hd.lower() not in domains:
        return False
    return email in _split_list(_env("ADMIN_EMAILS"))


# ── session cookie ────────────────────────────────────────────────────────────
_secret_cache: Optional[str] = None


def _serializer() -> URLSafeTimedSerializer:
    global _secret_cache
    secret = _env("ADMIN_SESSION_SECRET")
    if not secret:
        if _secret_cache is None:
            _secret_cache = secrets.token_hex(32)
            print("[admin] ADMIN_SESSION_SECRET 未設，改用啟動時隨機密鑰：重啟或多實例時 session 會失效")
        secret = _secret_cache
    return URLSafeTimedSerializer(secret, salt="nenai-admin")


def sign_session(email: str) -> str:
    return _serializer().dumps({"email": email, "iat": int(time.time())})


def read_session(token: Optional[str]) -> Optional[str]:
    """回 email；簽章壞掉／過期回 None。"""
    if not token:
        return None
    try:
        data = _serializer().loads(token, max_age=_SESSION_MAX_AGE)
        return data.get("email") or None
    except (BadSignature, SignatureExpired, Exception):
        return None


def _redirect_url(request: Request) -> str:
    override = _env("OAUTH_REDIRECT_URL")
    if override:
        return override
    # Cloud Run 在代理後面：scheme 看 X-Forwarded-Proto，host 看 Host
    proto = request.headers.get("x-forwarded-proto") or request.url.scheme
    host = request.headers.get("host") or request.url.netloc
    return f"{proto}://{host}/admin/callback"


def _basic_ok(request: Request) -> bool:
    auth = request.headers.get("authorization", "")
    if not auth.lower().startswith("basic "):
        return False
    try:
        user, _, pw = base64.b64decode(auth[6:]).decode().partition(":")
    except Exception:
        return False
    return secrets.compare_digest(user, _env("ADMIN_USER")) and secrets.compare_digest(pw, _env("ADMIN_PASS"))


def current_user(request: Request) -> Optional[str]:
    """已登入的身分（email 或 basic 使用者名），沒登入回 None。模式 off 一律 None。"""
    mode = admin_mode()
    if mode == "oauth":
        return read_session(request.cookies.get(_SESSION_COOKIE))
    if mode == "basic":
        return _env("ADMIN_USER") if _basic_ok(request) else None
    return None


def _require(request: Request) -> str:
    """守門：off → 404；basic 未登入 → 401 挑戰；oauth 未登入 → 302 去登入。"""
    mode = admin_mode()
    if mode == "off":
        raise HTTPException(status_code=404)
    user = current_user(request)
    if user:
        return user
    if mode == "basic":
        raise HTTPException(status_code=401, headers={"WWW-Authenticate": 'Basic realm="nenai-admin"'})
    raise HTTPException(status_code=302, headers={"Location": "/admin/login"})


# ── OAuth 流程 ────────────────────────────────────────────────────────────────
@router.get("/login")
async def admin_login(request: Request):
    mode = admin_mode()
    if mode == "off":
        raise HTTPException(status_code=404)
    if mode == "basic":
        return RedirectResponse("/admin", status_code=302)
    state = secrets.token_urlsafe(24)
    params = {
        "client_id": _env("GOOGLE_OAUTH_CLIENT_ID"),
        "redirect_uri": _redirect_url(request),
        "response_type": "code",
        "scope": "openid email",
        "state": state,
        "prompt": "select_account",
    }
    domains = _split_list(_env("OAUTH_EMAIL_DOMAINS", "highercloud.com.tw"))
    if len(domains) == 1:
        params["hd"] = domains[0]   # 只是 UI 提示，真正的檢查在 callback
    resp = RedirectResponse(f"{_GOOGLE_AUTH_URL}?{urlencode(params)}", status_code=302)
    resp.set_cookie(_STATE_COOKIE, _serializer().dumps(state), max_age=600, httponly=True,
                    secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https", samesite="lax")
    return resp


async def _exchange_code(code: str, redirect_uri: str) -> dict:
    async with httpx.AsyncClient(timeout=20.0) as client:
        r = await client.post(_GOOGLE_TOKEN_URL, data={
            "code": code, "client_id": _env("GOOGLE_OAUTH_CLIENT_ID"),
            "client_secret": _env("GOOGLE_OAUTH_CLIENT_SECRET"),
            "redirect_uri": redirect_uri, "grant_type": "authorization_code"})
        if r.status_code != 200:
            raise HTTPException(status_code=502, detail=f"Google token 交換失敗：{r.text[:200]}")
        id_token = r.json().get("id_token")
        if not id_token:
            raise HTTPException(status_code=502, detail="Google 沒有回 id_token")
        # 交給 Google 驗簽章與效期（tokeninfo），我們只看內容
        v = await client.get(_GOOGLE_TOKENINFO_URL, params={"id_token": id_token})
        if v.status_code != 200:
            raise HTTPException(status_code=401, detail="id_token 驗證失敗")
        return v.json()


def _reject_page(title: str, body: str, status: int = 403) -> HTMLResponse:
    return HTMLResponse(f"""<!doctype html><meta charset="utf-8"><title>{title}</title>
<style>body{{font-family:system-ui,-apple-system,sans-serif;background:#FBFAF8;color:#2B2724;margin:0;padding:48px}}
.box{{max-width:520px;margin:0 auto;padding:28px;border:1px solid #E6E2DB;border-radius:10px;background:#F6F2E9}}
a{{color:#3F625F}}</style><div class="box"><h1 style="font-weight:400;font-size:20px;margin:0 0 12px">{title}</h1>
<p style="color:#5C564F;line-height:1.7">{body}</p><p><a href="/admin/login">重新登入</a></p></div>""", status_code=status)


@router.get("/callback")
async def admin_callback(request: Request, code: str = "", state: str = "", error: str = ""):
    if admin_mode() != "oauth":
        raise HTTPException(status_code=404)
    if error:
        return _reject_page("登入未完成", f"Google 回報：{error}", 400)
    try:
        expected = _serializer().loads(request.cookies.get(_STATE_COOKIE, ""), max_age=600)
    except Exception:
        expected = None
    if not state or state != expected:
        return _reject_page("登入未完成", "登入流程的狀態對不上（可能逾時或被重放），請重新登入。", 400)
    info = await _exchange_code(code, _redirect_url(request))
    if str(info.get("aud")) != _env("GOOGLE_OAUTH_CLIENT_ID"):
        return _reject_page("登入未完成", "token 不是發給這個後台的。", 401)
    if str(info.get("email_verified")).lower() != "true":
        return _reject_page("無法登入", "這個 Google 帳號的 email 未驗證。", 403)
    email = (info.get("email") or "").lower()
    if not email_allowed(email, info.get("hd")):
        return _reject_page("沒有權限", f"{email} 不在管理員名單上。要開權限請把這個 email 加進 ADMIN_EMAILS。", 403)
    resp = RedirectResponse("/admin", status_code=302)
    resp.set_cookie(_SESSION_COOKIE, sign_session(email), max_age=_SESSION_MAX_AGE, httponly=True,
                    secure=request.headers.get("x-forwarded-proto", request.url.scheme) == "https", samesite="lax")
    resp.delete_cookie(_STATE_COOKIE)
    return resp


@router.get("/logout")
async def admin_logout():
    resp = RedirectResponse("/admin/login", status_code=302)
    resp.delete_cookie(_SESSION_COOKIE)
    return resp


# ── 資料：沿用 scripts/usage_stats.py ──────────────────────────────────────────
_stats_mod = None


def _usage_stats():
    """把 scripts/usage_stats.py 當模組載入（scripts/ 不是 package）。"""
    global _stats_mod
    if _stats_mod is None:
        spec = importlib.util.spec_from_file_location("usage_stats", ROOT / "scripts" / "usage_stats.py")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _stats_mod = mod
    return _stats_mod


_rows_cache: dict = {}   # days -> (ts, rows)
_ROWS_TTL = 120.0
_uid_map_cache: tuple[float, dict] = (0.0, {})
_UID_MAP_TTL = 300.0
_UID_MAP_KEY = os.environ.get("UID_MAP_KEY", "stats-meta/uid-map.json")


def _load_uid_map_sync() -> dict[str, str]:
    """uid → 使用者顯示名。優先讀 GCS（部署環境），沒有就讀本機 outputs/uid-map.json（開發）。

    值的形狀沿用 build_uid_map.py：{uid: {"user":…, "user_id":…, "token_name":…}}，
    這裡壓成 uid → "名稱（token 名）"，讓報表與後台顯示得出「哪一把 key 的哪個人」。
    """
    raw: dict = {}
    bucket_name = os.environ.get("GCS_BUCKET_NAME", "")
    if bucket_name:
        try:
            from google.cloud import storage as gcs_storage
            creds_json = os.environ.get("GCS_CREDENTIALS_JSON", "")
            if creds_json:
                import json as _json
                from google.oauth2 import service_account
                info = _json.loads(creds_json)
                client = gcs_storage.Client(
                    credentials=service_account.Credentials.from_service_account_info(info),
                    project=info.get("project_id"))
            else:
                client = gcs_storage.Client()
            blob = client.bucket(bucket_name).blob(_UID_MAP_KEY)
            if blob.exists():
                import json as _json
                raw = _json.loads(blob.download_as_text())
        except Exception as e:
            print(f"[admin] 讀取 uid 對照失敗（{type(e).__name__}: {e}）")
    if not raw:
        fp = ROOT / "outputs" / "uid-map.json"
        if fp.exists():
            import json as _json
            try:
                raw = _json.loads(fp.read_text(encoding="utf-8"))
            except Exception:
                raw = {}
    out: dict[str, str] = {}
    for uid, info in raw.items():
        if not isinstance(info, dict):
            continue
        name = (info.get("user") or "").strip()
        if not name:
            continue
        token = (info.get("token_name") or "").strip()
        out[uid] = f"{name}（{token}）" if token else name
    return out


async def uid_names() -> dict[str, str]:
    global _uid_map_cache
    now = time.time()
    if _uid_map_cache[1] and now - _uid_map_cache[0] < _UID_MAP_TTL:
        return _uid_map_cache[1]
    data = await asyncio.to_thread(_load_uid_map_sync)
    _uid_map_cache = (now, data)
    return data


# ── 來源 IP：查 Cloud Logging（容器裡沒有 gcloud CLI，直接打 REST API）─────────
_LOG_API = "https://logging.googleapis.com/v2/entries:list"
_ip_cache: dict = {}   # days -> (ts, logs)
_IP_TTL = 120.0


def _log_token_and_project() -> tuple[Optional[str], str]:
    """用執行身分（Cloud Run 服務帳戶）取 access token。取不到就回 (None, '')。"""
    try:
        import google.auth
        from google.auth.transport.requests import Request as GARequest
        creds, project = google.auth.default(scopes=["https://www.googleapis.com/auth/logging.read"])
        creds.refresh(GARequest())
        return creds.token, (os.environ.get("GCLOUD_PROJECT") or project or "")
    except Exception as e:
        # 本機開發常常只有 gcloud CLI 登入、沒有 ADC——退回問 CLI 要 token。
        # 容器裡沒有 gcloud，這條會直接失敗，回到「沒有 IP」的降級版本。
        import subprocess
        try:
            out = subprocess.run(["gcloud", "auth", "print-access-token"],
                                 capture_output=True, text=True, timeout=20)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip(), os.environ.get("GCLOUD_PROJECT", "ai-model-hub-newapi")
        except Exception:
            pass
        print(f"[admin] 取不到 Logging 憑證（{type(e).__name__}）——報表將不含來源 IP")
        return None, ""


def _fetch_logs_sync(days: int) -> list[dict]:
    """撈這段期間 /api/* 與 /login 的請求日誌（時間、路徑、狀態、IP、UA）。"""
    token, project = _log_token_and_project()
    if not token or not project:
        return []
    service = os.environ.get("CLOUD_RUN_SERVICE", "nenai-testing-platform")
    since = datetime.now(timezone.utc) - timedelta(days=days)
    fil = ('resource.type="cloud_run_revision" '
           f'AND resource.labels.service_name="{service}" '
           'AND (httpRequest.requestUrl:"/api/" OR httpRequest.requestUrl:"/login") '
           f'AND timestamp>="{since:%Y-%m-%dT%H:%M:%S}Z"')
    from urllib.parse import urlparse
    logs: list[dict] = []
    page: Optional[str] = None
    with httpx.Client(timeout=60.0) as client:
        for _ in range(10):   # 最多 10 頁（10k 筆），夠用且不會拖垮頁面
            body = {"resourceNames": [f"projects/{project}"], "filter": fil,
                    "orderBy": "timestamp desc", "pageSize": 1000}
            if page:
                body["pageToken"] = page
            r = client.post(_LOG_API, headers={"Authorization": f"Bearer {token}"}, json=body)
            if r.status_code != 200:
                print(f"[admin] Logging 查詢失敗 {r.status_code}: {r.text[:200]}")
                break
            data = r.json()
            for e in data.get("entries", []):
                hr = e.get("httpRequest") or {}
                url = hr.get("requestUrl", "")
                if not url:
                    continue
                try:
                    t = datetime.fromisoformat(e.get("timestamp", "").replace("Z", "+00:00")).replace(tzinfo=None)
                except ValueError:
                    continue
                logs.append({"t": t, "path": urlparse(url).path,
                             "status": int(hr.get("status", 0) or 0),
                             "ip": hr.get("remoteIp", ""), "ua": hr.get("userAgent", "")})
            page = data.get("nextPageToken")
            if not page:
                break
    return logs


async def attach_ips(rows: list[dict], days: int) -> int:
    """就地把 _ip／_ua 補進 rows（沿用 usage_stats._attach_ips 的對應規則）。回對上的筆數。"""
    global _ip_cache
    now = time.time()
    hit = _ip_cache.get(days)
    if hit and now - hit[0] < _IP_TTL:
        logs = hit[1]
    else:
        logs = await asyncio.to_thread(_fetch_logs_sync, days)
        _ip_cache[days] = (now, logs)
    if not logs:
        return 0
    # _attach_ips 會在 logs 上留 _used 標記，快取的清單要複製一份再用
    return _usage_stats()._attach_ips(rows, [dict(x) for x in logs])


def _load_rows_sync(days: int) -> list[dict]:
    us = _usage_stats()
    since = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)
    rows = us._load_local(since) + us._load_gcs(since)
    seen: set = set()
    uniq: list[dict] = []
    for r in rows:
        k = (r.get("ts"), r.get("uid"), r.get("endpoint"), r.get("ms"))
        if k in seen:
            continue
        seen.add(k)
        uniq.append(r)
    return uniq


async def load_rows(days: int) -> list[dict]:
    """讀 stats（GCS 是逐檔下載，30 天可能上千個小檔，所以放 thread 並快取 60 秒）。"""
    now = time.time()
    hit = _rows_cache.get(days)
    if hit and now - hit[0] < _ROWS_TTL:
        return hit[1]
    rows = await asyncio.to_thread(_load_rows_sync, days)
    _rows_cache[days] = (now, rows)
    return rows


def _model_names() -> dict[str, str]:
    """model id → 顯示名稱，直接讀已載入的 app.MODELS（不重新 import app）。"""
    import sys
    app_mod = sys.modules.get("app") or sys.modules.get("__main__")
    models = getattr(app_mod, "MODELS", None) or {}
    names: dict[str, str] = {}
    for v in models.values():
        for lst in (v.values() if isinstance(v, dict) else [v]):
            for m in lst:
                names.setdefault(m["id"], m.get("name") or m["id"])
    return names


def _clamp_days(days: int) -> int:
    return max(1, min(90, int(days or 7)))


@router.get("", response_class=HTMLResponse)
@router.get("/", response_class=HTMLResponse)
async def admin_home(request: Request, days: int = 7):
    user = _require(request)
    days = _clamp_days(days)
    html = (ROOT / "templates" / "admin.html").read_text(encoding="utf-8")
    return HTMLResponse(html.replace("{{USER}}", user).replace("{{DAYS}}", str(days))
                        .replace("{{MODE}}", admin_mode()))


# 舊版報表（usage_stats.build_html 嵌 iframe）已於 2026-09-09 移除——Levi：「不需要分新舊表，
# 把舊的移除」。同一份資料現在只有一種呈現；本機腳本 scripts/usage_stats.py 仍可自己產 HTML。


@router.get("/api/stats")
async def admin_api_stats(request: Request, days: int = 7):
    """給程式用的彙總：總筆數、成功率、每日、每模型、每使用者（uid）。"""
    _require(request)
    days = _clamp_days(days)
    rows = await load_rows(days)
    us = _usage_stats()
    rows = [dict(r) for r in rows]
    matched = await attach_ips(rows, days)
    names = await uid_names()
    model_names = _model_names()
    per_day: dict[str, int] = {}
    # 每個模型／使用者各自累計「總數、成功數、失敗的狀態碼」——版面用雙色細條表示
    # 成功率、只有真的有失敗時才展開狀態碼明細（官網 session 2026-09-09 的設計）
    agg_model: dict[str, dict] = {}
    agg_uid: dict[str, dict] = {}
    ok = 0
    for r in rows:
        t = us._ts(r) + timedelta(hours=8)          # 顯示一律台北時間
        per_day[t.strftime("%Y-%m-%d")] = per_day.get(t.strftime("%Y-%m-%d"), 0) + 1
        good = bool(r.get("ok"))
        ok += 1 if good else 0
        code = str(r.get("status", "?"))
        for bucket, key in ((agg_model, r.get("model")), (agg_uid, r.get("uid", "?"))):
            if not key:
                continue
            e = bucket.setdefault(key, {"calls": 0, "ok": 0, "statuses": {}, "ips": set()})
            e["calls"] += 1
            e["ok"] += 1 if good else 0
            if not good:
                e["statuses"][code] = e["statuses"].get(code, 0) + 1
            if bucket is agg_uid and (r.get("_ip") or ""):
                e["ips"].add(r["_ip"])

    def _rows_of(bucket: dict, name_of) -> list:
        out = []
        for k, e in sorted(bucket.items(), key=lambda x: -x[1]["calls"]):
            item = {"id": k, "name": name_of(k), "calls": e["calls"], "ok": e["ok"],
                    "statuses": dict(sorted(e["statuses"].items()))}
            if e["ips"]:
                item["ips"] = sorted(e["ips"])
            out.append(item)
        return out

    # 沒有呼叫的日子也要有一根 0 的柱子，否則時間軸的間距是假的（柱子會擠在一起、
    # 看起來像每天都有量）。以台北時間的今天往回補滿整個區間。
    today = (datetime.now(timezone.utc) + timedelta(hours=8)).date()
    per_day_full = {}
    for i in range(days - 1, -1, -1):
        d = (today - timedelta(days=i)).strftime("%Y-%m-%d")
        per_day_full[d] = per_day.get(d, 0)
    for d, n in per_day.items():        # 落在區間外的（時區邊界）不要丟掉
        per_day_full.setdefault(d, n)

    return JSONResponse({"days": days, "total": len(rows), "ok": ok, "ip_matched": matched,
                         "per_day": dict(sorted(per_day_full.items())),
                         "models": _rows_of(agg_model, lambda k: model_names.get(k, "")),
                         "users": _rows_of(agg_uid, lambda k: names.get(k, "")),
                         "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                        headers={"Cache-Control": "no-store"})


def _parse_tpe(v: str) -> Optional[datetime]:
    """把畫面上的台北時間（datetime-local 的 'YYYY-MM-DDTHH:MM'）轉成統計用的 naive UTC。

    統計紀錄的 ts 是容器的 datetime.now()＝UTC（Cloud Run 沒設 TZ），畫面一律顯示 +8，
    所以查詢條件要往回減 8 小時才對得上。只給日期時視為當天 00:00。
    """
    v = (v or "").strip()
    if not v:
        return None
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%dT%H:%M", "%Y-%m-%d %H:%M", "%Y-%m-%d"):
        try:
            return datetime.strptime(v, fmt) - timedelta(hours=8)
        except ValueError:
            continue
    return None


@router.get("/api/rows")
async def admin_api_rows(request: Request, days: int = 7, limit: int = 50, offset: int = 0,
                         start: str = "", end: str = "", uid: str = "", model: str = "",
                         ok: str = "", q: str = ""):
    """近期呼叫。支援時間區間與條件查詢（Levi 2026-09-09：「近期呼叫可以跟隨時間查詢」）。

    start／end 是台北時間；uid／model 精確比對；ok 是 "1"／"0"；q 對端點、使用者名稱、IP
    做不分大小寫的子字串比對。回傳 total（符合條件的總數）讓畫面能顯示「N / M」與分頁。
    """
    _require(request)
    days = _clamp_days(days)
    rows = await load_rows(days)
    us = _usage_stats()
    rows = [dict(r) for r in rows]
    await attach_ips(rows, days)
    who = await uid_names()
    for r in rows:
        r["user"] = who.get(r.get("uid", ""), "")
        r["ip"] = r.pop("_ip", "") or ""
        r.pop("_ua", None)

    t0, t1 = _parse_tpe(start), _parse_tpe(end)
    ql = q.strip().lower()
    def _keep(r: dict) -> bool:
        t = us._ts(r)
        if t0 and t < t0:
            return False
        if t1 and t > t1:
            return False
        if uid and r.get("uid") != uid:
            return False
        if model and r.get("model") != model:
            return False
        if ok in ("0", "1") and bool(r.get("ok")) != (ok == "1"):
            return False
        if ql and ql not in " ".join(str(r.get(k) or "") for k in ("endpoint", "user", "ip", "uid", "model")).lower():
            return False
        return True

    hits = sorted([r for r in rows if _keep(r)], key=us._ts, reverse=True)
    off = max(0, offset)
    page = hits[off: off + max(1, min(1000, limit))]
    return JSONResponse({"rows": page, "total": len(hits), "offset": off,
                         "model_names": _model_names()},
                        headers={"Cache-Control": "no-store"})
