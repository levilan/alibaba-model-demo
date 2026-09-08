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
outputs/stats/），所以後台看到的跟本機報表一模一樣；不查 Cloud Logging（那要 gcloud 憑證），
所以沒有來源 IP。uid 仍是 SHA256 後的識別碼，不可反推金鑰。

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
_ROWS_TTL = 60.0


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


@router.get("/report", response_class=HTMLResponse)
async def admin_report(request: Request, days: int = 7):
    _require(request)
    days = _clamp_days(days)
    rows = await load_rows(days)
    us = _usage_stats()
    html = us.build_html(rows, days, uid_names=us._load_uid_names(), model_names=_model_names())
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@router.get("/api/stats")
async def admin_api_stats(request: Request, days: int = 7):
    """給程式用的彙總：總筆數、成功率、每日、每模型、每使用者（uid）。"""
    _require(request)
    days = _clamp_days(days)
    rows = await load_rows(days)
    us = _usage_stats()
    per_day: dict[str, int] = {}
    per_model: dict[str, int] = {}
    per_uid: dict[str, int] = {}
    ok = 0
    for r in rows:
        t = us._ts(r)
        per_day[(t + timedelta(hours=8)).strftime("%Y-%m-%d")] = per_day.get((t + timedelta(hours=8)).strftime("%Y-%m-%d"), 0) + 1
        if r.get("model"):
            per_model[r["model"]] = per_model.get(r["model"], 0) + 1
        per_uid[r.get("uid", "?")] = per_uid.get(r.get("uid", "?"), 0) + 1
        ok += 1 if r.get("ok") else 0
    return JSONResponse({"days": days, "total": len(rows), "ok": ok, "per_day": dict(sorted(per_day.items())),
                         "per_model": dict(sorted(per_model.items(), key=lambda x: -x[1])),
                         "per_uid": dict(sorted(per_uid.items(), key=lambda x: -x[1])),
                         "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds")},
                        headers={"Cache-Control": "no-store"})


@router.get("/api/rows")
async def admin_api_rows(request: Request, days: int = 7, limit: int = 500):
    """原始紀錄（最新在前），給後台的「最近呼叫」表。"""
    _require(request)
    days = _clamp_days(days)
    rows = await load_rows(days)
    us = _usage_stats()
    rows = sorted(rows, key=us._ts, reverse=True)[: max(1, min(5000, limit))]
    names = _model_names()
    for r in rows:
        if r.get("model"):
            r = r  # 原地不改；名稱由前端查 /api/stats 的 model_names 也行，這裡直接附上
    return JSONResponse({"rows": rows, "model_names": names}, headers={"Cache-Control": "no-store"})
