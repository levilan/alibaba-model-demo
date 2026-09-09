"""管理後台（admin.py）的守門邏輯。每一條都對應一個「做錯會出事」的情境：
沒設環境變數卻露出後台、網域外的人登得進來、名單外的人登得進來、cookie 被竄改仍有效。
"""
import base64
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import admin  # noqa: E402
import app  # noqa: E402

_OAUTH_ENV = {"ADMIN_EMAILS": "levi@highercloud.com.tw, Ops@HigherCloud.com.tw",
              "GOOGLE_OAUTH_CLIENT_ID": "cid.apps.googleusercontent.com", "GOOGLE_OAUTH_CLIENT_SECRET": "sec",
              "ADMIN_SESSION_SECRET": "unit-test-secret"}
_ALL_KEYS = ["ADMIN_EMAILS", "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET", "ADMIN_SESSION_SECRET",
             "ADMIN_USER", "ADMIN_PASS", "OAUTH_EMAIL_DOMAINS", "OAUTH_REDIRECT_URL"]


@pytest.fixture
def env(monkeypatch):
    for k in _ALL_KEYS:
        monkeypatch.delenv(k, raising=False)
    return monkeypatch


def test_mode_is_fail_closed(env):
    assert admin.admin_mode() == "off"
    env.setenv("ADMIN_EMAILS", "a@highercloud.com.tw")          # 只有名單、沒有 OAuth client → 仍是 off
    assert admin.admin_mode() == "off"
    env.setenv("GOOGLE_OAUTH_CLIENT_ID", "x"); env.setenv("GOOGLE_OAUTH_CLIENT_SECRET", "y")
    assert admin.admin_mode() == "oauth"
    for k in ("ADMIN_EMAILS", "GOOGLE_OAUTH_CLIENT_ID", "GOOGLE_OAUTH_CLIENT_SECRET"):
        env.delenv(k)
    env.setenv("ADMIN_USER", "u"); env.setenv("ADMIN_PASS", "p")
    assert admin.admin_mode() == "basic"
    env.delenv("ADMIN_PASS")                                      # 帳密缺一個就整個關
    assert admin.admin_mode() == "off"


def test_email_allowed_requires_domain_and_list(env):
    for k, v in _OAUTH_ENV.items():
        env.setenv(k, v)
    assert admin.email_allowed("levi@highercloud.com.tw", "highercloud.com.tw")
    assert admin.email_allowed("OPS@highercloud.com.tw")            # 大小寫不分
    assert not admin.email_allowed("someone@highercloud.com.tw")     # 網域內但不在名單
    assert not admin.email_allowed("levi@gmail.com")                 # 名單格式對但網域不對
    assert not admin.email_allowed("levi@highercloud.com.tw", "gmail.com")   # hd 與網域不符
    assert not admin.email_allowed("")
    env.setenv("OAUTH_EMAIL_DOMAINS", "example.org")
    assert not admin.email_allowed("levi@highercloud.com.tw")


def test_session_roundtrip_and_tamper(env):
    env.setenv("ADMIN_SESSION_SECRET", "s1")
    tok = admin.sign_session("levi@highercloud.com.tw")
    assert admin.read_session(tok) == "levi@highercloud.com.tw"
    assert admin.read_session(tok[:-3] + "abc") is None
    assert admin.read_session(None) is None
    env.setenv("ADMIN_SESSION_SECRET", "s2")                        # 換密鑰，舊 cookie 失效
    assert admin.read_session(tok) is None


def test_routes_404_when_off(env):
    c = TestClient(app.app)
    for path in ("/admin", "/admin/", "/admin/login", "/admin/api/rows", "/admin/api/stats", "/admin/callback?code=x&state=y"):
        assert c.get(path, follow_redirects=False).status_code == 404, path


def test_basic_mode_challenges_then_serves(env):
    env.setenv("ADMIN_USER", "ops"); env.setenv("ADMIN_PASS", "pw")
    c = TestClient(app.app)
    r = c.get("/admin", follow_redirects=False)
    assert r.status_code == 401 and r.headers.get("www-authenticate", "").startswith("Basic")
    h = {"Authorization": "Basic " + base64.b64encode(b"ops:pw").decode()}
    r = c.get("/admin", headers=h)
    assert r.status_code == 200 and "使用紀錄" in r.text and "ops" in r.text
    r = c.get("/admin/api/stats?days=1", headers=h)
    assert r.status_code == 200 and set(r.json()) >= {"total", "ok", "per_day", "models", "users", "ip_matched"}
    for item in r.json()["models"] + r.json()["users"]:
        # 版面用 calls/ok 畫雙色細條、statuses 只在有失敗時展開，缺一個就畫不出來
        assert {"id", "name", "calls", "ok", "statuses"} <= set(item), item
    bad = {"Authorization": "Basic " + base64.b64encode(b"ops:wrong").decode()}
    assert c.get("/admin", headers=bad, follow_redirects=False).status_code == 401
    # 舊版報表已移除（Levi 2026-09-09：不需要分新舊表）——登入了也不該還在
    assert c.get("/admin/report", headers=h, follow_redirects=False).status_code == 404


def test_rows_filters(env):
    """呼叫紀錄的時間與條件查詢：start/end 是台北時間（要換回 UTC 才對得上統計的 ts）、
    uid/model 精確、ok 是 1/0、q 對端點與 IP 做子字串比對，並回 total 供分頁。"""
    env.setenv("ADMIN_USER", "ops"); env.setenv("ADMIN_PASS", "pw")
    from datetime import datetime, timedelta
    assert admin._parse_tpe("2026-09-09T10:30") == datetime(2026, 9, 9, 2, 30)
    assert admin._parse_tpe("2026-09-09") == datetime(2026, 9, 8, 16, 0)
    assert admin._parse_tpe("") is None and admin._parse_tpe("亂寫") is None

    # 天數上限 30（Levi 2026-09-09：「不需要 90 天 最多 30 天」）
    assert admin._clamp_days(90) == 30 and admin._clamp_days(30) == 30
    assert admin._clamp_days(0) == 7 and admin._clamp_days(-5) == 1

    c = TestClient(app.app)
    h = {"Authorization": "Basic " + base64.b64encode(b"ops:pw").decode()}
    base = c.get("/admin/api/rows?days=90&limit=1000", headers=h).json()
    assert {"rows", "total", "offset", "model_names"} <= set(base)
    assert len(base["rows"]) <= 1000 and base["total"] >= len(base["rows"])
    if base["total"]:
        first = base["rows"][0]
        # 只看失敗：回來的每一筆 ok 都必須是 false
        err = c.get("/admin/api/rows?days=90&limit=100&ok=0", headers=h).json()
        assert all(not r["ok"] for r in err["rows"])
        # 端點關鍵字
        q = c.get(f"/admin/api/rows?days=90&limit=100&q={first['endpoint']}", headers=h).json()
        assert q["total"] >= 1 and all(first["endpoint"] in r["endpoint"] for r in q["rows"])
        # 時間往未來設，結果必為空（證明 start 真的有生效，不是被忽略）
        future = c.get("/admin/api/rows?days=90&start=2099-01-01T00:00", headers=h).json()
        assert future["total"] == 0
        # 分頁：offset 前進後不會重複拿到第一筆
        if base["total"] > 1:
            p2 = c.get("/admin/api/rows?days=90&limit=1&offset=1", headers=h).json()
            assert p2["rows"][0]["ts"] <= first["ts"] and p2["offset"] == 1
    # 來源 IP 預設不查（那段要打平台日誌，是最慢的一環）；ip=1 才補。
    # 欄位一定在，值可能是空字串——前端靠欄位存在與否決定要不要畫那一欄。
    assert all(r["ip"] == "" for r in base["rows"]), "預設就不該去查 IP"
    called = []
    orig = admin.attach_ips

    async def spy(rows, days):
        called.append(days)
        return await orig(rows, days)

    monkeypatch_target = admin
    monkeypatch_target.attach_ips = spy
    try:
        c.get("/admin/api/rows?days=7&limit=1", headers=h)
        assert called == [], "沒帶 ip=1 就不該呼叫 attach_ips"
        c.get("/admin/api/stats?days=7", headers=h)
        assert called == [], "統計預設也不查 IP"
        c.get("/admin/api/rows?days=7&limit=1&ip=1", headers=h)
        assert called == [7], "帶了 ip=1 就要查"
    finally:
        monkeypatch_target.attach_ips = orig


def test_row_kind_excludes_polls_and_page_loads(env):
    """統計檔對每個 HTTP 請求都記一筆。一支非同步影片會產生數十筆 /status/ 輪詢、
    開一次頁會打 /api/models 與 /api/pricing——全算進去的話「使用者佔比」會嚴重失真
    （2026-09-09 實測：30 天 338 筆裡只有 82 筆是真的呼叫模型）。"""
    env.setenv("ADMIN_USER", "ops"); env.setenv("ADMIN_PASS", "pw")
    k = admin.row_kind
    assert k({"endpoint": "/api/muleai/status/w3.0-video/task_abc"}) == "poll"
    assert k({"endpoint": "/api/video/status/omni_123"}) == "poll"
    assert k({"endpoint": "/api/models"}) == "meta"
    assert k({"endpoint": "/api/pricing"}) == "meta"
    assert k({"endpoint": "/login"}) == "meta"
    # 未知端點一律當成真的呼叫——新增生成端點時才不會被靜默漏掉
    assert k({"endpoint": "/api/video/t2v"}) == "call"
    assert k({"endpoint": "/api/image/edit"}) == "call"
    assert k({"endpoint": "/api/some/brand/new/generate"}) == "call"

    c = TestClient(app.app)
    h = {"Authorization": "Basic " + base64.b64encode(b"ops:pw").decode()}
    st = c.get("/admin/api/stats?days=30", headers=h).json()
    assert {"polls", "meta"} <= set(st), "輪詢與頁面載入的筆數要照樣回報，不能只是消失"
    rows_call = c.get("/admin/api/rows?days=30&limit=1000", headers=h).json()
    assert all(r["kind"] == "call" for r in rows_call["rows"]), "呼叫紀錄預設只列真的呼叫"
    assert rows_call["total"] == st["total"], "摘要的總數要等於預設清單的總數"
    rows_all = c.get("/admin/api/rows?days=30&limit=1&kind=all", headers=h).json()
    assert rows_all["total"] == st["total"] + st["polls"] + st["meta"]
    if st["polls"]:
        only = c.get("/admin/api/rows?days=30&limit=50&kind=poll", headers=h).json()
        assert only["total"] == st["polls"] and all(r["kind"] == "poll" for r in only["rows"])


def test_oauth_mode_redirects_and_builds_google_url(env):
    for k, v in _OAUTH_ENV.items():
        env.setenv(k, v)
    c = TestClient(app.app)
    r = c.get("/admin", follow_redirects=False)
    assert r.status_code == 302 and r.headers["location"] == "/admin/login"
    r = c.get("/admin/login", follow_redirects=False, headers={"host": "play.example.com", "x-forwarded-proto": "https"})
    assert r.status_code == 302
    loc = r.headers["location"]
    assert loc.startswith("https://accounts.google.com/o/oauth2/v2/auth?")
    assert "redirect_uri=https%3A%2F%2Fplay.example.com%2Fadmin%2Fcallback" in loc
    assert "client_id=cid.apps.googleusercontent.com" in loc and "hd=highercloud.com.tw" in loc
    assert "nenai_admin_oauth" in r.headers.get("set-cookie", "")
    # state 對不上（沒有 cookie）→ 不會去換 token，直接拒
    r = c.get("/admin/callback?code=abc&state=zzz", follow_redirects=False)
    assert r.status_code == 400
    # 已登入的 cookie 直接放行
    c.cookies.set("nenai_admin", admin.sign_session("levi@highercloud.com.tw"))
    r = c.get("/admin", follow_redirects=False)
    assert r.status_code == 200 and "levi@highercloud.com.tw" in r.text
    env.setenv("OAUTH_REDIRECT_URL", "https://fixed.example/admin/callback")
    r = c.get("/admin/login", follow_redirects=False)
    assert "redirect_uri=https%3A%2F%2Ffixed.example%2Fadmin%2Fcallback" in r.headers["location"]
