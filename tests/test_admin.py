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
    for path in ("/admin", "/admin/", "/admin/login", "/admin/report", "/admin/api/stats", "/admin/callback?code=x&state=y"):
        assert c.get(path, follow_redirects=False).status_code == 404, path


def test_basic_mode_challenges_then_serves(env):
    env.setenv("ADMIN_USER", "ops"); env.setenv("ADMIN_PASS", "pw")
    c = TestClient(app.app)
    r = c.get("/admin", follow_redirects=False)
    assert r.status_code == 401 and r.headers.get("www-authenticate", "").startswith("Basic")
    h = {"Authorization": "Basic " + base64.b64encode(b"ops:pw").decode()}
    r = c.get("/admin", headers=h)
    assert r.status_code == 200 and "Playground 使用紀錄" in r.text and "ops" in r.text
    r = c.get("/admin/api/stats?days=1", headers=h)
    assert r.status_code == 200 and set(r.json()) >= {"total", "per_day", "per_model", "per_uid"}
    bad = {"Authorization": "Basic " + base64.b64encode(b"ops:wrong").decode()}
    assert c.get("/admin", headers=bad, follow_redirects=False).status_code == 401


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
