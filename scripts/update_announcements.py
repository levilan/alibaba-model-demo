#!/usr/bin/env python3
"""更新 nen.com.tw 的系統公告——新模型上架流程的最後一步。

用法：
    venv/bin/python scripts/update_announcements.py --show
    venv/bin/python scripts/update_announcements.py --add-models claude-opus-5,grok-4.3
    venv/bin/python scripts/update_announcements.py --add-models claude-opus-5 --confirm
    venv/bin/python scripts/update_announcements.py --to-english --confirm   # 一次性：舊公告轉英文

⚠️ 站上那個彈窗有**兩個頁籤**，對應兩個不同的設定項，很容易搞錯（我搞錯過一次）：

    「通知」    → option key `Notice`                      Markdown 字串，**維持中文**
    「系統公告」 → option key `console_setting.announcements` JSON 陣列，**這支腳本動的是這個**

🔒 這支腳本只寫 `console_setting.announcements` 一個設定項，不碰站台的任何其他 API。
   那把 `NEN_TOKEN` 是管理員層級的憑證，`GET /api/option/` 一次回 200+ 個設定項，
   寫錯任何一個都會影響整個站台。要寫入的 key 在程式碼裡寫死（`OPTION_KEY`），
   **沒有 CLI 旗標可以改**。

⚠️ **這個 API 的寫入不是即時生效的，而且回報成功不代表已經生效。**
   實測：PUT 回 `{"success":true}` 之後立刻回讀還是舊值，隔一段時間再讀就對了。
   所以本腳本寫入後**輪詢回讀**（最多等 56 秒）才判定成敗。

   ⚠️ **一讀不符時不要重送 PUT。** 早期版本會重送，看起來「第二次才成功」，
   其實是第二次的回讀剛好等到傳播完成、跟重送無關。這次的操作剛好是冪等的所以
   沒出事，但同樣的寫法用在「附加一則」上就會重複新增。

⚠️ 公開端點 `GET /api/status` 讀到的公告也可能是**快取**。要確認寫入結果請用
   認證端點 `/api/option/`（本腳本的做法）。
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import sys
import time
from pathlib import Path

import httpx

BACKUP_DIR = Path(__file__).resolve().parent.parent / "outputs" / "announcement-backups"

# 🔒 唯一允許寫入的設定項。刻意寫死、不開放參數化——見檔頭說明。
OPTION_KEY = "console_setting.announcements"

# 公告內容的固定格式。單一模型放同一行、多個模型換行列出（沿用站上既有的排版）：
#     New models：claude-opus-4-7
#     New models：
#     claude-opus-5
#     grok-4.3
# 全形冒號是指定格式（跟站上既有的「模型上線：」同一個形狀），不要換成半形。
PREFIX_EN = "New models："
PREFIX_ZH = "模型上線："

# 憑證放在 nen_ai_project 那層的 .env（三個 repo 共用）。
# ⚠️ 那份 env 有重複的鍵（`ip`/`user`/`passwd` 是不同伺服器的區塊），
# **不能用 shell 的 `source` 讀**，這裡自己解析。
NEN_ENV_PATH = Path("/Users/levi/nen_ai_project/.env")


# 那份 env 是 INI 風格的分區檔，而且**不同區塊有同名的鍵**（`endpoint`／`model_apikey`
# 在正式站與測試站各有一份）。先前這裡是整份平讀、後出現的蓋掉先出現的，等於「最後
# 一個區塊贏」——2026-09-01 新增 `[nen ai test site]` 區塊之後，`endpoint` 就被解析成
# 測試網關，公告會寫錯地方，而 `NEN_TOKEN` 只有正式站區塊有，等於**拿正式站的管理員
# token 去打測試網關**。所以改成分區解析，只認 `[nen ai product]` 這一區。
NEN_ENV_SECTION = "nen ai product"


def _load_env(section: str = NEN_ENV_SECTION) -> dict:
    if not NEN_ENV_PATH.exists():
        return {}
    out: dict = {}
    current = None
    for line in NEN_ENV_PATH.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("["):
            # 區塊標頭偶爾有多餘的結尾括號（`[nen ai test site]]`），一律剝掉
            current = line.strip("[]").strip()
            continue
        if "=" not in line or current != section:
            continue
        k, v = line.split("=", 1)
        out[k.strip()] = v.strip().strip("\"'")
    return out


_ENV = _load_env()
BASE = os.environ.get("NENAI_BASE") or _ENV.get("endpoint") or "https://nen.com.tw"


def _headers() -> dict:
    token = (os.environ.get("NEN_TOKEN") or _ENV.get("NEN_TOKEN") or "").strip()
    user = (os.environ.get("NEN_USER_ID") or _ENV.get("user_id") or "").strip()
    if not token or not user:
        sys.exit(
            f"缺少憑證。預期從 {NEN_ENV_PATH} 讀 `NEN_TOKEN` 與 `user_id`，\n"
            "也可以用環境變數 NEN_TOKEN / NEN_USER_ID 覆蓋。\n"
            "注意：呼叫模型的 API key（sk-...）在這裡不能用，會回「access token 无效」，\n"
            "而且 HTTP 狀態碼仍然是 200——只有 body 的 success 欄位看得出失敗。"
        )
    return {"Authorization": f"Bearer {token}", "New-Api-User": user,
            "Content-Type": "application/json"}


def read_items() -> list[dict]:
    """讀目前的公告陣列。用認證端點，不用公開端點——後者會有快取。"""
    r = httpx.get(f"{BASE}/api/option/", headers=_headers(), timeout=30)
    r.raise_for_status()
    raw = next((o.get("value") or "" for o in r.json().get("data", [])
                if o.get("key") == OPTION_KEY), "")
    return json.loads(raw) if raw.strip() else []


def backup(items: list[dict]) -> Path:
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    path = BACKUP_DIR / f"announcements-{stamp}.json"
    path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


# 回讀確認的等待節奏（秒）。這個 API 的寫入**不是即時生效的**：
# 實測 backfill 那次，PUT 回 success 之後立刻讀還是舊值，隔幾分鐘再讀就對了。
# 所以驗證要「等待重讀」，不能一讀不符就當失敗——更不能因此重送一次 PUT
# （這次剛好是冪等所以沒事，但同樣的寫法用在「附加一則」上就會重複新增）。
VERIFY_DELAYS = (0, 3, 8, 15, 30)


def _verify(expected: list[dict]) -> bool:
    """輪詢回讀直到與預期一致；全部等完仍不一致才回 False。"""
    for i, delay in enumerate(VERIFY_DELAYS):
        if delay:
            time.sleep(delay)
        if read_items() == expected:
            waited = sum(VERIFY_DELAYS[:i + 1])
            print(f"✅ 已寫入並回讀確認（等待 {waited} 秒）")
            return True
    return False


def write_items(items: list[dict]) -> None:
    """寫入後輪詢回讀確認。

    ⚠️ 只 PUT 一次。這個 API 回 `success: true` 不代表值已經生效，但**重送並不會
    讓它更快生效**——實測那次「重送才成功」其實是第二次的回讀剛好等到傳播完成，
    不是重送起的作用。重送對非冪等的操作反而危險，所以這裡改成只送一次、耐心等。
    """
    payload = json.dumps(items, ensure_ascii=False)
    r = httpx.put(f"{BASE}/api/option/", headers=_headers(), timeout=60,
                  json={"key": OPTION_KEY, "value": payload})
    if r.status_code != 200:
        sys.exit(f"寫入失敗 HTTP {r.status_code}：{r.text[:300]}")
    try:
        ok = r.json().get("success")
    except Exception:
        sys.exit(f"寫入回應不是 JSON：{r.text[:300]}")
    if not ok:
        sys.exit(f"寫入被拒：{r.json().get('message') or r.text[:300]}")

    if _verify(items):
        return
    sys.exit(
        f"⚠️ 寫入回報成功，但等了 {sum(VERIFY_DELAYS)} 秒回讀仍然不一致。\n"
        "**不要直接重跑**——這個 API 的寫入會延遲生效，很可能其實已經成功了。\n"
        f"請先用 --show 再確認一次，或到網頁後台看。備份檔在 {BACKUP_DIR}"
    )


def to_english(items: list[dict]) -> list[dict]:
    """把既有公告的「模型上線：」換成「New models：」。

    **只動 content 的那個前綴**，`id` / `publishDate` / `type` / `extra` 一律原樣保留
    ——上線日期是這些公告唯一的時間資訊，不能因為改語言而變動。
    """
    out = []
    for it in items:
        new = dict(it)                       # 淺拷貝，其餘欄位原樣帶過
        new["content"] = it.get("content", "").replace(PREFIX_ZH, PREFIX_EN)
        out.append(new)
    return out


def add_models(items: list[dict], models: list[str]) -> list[dict]:
    """新增一則模型上線公告。id 取現有最大值 +1（重複的 id 會讓前端顯示異常）。"""
    body = models[0] if len(models) == 1 else "\n" + "\n".join(models)
    content = PREFIX_EN + body
    if any(it.get("content") == content for it in items):
        return items                         # 一模一樣的內容已經在裡面
    next_id = max((it.get("id", 0) for it in items), default=0) + 1
    now = dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.") \
        + f"{dt.datetime.now().microsecond // 1000:03d}Z"
    return items + [{"id": next_id, "content": content, "publishDate": now,
                     "type": "default", "extra": ""}]


def backfill(items: list[dict], plan: list[dict]) -> list[dict]:
    """依計畫補上多則歷史公告，一次寫入。

    每筆計畫是 {"date": "YYYY-MM-DD", "models": [...]}。日期用當天 00:00:00Z——
    我們只知道到「哪一天」，不知道幾點，補一個假的時分秒會是憑空的精確度。

    既有公告的 `id` / `publishDate` **完全不動**；新的 id 從現有最大值往上接。
    最後依 publishDate 排序整個陣列（只改陣列順序，不改任何一則的內容），
    讓站上的時間軸讀起來是連貫的。
    """
    next_id = max((it.get("id", 0) for it in items), default=0) + 1
    added = []
    for entry in plan:
        models = entry["models"]
        body = models[0] if len(models) == 1 else "\n" + "\n".join(models)
        added.append({
            "id": next_id,
            "content": PREFIX_EN + body,
            "publishDate": f"{entry['date']}T00:00:00.000Z",
            "type": "default",
            "extra": "",
        })
        next_id += 1
    return sorted(items + added, key=lambda x: x.get("publishDate", ""))


def show(items: list[dict]) -> None:
    print(f"── {OPTION_KEY}（{BASE}）共 {len(items)} 則 ──\n")
    for it in items:
        print(f"  id={it.get('id')}  {it.get('publishDate')}  type={it.get('type')}")
        for ln in (it.get("content") or "").splitlines():
            print(f"      {ln}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--show", action="store_true", help="印出目前的公告")
    ap.add_argument("--add-models", help="新增一則公告（逗號分隔多個模型 id）")
    ap.add_argument("--backfill-file",
                    help="依 JSON 計畫補上多則歷史公告（[{\"date\":\"YYYY-MM-DD\",\"models\":[...]}]），"
                         "一次寫入。既有公告的 id 與 publishDate 完全不動")
    ap.add_argument("--to-english", action="store_true",
                    help="把既有公告的「模型上線：」改成「New models：」，"
                         "id / publishDate / type / extra 完全不動")
    ap.add_argument("--confirm", action="store_true",
                    help="真的寫入。不加只做預演")
    args = ap.parse_args()

    items = read_items()

    if args.show or not (args.add_models or args.to_english or args.backfill_file):
        show(items)
        if not (args.add_models or args.to_english or args.backfill_file):
            return 0

    new_items = items
    if args.backfill_file:
        plan = json.loads(Path(args.backfill_file).read_text(encoding="utf-8"))
        new_items = backfill(new_items, plan)
    if args.to_english:
        new_items = to_english(new_items)
    if args.add_models:
        models = [m.strip() for m in args.add_models.split(",") if m.strip()]
        if not models:
            sys.exit("--add-models 沒有解析出任何模型 id。")
        if re.search(r"[一-鿿]", args.add_models):
            sys.exit("模型 id 看起來含中文，請確認。公告內容一律英文。")
        new_items = add_models(new_items, models)

    if new_items == items:
        print("內容沒有變化，不需要寫入。")
        return 0

    print("── 寫入後會變成 ──\n")
    show(new_items)

    if not args.confirm:
        print("⚠️ 這是預演，尚未寫入。確認無誤後加 --confirm 再執行一次。")
        return 0

    path = backup(items)
    print(f"已把目前的內容備份到：{path}")
    print("（這個 API 沒有版本歷史，備份是唯一的還原方式）")
    write_items(new_items)
    return 0


if __name__ == "__main__":
    sys.exit(main())
