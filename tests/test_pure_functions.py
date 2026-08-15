"""鎖住幾個容易被改壞的純函式。

這些不是為了追求覆蓋率——每一條測試都對應一個**實際發生過的 bug**，寫下來是為了
避免同樣的錯誤再犯一次。改動這些函式前先跑：

    venv/bin/python -m pytest tests/ -q

（pytest 不在 requirements.txt 裡，需要時 `venv/bin/pip install pytest`。）
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import app  # noqa: E402


class _Usage:
    """模擬 openai SDK 的 usage 物件。"""

    def __init__(self, prompt, completion, total):
        self.prompt_tokens = prompt
        self.completion_tokens = completion
        self.total_tokens = total


class _Req:
    def __init__(self, headers, scheme="http"):
        self.headers = headers
        self.url = type("U", (), {"scheme": scheme})()


# ─────────────────────────────────────────────────────────────────────────────
# _openai_usage：兩種 token 帳法必須都正確
# 起因：Grok 的推理 token **不計入** completion_tokens（實測 grok-4.3：
# prompt 31 + completion 1 + reasoning 844 = total 876），但照樣收費。只讀
# completion_tokens 會把 844 個推理 token 算成 1 個，「本次花費」嚴重低估。
# 而 GLM/DeepSeek/Seed 的推理已經含在 completion 裡，直接相加會變成兩倍。
# 所以用 total - prompt 反推，對兩種帳法都正確。
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("prompt,completion,total,expected,why", [
    (31, 1, 876, 845, "Grok：推理不含在 completion，要補回來"),
    (26, 139, 165, 139, "GLM：推理已含在 completion，不可重複加"),
    (10, 20, 0, 20, "沒有 total 時退回原值"),
    (10, 20, 25, 20, "total 比 prompt+completion 小時不要算出更小的值"),
])
def test_openai_usage(prompt, completion, total, expected, why):
    got = app._openai_usage(_Usage(prompt, completion, total))
    assert got["completion_tokens"] == expected, why
    assert got["prompt_tokens"] == prompt


# ─────────────────────────────────────────────────────────────────────────────
# _public_base_url：推導對外網址
# 起因：上游不收 base64 影片，必須給可下載的 URL。沒有雲端儲存時退回本站的
# /outputs 公開路徑，所以要能正確推導出對外網址。
# 最容易錯的一點：**不能用 request.url.scheme**——在 LB 後面那是內部的 http，
# 給出 http:// 的網址有機會被上游拒絕。
# ─────────────────────────────────────────────────────────────────────────────
@pytest.mark.parametrize("headers,expected,why", [
    ({"x-forwarded-host": "playground.nen.com.tw", "x-forwarded-proto": "https", "host": "internal"},
     "https://playground.nen.com.tw", "LB 轉發時以 X-Forwarded-* 為準"),
    ({"host": "playground.nen.com.tw"},
     "https://playground.nen.com.tw", "沒有 X-Forwarded-Proto 時預設 https，不可用內部的 http"),
    ({"x-forwarded-host": "playground.nen.com.tw, internal", "x-forwarded-proto": "https, http"},
     "https://playground.nen.com.tw", "多值標頭取第一段"),
    ({"host": "127.0.0.1:5050"}, None, "本機位址外部抓不到，要回 None 讓呼叫端報錯"),
    ({"host": "localhost:5050"}, None, "同上"),
    ({}, None, "沒有標頭就推導不出來"),
])
def test_public_base_url(headers, expected, why, monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert app._public_base_url(_Req(headers)) == expected, why


def test_public_base_url_env_wins(monkeypatch):
    """明確設定的 PUBLIC_BASE_URL 優先於推導，且尾斜線要去掉。"""
    monkeypatch.setenv("PUBLIC_BASE_URL", "https://custom.example.com/")
    assert app._public_base_url(_Req({"host": "127.0.0.1"})) == "https://custom.example.com"


def test_public_base_url_none_request(monkeypatch):
    monkeypatch.delenv("PUBLIC_BASE_URL", raising=False)
    assert app._public_base_url(None) is None


# ─────────────────────────────────────────────────────────────────────────────
# _apply_res_and_duration：三家上游取值的欄位都不同，漏送不會報錯、只會靜默用預設值
# 起因：只送頂層 size 時，Veo 的 "1080P" 切不開會 fallback 成 720p 且照 720p 計費；
# doubao 則連頂層 size 都不讀。所以三種形式一律都送。
# ─────────────────────────────────────────────────────────────────────────────
def test_apply_res_and_duration_sends_all_forms():
    payload, meta = {}, {}
    app._apply_res_and_duration(payload, meta, "1080P", 5, "16:9")
    assert payload["size"] == "1080P", "阿里讀頂層 size"
    assert meta["resolution"] == "1080p", "Veo/doubao 讀 metadata.resolution，且要小寫"
    assert payload["duration"] == 5, "阿里/Veo 讀頂層 duration(int)"
    assert payload["seconds"] == "5", "doubao 只吃頂層 seconds(字串)"
    assert meta["ratio"] == "16:9", "doubao 的比例欄位"
    assert meta["aspectRatio"] == "16:9", "Veo 的比例欄位名不同"


def test_apply_res_and_duration_skips_duration_when_none():
    """視頻編輯不指定時長（保留來源長度），此時不可送 duration。"""
    payload, meta = {}, {}
    app._apply_res_and_duration(payload, meta, "720P", None, "")
    assert "duration" not in payload and "seconds" not in payload
    assert "ratio" not in meta, "沒有比例時不要送空字串"


# ─────────────────────────────────────────────────────────────────────────────
# _image_usage：兩條圖片路徑的 usage 欄位命名不同
# 起因：9 個按 token 計費的圖片模型先前完全沒被計入花費。
# ─────────────────────────────────────────────────────────────────────────────
def test_image_usage_openai_style():
    got = app._image_usage({"usage": {"num_input_text_tokens": 9, "num_input_image_tokens": 3,
                                      "num_output_tokens": 1024}})
    assert got == {"prompt_tokens": 12, "completion_tokens": 1024}, "輸入的文字與圖片 token 要相加"


def test_image_usage_standard_names():
    got = app._image_usage({"usage": {"prompt_tokens": 5, "completion_tokens": 100}})
    assert got == {"prompt_tokens": 5, "completion_tokens": 100}


def test_image_usage_absent_returns_none():
    """按次計費的模型沒有 usage，要回 None 而不是零——零會讓呼叫端誤以為免費。"""
    assert app._image_usage({"data": []}) is None
    assert app._image_usage({"usage": {}}) is None


# ─────────────────────────────────────────────────────────────────────────────
# 從 MODELS 推導出來的限制集合：改 MODELS 時這些要跟著對
# ─────────────────────────────────────────────────────────────────────────────
def test_edit_max_ref_reflects_models():
    """參考圖上限以 MODELS 的 max_ref 為單一來源，前後端讀同一份資料。"""
    assert app._EDIT_MAX_REF["wan2.6-image"] == 4, "萬相 2.6 上限 4（送 5 張回 must contain 1 to 4）"
    assert app._EDIT_MAX_REF["qwen-image-2.0"] == 3, "千問融合編輯上限 3"
    assert app._EDIT_MAX_REF["MAI-Image-2.5"] == 1, "MAI 只接受剛好一張"
    assert "wan2.7-image" in app._EDIT_MAX_REF and app._EDIT_MAX_REF["wan2.7-image"] == 9, \
        "萬相 2.7 實測 9 張都生效——曾誤依 Go struct 設成 2，導致使用者反映『只能傳兩張』"


def test_first_frame_only_models_derived():
    """i2v 只讀 images[0] 的模型要限制成單一模式，否則尾幀會被靜默丟棄。"""
    assert "wan2.7-i2v" in app._FIRST_FRAME_ONLY_I2V_MODELS
    assert "happyhorse-1.1-i2v" in app._FIRST_FRAME_ONLY_I2V_MODELS


def test_gemini_thinking_exception_sets_are_disjoint_in_meaning():
    """兩個例外集合各自代表不同的限制，不要混淆：
    NO_THINKING_OFF   = 思考關不掉（但過程看得到）
    NO_INCLUDE_THOUGHTS = 過程拿不到（但關得掉）
    """
    assert "gemini-2.5-pro" in app._GEMINI_NO_THINKING_OFF
    assert "gemini-2.5-flash-lite" in app._GEMINI_NO_INCLUDE_THOUGHTS
    assert "gemini-2.5-pro" not in app._GEMINI_NO_INCLUDE_THOUGHTS


def test_gemini_thinking_off_by_default_needs_budget():
    """這些型號思考預設是關的，要送 thinkingBudget=-1 才會啟動，
    否則「思考開」那一檔完全沒有作用（實測開/關都是 1 個 token）。"""
    assert "gemini-2.5-flash-lite" in app._GEMINI_THINKING_OFF_BY_DEFAULT
    assert "gemini-3.5-flash-lite" in app._GEMINI_THINKING_OFF_BY_DEFAULT


def test_kimi_must_not_receive_enable_thinking():
    """kimi/kimi-k3 送 enable_thinking:false 會 400，且錯誤訊息指向 temperature。

    起因：它是純思考模型、MODELS 標 thinking=False，前端因此會送 enable_thinking:false。
    後端原本對所有非 GPT 模型一律帶這個欄位，不排除就是**每一次呼叫都失敗**。
    錯誤訊息是 "invalid temperature: only 0.6 is allowed for this model"——完全沒有
    提到 thinking，光看訊息會往 temperature 的方向查（實測 temperature=0.7 反而正常）。
    """
    assert "kimi/kimi-k3" in app._NO_ENABLE_THINKING_MODELS


def test_gemini_37_flash_thinking_toggle_is_effective():
    """gemini-3.7-flash 的 thinkingBudget=0 有作用，所以要給開關。

    起因：一度誤判成「關不掉、靜默忽略」而不給開關——那是**只用一個提示詞**測出來的。
    用會觸發大量思考的題目背對背重測（正式環境各 5 次）才看得出來：
        多步算術題  budget=0 → 76~94   不帶設定 → 180~294   完全不重疊
        短句陷阱題  budget=0 → 0~158   不帶設定 → 121~183   重疊
    陷阱題基準思考量本來就低，沒有下降空間。開關能省約一半 thinking token。
    """
    assert "gemini-3.7-flash" not in app._GEMINI_NO_THINKING_OFF, \
        "budget=0 有作用（208 → 89），不該歸在關不掉那一類"
    assert "gemini-3.7-flash" not in app._GEMINI_NO_INCLUDE_THOUGHTS, "它的思考過程看得到"
    ids = {m["id"] for m in app.MODELS["text"]}
    assert "gemini-3.7-flash" in ids and "kimi/kimi-k3" in ids
    meta = {m["id"]: m for m in app.MODELS["text"]}
    assert meta["gemini-3.7-flash"]["thinking"] is True, "budget=0 有作用，該給開關"
    assert meta["kimi/kimi-k3"]["thinking"] is False, "純思考模型，關不掉"
    assert meta["kimi/kimi-k3"].get("vision") is True, \
        "kimi 的 data URI 圖片輸入正式環境實測 9/9 可用（三種尺寸各 3 次）"


def test_proxy_whitelist_covers_upstream_output_hosts():
    """白名單要涵蓋上游直接回傳產出網址的網域，否則 Canvas 接續會被擋。"""
    def allowed(host):
        return any(host == s.lstrip(".") or host.endswith(s) for s in app._PROXY_ALLOWED_SUFFIXES)

    assert allowed("dashscope-463f.oss-accelerate.aliyuncs.com"), "萬相/千問的產出"
    assert allowed("ark-acg-ap-southeast-1.tos-ap-southeast-1.volces.com"), "Seedance 的產出"
    assert allowed("storage.googleapis.com")
    assert not allowed("evil.com")
    assert not allowed("volces.com.evil.com"), "偽裝後綴必須擋下（SSRF 防護）"


# ─────────────────────────────────────────────────────────────────────────────
# Veo 的每秒單價表：唯一的防線是註解，補一條測試把數字釘住
# 那張表在 `static/js/app.js`（前端顯示用的人工快照），閘道的 /api/pricing 只給
# **基準價**、不給檔次倍率，所以沒有任何 API 能驗證它——這條測試就是驗證。
#
# 起因有三個都實際發生過：
#   1. Lite 一度註明「不支援配音」而漏了有聲檔次，有聲時顯示比實付低 66%
#   2. Fast 的基準價一度被填成官方的 3 倍，差點反過來把顯示改成 $0.24 那組
#   3. 閘道端的 Lite 修正部署時程未定，**這段期間拿正式站帳單來對照會發現這裡偏高**
#      ——那不是錯。沒有這條測試的話，下一個比對帳單的人很可能把正確值改回錯的。
#
# ⚠️ **價格驗證一律以 `nen-ai-platform`（閘道）專案為準。** 這裡的數字是跟著它走的
# 快照，不是獨立的第二意見——兩邊對不上時，以他們的為準、我們跟著改，不要在這裡
# 自行定案。理由是實際扣款發生在閘道端，我們只負責顯示。
#
# 數字來源：Google Cloud 官方價格頁（Vertex AI ＞ Other Gemini models ＞ Veo），
# 2026-08-15 由本專案直接對照截圖逐格核對，18 格與 nen-ai-platform 的核對結果
# 全數相符。我們的核對是複驗，不是背書。
#
# 那份截圖的單位欄只寫「/ 1 count」、且在 Lite 的 1080p 那列被截斷，所以另外查了
# 第二份官方來源補這兩格：https://ai.google.dev/gemini-api/docs/pricing（本專案第一
# 手抓取，2026-08-15）——
#   1. 計價單位：欄標逐字是「Paid Tier, per second in USD」。**確定是每秒**，不再
#      是「公式與已知牌價吻合」的間接推論。這格很重要：若它其實是每支影片計價，
#      我們（與閘道）把單價乘上秒數就是對使用者**超收**約 8 倍。
#   2. Lite 的 4K：該頁在 Lite 那列直接標注「(4k output not supported)」。**確定
#      沒有這一檔**，所以表上 Lite 的 `4K` 填什麼都不會被用到（UI 也不開 4K）。
# 該頁列的是「含音訊」那一檔（0.40 / 0.10・0.12・0.30 / 0.05・0.08），與上面的
# _withAudio 完全相符；無聲那一檔由前述 Vertex 頁面覆蓋。兩份來源互相獨立。
#
# ⚠️ 搜尋時會撞到一批第三方部落格說 Veo 3.1 是 $0.50／$0.75 每秒、Fast $0.15——
# 那些與官方頁面不符（多半是 Veo 3 時期或推測值），不要拿來當來源。
#
# 這條測試失敗時該怎麼辦（三種情況，都不是刪掉它）：
#   1. 官方調價 → **先跟 nen-ai-platform 對齊**，以他們的值改期望值並更新上面的日期。
#      不要只憑自己查到的官方頁面就改完收工——真正扣款的是他們，顯示與扣款分岔比
#      顯示成舊價更糟。
#   2. 我們某一格核對錯了 → 同上，跟他們對齊；查證時**回頭比對官方頁面**，不要照著
#      實際扣款反推（實際扣款可能正處在某個修正尚未部署的落差裡，見上面第 3 點）
#   3. 有人把 app.js 的表改壞了 → 改回來
_VEO_EXPECTED = {
    "veo-3.1-generate-001":      ({"720P": 0.20, "1080P": 0.20, "4K": 0.40},
                                  {"720P": 0.40, "1080P": 0.40, "4K": 0.60}),
    "veo-3.1-fast-generate-001": ({"720P": 0.08, "1080P": 0.10, "4K": 0.25},
                                  {"720P": 0.10, "1080P": 0.12, "4K": 0.30}),
    # Lite 沒有 4K 檔次，4K 請求會落到 1080P 的價，所以兩者相同
    "veo-3.1-lite-generate-001": ({"720P": 0.03, "1080P": 0.05, "4K": 0.05},
                                  {"720P": 0.05, "1080P": 0.08, "4K": 0.08}),
}


def _veo_price_entry(model_id):
    """從 app.js 的 _VIDEO_SEC_PRICE 取出某個 veo 型號的（無聲, 有聲）單價。"""
    import re

    src = (Path(__file__).resolve().parent.parent / "static" / "js" / "app.js").read_text("utf-8")
    table = src.split("const _VIDEO_SEC_PRICE = {", 1)[1].split("\n};", 1)[0]
    # 取這個 id 到下一個 id（或表尾）之間的片段，去掉空白與註解行後再解析
    body = table.split(f"'{model_id}':", 1)[1]
    body = re.split(r"\n\s*(?://|')", body, maxsplit=1)[0]
    base_txt, _, audio_txt = body.partition("_withAudio")

    def tiers(txt):
        return {k: float(v) for k, v in re.findall(r"'(\d+P|4K)':\s*([0-9.]+)", txt)}

    return tiers(base_txt), tiers(audio_txt)


@pytest.mark.parametrize("model_id", sorted(_VEO_EXPECTED))
def test_veo_per_second_prices(model_id):
    expected_base, expected_audio = _VEO_EXPECTED[model_id]
    base, audio = _veo_price_entry(model_id)
    assert base == expected_base, f"{model_id} 的無聲每秒單價與官方價格表不符"
    assert audio == expected_audio, f"{model_id} 的有聲每秒單價與官方價格表不符"


def test_veo_resolutions_exclude_480p():
    """480P 不是 Veo 的檔次：閘道計費端用 == "4k" / == "720p" 判斷，480p 兩個都不中，
    會落進 else 的 1080P 檔——使用者被以較高檔次預扣，然後請求被上游擋掉拿不到影片。
    4K 不列入：UI 的解析度選單裡沒有 4K，我們也沒實測過。
    """
    veo = [m for m in app.MODELS["video"] if m["id"].startswith("veo-")]
    assert len(veo) == 9, "t2v/i2v/r2v × 三個型號"
    for m in veo:
        assert m.get("resolutions") == ["720P", "1080P"], f"{m['id']}（{m['type']}）"
