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


def test_gemini_37_flash_thinking_cannot_be_turned_off():
    """gemini-3.7-flash 送 thinkingBudget=0 會**收下但照樣思考**，不報錯。

    跟 gemini-2.5-pro 同一個集合、但失敗方式不同：2.5-pro 直接 400，3.7-flash 是
    靜默忽略（兩種題目各 5 次，需推理題 5/5、簡單題 4/5 仍有 thoughtsTokenCount）。
    靜默忽略比報錯危險——不實測就會以為開關有效。
    """
    assert "gemini-3.7-flash" in app._GEMINI_NO_THINKING_OFF
    assert "gemini-3.7-flash" not in app._GEMINI_NO_INCLUDE_THOUGHTS, "它的思考過程看得到"
    ids = {m["id"] for m in app.MODELS["text"]}
    assert "gemini-3.7-flash" in ids and "kimi/kimi-k3" in ids
    meta = {m["id"]: m for m in app.MODELS["text"]}
    assert meta["gemini-3.7-flash"]["thinking"] is False, "關不掉就不該給開關"
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
