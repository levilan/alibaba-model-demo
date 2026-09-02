"""鎖住幾個容易被改壞的純函式。

這些不是為了追求覆蓋率——每一條測試都對應一個**實際發生過的 bug**，寫下來是為了
避免同樣的錯誤再犯一次。改動這些函式前先跑：

    venv/bin/python -m pytest tests/ -q

（pytest 不在 requirements.txt 裡，需要時 `venv/bin/pip install pytest`。）
"""
import json
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
    """i2v 只讀 images[0] 的模型要限制成單一模式，否則尾幀會被靜默丟棄。
    wan2.7-i2v 自閘道 abc0a8b7b（2026-08-28 部署）起支援 audio_url→driving_audio
    口型同步，開放 first_frame_audio 模式、不再是首幀限定。"""
    assert "wan2.7-i2v" not in app._FIRST_FRAME_ONLY_I2V_MODELS
    ff_audio = [m for m in app.MODELS["video"]
                if m["id"] == "wan2.7-i2v" and m["type"] == "i2v"][0]
    assert ff_audio["i2v_modes"] == ["first_frame", "first_frame_audio"]
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


def test_dola_seed_21_turbo_flags():
    """Seed 2.1 Turbo 的旗標鎖住 2026-08-21 的實測結論（正式環境）。

    enable_thinking true/false 各 3 次全部回 reasoning_content——開關無效，
    所以 thinking 必須是 False（不顯示沒作用的開關）。但 reasoning_effort:none
    實測 3 次 reasoning 全 0、能真正關閉思考，是 seed 家族唯一能控制思考的
    型號，所以 reasoning_effort 選單必須開、且清單必須含 none。
    vision：data URI 圖片輸入實測可用。
    """
    meta = {m["id"]: m for m in app.MODELS["text"]}
    m = meta["dola-seed-2.1-turbo"]
    assert m["thinking"] is False, "enable_thinking 實測無效，不給誤導性開關"
    assert m.get("reasoning_effort") is True, "reasoning_effort 是它唯一有效的思考控制"
    assert "none" in m.get("reasoning_efforts", []), "none 是唯一能關思考的值，不能拿掉"
    assert m.get("vision") is True, "data URI 圖片輸入實測可用"


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


# ─────────────────────────────────────────────────────────────────────────────
# _pricing_entry：全模態模型的音訊單價要一起帶出來
# 起因：`audio_ratio` / `audio_completion_ratio` 在網關是 `*float64` + `omitempty`，
# **只有該模型真的設定了才會出現**。我一度查了幾個模型都沒看到就判定「API 不吐」，
# 差點為此做一份會過期的人工快照——實際上 qwen3.5-omni-plus-realtime 有這兩個欄位。
# （更正一次自己的錯誤流程：那次是我的查詢只印固定幾個欄位，把答案自己濾掉了。）
#
# 數字對照 2026-08-16 正式站 /api/pricing 的實際回應，以及阿里雲官方牌價：
#   輸入文字 $2.1/1M、輸出文字 $12.4/1M、輸入音訊 $16.5/1M、輸出音訊 $62/1M
def test_pricing_entry_carries_audio_rates():
    m = {
        "model_name": "qwen3.5-omni-plus-realtime", "quota_type": 0,
        "model_ratio": 1.05, "completion_ratio": 5.904761904762,
        "audio_ratio": 7.857142857143, "audio_completion_ratio": 3.757575757576,
    }
    e = app._pricing_entry(m)
    assert e["type"] == "token"
    assert round(e["input"], 4) == 2.1
    assert round(e["output"], 4) == 12.4
    assert round(e["audio_input"], 4) == 16.5, "音訊輸入是文字的 7.86 倍，漏掉會低估"
    assert round(e["audio_output"], 4) == 62.0, "音訊輸出最貴的一檔"


def test_pricing_entry_without_audio_rates():
    """沒有音訊倍率的模型不該憑空生出音訊欄位（前端會據此決定要不要用預設值）。"""
    e = app._pricing_entry({"quota_type": 0, "model_ratio": 0.75, "completion_ratio": 4})
    assert "audio_input" not in e and "audio_output" not in e


def test_realtime_voices_verified():
    """音色清單都是逐一實測過的；Chelsie 實測不支援，不可以再列回來。

    驗法：session.update 帶音色 + response.create，有效的會開始回傳音訊、無效的回
    `Voice 'X' is not supported.`。**不要用 session.update 的回應驗**——它對任何
    字串都回 session.updated，連亂編的名字都照收。也不要用音訊位元比對——同音色
    同輸入重跑兩次的位元並不相同。

    2026-08-16 起共四個模型（scripts/probe_realtime.py 對測試網關實測）：
    omni 兩個共用同一組 56 音色（plus 對正式環境、flash 對測試網關各全掃一次）；
    audio-3.0 兩個共用另一組 15 音色（非法音色的錯誤訊息會列出完整合法清單，
    再逐一驗證出聲），與 omni 的音色完全不互通。
    """
    rt = app.MODELS["voice"]["realtime"]
    assert [m["id"] for m in rt] == [
        "qwen3.5-omni-plus-realtime",
        "qwen3.5-omni-flash-realtime",
        "qwen-audio-3.0-realtime-plus",
        "qwen-audio-3.0-realtime-flash",
    ]
    by_id = {m["id"]: m for m in rt}
    for m in rt:
        ids = {v["id"] for v in m["voices"]}
        assert m["default_voice"] in ids, m["id"]
        assert (m["input_rate"], m["output_rate"]) == (16000, 24000), m["id"]

    omni = [by_id["qwen3.5-omni-plus-realtime"], by_id["qwen3.5-omni-flash-realtime"]]
    for m in omni:
        ids = {v["id"] for v in m["voices"]}
        assert len(ids) == 56, m["id"]
        assert "Chelsie" not in ids, "實測不支援（那是 qwen2.5-omni 的音色）"
    assert omni[0]["voices"] is omni[1]["voices"], "兩個 omni 型號實測同一組音色，共用同一份清單"

    audio = [by_id["qwen-audio-3.0-realtime-plus"], by_id["qwen-audio-3.0-realtime-flash"]]
    for m in audio:
        ids = {v["id"] for v in m["voices"]}
        assert len(ids) == 15, m["id"]
        assert m["default_voice"] == "longanqian", "官方文件與實測的預設音色"
        assert not ids & {"Tina", "Ethan", "Serena", "Cherry"}, "omni 音色在 audio-3.0 上實測全被拒"
        assert m.get("audio_only") is True, "純語音模型；圖片會被上游靜默忽略，前端據此藏附件鈕"


def test_realtime_turn_modes_verified():
    """turn_detection 的合法值兩個家族不同，都是對測試網關實測的（2026-08-16）：

    omni 收 semantic_vad / server_vad；audio-3.0 送 semantic_vad 會回
    `Unsupported turn_detection.type: 'semantic_vad'. Supported values: server_vad,
    smart_turn.`，null（前端以 "none" 表示）兩邊都收。前端會拿 turn_modes 重建
    選單，所以這裡鎖住的是「不要把某家族不收的值放進另一家族」。
    """
    rt = {m["id"]: m for m in app.MODELS["voice"]["realtime"]}
    for mid in ("qwen3.5-omni-plus-realtime", "qwen3.5-omni-flash-realtime"):
        assert rt[mid]["turn_modes"] == ["semantic_vad", "server_vad", "none"], mid
    for mid in ("qwen-audio-3.0-realtime-plus", "qwen-audio-3.0-realtime-flash"):
        assert rt[mid]["turn_modes"] == ["server_vad", "smart_turn", "none"], mid


def test_lyria_music_models():
    """Lyria 三個模型的能力旗標（2026-08-17，閘道端逐一實測轉知＋本平台對 clip 與
    圖片輸入複驗）：圖片生音樂只有 lyria-3 兩個支援，lyria-002 帶圖上游會明確報錯
    ——image_input 旗標決定前端顯不顯示靈感圖片上傳欄，把 002 打開等於做出一個
    必定失敗的選項。"""
    music = app.MODELS["voice"]["music"]
    assert [m["id"] for m in music] == ["lyria-3-clip-preview", "lyria-3-pro-preview", "lyria-002"]
    by_id = {m["id"]: m for m in music}
    assert by_id["lyria-3-clip-preview"].get("image_input") is True
    assert by_id["lyria-3-pro-preview"].get("image_input") is True
    assert not by_id["lyria-002"].get("image_input")


def test_seedance_25_video_entries():
    """Seedance 2.5 的三個影片條目共用同一組實測約束（t2v 2026-08-11、i2v/r2v
    2026-08-17 對測試網關端到端驗證）：時長 [4,30]、解析度只有 480P/720P（1080P/4K
    上游拒絕）。r2v 的 max_ref=30（官方參考素材上限）；vedit 不列（家族既有結論是
    上游拒絕）。上游按圖片張數分類模式（1→i2v、2→首尾幀、3+→r2v），r2v 少於 3 張
    由後端擋下，這裡鎖住條目本身的形狀。"""
    entries = [m for m in app.MODELS["video"] if m["id"] == "dreamina-seedance-2.5"]
    assert sorted(m["type"] for m in entries) == ["i2v", "r2v", "t2v"]
    for m in entries:
        assert (m["min_dur"], m["max_dur"]) == (4, 30), m["type"]
        assert m["resolutions"] == ["480P", "720P"], m["type"]
    r2v = next(m for m in entries if m["type"] == "r2v")
    assert r2v["max_ref"] == 30


def test_mai_image_n_locked_to_one():
    """MAI 三個 t2i 的 max_n 鎖 1，**長期維持**（2026-08-16）：n=3 實測 data[] 只回
    1 筆 b64_json、無 metadata 可補，且閘道曾照 3 張計費（增幅 $0.3257 與 3 張的
    算式完全吻合，多收錢的部分閘道端已修）。（📄 轉述自閘道端）Azure 上游本來就
    靜默忽略 n、永遠只回 1 張——所以「n=3 回 3 張」這個解鎖條件永遠不會發生，
    把這裡改回去只會做出一個沒有作用的選項。"""
    mai = [m for m in app.MODELS["image"]
           if m["id"].startswith("MAI-Image") and m.get("type") == "t2i"]
    assert len(mai) == 3
    for m in mai:
        assert m["max_n"] == 1, m["id"]


def test_debug_req_never_leaks_api_key():
    """「查看實際請求」的摘要永遠不能帶出真實金鑰。

    這個面板會出現在使用者畫面上、會被截圖與螢幕分享，所以 auth 欄位固定是佔位字串，
    而且整個結構裡不應該出現任何 sk- 開頭的東西。
    """
    from app import _debug_req
    out = _debug_req("/v1/images/generations", {"model": "x", "prompt": "hi"})
    assert out["auth"] == "Bearer $NENAI_API_KEY"
    assert "sk-" not in json.dumps(out)


def test_debug_req_summarizes_base64():
    """base64 內容換成長度摘要——塞爆畫面而且沒有參考價值，但欄位結構要留著。"""
    from app import _debug_req
    big = "data:image/png;base64," + ("A" * 5000)
    out = _debug_req("/v1/images/edits", {"image": big, "prompt": "keep me"})
    assert out["body"]["prompt"] == "keep me"
    assert "5000 chars" in out["body"]["image"]
    assert "A" * 100 not in json.dumps(out)


# ── _apply_video_extra_params：seed/watermark/prompt_extend 的放置層級 ──────
# 對應 2026-08-25 平台端回報的 bug：ali 系 adaptor（wan*/happyhorse*）只讀巢狀
# metadata.parameters.*，扁平的 metadata.seed 被靜默忽略——playground 的 seed/
# 浮水印/prompt_extend 三個開關在這兩家等於沒作用。其他家族行為未驗證維持扁平。

def test_video_extra_params_wan_goes_nested():
    meta = {}
    app._apply_video_extra_params(meta, "wan2.7-t2v", 42, True, True)
    assert meta == {"parameters": {"seed": 42, "watermark": True, "prompt_extend": True}}


def test_video_extra_params_happyhorse_goes_nested():
    meta = {}
    app._apply_video_extra_params(meta, "happyhorse-1.1-t2v", 7, False, False)
    assert meta == {"parameters": {"seed": 7}}


def test_video_extra_params_veo_stays_flat():
    # veo 的 adaptor 行為未驗證，維持原本的扁平寫法——不要「順手統一」
    meta = {}
    app._apply_video_extra_params(meta, "veo-3.1-generate-001", 42, True, False)
    assert meta == {"seed": 42, "watermark": True}
    assert "parameters" not in meta


def test_video_extra_params_noop_leaves_meta_clean():
    # 三個都沒設時不能留下空的 parameters 鍵——多餘欄位可能改變上游行為
    meta = {"negative_prompt": "x"}
    app._apply_video_extra_params(meta, "wan2.7-t2v", None, False, False)
    assert meta == {"negative_prompt": "x"}


def test_video_extra_params_seed_zero_is_sent():
    # seed=0 是合法值，不能被當成「沒設」丟掉（與前端「留空≠0」同一條原則）
    meta = {}
    app._apply_video_extra_params(meta, "wan3.0-video", 0, False, False)
    assert meta["parameters"]["seed"] == 0


def test_video_extra_params_negative_prompt_wan_goes_input_layer():
    # negative_prompt 在 ali 系走 metadata.input.*（不是 parameters！）——
    # 平台端 2026-08-25 實測：扁平與 parameters 層都無效，只有 input 層生效
    meta = {}
    app._apply_video_extra_params(meta, "wan2.7-t2v", None, False, False, "blurry")
    assert meta == {"input": {"negative_prompt": "blurry"}}


def test_video_extra_params_negative_prompt_veo_stays_flat():
    meta = {}
    app._apply_video_extra_params(meta, "veo-3.1-generate-001", None, False, False, "blurry")
    assert meta == {"negative_prompt": "blurry"}


def test_video_extra_params_input_layer_merges_with_existing():
    # vedit 的 meta 可能已有 input（_apply_explicit_media 的 media 陣列），要合併不能蓋掉
    meta = {"input": {"media": [{"type": "reference_image", "url": "x"}]}}
    app._apply_video_extra_params(meta, "wan3.0-video", None, False, False, "ugly")
    assert meta["input"]["media"] and meta["input"]["negative_prompt"] == "ugly"


def test_default_ratio_per_generation():
    # 平台端 2026-08-25 實測的各代行為：wan3.0 主動 adaptive、wan2.7/hh 不送、
    # wan2.6 不走 ratio、其他家族維持 16:9
    assert app._default_ratio("wan3.0-video") == "adaptive"
    assert app._default_ratio("wan2.7-t2v") == ""
    assert app._default_ratio("happyhorse-1.1-t2v") == ""
    assert app._default_ratio("wan2.6-t2v") == ""
    assert app._default_ratio("veo-3.1-generate-001") == "16:9"


# ── _apply_image_negative_prompt：圖像 negative_prompt 的家族分流 ──────────
# 平台 marshal 實測 2026-08-25：千問 3.0 扁平靜默失效、巢狀 parameters 有效；
# MAI 請求被重建怎麼寫都到不了上游。與 wan 影片 metadata 那批同型的坑。

def test_image_neg_prompt_qwen3_goes_parameters():
    payload = {"model": "qwen-image-3.0", "prompt": "x"}
    app._apply_image_negative_prompt(payload, "qwen-image-3.0", "blurry")
    assert payload["parameters"] == {"negative_prompt": "blurry"}
    assert "negative_prompt" not in payload


def test_image_neg_prompt_mai_not_sent():
    payload = {"model": "MAI-Image-2.5", "prompt": "x"}
    app._apply_image_negative_prompt(payload, "MAI-Image-2.5", "blurry")
    assert "negative_prompt" not in payload and "parameters" not in payload


def test_image_neg_prompt_others_stay_flat():
    # 其他家族維持扁平（未驗證前不改——各代巢狀位置不同，等平台批次結果）
    payload = {"model": "z-image-turbo", "prompt": "x"}
    app._apply_image_negative_prompt(payload, "z-image-turbo", "blurry")
    assert payload["negative_prompt"] == "blurry"


def test_mai_models_flagged_all_four():
    # 平台 marshal 實測：MAI 重建請求只留 model/prompt/size/n（generations 與 edits
    # 同一個函式）——六筆條目（t2i＋i2i）四旗標都要在，少一顆 UI 就會騙人
    for m in app.MODELS["image"]:
        if m["id"].startswith("MAI-Image"):
            for flag in ("no_seed", "no_negative_prompt", "no_watermark", "no_prompt_extend"):
                assert m.get(flag), m["name"] + " 缺 " + flag


def test_official_doc_flags_wan27_image_and_happyhorse():
    # 📄官方（reference §2.3.5/2.3.6/2.3.10b，2026-08-25）：wan2.7 圖像不支援
    # prompt_extend/negative_prompt；happyhorse 四接口皆無 prompt_extend。
    # 旗標少一顆，UI 就會顯示一個「填了沒作用」的控制項。
    for m in app.MODELS["image"]:
        if m["id"].startswith("wan2.7-image"):
            assert m.get("no_negative_prompt") and m.get("no_prompt_extend"), m["name"]
    for m in app.MODELS["video"]:
        if m["id"].startswith("happyhorse"):
            assert m.get("no_prompt_extend"), m["name"]


# ── 阿里影片解析度白名單與 no_ratio（閘道 PR #64 起顯式非法值 422）─────────

def test_ali_video_resolution_whitelists_match_official():
    """依 reference/api-params-official.md §2.3.13~2.3.16：wan2.6/2.7 全接口與
    happyhorse-1.0 系只有 720P/1080P（1.0 的 480P 由閘道刻意擋下）；
    happyhorse-1.1 系才有 480P。閘道 2026-08-28 起對非法檔位回 422，
    這條鎖住 MODELS 白名單不被之後的新條目照「480/720/1080 直覺」誤加。"""
    import app
    two_tier = {
        "wan2.7-t2v", "wan2.7-i2v", "wan2.7-r2v", "wan2.7-videoedit",
        "wan2.6-t2v", "wan2.6-i2v", "wan2.6-i2v-flash", "wan2.6-r2v", "wan2.6-r2v-flash",
        "happyhorse-1.0-t2v", "happyhorse-1.0-i2v", "happyhorse-1.0-r2v",
        "happyhorse-1.0-video-edit",
    }
    three_tier = {"happyhorse-1.1-t2v", "happyhorse-1.1-i2v", "happyhorse-1.1-r2v"}
    seen = set()
    for m in app.MODELS["video"]:
        if m["id"] in two_tier:
            assert m.get("resolutions") == ["720P", "1080P"], m["id"]
            seen.add(m["id"])
        elif m["id"] in three_tier:
            assert m.get("resolutions") == ["480P", "720P", "1080P"], m["id"]
            seen.add(m["id"])
    assert seen == two_tier | three_tier


def test_video_no_ratio_set():
    """官方沒有 ratio 的接口（wan2.7-i2v／happyhorse i2v／happyhorse-1.0-video-edit）
    必須標 no_ratio：UI 藏下拉、後端剝除，顯式送出會被閘道 422。"""
    import app
    assert app._VIDEO_NO_RATIO == {
        ("wan2.7-i2v", "i2v"),
        ("happyhorse-1.1-i2v", "i2v"),
        ("happyhorse-1.0-i2v", "i2v"),
        ("happyhorse-1.0-video-edit", "vedit"),
    }


# ── 百煉 Chat 方言四參數的適用集合（閘道 PR #65 起透傳；reference §2.3.24）─────

def test_bailian_dialect_param_sets():
    """適用清單鎖定：thinking_budget 對 kimi/kimi-k3 不開（官方明列不支援）、
    clear_thinking 僅 GLM、preserve_thinking 僅 qwen3.7/3.6 系（不含 GLM 與 3.8）、
    repetition_penalty 僅百煉文字家族（Claude/GPT/Gemini/Grok/Seed 都不送）。"""
    assert "kimi/kimi-k3" not in app._TEXT_THINKING_BUDGET
    assert "glm-5.2" in app._TEXT_THINKING_BUDGET
    assert "qwen3.8-max" in app._TEXT_THINKING_BUDGET
    assert app._TEXT_CLEAR_THINKING == {"glm-5.1", "glm-5.2"}
    assert app._TEXT_PRESERVE_THINKING == {
        "qwen3.7-max", "qwen3.7-plus", "qwen3.6-max-preview", "qwen3.6-plus", "qwen3.6-flash"}
    assert "glm-5.2" not in app._TEXT_PRESERVE_THINKING
    assert "kimi/kimi-k3" in app._TEXT_REPETITION_PENALTY
    for outsider in ("claude-opus-5", "gpt-5.5", "grok-4.3", "dola-seed-2.1-turbo"):
        assert outsider not in app._TEXT_REPETITION_PENALTY, outsider


def test_text_top_p_is_optional():
    """top_p 未帶＝不送（None），不要在請求層補預設——閘道曾把未帶硬補成 0.001，
    修掉後「沒填就不送」才是正確語意，這條防止之後有人把預設值加回來。"""
    req = app.TextGenerateRequest(model="qwen3.5-flash", prompt="hi")
    assert req.top_p is None


def test_drift_collects_every_model_category_but_not_voices():
    """`probe_model.collect_model_ids()`：**每個分類都要收到、音色不能收**。

    對應兩個實際發生過的 bug（2026-09-02 同一次修改的前後兩版），兩個都是這支
    「偵測清單漂移」的工具**自己漂移**：
      1. 原本寫死 text/image/video/muleai ＋ voice.asr/voice.tts，**漏了
         voice.realtime 與 voice.music**，那 7 顆明明有收錄卻被報成「閘道有、我們
         沒收錄」（實際 16 顆報成 23 顆）——會催人去補上根本已經在的東西。
      2. 改成遞迴走訪後**收過頭**：TTS 底下的音色也長成 {"id","name","desc"}，
         loongjohn／longanlingxin 這些音色 id 被當成模型，變成一堆假警報。
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "probe_model", Path(__file__).resolve().parent.parent / "scripts" / "probe_model.py")
    probe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(probe)

    ours = probe.collect_model_ids(app.MODELS)

    # 每個分類至少要有一顆被收到（含最容易被漏掉的 voice 子分類）
    for cat in ("text", "image", "video", "muleai"):
        assert any(m["id"] in ours for m in app.MODELS[cat]), cat
    for sub in app.MODELS["voice"]:
        assert any(m["id"] in ours for m in app.MODELS["voice"][sub]), f"voice.{sub}"

    # 音色不能被當成模型
    voice_ids = {v["id"] for m in app.MODELS["voice"]["tts"] for v in m.get("voices", [])}
    assert voice_ids, "測試前提不成立：TTS 模型底下沒有音色"
    assert not (voice_ids & ours), sorted(voice_ids & ours)[:5]


# ─────────────────────────────────────────────────────────────────────────────
# scripts/blender_greybox_export.py：與 canvas.js 的灰模 DSL 是同一套語法，兩邊各有
# 一份解析器（JS 跑在瀏覽器、Python 跑在 Blender）。2026-09-03 加 figure／move to／
# camera 行時兩邊一起改；這裡鎖住「Python 版讀得懂 JS 版會產出的每一種行」，以及
# 兩邊共用的常數表（比例、量體參數數量）與運鏡公式沒有漂移。
import re as _re

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))
import blender_greybox_export as gb  # noqa: E402

_CANVAS_JS = (Path(__file__).resolve().parent.parent / "static/js/canvas.js").read_text()


def test_greybox_spec_parser_handles_every_line_kind():
    spec = """# move: orbit_left
# distance: 20
box 6 10 6 at -8 5 -14
figure 1.7 at -5 0 -10 move to 5 0 -10
cyl 0.4 2 at -2 1 -4 rot 30
camera from 0 3 14 to 0 2 4 look 0 1.2 -10
box 1 2 at 0 0 0
cone 1 2 at 0 x 0
"""
    shapes, cam, settings, errors = gb.parse_spec(spec)
    assert [s["kind"] for s in shapes] == ["box", "figure", "cyl"]
    assert shapes[1]["to"] == [5.0, 0.0, -10.0]
    assert shapes[2]["rot"] == 30.0 and shapes[2]["to"] is None
    assert cam == {"from": [0.0, 3.0, 14.0], "to": [0.0, 2.0, 4.0], "look": [0.0, 1.2, -10.0]}
    assert settings == {"move": "orbit_left", "distance": "20"}
    assert len(errors) == 2   # box 少一個維度、cone 座標不是數字


def test_greybox_constants_match_canvas_js():
    js_ratios = dict(_re.findall(r"'(\d+:\d+)': \[(\d+, \d+)\]", _CANVAS_JS.split("const GREYBOX_RATIOS")[1].split("};")[0]))
    assert {k: f"{w}, {h}" for k, (w, h) in gb.RATIOS.items()} == js_ratios
    js_dims = _re.search(r"const GREYBOX_DIMS = \{([^}]+)\}", _CANVAS_JS).group(1)
    assert dict(_re.findall(r"(\w+): (\d)", js_dims)) == {k: str(v) for k, v in gb.DIMS.items()}


def test_greybox_camera_moves_match_canvas_js_endpoints():
    # 與 canvas.js GREYBOX_MOVES 手算對照：t=0 與 t=1 的相機位置
    d, h = 14, 3
    assert gb.camera_move("dolly_in", 0, d, h)[0] == [0, h, d]
    assert gb.camera_move("dolly_in", 1, d, h)[0] == [0, h, pytest.approx(d * 0.35)]
    assert gb.camera_move("push_through", 1, d, h)[0][2] == pytest.approx(d - d * 1.8)
    assert gb.camera_move("pan_right", 1, d, h)[1][0] == pytest.approx(d * 0.6)
    assert gb.camera_move("orbit_left", 1, d, h)[0][0] == pytest.approx(-d * (2 ** 0.5) / 2)
    with pytest.raises(ValueError):
        gb.camera_move("nope", 0, d, h)


def test_greybox_to_blender_is_proper_rotation():
    # three.js（Y 上、看 -Z）→ Blender（Z 上、看 +Y）：主體在我們的 -Z 要落在 Blender 的 +Y
    assert gb.to_blender([1, 2, -3]) == (1, 3, 2)
    # 行列式 +1 → rot 角度不用反號
    import itertools
    m = [list(gb.to_blender(e)) for e in ([1, 0, 0], [0, 1, 0], [0, 0, 1])]
    det = sum(
        (1 if (perm in ((0, 1, 2), (1, 2, 0), (2, 0, 1))) else -1) * m[0][perm[0]] * m[1][perm[1]] * m[2][perm[2]]
        for perm in itertools.permutations(range(3))
    )
    assert det == 1
