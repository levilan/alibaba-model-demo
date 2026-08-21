// 模型介紹資料 —— AI Canvas 節點選模型時的彈出介紹視窗用。
//
// 特色文案以「模型家族」為單位手寫（這是客戶看得到的文案，遵守 CLAUDE.md 的規範：
// 只描述能力、不寫內部術語/價格比較/驗證狀態）；型號層級的規格（尺寸、時長、
// 張數上限……）由 canvas.js 從 /api/models 回傳的資料動態產生，這裡不重複。
//
// 規則順序重要：前綴較特定的（qwen-image / qwen-audio / dola-seedream…）
// 必須排在較泛用的（qwen / dola-seed…）前面，否則會被泛用規則先攔走。
(function () {
    'use strict';

    const FAMILIES = [
        // ── NenAI Spicy 專區（含換臉）──
        {
            match: /^(wan2\.7-i2v-spicy|z-image-spicy|qwen-image-edit-spicy|face-swap)/,
            name: 'NenAI Spicy 系列',
            points: [
                '創作自由度更高的生成模型專區',
                '涵蓋圖片生成、圖像編輯、換臉與影片生成',
            ],
        },

        // ── 語音 ──
        {
            match: /^qwen-audio/,
            name: '通義千問語音（Qwen Audio）',
            points: [
                '涵蓋語音合成（TTS）與語音辨識（ASR）的語音家族',
                '多種擬真音色，支援中文與英文',
            ],
        },
        {
            match: /^lyria/,
            name: 'Lyria（Google DeepMind）',
            points: [
                'Google DeepMind 的音樂生成家族',
                '以文字描述生成器樂音樂',
            ],
        },
        {
            match: /^gemini-.*tts/,
            name: 'Gemini 語音合成',
            points: [
                'Google Gemini 家族的語音合成模型',
                '聲線自然，支援多語言朗讀',
            ],
        },

        // ── 圖片 ──
        {
            match: /^qwen-image/,
            name: '千問圖像（Qwen-Image）',
            points: [
                '以文字渲染見長——海報、招牌等中英文字可直接入畫',
                '同系列同時提供文生圖與圖像編輯',
            ],
        },
        {
            match: /^(wan2\.6-t2i|wan2\.[67]-image)/,
            name: '通義萬相圖像（Wan）',
            points: [
                '通義萬相家族的圖像生成與編輯',
                '多張參考圖融合、風格遷移',
            ],
        },
        {
            match: /^z-image/,
            name: 'Z-Image',
            points: [
                '輕量級快速文生圖',
                '適合快速出圖與大量嘗試',
            ],
        },
        {
            match: /^MAI-Image/,
            name: 'Microsoft AI Image（MAI）',
            points: [
                'Microsoft AI 的圖像生成與編輯家族',
                '寫實光影與細節表現',
            ],
        },
        {
            match: /^gpt-image/,
            name: 'GPT Image（OpenAI）',
            points: [
                'OpenAI 的圖像生成與編輯家族',
                '提示詞理解精準，可用文字描述直接修改圖片',
            ],
        },
        {
            match: /^dola-seedream/,
            name: 'Seedream（字節跳動）',
            points: [
                '字節跳動 Seed 的圖像生成家族',
                '大畫面高解析度輸出',
            ],
        },
        {
            match: /^(gemini-.*image|gemini-3-pro-image)/,
            name: 'Gemini 圖像（Google）',
            points: [
                'Google Gemini 的圖像生成與編輯',
                '擅長依對話逐步修圖與多圖組合',
            ],
        },

        // ── 影片 ──
        {
            match: /^wan2\.2-animate/,
            name: '萬相動作動畫（Wan Animate）',
            points: [
                '視頻換人：把影片中的角色換成你的人物圖片，保留原場景與動作',
                '圖生動作：把參考影片的動作與表情遷移到人物圖片',
            ],
        },
        {
            match: /^(wan3\.0-video|wan2\.[67]-(t2v|i2v|r2v)|wan2\.7-videoedit)/,
            name: '通義萬相影片（Wan）',
            points: [
                '文生、圖生、參考生與影片編輯全系列',
                '可自動配音，畫面與聲音一次生成',
            ],
        },
        {
            match: /^happyhorse/,
            name: 'HappyHorse',
            points: [
                '高還原度影片生成家族，畫面貼合提示詞',
                '支援文生、圖生與多圖參考',
            ],
        },
        {
            match: /^veo/,
            name: 'Veo（Google DeepMind）',
            points: [
                'Google DeepMind 旗艦影片生成',
                '原生配音——對白、音效與配樂隨影片一次生成',
                '電影感運鏡與高畫質',
            ],
        },
        {
            match: /^(dreamina-seedance|bytedance-seedance)/,
            name: 'Seedance（字節跳動）',
            points: [
                '字節跳動 Seed 的影片生成家族',
                '動作流暢、多鏡頭敘事',
            ],
        },
        {
            match: /^gemini-omni/,
            name: 'Gemini 影音（Google）',
            points: [
                'Gemini 家族的影音生成模型',
                '影片長度與畫面由模型依內容自動決定',
            ],
        },

        // ── 文字（放在最後，前綴最泛用）──
        {
            match: /^qwen/,
            name: '通義千問（Qwen）',
            points: [
                '通義千問大語言模型家族',
                '涵蓋旗艦推理、均衡、極速，以及代碼、視覺語言、角色扮演等特化型號',
                '長上下文與多語言支援',
            ],
        },
        {
            match: /^deepseek/,
            name: 'DeepSeek',
            points: [
                '以深度推理見長的模型家族',
                '數學與程式推導表現突出',
            ],
        },
        {
            match: /^glm/,
            name: 'GLM（智譜）',
            points: [
                '智譜 AI 的 GLM 家族',
                '中文理解出色，支援超長上下文',
            ],
        },
        {
            match: /^dola-seed/,
            name: '豆包 Seed（字節跳動）',
            points: [
                '字節跳動豆包大模型家族',
                '對話自然，中文口語表現佳',
            ],
        },
        {
            match: /^kimi/,
            name: 'Kimi（月之暗面）',
            points: [
                '以長文本理解見長，支援百萬字上下文',
                '深度推理並支援看圖',
            ],
        },
        {
            match: /^claude/,
            name: 'Claude（Anthropic）',
            points: [
                'Anthropic 的 Claude 家族',
                '以寫作品質與程式能力著稱，推理沉穩',
            ],
        },
        {
            match: /^gpt-5/,
            name: 'GPT（OpenAI）',
            points: [
                'OpenAI 的 GPT 家族',
                '通用能力均衡，生態成熟',
            ],
        },
        {
            match: /^grok/,
            name: 'Grok（xAI）',
            points: [
                'xAI 的 Grok 模型家族',
                '提供推理型與極速型兩條產品線，依需求選擇',
            ],
        },
        // gemini 文字放在 gemini-*-tts / gemini-*-image / gemini-omni 之後
        {
            match: /^gemini/,
            name: 'Gemini（Google）',
            points: [
                'Google 的原生多模態家族',
                '文字與圖片理解一體，回應速度快',
            ],
        },
    ];

    window.NENAI_MODEL_FAMILY = function (modelId) {
        for (const f of FAMILIES) {
            if (f.match.test(modelId)) return f;
        }
        return null;
    };
})();
