/**
 * NenAI Testing Platform
 * Frontend JS — API Key auth, SSE streaming, polling
 */

// ── Lightbox ──────────────────────────────────────────────────
function openLightbox(src, type) {
    const img = document.getElementById('lightboxImg');
    const vid = document.getElementById('lightboxVid');
    if (type === 'video') {
        img.style.display = 'none';
        vid.src = src; vid.style.display = '';
    } else {
        vid.pause(); vid.src = ''; vid.style.display = 'none';
        img.src = src; img.style.display = '';
    }
    document.getElementById('lightbox').classList.add('open');
}
function closeLightbox() {
    const vid = document.getElementById('lightboxVid');
    if (vid) { vid.pause(); vid.src = ''; }
    document.getElementById('lightbox')?.classList.remove('open');
}
document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── State ─────────────────────────────────────────────────────
let apiKey = sessionStorage.getItem('nenai_api_key') || '';
let models = { text: [], image: [], video: [], muleai: [], voice: { asr: [], tts: [] } };
let pricingMap = {}; // model id -> {type:'token', input, output} 或 {type:'fixed', price}，僅供參考
// 文字生成多輪對話歷史，[{role, content}]，隨對話累積、清除對話時清空。
// 注意這裡一律累積，「記住上下文」開關（#textRememberContext）只決定送出時要不要帶上，
// 不會影響累積本身——關掉再打開能接著先前的內容繼續。
// 每一輪都必須重送完整歷史，所以 prompt_tokens 是累加的：聊到第 N 輪要付前 N-1 輪的錢，
// 總花費大致隨輪數平方成長。那個開關存在的主要理由就是這個。
let textChatHistory = [];
let sessionCost = 0; // 本次瀏覽器分頁累積的估計花費（USD），僅供參考、重新整理後歸零
let refFiles = [];
let editRefFiles = [];  // for video editing reference images
let imgRefFiles = [];   // for image edit reference images (up to 9)
let muleaiImgRefFiles = [];
let loadingTimerInterval = null;


// ── History Persistence ───────────────────────────────────────
const TaskHistory = {
    save(type, model, prompt, url) {
        try {
            const hist = JSON.parse(localStorage.getItem('ai_tester_history') || '[]');
            if (hist.length > 0 && hist[0].url === url) return;
            hist.unshift({ type, model, prompt, url, ts: Date.now() });
            localStorage.setItem('ai_tester_history', JSON.stringify(hist.slice(0, 50)));
        } catch(e) { console.error('History save error', e); }
    },
    load() {
        try {
            const hist = JSON.parse(localStorage.getItem('ai_tester_history') || '[]');
            hist.reverse().forEach(item => {
                if (item.type === 'video') addVideoResult(item.model, item.prompt, item.url, true);
            });
        } catch(e) { console.error('History load error', e); }
    }
};

function addMuleAIVideoResult(model, prompt, src, isHistory = false) {
    const cont = document.getElementById('muleaiVideoResults');
    if (cont) {
        const empty = cont.querySelector('.empty-state');
        if (empty) empty.remove();
        const card = el('div', { className: 'video-task-card' });
        card.innerHTML = '<div class="vtc-header"><span class="vtc-model">' + model + '</span><span class="vtc-status succeeded">SUCCEEDED</span></div><div class="vtc-prompt">' + prompt.substring(0, 120) + '</div><video class="video-player" controls src="' + src + '"></video><div class="video-card-actions"><a href="' + src + '" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a><button class="btn btn-ghost btn-sm" onclick="openLightbox(\'' + src + '\', \'video\')">展開預覽</button></div>';
        cont.insertBefore(card, cont.firstChild);
    }
}

function addMuleAIImageResult(model, prompt, src, isHistory = false) {
    const cont = document.getElementById('muleaiVideoResults');
    if (cont) {
        const empty = cont.querySelector('.empty-state');
        if (empty) empty.remove();
        const card = el('div', { className: 'video-task-card' });
        card.innerHTML = '<div class="vtc-header"><span class="vtc-model">' + model + '</span><span class="vtc-status succeeded">SUCCEEDED</span></div><div class="vtc-prompt">' + prompt.substring(0, 120) + '</div><img src="' + src + '" alt="Generated Image" class="muleai-img-result" onclick="openLightbox(\'' + src + '\')"><div class="video-card-actions"><a href="' + src + '" download target="_blank" rel="noopener noreferrer" class="img-dl">下載圖片</a></div>';
        cont.insertBefore(card, cont.firstChild);
    }
}

// ── Omni Voice Map ────────────────────────────────────────────
const OMNI_VOICE_MAP = {
    'qwen3.5-omni': {
        default: 'Tina',
        groups: [
            { label: '通用普通話', voices: [
                ['Tina',       'Tina 甜甜 — 甜美暖心'],
                ['Cindy',      'Cindy 林欣宜 — 台灣嗲嗲'],
                ['Liora Mira', 'Liora Mira 清歡 — 烟火溫柔'],
                ['Sunnybobi',  'Sunnybobi 知芝 — 大咧咧'],
                ['Raymond',    'Raymond 林川野 — 宅男清亮'],
                ['Ethan',      'Ethan 晨煦 — 陽光活力'],
                ['Theo Calm',  'Theo Calm 予安 — 療癒靜默'],
                ['Serena',     'Serena 蘇瑤 — 溫柔小姐姐'],
                ['Harvey',     'Harvey 厚 — 低沉歲月感'],
                ['Maia',       'Maia 四月 — 知性溫柔'],
                ['Evan',       'Evan 江晨 — 年下奶狗'],
                ['Qiao',       'Qiao 小喬妹 — 台灣甜妹個性'],
                ['Momo',       'Momo 茉兔 — 撒嬌搞怪'],
                ['Wil',        'Wil 偉倫 — 港台腔小哥'],
                ['Angel',      'Angel 安琪 — 台式口音甜美'],
                ['Li Cassian', 'Li Cassian 李公公 — 察言觀色'],
                ['Mia',        'Mia 舒然 — 慢生活博主'],
                ['Joyner',     'Joyner 阿逗 — 搞笑接地氣'],
                ['Gold',       'Gold 金爺 — Rapper'],
                ['Katerina',   'Katerina 卡捷琳娜 — 御姐韻律'],
                ['Ryan',       'Ryan 甜茶 — 戲感張力'],
                ['Jennifer',   'Jennifer 詹妮弗 — 電影質感美語'],
                ['Aiden',      'Aiden 艾登 — 廚藝大男孩'],
                ['Mione',      'Mione 敏兒 — 英式知性'],
                ['Roya',       'Roya 蘿雅 — 熱愛運動'],
            ]},
            { label: '方言', voices: [
                ['Sunny',        'Sunny 四川晴兒 — 甜川妹'],
                ['Dylan',        'Dylan 北京曉東 — 北京少年'],
                ['Eric',         'Eric 四川程川 — 成都男子'],
                ['Peter',        'Peter 天津李彼得 — 相聲捧哏'],
                ['Joseph Chen',  'Joseph Chen 阿樸伯 — 閩南老華僑'],
                ['Marcus',       'Marcus 陝西秦川 — 老陝沉聲'],
                ['Li',           'Li 南京老李 — 罵罵咧咧'],
                ['Kiki',         'Kiki 粵語阿清 — 港妹閨蜜'],
                ['Rocky',        'Rocky 粵語阿強 — 幽默在線陪聊'],
            ]},
            { label: '國際', voices: [
                ['Sohee',      'Sohee 素熙 — 韓國歐尼'],
                ['Lenn',       'Lenn 萊恩 — 德國叛逆青年'],
                ['Ono Anna',   'Ono Anna 小野杏 — 日本鬼靈精'],
                ['Sonrisa',    'Sonrisa 索尼莎 — 拉美熱情大姐'],
                ['Bodega',     'Bodega 博德加 — 西班牙大叔'],
                ['Emilien',    'Emilien 埃米爾安 — 法國浪漫'],
                ['Andre',      'Andre 安德雷 — 磁性沉穩'],
                ['Radio Gol',  'Radio Gol — 葡語足球詩人'],
                ['Alek',       'Alek 阿列克 — 俄羅斯冷暖'],
                ['Rizky',      'Rizky 阿力 — 印尼個性青年'],
                ['Arda',       'Arda 阿爾達 — 土耳其溫潤'],
                ['Hana',       'Hana 阿幸 — 越南成熟姐姐'],
                ['Dolce',      'Dolce 多爾切 — 義大利慵懶'],
                ['Jakub',      'Jakub 雅克 — 波蘭磁性'],
                ['Griet',      'Griet 海娜 — 荷蘭文藝'],
                ['Eliška',     'Eliška 艾莉卡 — 捷克匠心'],
                ['Marina',     'Marina 瑪麗娜 — 多元文化'],
                ['Siiri',      'Siiri 西芮 — 芬蘭舒緩'],
                ['Ingrid',     'Ingrid 林恩 — 挪威鄉村'],
                ['Sigga',      'Sigga 海娜 — 冰島知性'],
                ['Bea',        'Bea 雅娜 — 菲律賓甜甜'],
                ['Chloe',      'Chloe 思怡 — 馬來西亞白領'],
            ]},
        ]
    },
    'qwen2.5-omni': {
        default: 'Ethan',
        groups: [
            { label: '音色', voices: [
                ['Ethan',   'Ethan 晨煦 — 男・陽光'],
                ['Chelsie', 'Chelsie 千雪 — 女・二次元'],
            ]}
        ]
    }
};

function updateOmniVoices(model) {
    const sel = document.getElementById('omniVoice');
    if (!sel) return;
    const key = model.includes('qwen2.5-omni') ? 'qwen2.5-omni' : 'qwen3.5-omni';
    const data = OMNI_VOICE_MAP[key];
    const prev = sel.value;
    sel.innerHTML = '';
    data.groups.forEach(g => {
        const og = document.createElement('optgroup');
        og.label = g.label;
        g.voices.forEach(([v, label]) => {
            const opt = document.createElement('option');
            opt.value = v;
            opt.textContent = label;
            og.appendChild(opt);
        });
        sel.appendChild(og);
    });
    // 保留原選擇，否則用預設
    const exists = [...sel.options].some(o => o.value === prev);
    sel.value = exists ? prev : data.default;
}

// ── Theme（淺色 / 深色 / 自動）───────────────────────────────────
const THEME_STORAGE_KEY = 'nenai_theme_pref';
const darkMediaQuery = window.matchMedia ? window.matchMedia('(prefers-color-scheme: dark)') : null;

function getThemePref() {
    return localStorage.getItem(THEME_STORAGE_KEY) || 'auto';
}

function resolveEffectiveTheme(pref) {
    if (pref === 'auto') return darkMediaQuery && darkMediaQuery.matches ? 'dark' : 'light';
    return pref;
}

function applyTheme() {
    const pref = getThemePref();
    const effective = resolveEffectiveTheme(pref);
    document.documentElement.setAttribute('data-theme', effective);

    document.getElementById('themeToggleIconSun').style.display = effective === 'dark' ? 'none' : '';
    document.getElementById('themeToggleIconMoon').style.display = effective === 'dark' ? '' : 'none';

    document.querySelectorAll('.theme-menu-item').forEach(btn => {
        btn.classList.toggle('active', btn.dataset.themeChoice === pref);
    });
}

function setThemePref(pref) {
    localStorage.setItem(THEME_STORAGE_KEY, pref);
    applyTheme();
    document.getElementById('themeMenu').classList.add('hidden');
}

function toggleThemeMenu() {
    document.getElementById('themeMenu').classList.toggle('hidden');
}

document.addEventListener('click', e => {
    const switcher = document.querySelector('.theme-switcher');
    if (switcher && !switcher.contains(e.target)) {
        document.getElementById('themeMenu')?.classList.add('hidden');
    }
});

if (darkMediaQuery) {
    darkMediaQuery.addEventListener('change', () => {
        if (getThemePref() === 'auto') applyTheme();
    });
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    applyTheme();
    initLoginFx();
    if (apiKey) attemptAutoLogin();
    updateOmniVoices(document.getElementById('omniModel')?.value || 'qwen3.5-omni-flash-realtime');

    document.getElementById('apiKeyInput').addEventListener('keydown', e => {
        if (e.key === 'Enter') handleLogin();
    });
        document.getElementById('textPrompt').addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') sendText();
    });
});

async function attemptAutoLogin() {
    try {
        const res = await fetch('/api/models', { headers: authHeader() });
        if (res.ok) {
            models = await res.json();
            showApp();
        } else {
            apiKey = '';
            sessionStorage.removeItem('nenai_api_key');
        }
    } catch (_) { /* show login */ }
}

// ── 登入頁裝飾檯燈：純裝飾互動，跟登入邏輯無關，點一下切換開/關 ──────────
// 拉燈＝切換淺色/深色模式（深色＝燈亮）。直接寫死 light/dark 兩個選項，
// 不接「自動」——拉繩開關本身就是一個明確的二元動作，銜接自動模式的語意
// 反而會讓「拉一下卻沒變」的情況發生（系統剛好已經是那個主題時）。
function toggleLoginLamp() {
    const lamp = document.getElementById('loginLamp');
    if (!lamp) return;
    const isDark = resolveEffectiveTheme(getThemePref()) === 'dark';
    setThemePref(isDark ? 'light' : 'dark');
    const cord = lamp.querySelector('.lamp-cord');
    // 每次點擊都要重新播放「往下拉」的動畫——class 已經在身上時直接再 add
    // 一次不會觸發 animation，要先移除、等下一個動畫幀再加回去強制重播
    cord.classList.remove('pulling');
    requestAnimationFrame(() => cord.classList.add('pulling'));
}

// 登入頁背景動態效果的粒子——深色模式的星星、淺色模式的暖色光塵。兩組都
// 常駐在 DOM 裡，用 CSS 的 [data-theme] 選擇器切換顯示/隱藏，切換主題時
// 不用重新產生，位置/延遲用 CSS 變數帶隨機值，只需要產生一次
function initLoginFx() {
    const container = document.getElementById('loginFxParticles');
    if (!container || container.childElementCount) return; // 已經產生過就跳過
    const frag = document.createDocumentFragment();
    for (let i = 0; i < 40; i++) {
        const star = document.createElement('span');
        star.className = 'fx-star';
        const size = 1 + Math.random() * 1.8;
        star.style.width = star.style.height = size + 'px';
        star.style.left = Math.random() * 100 + '%';
        star.style.top = Math.random() * 70 + '%';
        star.style.setProperty('--fx-dur', (2 + Math.random() * 3).toFixed(2) + 's');
        star.style.setProperty('--fx-delay', (Math.random() * 4).toFixed(2) + 's');
        frag.appendChild(star);
    }
    for (let i = 0; i < 16; i++) {
        const mote = document.createElement('span');
        mote.className = 'fx-mote';
        const size = 4 + Math.random() * 7;
        mote.style.width = mote.style.height = size + 'px';
        mote.style.left = Math.random() * 100 + '%';
        mote.style.top = 40 + Math.random() * 55 + '%';
        mote.style.setProperty('--fx-dur', (6 + Math.random() * 6).toFixed(2) + 's');
        mote.style.setProperty('--fx-delay', (Math.random() * 6).toFixed(2) + 's');
        frag.appendChild(mote);
    }
    // 淺色模式的雲：只放 3 片、走得很慢（90 秒以上跨過整個畫面），
    // 目的是讓白天的天空「有在動」但不會吸引注意力離開登入表單
    // 只給寬度，高度由 CSS 的 aspect-ratio 自動換算，維持雲的形狀比例
    // top 控制在 4%～23%：再往下就會飄到檯燈/登入卡片的高度，跟近景的物件
    // 打架（雲應該只出現在遠景的天空區域）
    const clouds = [
        { top: 13, w: 210, dur: 96,  delay: -20  },
        { top: 23, w: 150, dur: 132, delay: -70  },
        { top: 4,  w: 116, dur: 112, delay: -110 },
        { top: 18, w: 178, dur: 148, delay: -46  },
    ];
    clouds.forEach(c => {
        const cloud = document.createElement('span');
        cloud.className = 'fx-cloud';
        cloud.style.top = c.top + '%';
        // 寬度透過 CSS 變數傳遞（而不是直接設 style.width），CSS 才有辦法
        // 在窄螢幕的 media query 裡等比例縮小——inline style 的優先權比
        // media query 高，直接設 width 的話就得靠 !important 才蓋得掉
        cloud.style.setProperty('--fx-w', c.w + 'px');
        cloud.style.setProperty('--fx-dur', c.dur + 's');
        cloud.style.setProperty('--fx-delay', c.delay + 's');
        frag.appendChild(cloud);
    });
    container.appendChild(frag);
}

// ── Auth ──────────────────────────────────────────────────────
// 登入鎖定倒數計時器——後端依 IP 記錄失敗次數，超過 5 次會回 429 + retry_after
// 秒數，這裡在鎖定期間停用登入按鈕並即時倒數，避免使用者一直狂點碰壁
let _loginLockoutTimer = null;

function _startLoginLockoutCountdown(seconds) {
    const btn = document.getElementById('loginBtn');
    const errEl = document.getElementById('loginError');
    clearInterval(_loginLockoutTimer);
    let remaining = seconds;
    const render = () => {
        btn.disabled = true;
        btn.innerHTML = `<span>請 ${remaining} 秒後再試</span>`;
        errEl.textContent = `登入失敗次數過多，已暫時鎖定，請 ${remaining} 秒後再試`;
    };
    render();
    _loginLockoutTimer = setInterval(() => {
        remaining -= 1;
        if (remaining <= 0) {
            clearInterval(_loginLockoutTimer);
            btn.disabled = false;
            errEl.textContent = '';
            btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4M10 17l5-5-5-5M15 12H3"/></svg><span>登入</span>';
            return;
        }
        render();
    }, 1000);
}

async function handleLogin() {
    const key = document.getElementById('apiKeyInput').value.trim();
    const errEl = document.getElementById('loginError');
    errEl.textContent = '';

    if (!key) { errEl.textContent = '請輸入 NenAI API Key'; return; }

    const btn = document.getElementById('loginBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg><span>驗證中...</span>';

    try {
        const res = await fetch('/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ api_key: key }),
        });
        const data = await res.json();
        if (data.success) {
            apiKey = key;
            sessionStorage.setItem('nenai_api_key', key);
            const mRes = await fetch('/api/models', { headers: authHeader() });
            models = await mRes.json();
            showApp();
            return;
        }
        if (data.locked && data.retry_after) {
            _startLoginLockoutCountdown(data.retry_after);
            return;
        }
        errEl.textContent = data.message || '驗證失敗，請確認 API Key';
    } catch (e) {
        errEl.textContent = '網路錯誤，請重試';
    }
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4M10 17l5-5-5-5M15 12H3"/></svg><span>登入</span>';
}

function showApp() {
    document.getElementById('loginOverlay').style.display = 'none';
    const app = document.getElementById('mainApp');
    app.classList.remove('hidden');
    app.style.display = 'flex';
    const masked = apiKey.slice(0, 6) + '****' + apiKey.slice(-4);
    document.getElementById('apiKeyLabel').textContent = masked;
    try { populateSelectors(); TaskHistory.load(); resumePendingTasks(); } catch(e) { toast('UI 載入發生錯誤，請聯絡開發者', 'error'); }
    loadPricing();
}

// 價格是輔助參考資訊，非同步背景載入即可，失敗也不影響主要功能——載入完成後
// 重新跑一次 populateSelectors() 讓已經產生的選單補上價格顯示
async function loadPricing() {
    try {
        const res = await fetch('/api/pricing', { headers: authHeader() });
        if (res.ok) {
            pricingMap = await res.json();
            populateSelectors();
        }
    } catch (_) { /* 價格載入失敗就不顯示，不影響其他功能 */ }
}

function handleLogout() {
    apiKey = '';
    sessionStorage.removeItem('nenai_api_key');
    location.reload();
}

function authHeader() {
    return {
        'Authorization': 'Bearer ' + apiKey,
        'Content-Type': 'application/json',
    };
}

// ── Selectors ─────────────────────────────────────────────────
function onMuleaiModelChange() {
    const model = document.getElementById('muleaiModel').value;
    updateModelPriceHint('muleaiModelPrice', model);
    const isZImage   = model.includes('z-image');
    const isImgEdit  = model === 'qwen-image-edit-spicy';
    const isFaceSwap = model === 'face-swap';
    const isImageModel = isZImage || isImgEdit || isFaceSwap;
    const isVideoModel = !isImageModel;

    // 解析度 / 時長 / 圖片尺寸
    document.getElementById('muleaiVidResGroup').style.display  = isVideoModel ? '' : 'none';
    document.getElementById('muleaiImgResGroup').style.display  = isZImage     ? '' : 'none';
    document.getElementById('muleaiVidDurGroup').style.display  = isVideoModel ? '' : 'none';

    // 首幀 / 來源圖上傳區
    document.getElementById('muleaiImgUploadSection').style.display = (isVideoModel || isImgEdit || isFaceSwap) ? '' : 'none';

    const uploadTitle = document.getElementById('muleaiImgUploadTitle');
    if (uploadTitle) {
        if (isFaceSwap)     uploadTitle.textContent = '來源圖片 (必填)';
        else if (isImgEdit) uploadTitle.textContent = '來源圖片 (必填)';
        else                uploadTitle.textContent = '首幀圖片 (影片必填)';
    }

    // 換臉參考圖
    document.getElementById('muleaiFaceImgSection').style.display = isFaceSwap ? '' : 'none';

    // Prompt 區（face-swap 不需要）
    const promptSection = document.getElementById('muleaiPromptSection');
    if (promptSection) promptSection.style.display = isFaceSwap ? 'none' : '';
    document.getElementById('muleaiPromptExtendGroup').style.display = isFaceSwap ? 'none' : '';

    // 配音（僅影片）
    document.getElementById('muleaiAudioSection').style.display = isVideoModel ? '' : 'none';
    if (!isVideoModel) {
        const cb = document.getElementById('muleaiAudioEnable');
        if (cb) cb.checked = false;
        document.getElementById('muleaiAudioUploadSection').style.display = 'none';
    }

    const promptInput = document.getElementById('muleaiVidPrompt');
    if (promptInput) {
        if (isImgEdit)       promptInput.placeholder = '描述編輯效果（例：將人物改為紅髮）...';
        else if (isImageModel) promptInput.placeholder = '描述圖片畫面與細節...';
        else                   promptInput.placeholder = '描述影片動作與細節...';
    }

    const sendBtn = document.getElementById('muleaiVidSendBtn');
    if (sendBtn) {
        const label = isVideoModel ? '生成影片' : '生成圖片';
        sendBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> ' + label;
    }
}

function onMuleaiAudioToggle() {
    const enabled = document.getElementById('muleaiAudioEnable').checked;
    document.getElementById('muleaiAudioUploadSection').style.display = enabled ? '' : 'none';
    if (!enabled) clearMuleaiAudio();
}

function onMuleaiAudioChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    document.getElementById('muleaiAudioLabel').innerHTML = `<strong>${file.name}</strong><br><span style="font-size:11px;color:var(--text-muted)">${(file.size/1024).toFixed(0)} KB</span>`;
    document.getElementById('muleaiAudioIcon').textContent = '🎵';
    document.getElementById('muleaiAudioClearBtn').style.display = '';
}

function clearMuleaiAudio() {
    document.getElementById('muleaiAudioInput').value = '';
    document.getElementById('muleaiAudioLabel').innerHTML = '上傳音頻（選填）<br><span style="font-size:11px;color:var(--text-muted)">支援 WAV, MP3, OGG</span>';
    document.getElementById('muleaiAudioIcon').textContent = '🎵';
    document.getElementById('muleaiAudioClearBtn').style.display = 'none';
}

function populateSelectors() {
    populateSelect('textModel', models.text);
    populateSelect('muleaiModel', models.muleai || []);
    onTextModelChange();
    onImgTaskChange();
    onVidTaskChange();
    onMuleaiModelChange();
    onVoiceTaskChange();
}

// 「思考模式」（enable_thinking 布林值）跟 GPT 的「推理強度」（reasoning_effort
// 字串）是兩種互斥的機制：Qwen/DeepSeek/GLM 用前者，GPT 用後者，送錯機制給錯的
// 家族會直接 400（例如對 GPT 送 enable_thinking 會回 "Unknown parameter"）。
// Claude/Gemini 兩組都不支援（Claude 送了沒反應，Gemini 一律無條件思考關不掉），
// 所以兩個欄位都隱藏。
const _EFFORT_LABELS = {
    none: 'none（不推理）', minimal: 'minimal（最少）', low: 'low',
    medium: 'medium', high: 'high', xhigh: 'xhigh', max: 'max（最高）',
};

// 視覺語言模型帶入的圖片（data URI）。只附在當下這一輪的提問上，不進對話歷史
let textVisionImages = [];

function onTextVisionUpload(e) {
    const files = Array.from(e.target.files || []);
    Promise.all(files.map(f => new Promise(res => {
        const r = new FileReader();
        r.onload = () => res({ name: f.name, url: r.result });
        r.readAsDataURL(f);
    }))).then(added => {
        textVisionImages = [...textVisionImages, ...added];
        renderTextVisionList();
    });
    e.target.value = '';   // 讓同一個檔案能重複選取
}

function removeTextVisionImage(i) { textVisionImages.splice(i, 1); renderTextVisionList(); }

function renderTextVisionList() {
    document.getElementById('textVisionList').innerHTML = textVisionImages.map((f, i) => `
        <div class="ref-item">
            <span>${f.name}</span>
            <button onclick="removeTextVisionImage(${i})">✕</button>
        </div>`).join('');
}

function onTextModelChange() {
    const modelId = document.getElementById('textModel').value;
    const modelInfo = models.text.find(m => m.id === modelId) || {};
    document.getElementById('textThinkingGroup').style.display = modelInfo.thinking ? '' : 'none';

    // 圖片輸入只有視覺語言模型支援；切到不支援的模型時把已選的圖清掉，
    // 否則會靜默夾帶到不吃圖的模型上（那些模型會直接 400）
    const visionGroup = document.getElementById('textVisionGroup');
    if (visionGroup) {
        visionGroup.style.display = modelInfo.vision ? '' : 'none';
        if (!modelInfo.vision && textVisionImages.length) {
            textVisionImages = [];
            renderTextVisionList();
        }
    }
    document.getElementById('textReasoningEffortGroup').style.display = modelInfo.reasoning_effort ? '' : 'none';

    // 可用的推理強度各家族不同（GLM 5.2 有 minimal/max、GLM 5.1 沒有 max、GPT 兩者都沒有），
    // 送出不支援的值上游會直接 400，所以選項改成由 MODELS 的 reasoning_efforts 產生，
    // 不再寫死一份可能跟後端不同步的清單
    const efforts = modelInfo.reasoning_efforts || [];
    if (efforts.length) {
        const el = document.getElementById('textReasoningEffort');
        const cur = el.value;
        el.innerHTML = '<option value="">預設</option>' + efforts.map(v =>
            `<option value="${v}"${v === cur ? ' selected' : ''}>${_EFFORT_LABELS[v] || v}</option>`
        ).join('');
        if (cur && !efforts.includes(cur)) el.value = '';
    }
    updateModelPriceHint('textModelPrice', modelId);
}

// 價格資料來自網關自己的計費表（/api/pricing），只當參考用，不是精確帳單金額
// （已假設帳號分組倍率 group_ratio=1，實測目前所有分組確實都是 1）
// 後端不對價格做固定小數位 round（避免像語音辨識 $0.000035/次 這種極小值被
// 捨去顯示成 0，讓人誤以為免費），改成這裡依數值大小動態決定要顯示幾位小數——
// 至少抓到第一個非零小數位、再多留一位，取到合理的精度
// 顯示完整價格，不做四捨五入。原本 >= 0.01 的金額會被 round 到小數第二位，
// 導致 wan2.7 的 $0.075 顯示成 $0.08——使用者看到會以為這個平台比原廠貴，
// 是實際被回報過的問題。後端的 /api/pricing 本來就保留原始精度，捨入只發生在這裡。
function formatUsd(n) {
    if (!n || !isFinite(n)) return '0';
    // toPrecision(12) 的用途只是消掉浮點誤差（例如 0.1+0.2 會得到
    // 0.30000000000000004），12 位有效數字遠超過任何實際單價需要的精度，
    // 不會改變真正的數值
    const v = Number(n.toPrecision(12));
    // 小於 1e-6 時 String() 會輸出 "1e-7" 這種科學記號，展開成一般小數再去掉尾隨的 0
    if (Math.abs(v) < 1e-6) return v.toFixed(12).replace(/0+$/, '').replace(/\.$/, '');
    return String(v);
}

// ── 即時花費統計 ──────────────────────────────────────────────
// 只是根據網關自己的計費表（pricingMap）粗略估算，不是精確帳單；固定價格以外
// 的計費方式（例如少數以 token 計費的圖片模型）目前沒有對應資料，故不計入
function addCost(amount) {
    if (!amount || !isFinite(amount)) return;
    sessionCost += amount;
    const disp = document.getElementById('sessionCostDisplay');
    if (disp) disp.textContent = '本次花費：$' + formatUsd(sessionCost);
    const indicator = document.getElementById('sessionCostIndicator');
    if (indicator) {
        // 重新觸發 CSS animation：先移除 class 讓瀏覽器有機會回到起始狀態，
        // 下一輪動畫幀再加回去，否則連續呼叫時 class 已經在身上，animation 不會重播
        indicator.classList.remove('cost-flash');
        requestAnimationFrame(() => indicator.classList.add('cost-flash'));
    }
}

// 影片的計費入口：按次的用固定價，按 token 的（Seedance 系列）用解析度＋時長換算。
// 先前只呼叫 addFixedCost()，所以 8 個按 token 計費的影片模型完全沒被計入「本次花費」
// ── 昂貴任務的送出前確認 ────────────────────────────────────────────────────
// 影片單次最貴可以到 $6.94（Seedance 2.5 / 720P / 30 秒）。誤點一下就是實際扣款，
// 而且非同步任務一旦送出就無法取消。超過門檻時先讓使用者確認，並把換算依據講清楚
// ——不是只丟一個數字，使用者要能判斷這個估算合不合理。
const COST_CONFIRM_THRESHOLD = 1.0;   // USD

function estimateVideoCost(modelId, costInfo) {
    const p = pricingMap[modelId];
    if (!p) return null;
    if (p.type === 'fixed') return p.price;
    return estimateVideoTokenCost(modelId, costInfo?.resolution, costInfo?.seconds);
}

// 回傳 true 表示可以送出。算不出價格時**不攔**——寧可放行，也不要用一個猜測的
// 數字嚇阻使用者（那比沒有提示更糟）
function confirmIfExpensive(modelId, costInfo) {
    const est = estimateVideoCost(modelId, costInfo);
    if (est == null || est < COST_CONFIRM_THRESHOLD) return true;
    const basis = costInfo?.resolution && costInfo?.seconds
        ? `${costInfo.resolution}、${costInfo.seconds} 秒`
        : '目前設定';
    return window.confirm(
        `這次生成的預估費用約 $${formatUsd(Number(est.toFixed(4)))}（${modelId}，${basis}）。\n\n` +
        `任務送出後無法取消，費用照實際用量計算、可能與估算略有出入。\n確定要繼續嗎？`
    );
}

// 圖片的計費入口：按次的用「單價 × 張數」，按 token 的用上游回報的實際 token 數。
// 有 9 個圖片模型是按 token 計費（MAI 三個、GPT Image 兩個、Gemini Image 四個），
// 先前只呼叫 addFixedCost() 而它只認 type==='fixed'，所以那九個完全沒被計入花費。
function addImageCost(modelId, count, usage) {
    const p = pricingMap[modelId];
    if (!p) return;
    if (p.type === 'fixed') { addCost(p.price * count); return; }
    if (usage) { addTokenTextCost(modelId, usage); return; }
    // 按 token 但上游沒回 usage 就不計——不要用猜的數字
}

function addVideoCost(modelId, costInfo) {
    const p = pricingMap[modelId];
    if (!p) return;
    if (p.type === 'fixed') { addCost(p.price); return; }
    const est = estimateVideoTokenCost(modelId, costInfo?.resolution, costInfo?.seconds);
    if (est) addCost(est);
    // 算不出來就不計——寧可少算也不要顯示一個編出來的數字
}

function addFixedCost(modelId, count = 1) {
    const p = pricingMap[modelId];
    if (p && p.type === 'fixed') addCost(p.price * count);
}

function addTokenTextCost(modelId, usage) {
    const p = pricingMap[modelId];
    if (!p || p.type !== 'token' || !usage) return;
    const cost = (usage.prompt_tokens || 0) / 1e6 * p.input + (usage.completion_tokens || 0) / 1e6 * p.output;
    addCost(cost);
}

// ── 按 token 計費的影片模型 ────────────────────────────────────────────────
// 27 個影片模型裡有 8 個是 quota_type=0（按 token），不是按次固定價。對這些模型：
//   1. 顯示「$X→$Y/1M」對使用者毫無意義——沒人能心算一支影片是幾個 token
//   2. addFixedCost() 只計 type==='fixed'，所以它們的花費**完全不會**累加到「本次花費」
// Seedance 的 token 數有明確公式（網關端反推、我們端到端驗證過）：
//   tokens = 寬 × 高 × 幀數 / 1024，幀數 = 要求秒數 × 24 + 1（fps 固定 24）
// 我實測 480P/4 秒得到 854×480、97 幀，代入 = 38,830.31，與上游回傳的 38,830 吻合。
//
// ⚠️ **480p 的尺寸各世代不同**，這是最容易算錯 4.5% 的地方：
//   Seedance 2.5      → 854×480
//   2.0 系列 / 1.5-pro → 864×496
// 720p 以上各世代都是 1280×720，沒有差異。
const _SEEDANCE_DIMS = {
    'dreamina-seedance-2.5':      { '480P': [854, 480],  '720P': [1280, 720] },
    'dreamina-seedance-2.0':      { '480P': [864, 496],  '720P': [1280, 720], '1080P': [1920, 1080] },
    'dreamina-seedance-2.0-fast': { '480P': [864, 496],  '720P': [1280, 720] },
    'bytedance-seedance-1.5-pro': { '480P': [864, 496],  '720P': [1280, 720], '1080P': [1920, 1080] },
};

// 解析度倍率：純用像素數算會低估 1080p、高估 4K。倍率取自網關的計費設定
// （2.0 系列的 1080p ×1.1、4k ×4/7，基準是 720p），我用 peer 的精確總價表反推
// 驗證過：純公式 ÷ 精確表 = 1.0997 與 0.5715，與 1.1 和 4/7 吻合。
// 2.5 只有 480p/720p 兩檔、都不套倍率（兩檔實測與精確表完全相符）。
const _SEEDANCE_RES_RATIO = { '1080P': 1.1, '4K': 4 / 7 };

// 回傳預估費用（USD），算不出來就回 null（例如沒有該模型的尺寸表）
function estimateVideoTokenCost(modelId, resolution, seconds) {
    const p = pricingMap[modelId];
    if (!p || p.type !== 'token') return null;
    const dims = _SEEDANCE_DIMS[modelId]?.[resolution];
    if (!dims || !seconds) return null;
    const frames = seconds * 24 + 1;              // fps 固定 24，實際幀數比要求秒數多 1 幀
    const tokens = dims[0] * dims[1] * frames / 1024;
    const ratio = _SEEDANCE_RES_RATIO[resolution] || 1;
    return tokens / 1e6 * p.output * ratio;       // input/output 同價，用 output 即可
}

function formatPriceSuffix(modelId) {
    const p = pricingMap[modelId];
    if (!p) return '';
    if (p.type === 'fixed') return ` ・ $${formatUsd(p.price)}/次`;
    // 影片模型按 token 計費時，改成用目前選到的解析度與時長換算成「這一次大約多少錢」，
    // 那才是使用者看得懂的資訊；換算不出來才退回原本的每 1M 顯示
    if (_SEEDANCE_DIMS[modelId]) {
        const res = document.getElementById('videoResolution')?.value;
        const sec = parseInt(document.getElementById('videoDuration')?.value) || 0;
        const est = estimateVideoTokenCost(modelId, res, sec);
        if (est) return ` ・ 約 $${formatUsd(Number(est.toFixed(4)))}/次（${res} ${sec}秒）`;
    }
    return ` ・ $${formatUsd(p.input)}→$${formatUsd(p.output)}/1M`;
}

// 下拉選單收合狀態下常因側欄寬度不夠被截斷看不到價格，所以在「模型」label 旁邊
// 另外放一個不會被截斷的價格提示，跟著目前選到的模型即時更新
function updateModelPriceHint(hintElId, modelId) {
    const el = document.getElementById(hintElId);
    if (!el) return;
    const suffix = formatPriceSuffix(modelId);
    el.textContent = suffix ? '（' + suffix.replace(/^\s*・\s*/, '') + '）' : '';
}

function populateSelect(id, list, filterFn = null) {
    const sel = document.getElementById(id);
    // 價格背景載入完成後會重新呼叫這裡補上價格顯示——重建選單前先記住目前選到
    // 哪個值，重建後試著設回去，否則使用者已經手動選的模型會被打回第一個選項
    const prevValue = sel.value;
    sel.innerHTML = '';
    const filtered = filterFn ? list.filter(filterFn) : list;
    let group = '';
    filtered.forEach(m => {
        if (m.group !== group) {
            sel.appendChild(Object.assign(document.createElement('optgroup'), { label: m.group }));
            group = m.group;
        }
        sel.lastElementChild.appendChild(
            Object.assign(document.createElement('option'), { value: m.id, textContent: `${m.name} — ${m.desc}${formatPriceSuffix(m.id)}` })
        );
    });
    // 只有新清單裡真的還有這個值才恢復，否則保持瀏覽器預設行為（自動選第一項）——
    // 直接無條件 sel.value = prevValue 在新清單不包含舊值時會讓選單整個變成沒有
    // 任何選項被選中（selectedIndex=-1，value 變空字串），而不是退回選第一項
    if (prevValue && [...sel.options].some(o => o.value === prevValue)) {
        sel.value = prevValue;
    }
}

// ── Image 任務/模型切換 ────────────────────────────────────────
let imgMaxRef = 9; // 參考圖上限，依模型動態調整（qwen-image-2.0 系列為 3，其餘為 9）

function onImgTaskChange() {
    const t = document.getElementById('imageTaskType').value;
    populateSelect('imageModel', models.image, m => m.type === t);
    document.getElementById('imgUploadSection').classList.toggle('hidden', t !== 'i2i');
    if (t !== 'i2i') { imgRefFiles = []; renderImgThumbs(); }
    onImgModelChange();
}

// ── 自訂尺寸（目前只有 MAI 支援任意尺寸）─────────────────────────────────────
// 送出的仍然是 size 字串（"1200x800"），不是 width/height 兩個欄位——實測正式環境的
// 頂層 width/height 會被靜默丟棄、退回 1024x1024，而 size 這條現在就能用。
const CUSTOM_SIZE_VALUE = '__custom__';
let _imgCustomSizeSpec = null;   // { min_side, max_pixels, align }

function onImgSizeChange() {
    const on = document.getElementById('imageSize').value === CUSTOM_SIZE_VALUE && !!_imgCustomSizeSpec;
    document.getElementById('imgCustomSizeGroup').style.display = on ? '' : 'none';
    if (!on) return;
    const spec = _imgCustomSizeSpec;
    document.getElementById('imgCustomSizeHint').textContent =
        `自訂寬高（每邊至少 ${spec.min_side}，總像素不超過 ${spec.max_pixels.toLocaleString()}）`;
    const w = document.getElementById('imgCustomW'), h = document.getElementById('imgCustomH');
    if (!w.value) w.value = 1024;
    if (!h.value) h.value = 1024;
    onImgCustomSizeInput();
}

// 回傳 { size, error }。size 為 null 代表目前的輸入不能用。
function currentCustomSize() {
    const spec = _imgCustomSizeSpec;
    if (!spec) return { size: null, error: '這個模型不支援自訂尺寸' };
    const w = parseInt(document.getElementById('imgCustomW').value, 10);
    const h = parseInt(document.getElementById('imgCustomH').value, 10);
    if (!Number.isFinite(w) || !Number.isFinite(h)) return { size: null, error: '請輸入寬與高' };
    // 先往下對齊到 align 的倍數再檢查——上游本來就會這樣對齊，先算好使用者才不會
    // 拿到跟自己輸入不同的尺寸（例如輸入 1366 實際拿到 1360）
    const aw = Math.floor(w / spec.align) * spec.align;
    const ah = Math.floor(h / spec.align) * spec.align;
    if (aw < spec.min_side || ah < spec.min_side)
        return { size: null, error: `每邊至少 ${spec.min_side} 像素（目前 ${aw}×${ah}）` };
    if (aw * ah > spec.max_pixels)
        return { size: null, error: `總像素 ${(aw * ah).toLocaleString()} 超過上限 ${spec.max_pixels.toLocaleString()}` };
    return { size: `${aw}x${ah}`, aligned: (aw !== w || ah !== h), w: aw, h: ah, error: null };
}

function onImgCustomSizeInput() {
    const msg = document.getElementById('imgCustomSizeMsg');
    const r = currentCustomSize();
    if (r.error) {
        msg.style.color = 'var(--red)';
        msg.textContent = r.error;
    } else if (r.aligned) {
        // 明確告知已對齊，而不是默默改掉使用者輸入的值
        msg.style.color = 'var(--text-muted)';
        msg.textContent = `實際輸出 ${r.w}×${r.h}（尺寸會對齊到 ${_imgCustomSizeSpec.align} 的倍數）`;
    } else {
        msg.style.color = 'var(--text-muted)';
        msg.textContent = `實際輸出 ${r.w}×${r.h}`;
    }
}

function onImgModelChange() {
    const t = document.getElementById('imageTaskType').value;
    const modelId = document.getElementById('imageModel').value;
    updateModelPriceHint('imageModelPrice', modelId);
    // 同一 model id 可能同時存在 t2i 與 i2i 兩筆資料（如 qwen-image-2.0），需依 type 一併比對避免混淆
    const modelInfo = models.image.find(m => m.id === modelId && m.type === t) || {};

    // 更新尺寸選單（Gemini 圖像模型走 chat/completions，不支援 size 參數，隱藏此選項）
    document.getElementById('imgSizeGroup').style.display = modelInfo.no_size ? 'none' : '';
    const sizeEl = document.getElementById('imageSize');
    const currentSize = sizeEl.value;
    const sizes = modelInfo.sizes || ["1024*1024","1280*720","720*1280","1024*768","768*1024"];
    const sizeLabels = {
        "1024*1024": "1024×1024 (1:1)", "1280*720": "1280×720 (16:9)", "720*1280": "720×1280 (9:16)",
        "1024*768": "1024×768 (4:3)", "768*1024": "768×1024 (3:4)",
        "960*1280": "960×1280 (3:4)", "1280*960": "1280×960 (4:3)",
        "960*1696": "960×1696 (9:16)", "1696*960": "1696×960 (16:9)",
        "2048*2048": "2048×2048 (2K)", "4096*4096": "4096×4096 (4K)",
        // 規格值：萬相 2.7 走 size 的「方式一」（1K=1024*1024 等效總像素，比例
        // 跟隨輸入圖、無輸入圖則為正方形）；Gemini 走 imageConfig.imageSize。
        // 兩家的實際輸出像素算法不同，標籤只寫規格本身、不寫死換算結果
        "1K": "1K", "2K": "2K（預設）", "4K": "4K",
    };
    // 組圖模式（萬相 2.7）下 4K 不可用——官方文件寫明組圖只到 2K，但網關的驗證
    // 擋不住這個組合（實測送 4K + enable_sequential 一樣通過驗證），所以要在這裡擋，
    // 否則使用者要等到生成階段才會失敗
    const seqOn = document.getElementById('imgEnableSequential').checked
        && document.getElementById('imgSequentialGroup').style.display !== 'none';
    const seqCap = seqOn ? modelInfo.sequential_max_size : null;
    const usable = (seqCap === '2K') ? sizes.filter(s => s !== '4K' && s !== '4096*4096') : sizes;
    sizeEl.innerHTML = usable.map(s =>
        `<option value="${s}"${s === currentSize ? ' selected' : ''}>${sizeLabels[s] || s}</option>`
    ).join('');
    // 支援任意尺寸的模型（目前只有 MAI）多給一個「自訂」選項
    if (modelInfo.custom_size) {
        sizeEl.insertAdjacentHTML('beforeend',
            `<option value="${CUSTOM_SIZE_VALUE}"${currentSize === CUSTOM_SIZE_VALUE ? ' selected' : ''}>自訂尺寸…</option>`);
    }
    const selectable = modelInfo.custom_size ? usable.concat(CUSTOM_SIZE_VALUE) : usable;
    if (!selectable.includes(currentSize) && selectable.length) sizeEl.value = selectable[0];
    _imgCustomSizeSpec = modelInfo.custom_size || null;
    onImgSizeChange();

    // 圖片比例（僅 Gemini T2I 支援，用自然語言注入 prompt 的方式模擬比例控制）
    const aspectRatioGroup = document.getElementById('imgAspectRatioGroup');
    const aspectRatios = modelInfo.aspect_ratios || [];
    aspectRatioGroup.style.display = aspectRatios.length ? '' : 'none';
    if (aspectRatios.length) {
        const arEl = document.getElementById('imageAspectRatio');
        const currentAr = arEl.value;
        arEl.innerHTML = aspectRatios.map(r =>
            `<option value="${r}"${r === currentAr ? ' selected' : ''}>${r}</option>`
        ).join('');
    }

    // 更新張數上限（i2i 模式下，僅 max_n > 1 的模型如 qwen-image-2.0 系列才顯示張數選擇）
    const maxN = modelInfo.max_n || 4;
    const nSlider = document.getElementById('imgN');
    const sequentialChecked = document.getElementById('imgEnableSequential').checked;
    if (!sequentialChecked) {
        nSlider.max = maxN;
        if (parseInt(nSlider.value) > maxN) {
            nSlider.value = maxN;
            document.getElementById('imgNVal').textContent = maxN;
        }
    }
    document.getElementById('imgNGroup').style.display = (maxN > 1) ? '' : 'none';

    // 萬相 2.7 組圖模式（enable_sequential）僅 T2I 支援
    const sequentialGroup = document.getElementById('imgSequentialGroup');
    sequentialGroup.style.display = (t === 't2i' && modelInfo.supports_sequential) ? '' : 'none';
    if (!modelInfo.supports_sequential) {
        document.getElementById('imgEnableSequential').checked = false;
        onImgSequentialToggle();
    }

    // ref_strength 僅 Wan 圖像編輯系列支援，qwen-image-2.0 系列無此參數
    document.getElementById('imgRefStrengthGroup').style.display =
        (t === 'i2i' && !modelInfo.no_ref_strength) ? '' : 'none';

    // prompt_extend 僅 T2I 與 qwen-image-2.0 系列（i2i 融合模型）支援，其餘 I2I 圖像編輯模型後端不支援此參數
    document.getElementById('imgPromptExtendGroup').style.display =
        (t === 't2i' || modelInfo.fusion_edit) ? '' : 'none';

    // GPT Image 專屬參數（quality/background/output_format），T2I/I2I 皆適用
    document.getElementById('imgGptParamsSection').style.display = modelInfo.supports_gpt_params ? '' : 'none';

    // 參考圖張數上限（qwen-image-2.0 系列最多 3 張，其餘模型最多 9 張）
    imgMaxRef = modelInfo.max_ref || 9;
    if (imgRefFiles.length > imgMaxRef) imgRefFiles = imgRefFiles.slice(0, imgMaxRef);
    renderImgThumbs();
}

// 組圖模式開啟時，n 上限由 4 提高到 12（實際生成張數由模型決定、不保證等於設定值）
let _imgSizeRefreshing = false;   // 見 onImgSequentialToggle 末端的遞迴保護

function onImgSequentialToggle() {
    const on = document.getElementById('imgEnableSequential').checked;
    const nSlider = document.getElementById('imgN');
    const modelId = document.getElementById('imageModel').value;
    const modelInfo = models.image.find(m => m.id === modelId && m.type === 't2i') || {};
    const maxN = on ? 12 : (modelInfo.max_n || 4);
    nSlider.max = maxN;
    if (parseInt(nSlider.value) > maxN) {
        nSlider.value = maxN;
        document.getElementById('imgNVal').textContent = maxN;
    }
    document.getElementById('imgNLabel').innerHTML =
        (on ? '最大張數（組圖模式，實際張數由模型決定）' : '生成張數') +
        ` <span class="param-val" id="imgNVal">${nSlider.value}</span>`;
    // 組圖模式會限制可用的解析度規格（4K 不可用），重算尺寸選單。
    // onImgModelChange() 在模型不支援組圖時會反過來呼叫這裡，用旗標擋掉無限遞迴
    if (!_imgSizeRefreshing) {
        _imgSizeRefreshing = true;
        try { onImgModelChange(); } finally { _imgSizeRefreshing = false; }
    }
}

// ── Video 任務/模型切換 ────────────────────────────────────────
function onVidTaskChange() {
    const t = document.getElementById('videoTaskType').value;
    populateSelect('videoModel', models.video, m => m.type === t);

    document.getElementById('vidI2VUpload').classList.toggle('hidden', t !== 'i2v');
    document.getElementById('vidR2VUpload').classList.toggle('hidden', t !== 'r2v');
    document.getElementById('vidEditUpload').classList.toggle('hidden', t !== 'vedit');
    document.getElementById('vidAnimateUpload').classList.toggle('hidden', t !== 'animate');

    // vedit-specific controls
    document.getElementById('vidRatioGroup').style.display = (t === 'vedit') ? '' : 'none';
    document.getElementById('vidAudioSettingGroup').style.display = (t === 'vedit') ? '' : 'none';

    // i2v-specific controls
    document.getElementById('vidI2VModeGroup').style.display = (t === 'i2v') ? '' : 'none';

    // animate-specific controls（無 prompt / 解析度 / 時長，改用 mode + check_image）
    document.getElementById('vidAnimateModeGroup').style.display = (t === 'animate') ? '' : 'none';
    document.getElementById('vidAnimateCheckImgRow').style.display = (t === 'animate') ? '' : 'none';
    document.getElementById('vidResolutionGroup').style.display = (t === 'animate') ? 'none' : '';
    document.getElementById('vidDurationGroup').style.display = (t === 'animate') ? 'none' : '';
    document.getElementById('vidPromptCol').style.display = (t === 'animate') ? 'none' : '';
    document.getElementById('vidPromptExtendGroup').style.display = (t === 'animate') ? 'none' : '';

    // vedit duration hint（僅 Wan 的 min_dur=0 才顯示）
    const _veditModel = document.getElementById('videoModel').value;
    const _veditInfo  = models.video.find(m => m.id === _veditModel) || {};
    document.getElementById('durHintZero').style.display =
        (t === 'vedit' && (_veditInfo.min_dur === 0 || _veditInfo.min_dur === undefined)) ? '' : 'none';

    if (t === 'i2v') onI2VModeChange();
    onVidModelChange();
}

function onVidModelChange() {
    const taskType = document.getElementById('videoTaskType').value;
    const modelId  = document.getElementById('videoModel').value;
    updateModelPriceHint('videoModelPrice', modelId);
    const modelInfo = models.video.find(m => m.id === modelId) || {};

    // 顯示/隱藏自動配音
    const audioRow = document.getElementById('vidAudioRow');
    audioRow.style.display = modelInfo.audio ? '' : 'none';
    if (!modelInfo.audio) document.getElementById('vidAudio').checked = false;

    // 上游不接受這兩個參數的型號（Seedance 家族）要把控制項整組藏起來——留著會讓人
    // 以為填了有作用。清空內容/取消勾選，避免隱藏後仍把值送出去。
    // 注意 onVidTaskChange() 結尾會呼叫本函式，所以這裡的判斷會蓋過它依任務類型設的
    // display；animate 那個條件必須一併帶上，不然切到 animate 會把它重新顯示出來。
    const negGroup = document.getElementById('videoNegPromptGroup');
    negGroup.style.display = modelInfo.no_negative_prompt ? 'none' : '';
    if (modelInfo.no_negative_prompt) document.getElementById('videoNegPrompt').value = '';

    const extGroup = document.getElementById('vidPromptExtendGroup');
    extGroup.style.display = (taskType === 'animate' || modelInfo.no_prompt_extend) ? 'none' : '';
    if (modelInfo.no_prompt_extend) document.getElementById('vidPromptExtend').checked = false;

    // 調整時長範圍（gemini-omni-flash-preview 等模型自行決定長度與解析度，不支援 duration/resolution 參數）
    document.getElementById('vidDurationGroup').style.display = modelInfo.no_duration ? 'none' : '';
    document.getElementById('vidResolutionGroup').style.display =
        (modelInfo.no_duration || taskType === 'animate') ? 'none' : '';
    const dur    = document.getElementById('videoDuration');
    const minD   = modelInfo.min_dur ?? 3;
    const maxD   = modelInfo.max_dur || 10;
    const stepD  = modelInfo.dur_step || 1;
    dur.min  = minD;
    dur.max  = maxD;
    dur.step = stepD;
    let curVal = parseInt(dur.value);
    if (curVal < minD) { curVal = minD; }
    if (curVal > maxD) { curVal = maxD; }
    curVal = minD + Math.round((curVal - minD) / stepD) * stepD;
    dur.value = curVal;
    document.getElementById('durVal').textContent = curVal;
    const rangeEl = document.getElementById('durRange');
    if (rangeEl) rangeEl.textContent = `（${minD} ~ ${maxD} 秒）`;

    // 解析度選項：vedit 不支援 480P；另外部分模型的上游有自己的支援範圍
    // （例如 dreamina-seedance-2.0-fast 只到 720P，送 1080P 會直接被拒），
    // 由 MODELS 的 resolutions 明確指定
    const resEl = document.getElementById('videoResolution');
    // 模型有明確的 resolutions 清單時以它為準（例如萬相 3.0 的視頻編輯確實支援
    // 480P，那是它的基準價位，不該被 vedit 的通則擋掉）
    const allowedRes = modelInfo.resolutions;
    Array.from(resEl.options).forEach(o => {
        o.hidden = allowedRes ? !allowedRes.includes(o.value)
                              : (taskType === 'vedit' && o.value === '480P');
    });
    if (resEl.selectedOptions[0] && resEl.selectedOptions[0].hidden) {
        const first = Array.from(resEl.options).find(o => !o.hidden);
        if (first) resEl.value = first.value;
    }

    // I2V 模式：部分模型的上游只讀取首幀，尾幀／驅動音訊／影片延伸送過去會被
    // 靜默丟棄（拿到的影片看起來就是沒照做、但不會有任何錯誤），所以直接把
    // 其餘模式從選單收起來
    if (taskType === 'i2v') {
        const allowed = modelInfo.i2v_modes;
        const modeEl  = document.getElementById('videoI2VMode');
        Array.from(modeEl.options).forEach(o => {
            o.hidden = allowed ? !allowed.includes(o.value) : false;
        });
        if (allowed && !allowed.includes(modeEl.value)) modeEl.value = allowed[0];
        onI2VModeChange();
    }

    // R2V 參考檔案：只吃圖片的模型不要讓使用者挑到影片
    const refInput = document.getElementById('vidRefInput');
    if (refInput) {
        refInput.accept = modelInfo.ref_images_only ? 'image/*' : 'image/*,video/*';
    }
}

// I2V 模式切換（顯示對應上傳區）
function onI2VModeChange() {
    const mode = document.getElementById('videoI2VMode').value;
    document.getElementById('vidFirstFrameZone').style.display =
        (mode === 'first_clip' || mode === 'first_clip_last_frame') ? 'none' : '';
    document.getElementById('vidLastFrameZone').style.display  =
        (mode === 'first_last_frame' || mode === 'first_clip_last_frame') ? '' : 'none';
    document.getElementById('vidAudioZone').style.display   =
        (mode === 'first_frame_audio') ? '' : 'none';
    document.getElementById('vidClipZone').style.display    =
        (mode === 'first_clip' || mode === 'first_clip_last_frame') ? '' : 'none';
}

// T2V 自動配音開關 → 展開/收合上傳區
function onVidAudioToggle(cb) {
    const zone = document.getElementById('vidT2VAudioZone');
    if (zone) zone.style.display = cb.checked ? '' : 'none';
    if (!cb.checked) {
        const inp = document.getElementById('vidT2VAudioInput');
        if (inp) inp.value = '';
        const hint = document.getElementById('vidT2VAudioHint');
        if (hint) hint.innerHTML = '上傳音訊（可選）<br><span style="font-size:11px;color:var(--text-muted)">留空由模型自動配音</span>';
    }
}
function onT2VAudioUpload(e) {
    const f = e.target.files[0];
    const hint = document.getElementById('vidT2VAudioHint');
    if (f && hint) hint.innerHTML = `<strong>${f.name}</strong><br><span style="font-size:11px;color:var(--text-muted)">${(f.size/1024).toFixed(0)} KB</span>`;
}

// 音訊 / 片段上傳名稱顯示
function onAudioUpload(e) {
    const f = e.target.files[0];
    if (f) document.getElementById('vidAudioFileName').textContent = f.name;
}
function onVidAudioToggle(cb) {
    const zone = document.getElementById('vidT2VAudioZone');
    if (zone) zone.style.display = cb.checked ? '' : 'none';
    if (!cb.checked) {
        const inp = document.getElementById('vidT2VAudioInput');
        if (inp) inp.value = '';
        const hint = document.getElementById('vidT2VAudioHint');
        if (hint) hint.innerHTML = '上傳音訊（可選）<br><span style="font-size:11px;color:var(--text-muted)">留空由模型自動配音</span>';
    }
}
function onT2VAudioUpload(e) {
    const f = e.target.files[0];
    const hint = document.getElementById('vidT2VAudioHint');
    if (f && hint) hint.innerHTML = `<strong>${f.name}</strong><br><span style="font-size:11px;color:var(--text-muted)">${(f.size/1024).toFixed(0)} KB</span>`;
}
function onAnimateVideoUpload(e) {
    const f = e.target.files[0];
    if (f) document.getElementById('vidAnimateVideoName').textContent = f.name;
}
function onClipUpload(e) {
    const f = e.target.files[0];
    if (f) document.getElementById('vidClipFileName').textContent = f.name;
}

// 影片編輯上傳
function onEditVideoUpload(e) {
    const f = e.target.files[0];
    if (f) document.getElementById('vidEditVideoName').textContent = f.name;
}
// 參考圖張數上限依模型而定（MODELS 的 max_ref）。原本這裡跟 r2v 都是寫死／沒有上限，
// 與後端實際會讀取的張數不一致——超出的檔案會被靜默丟棄（使用者看不出來），
// 不足的則是白白少了模型支援的欄位
function vidRefLimit(taskType, fallback) {
    const id = document.getElementById('videoModel').value;
    const info = (models.video || []).find(m => m.id === id && m.type === taskType);
    return (info && info.max_ref) || fallback;
}

function onEditRefUpload(e) {
    const newFiles = Array.from(e.target.files);
    editRefFiles = [...editRefFiles, ...newFiles].slice(0, vidRefLimit('vedit', 3));
    renderEditRefList();
}
function renderEditRefList() {
    document.getElementById('vidEditRefList').innerHTML = editRefFiles.map((f, i) => `
        <div class="ref-item">
            <span>${f.name}</span>
            <button onclick="removeEditRef(${i})">✕</button>
        </div>`).join('');
}
function removeEditRef(i) { editRefFiles.splice(i, 1); renderEditRefList(); }

// ── Tab ───────────────────────────────────────────────────────
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
}


// ── Omni ──────────────────────────────────────────────────────
let omniAudioContext;
let omniMicrophone;
let omniProcessor;
let omniWebSocket;
let outAudioCtx;
let nextPlayTime = 0;
let omniChatHistory = [];   // for non-realtime models

function onOmniModelChange() {
    const model = document.getElementById('omniModel').value;
    const isRealtime = model.includes('realtime');
    document.getElementById('omniRealtimeControls').style.display = isRealtime ? '' : 'none';
    document.getElementById('omniChatInputArea').style.display   = isRealtime ? 'none' : '';
    // 切換模型時清空歷史
    omniChatHistory = [];
    const area = document.getElementById('omniTranscriptionArea');
    area.innerHTML = '';
    const hint = isRealtime
        ? '點擊「開始通話」並允許麥克風權限，即可與 AI 即時對話'
        : '在下方輸入訊息，AI 將以文字＋語音回覆（Ctrl+Enter 發送）';
    area.innerHTML = `<div class="empty-state" style="width:100%"><svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm-1 14.5v-9l6 4.5-6 4.5z"/></svg><p>${hint}</p></div>`;
    updateOmniVoices(model);
}

async function sendOmniChat() {
    const text = document.getElementById('omniChatInput').value.trim();
    if (!text) return;
    if (!apiKey) { toast('請先設定 API Key', 'error'); return; }

    const model = document.getElementById('omniModel').value;
    const voice = document.getElementById('omniVoice').value;
    const instructions = document.getElementById('omniInstructions').value.trim();

    omniChatHistory.push({ role: 'user', content: text });

    const area = document.getElementById('omniTranscriptionArea');
    const empty = area.querySelector('.empty-state');
    if (empty) empty.remove();
    area.innerHTML += `<div style="margin-bottom:8px"><strong style="color:var(--primary-color)">[User]</strong> ${text}</div>`;
    area.scrollTop = area.scrollHeight;

    document.getElementById('omniChatInput').value = '';
    const sendBtn = document.getElementById('omniChatSendBtn');
    sendBtn.disabled = true;

    // 建立 AI 回覆容器
    const replyId = 'omni-reply-' + Date.now();
    area.innerHTML += `<div id="${replyId}" style="margin-bottom:12px"><strong style="color:#00c853">[AI]</strong> <span id="${replyId}-text"></span></div>`;
    area.scrollTop = area.scrollHeight;

    // 收集 PCM16 音訊 chunks
    const audioChunks = [];
    let textContent = '';

    try {
        const resp = await fetch('/api/omni/chat', {
            method: 'POST',
            headers: { ...authHeader(), 'Content-Type': 'application/json' },
            body: JSON.stringify({ model, messages: omniChatHistory, voice, instructions }),
        });

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf = '';

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;
            buf += decoder.decode(value, { stream: true });
            const lines = buf.split('\n');
            buf = lines.pop();
            for (const line of lines) {
                if (!line.startsWith('data: ')) continue;
                const msg = JSON.parse(line.slice(6));
                if (msg.type === 'text') {
                    textContent += msg.content;
                    document.getElementById(replyId + '-text').textContent = textContent;
                    area.scrollTop = area.scrollHeight;
                } else if (msg.type === 'transcript') {
                    textContent += msg.content;
                    document.getElementById(replyId + '-text').textContent = textContent;
                    area.scrollTop = area.scrollHeight;
                } else if (msg.type === 'audio') {
                    audioChunks.push(msg.data);
                } else if (msg.type === 'error') {
                    logOmniMessage('Error', msg.content);
                }
            }
        }

        // 更新對話歷史
        omniChatHistory.push({ role: 'assistant', content: textContent || '（語音回覆）' });

        // 播放音訊
        if (audioChunks.length > 0) {
            _playOmniPCM(audioChunks);
        }
    } catch (e) {
        logOmniMessage('Error', e.message);
    }
    sendBtn.disabled = false;
}

function _playOmniPCM(chunks) {
    // 每個 chunk 是獨立的 base64，需分開 decode 再合併 binary
    const decoded = chunks.map(b64 => {
        const bin = atob(b64);
        const arr = new Uint8Array(bin.length);
        for (let i = 0; i < bin.length; i++) arr[i] = bin.charCodeAt(i);
        return arr;
    });
    const totalLen = decoded.reduce((s, a) => s + a.length, 0);
    const combined = new Uint8Array(totalLen);
    let offset = 0;
    for (const arr of decoded) { combined.set(arr, offset); offset += arr.length; }

    const int16 = new Int16Array(combined.buffer);
    const float32 = new Float32Array(int16.length);
    for (let i = 0; i < int16.length; i++) float32[i] = int16[i] / 32768.0;

    const ctx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    const buf = ctx.createBuffer(1, float32.length, 24000);
    buf.getChannelData(0).set(float32);
    const src = ctx.createBufferSource();
    src.buffer = buf;
    src.connect(ctx.destination);
    src.start(0);
    src.onended = () => ctx.close();
}

async function startOmniConversation() {
    if(!apiKey) { alert('請先設定 API Key'); return; }
    
    const protocol = window.location.protocol === 'https:' ? 'wss' : 'ws';
    const model = document.getElementById('omniModel').value;
    const wsUrl = `${protocol}://${window.location.host}/ws/omni?api_key=${apiKey}&model=${model}`;
    omniWebSocket = new WebSocket(wsUrl);
    
    omniWebSocket.onopen = async () => {
        logOmniMessage('System', 'Connected. Setting up audio...');
        
        const setupMsg = {
            "event_id": crypto.randomUUID(),
            "type": "session.update",
            "session": {
                "modalities": ["text", "audio"],
                "voice": document.getElementById('omniVoice').value,
                "input_audio_format": "pcm16",
                "output_audio_format": "pcm16",
                "instructions": document.getElementById('omniInstructions').value,
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.5,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 600
                }
            }
        };
        omniWebSocket.send(JSON.stringify(setupMsg));
        
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            omniAudioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 16000 });
            omniMicrophone = omniAudioContext.createMediaStreamSource(stream);
            
            omniProcessor = omniAudioContext.createScriptProcessor(4096, 1, 1);
            omniProcessor.onaudioprocess = (e) => {
                if (omniWebSocket.readyState !== WebSocket.OPEN) return;
                
                const float32Array = e.inputBuffer.getChannelData(0);
                const int16Array = new Int16Array(float32Array.length);
                for (let i = 0; i < float32Array.length; i++) {
                    let s = Math.max(-1, Math.min(1, float32Array[i]));
                    int16Array[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
                }
                
                const bytes = new Uint8Array(int16Array.buffer);
                let binary = '';
                for (let i = 0; i < bytes.byteLength; i++) {
                    binary += String.fromCharCode(bytes[i]);
                }
                const base64 = btoa(binary);
                
                omniWebSocket.send(JSON.stringify({
                    "event_id": crypto.randomUUID(),
                    "type": "input_audio_buffer.append",
                    "audio": base64
                }));
            };
            
            omniMicrophone.connect(omniProcessor);
            omniProcessor.connect(omniAudioContext.destination);
            
            document.getElementById('omniStartBtn').disabled = true;
            document.getElementById('omniStopBtn').disabled = false;
            logOmniMessage('System', 'Microphone active. Start speaking...');
        } catch (err) {
            logOmniMessage('Error', 'Microphone setup failed: ' + err.message);
            stopOmniConversation();
        }
    };
    
    outAudioCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 24000 });
    nextPlayTime = 0;
    
    omniWebSocket.onmessage = (e) => {
        const msg = JSON.parse(e.data);
        if (msg.type === 'response.audio.delta' && msg.delta) {
            const binary = atob(msg.delta);
            const int16Array = new Int16Array(binary.length / 2);
            for (let i = 0; i < binary.length; i+=2) {
                int16Array[i/2] = binary.charCodeAt(i) | (binary.charCodeAt(i+1) << 8);
            }
            
            const float32Array = new Float32Array(int16Array.length);
            for (let i = 0; i < int16Array.length; i++) {
                float32Array[i] = int16Array[i] / 32768.0;
            }
            
            const audioBuffer = outAudioCtx.createBuffer(1, float32Array.length, 24000);
            audioBuffer.getChannelData(0).set(float32Array);
            
            const source = outAudioCtx.createBufferSource();
            source.buffer = audioBuffer;
            source.connect(outAudioCtx.destination);
            
            const currentTime = outAudioCtx.currentTime;
            if (nextPlayTime < currentTime) nextPlayTime = currentTime + 0.1;
            source.start(nextPlayTime);
            nextPlayTime += audioBuffer.duration;
            
        } else if (msg.type === 'conversation.item.input_audio_transcription.completed') {
            logOmniMessage('User', msg.transcript);
        } else if (msg.type === 'response.audio_transcript.done') {
            logOmniMessage('LLM', msg.transcript);
        } else if (msg.type === 'error') {
            logOmniMessage('Error', JSON.stringify(msg.error || msg));
        }
    };
    
    omniWebSocket.onclose = () => {
        logOmniMessage('System', 'Connection closed');
        stopOmniConversation();
    };
}

function stopOmniConversation() {
    if (omniProcessor) {
        omniProcessor.disconnect();
        omniProcessor = null;
    }
    if (omniMicrophone) {
        omniMicrophone.disconnect();
        omniMicrophone = null;
    }
    if (omniAudioContext) {
        omniAudioContext.close();
        omniAudioContext = null;
    }
    if (omniWebSocket) {
        omniWebSocket.close();
        omniWebSocket = null;
    }
    if (outAudioCtx) {
        outAudioCtx.close();
        outAudioCtx = null;
    }
    document.getElementById('omniStartBtn').disabled = false;
    document.getElementById('omniStopBtn').disabled = true;
}

function logOmniMessage(role, text) {
    const area = document.getElementById('omniTranscriptionArea');
    const empty = area.querySelector('.empty-state');
    if (empty) empty.remove();
    const roleColor = role === 'User' ? 'var(--primary-color)' : role === 'LLM' ? '#00c853' : '#757575';
    area.innerHTML += `<div style="margin-bottom:5px"><strong style="color:${roleColor}">[${role}]</strong> ${text}</div>`;
    area.scrollTop = area.scrollHeight;
}

// ── Text Generation ───────────────────────────────────────────
async function sendText() {
    const prompt = document.getElementById('textPrompt').value.trim();
    if (!prompt) { toast('請輸入提示詞', 'error'); return; }

    const model             = document.getElementById('textModel').value;
    const systemPrompt      = document.getElementById('textSystemPrompt').value;
    const temperature       = parseFloat(document.getElementById('textTemperature').value);
    const topP              = parseFloat(document.getElementById('textTopP').value);
    const topK              = parseInt(document.getElementById('textTopK').value);
    const maxTokens         = parseInt(document.getElementById('textMaxTokens').value);
    const presencePenalty   = parseFloat(document.getElementById('textPresencePenalty').value);
    const frequencyPenalty  = parseFloat(document.getElementById('textFrequencyPenalty').value);
    const seedRaw           = document.getElementById('textSeed').value.trim();
    const seed              = seedRaw !== '' ? parseInt(seedRaw) : null;
    const stopRaw           = document.getElementById('textStop').value.trim();
    const stop              = stopRaw ? stopRaw.split('\n').map(s => s.trim()).filter(Boolean).slice(0, 4) : [];
    const enableThinking    = document.getElementById('textThinking').checked;
    const rememberContext   = document.getElementById('textRememberContext').checked;
    const reasoningEffort   = document.getElementById('textReasoningEffort').value;
    const useStream         = document.getElementById('textStream').checked;
    const modelInfo         = models.text.find(m => m.id === model);

    const output = document.getElementById('textOutput');
    output.querySelector('.empty-state')?.remove();

    const uDiv = el('div', { className: 'chat-message user', textContent: prompt });
    const uMeta = el('div', { className: 'msg-meta', style: 'color: rgba(255,255,255,0.7); justify-content: flex-end;' });
    uMeta.innerHTML = '<span>' + new Date().toLocaleTimeString() + '</span>';
    uDiv.appendChild(uMeta);
    output.appendChild(uDiv);

    const aDiv = el('div', { className: 'chat-message assistant streaming-cursor' });
    const contentDiv = el('span');
    aDiv.appendChild(contentDiv);
    output.appendChild(aDiv);
    output.scrollTop = output.scrollHeight;

    // 思考過程（reasoning_content）跟正式回答是分開的兩股文字，用獨立的可收合區塊
    // 顯示、且第一次收到內容時才插入 DOM——大多數模型/情境完全不會有這段內容
    let reasoningEl = null, reasoningBodyEl = null, reasoningFull = '';
    const ensureReasoningEl = () => {
        if (reasoningEl) return;
        reasoningEl = el('details', { className: 'msg-reasoning', open: true });
        const summary = el('summary', { textContent: '思考過程' });
        reasoningBodyEl = el('div', { className: 'msg-reasoning-body' });
        reasoningEl.appendChild(summary);
        reasoningEl.appendChild(reasoningBodyEl);
        aDiv.insertBefore(reasoningEl, contentDiv);
    };

    const btn = document.getElementById('textSendBtn');
    btn.disabled = true;
    document.getElementById('textPrompt').value = '';

    try {
        const body = {
            model, prompt, system_prompt: systemPrompt,
            temperature, top_p: topP, max_tokens: maxTokens,
            presence_penalty: presencePenalty, frequency_penalty: frequencyPenalty,
            stream: useStream,
            enable_thinking: enableThinking && !!modelInfo?.thinking,
            // 「記住上下文」關閉時不帶歷史訊息，每一輪都是全新對話。畫面上的對話紀錄
            // 與 textChatHistory 照常累積——開關只決定「這一輪送什麼給模型」，重新
            // 打開後就能接著先前的內容繼續，不必重新問一次
            history: rememberContext ? textChatHistory : [],
        };
        if (topK > 0) body.top_k = topK;
        if (seed !== null) body.seed = seed;
        if (stop.length > 0) body.stop = stop;
        if (reasoningEffort && modelInfo?.reasoning_effort) body.reasoning_effort = reasoningEffort;
        if (modelInfo?.vision && textVisionImages.length) body.images = textVisionImages.map(f => f.url);

        const startTime = Date.now();
        const res = await fetch('/api/text/generate', {
            method: 'POST',
            headers: authHeader(),
            body: JSON.stringify(body),
        });

        aDiv.classList.remove('streaming-cursor');

        if (!useStream) {
            const data = await res.json();
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            if (data.error) {
                contentDiv.textContent = '⚠ 錯誤：' + data.error;
            } else {
                if (data.reasoning_content) {
                    ensureReasoningEl();
                    reasoningEl.open = false;
                    reasoningBodyEl.textContent = data.reasoning_content;
                }
                contentDiv.textContent = data.content || '';
                const meta = el('div', { className: 'msg-meta' });
                meta.innerHTML = '<span>' + model + ' (耗時 ' + elapsed + 's)</span><span>' + new Date().toLocaleTimeString() + '</span>';
                aDiv.appendChild(meta);
                textChatHistory.push({ role: 'user', content: prompt });
                textChatHistory.push({ role: 'assistant', content: data.content || '' });
                addTokenTextCost(model, data.usage);
            }
        } else {
            const reader  = res.body.getReader();
            const decoder = new TextDecoder();
            let full = '', buf = '', usage = null;
            aDiv.classList.add('streaming-cursor');

            while (true) {
                const { value, done } = await reader.read();
                if (done) break;
                buf += decoder.decode(value, { stream: true });
                const lines = buf.split('\n');
                buf = lines.pop();

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const d = JSON.parse(line.slice(6));
                        if (d.reasoning) {
                            ensureReasoningEl();
                            reasoningFull += d.reasoning;
                            reasoningBodyEl.textContent = reasoningFull;
                            output.scrollTop = output.scrollHeight;
                        } else if (d.content) {
                            if (reasoningEl) reasoningEl.open = false; // 開始輸出正式回答，收合思考過程
                            full += d.content;
                            contentDiv.textContent = full;
                            output.scrollTop = output.scrollHeight;
                        } else if (d.error) {
                            contentDiv.textContent = '⚠ 錯誤：' + d.error;
                            aDiv.classList.remove('streaming-cursor');
                        } else if (d.done) {
                            usage = d.usage || null;
                        }
                    } catch (_) { /* skip */ }
                }
            }
            aDiv.classList.remove('streaming-cursor');
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            const meta = el('div', { className: 'msg-meta' });
            meta.innerHTML = '<span>' + model + ' (耗時 ' + elapsed + 's)</span><span>' + new Date().toLocaleTimeString() + '</span>';
            aDiv.appendChild(meta);
            if (full) {
                textChatHistory.push({ role: 'user', content: prompt });
                textChatHistory.push({ role: 'assistant', content: full });
            }
            addTokenTextCost(model, usage);
        }
    } catch (e) {
        contentDiv.textContent = '⚠ 錯誤：' + e.message;
        aDiv.classList.remove('streaming-cursor');
    }
    btn.disabled = false;
}

function clearChat() {
    textChatHistory = [];
    document.getElementById('textOutput').innerHTML = `
        <div class="empty-state">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.2"><path d="M21 15a2 2 0 01-2 2H7l-4 4V5a2 2 0 012-2h14a2 2 0 012 2z"/></svg>
            <p>輸入提示詞，按 Ctrl+Enter 發送</p>
        </div>`;
}

// ── Image Generation ──────────────────────────────────────────
async function sendImage() {
    const taskType = document.getElementById('imageTaskType').value;
    const model    = document.getElementById('imageModel').value;
    const prompt   = document.getElementById('imagePrompt').value.trim();
    const negPrompt= document.getElementById('imageNegPrompt').value.trim();
    let   size     = document.getElementById('imageSize').value;
    // 自訂尺寸：把兩個輸入框組成 size 字串。不合法就擋在這裡，省下一次必定失敗的呼叫
    if (size === CUSTOM_SIZE_VALUE) {
        const r = currentCustomSize();
        if (r.error) { toast(r.error, "error"); return; }
        size = r.size;
    }
    const extend      = document.getElementById('imgPromptExtend').checked;
    const watermark   = document.getElementById('imgWatermark').checked;
    const n           = parseInt(document.getElementById('imgN').value) || 1;
    const imgSeedRaw  = document.getElementById('imgSeed').value.trim();
    const imgSeed     = imgSeedRaw !== '' ? parseInt(imgSeedRaw) : null;
    const refStrength = parseFloat(document.getElementById('imgRefStrength').value);
    const aspectRatio = document.getElementById('imgAspectRatioGroup').style.display !== 'none'
        ? document.getElementById('imageAspectRatio').value : '';
    const enableSequential = document.getElementById('imgSequentialGroup').style.display !== 'none'
        && document.getElementById('imgEnableSequential').checked;
    const gptParamsVisible = document.getElementById('imgGptParamsSection').style.display !== 'none';
    const quality      = gptParamsVisible ? document.getElementById('imgQuality').value : '';
    const background   = gptParamsVisible ? document.getElementById('imgBackground').value : '';
    const outputFormat = gptParamsVisible ? document.getElementById('imgOutputFormat').value : '';

    if (!prompt) { toast('請輸入 Prompt', 'error'); return; }

    if (taskType === 'i2i' && !imgRefFiles.length) { toast('請先上傳至少一張參考圖片', 'error'); return; }

    // 改成跟影片一樣的「行內佔位卡」而不是全螢幕遮罩：圖片生成動輒數十秒，
    // 遮罩會把整個平台鎖住，連切到別的頁籤看結果都不行。改成插入一張生成中的
    // 卡片、送出鈕也不再鎖定，使用者可以同時做別的事、甚至再送下一張（卡片會
    // 立刻出現，本身就是「已收到」的回饋，不需要靠鎖按鈕來防重複點）。
    const card = addImagePendingCard(model, prompt);
    const startTime = Date.now();

    try {
        let res;
        if (taskType === 't2i') {
            const body = { model, prompt, negative_prompt: negPrompt, size, n, prompt_extend: extend, watermark };
            if (imgSeed !== null) body.seed = imgSeed;
            if (aspectRatio) body.aspect_ratio = aspectRatio;
            if (enableSequential) body.enable_sequential = true;
            if (quality) body.quality = quality;
            if (background) body.background = background;
            if (outputFormat) body.output_format = outputFormat;
            res = await apiPost('/api/image/generate', body);
        } else {
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('size', size);
            fd.append('watermark', watermark); fd.append('ref_strength', refStrength);
            fd.append('n', n); fd.append('prompt_extend', extend);
            if (imgSeed !== null) fd.append('seed', imgSeed);
            if (aspectRatio) fd.append('aspect_ratio', aspectRatio);
            if (quality) fd.append('quality', quality);
            if (background) fd.append('background', background);
            if (outputFormat) fd.append('output_format', outputFormat);
            imgRefFiles.forEach((f, i) => fd.append(`image_${i + 1}`, f));
            res = await apiPostForm('/api/image/edit', fd);
        }

        if (res.success && res.images?.length) {
            const elapsed = fmtElapsed(Date.now() - startTime);
            finishImagePendingCard(card, res, elapsed);
            toast(`圖片生成完成！共 ${res.images.length} 張`, 'success');
            addImageCost(res.model, res.images.length, res.usage);
        } else {
            const errMsg = res.error || '生成失敗';
            failImagePendingCard(card, errMsg);
            toast(errMsg, 'error');
            console.error('Image generation error:', res);
        }
    } catch (e) {
        failImagePendingCard(card, e.message);
        toast(`錯誤：${e.message}`, 'error');
    }
}

// ── 圖片生成的行內佔位卡（沿用影片那組 vtc-* 樣式，維持兩個頁籤觀感一致）──
function addImagePendingCard(model, prompt) {
    const gallery = document.getElementById('imageResults');
    gallery.querySelector('.empty-state')?.remove();
    const card = el('div', { className: 'video-task-card' });
    card.innerHTML = `
        <div class="vtc-header">
            <span class="vtc-model">${model}</span>
            <span class="vtc-timer">(耗時 0s)</span>
            <span class="vtc-status pending">生成中</span>
        </div>
        <div class="vtc-prompt">${prompt.substring(0, 120)}${prompt.length > 120 ? '...' : ''}</div>
        <div class="vtc-progress"><div class="vtc-progress-bar indeterminate"></div></div>
        <div class="img-pending-body"><span class="spinner-sm"></span>圖片生成中…</div>`;
    gallery.insertBefore(card, gallery.firstChild);

    // 計時器掛在卡片上，完成/失敗時要清掉，否則分頁一直開著會累積 interval
    const startTime = Date.now();
    const timerEl = card.querySelector('.vtc-timer');
    card._timer = setInterval(() => {
        timerEl.textContent = `(耗時 ${fmtElapsed(Date.now() - startTime)})`;
    }, 1000);
    return card;
}

function _stopPendingTimer(card) {
    if (card && card._timer) { clearInterval(card._timer); card._timer = null; }
}

function finishImagePendingCard(card, res, elapsed) {
    _stopPendingTimer(card);
    const gallery = document.getElementById('imageResults');
    const frag = document.createDocumentFragment();
    res.images.forEach(img => {
        const src = img.local_path || img.url;
        const c = el('div', { className: 'img-card' });
        c.innerHTML = `
            <img src="${src}" alt="Generated" loading="lazy" onclick="openLightbox('${src}')">
            <div class="img-card-footer">
                <span class="img-model-tag">${res.model}（耗時 ${elapsed}）</span>
                <a href="${src}" download class="img-dl">下載</a>
            </div>`;
        if (img.actual_prompt) {
            const extEl = el('div', { className: 'img-actual-prompt' });
            extEl.textContent = 'Prompt Extend 擴充後：' + img.actual_prompt;
            c.appendChild(extEl);
        }
        frag.appendChild(c);
    });
    // 就地換掉佔位卡，結果才會留在當初送出的位置，不會插到後來完成的任務前面
    gallery.replaceChild(frag, card);
}

function failImagePendingCard(card, message) {
    _stopPendingTimer(card);
    card.querySelector('.vtc-status').className = 'vtc-status failed';
    card.querySelector('.vtc-status').textContent = '失敗';
    card.querySelector('.vtc-progress')?.remove();
    const body = card.querySelector('.img-pending-body');
    body.style.color = 'var(--red)';
    body.textContent = message || '生成失敗';
}

// ── Video Generation ──────────────────────────────────────────
async function sendVideo() {
    const taskType  = document.getElementById('videoTaskType').value;
    const model     = document.getElementById('videoModel').value;
    const prompt    = document.getElementById('videoPrompt').value.trim();
    const negPrompt = document.getElementById('videoNegPrompt').value.trim();
    const resolution= document.getElementById('videoResolution').value;
    const duration  = parseInt(document.getElementById('videoDuration').value);
    const audio         = document.getElementById('vidAudio').checked;
    const vidExtend     = document.getElementById('vidPromptExtend').checked;
    const vidWatermark  = document.getElementById('vidWatermark').checked;
    const vidSeedRaw    = document.getElementById('vidSeed').value.trim();
    const vidSeed       = vidSeedRaw !== '' ? parseInt(vidSeedRaw) : null;

    if (!prompt && taskType !== 'vedit' && taskType !== 'animate') { toast('請輸入 Prompt', 'error'); return; }

    // 昂貴任務先確認（門檻與理由見 confirmIfExpensive）。放在按鈕鎖定之前，
    // 使用者取消時不需要再把按鈕解鎖
    if (!confirmIfExpensive(model, { resolution, seconds: duration })) return;

    const btn = document.getElementById('videoSendBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> 提交中...';

    const startTime = Date.now();
    try {
        let res;
        if (taskType === 't2v') {
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('resolution', resolution);
            fd.append('duration', duration); fd.append('audio', audio);
            fd.append('prompt_extend', vidExtend); fd.append('watermark', vidWatermark);
            if (vidSeed !== null) fd.append('seed', vidSeed);
            if (audio) {
                const audioFile = document.getElementById('vidT2VAudioInput').files[0];
                if (audioFile) fd.append('audio_file', audioFile);
            }
            res = await apiPostForm('/api/video/t2v', fd);

        } else if (taskType === 'i2v') {
            const i2vMode = document.getElementById('videoI2VMode').value;
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('resolution', resolution);
            fd.append('duration', duration); fd.append('i2v_mode', i2vMode);
            fd.append('prompt_extend', vidExtend); fd.append('watermark', vidWatermark);
            if (vidSeed !== null) fd.append('seed', vidSeed);

            fd.append('audio', audio);
            if (audio) {
                const bgmFile = document.getElementById('vidT2VAudioInput')?.files[0];
                if (bgmFile) fd.append('audio_file', bgmFile);
            }
            if (i2vMode === 'first_clip' || i2vMode === 'first_clip_last_frame') {
                const clipFile = document.getElementById('vidClipInput').files[0];
                if (!clipFile) { toast('請上傳首段影片片段', 'error'); btn.disabled = false; btn.innerHTML = _vidBtnHTML(); return; }
                fd.append('first_clip', clipFile);
                if (i2vMode === 'first_clip_last_frame') {
                    const lastFile = document.getElementById('vidLastFrameInput').files[0];
                    if (lastFile) fd.append('last_frame', lastFile);
                }
            } else {
                const firstFile = document.getElementById('vidFirstFrameInput').files[0];
                if (!firstFile) { toast('請上傳首幀圖片', 'error'); btn.disabled = false; btn.innerHTML = _vidBtnHTML(); return; }
                fd.append('first_frame', firstFile);
                if (i2vMode === 'first_last_frame') {
                    const lastFile = document.getElementById('vidLastFrameInput').files[0];
                    if (lastFile) fd.append('last_frame', lastFile);
                }
                if (i2vMode === 'first_frame_audio') {
                    const audioFile = document.getElementById('vidAudioFileInput').files[0];
                    if (audioFile) fd.append('driving_audio', audioFile);
                }
            }
            res = await apiPostForm('/api/video/i2v', fd);

        } else if (taskType === 'vedit') {
            const editVideoFile = document.getElementById('vidEditVideoInput').files[0];
            if (!editVideoFile) { toast('請上傳來源影片', 'error'); btn.disabled = false; btn.innerHTML = _vidBtnHTML(); return; }
            const ratio        = document.getElementById('videoRatio').value;
            const audioSetting = document.getElementById('videoAudioSetting').value;
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('resolution', resolution);
            fd.append('duration', duration); fd.append('audio_setting', audioSetting);
            fd.append('prompt_extend', vidExtend); fd.append('watermark', vidWatermark);
            if (ratio) fd.append('ratio', ratio);
            if (vidSeed !== null) fd.append('seed', vidSeed);
            fd.append('video', editVideoFile);
            editRefFiles.forEach((f, i) => fd.append(`reference_image_${i + 1}`, f));
            res = await apiPostForm('/api/video/vedit', fd);

        } else if (taskType === 'animate') {
            const imgFile = document.getElementById('vidAnimateImgInput').files[0];
            const vidFile = document.getElementById('vidAnimateVideoInput').files[0];
            if (!imgFile) { toast('請上傳人物圖片', 'error'); btn.disabled = false; btn.innerHTML = _vidBtnHTML(); return; }
            if (!vidFile) { toast('請上傳參考影片', 'error'); btn.disabled = false; btn.innerHTML = _vidBtnHTML(); return; }
            const animateMode  = document.getElementById('videoAnimateMode').value;
            const checkImage   = document.getElementById('vidAnimateCheckImage').checked;
            const fd = new FormData();
            fd.append('model', model); fd.append('mode', animateMode);
            fd.append('watermark', vidWatermark); fd.append('check_image', checkImage);
            fd.append('image', imgFile); fd.append('video', vidFile);
            res = await apiPostForm('/api/video/animate', fd);

        } else {
            // r2v
            if (!refFiles.length) { toast('請上傳參考文件', 'error'); btn.disabled = false; btn.innerHTML = _vidBtnHTML(); return; }
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('resolution', resolution); fd.append('duration', duration);
            fd.append('prompt_extend', vidExtend); fd.append('watermark', vidWatermark);
            fd.append('audio', audio);
            if (vidSeed !== null) fd.append('seed', vidSeed);
            if (audio) {
                const bgmFile = document.getElementById('vidT2VAudioInput')?.files[0];
                if (bgmFile) fd.append('audio_file', bgmFile);
            }
            refFiles.forEach(f => fd.append('reference_files', f));
            res = await apiPostForm('/api/video/r2v', fd);
        }

        if (res.success && res.task_id) {
            addVideoTask(res.task_id, model, prompt, res.status, { resolution, seconds: duration });
            toast('任務已提交，輪詢中...', 'info');
        } else if (res.success && res.video_url) {
            addVideoResult(model, prompt, res.local_path || res.video_url, false, fmtElapsed(Date.now() - startTime));
            TaskHistory.save('video', model, prompt, res.local_path || res.video_url);
            toast('影片生成完成！', 'success');
            addVideoCost(model, { resolution, seconds: duration });
        } else {
            toast(res.error || '生成失敗', 'error');
            console.error('Video generation error:', res);
        }
    } catch (e) {
        toast(`錯誤：${e.message}`, 'error');
    }
    btn.disabled = false;
    btn.innerHTML = _vidBtnHTML();
}

function _vidBtnHTML() {
    return '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> 生成';
}

// ── 進行中的任務持久化 ──────────────────────────────────────────────────────
// 影片是非同步任務：提交後由前端輪詢。先前 task_id 只存在記憶體裡，使用者一重新
// 整理（或分頁被瀏覽器回收）輪詢就永久停止——**任務照樣跑完、照樣計費，但結果
// 再也拿不到**。一支 30 秒的影片可能要 $6.94，這是實打實的損失。
// 這裡把進行中的任務寫進 localStorage，載入時自動重建卡片並接續輪詢。
const PENDING_TASKS_KEY = 'nenai_pending_tasks';
const PENDING_TASK_MAX_AGE = 60 * 60 * 1000;   // 超過 1 小時的視為過期，不再嘗試

function _readPendingTasks() {
    try { return JSON.parse(localStorage.getItem(PENDING_TASKS_KEY) || '[]'); }
    catch (_) { return []; }
}

function _writePendingTasks(list) {
    try { localStorage.setItem(PENDING_TASKS_KEY, JSON.stringify(list.slice(0, 30))); }
    catch (_) { /* 儲存空間滿了就算了，不影響當下這次生成 */ }
}

function savePendingTask(entry) {
    const list = _readPendingTasks().filter(t => t.taskId !== entry.taskId);
    list.unshift({ ...entry, savedAt: Date.now() });
    _writePendingTasks(list);
}

function clearPendingTask(taskId) {
    _writePendingTasks(_readPendingTasks().filter(t => t.taskId !== taskId));
}

// 頁面載入時呼叫：把還沒結束的任務重新掛回畫面並接續輪詢
function resumePendingTasks() {
    const now = Date.now();
    const list = _readPendingTasks();
    const alive = list.filter(t => now - (t.savedAt || 0) < PENDING_TASK_MAX_AGE);
    if (alive.length !== list.length) _writePendingTasks(alive);
    if (!alive.length) return;
    alive.forEach(t => {
        try {
            if (t.kind === 'muleai') {
                addMuleAIVideoTask(t.taskId, t.model, t.prompt || '', '恢復中', true);
            } else {
                addVideoTask(t.taskId, t.model, t.prompt || '', '恢復中', t.costInfo, true);
            }
        } catch (e) { console.warn('恢復任務失敗', t.taskId, e); }
    });
    toast(`已恢復 ${alive.length} 個進行中的任務`, 'info');
}

function addVideoTask(taskId, model, prompt, status, costInfo, isResume = false) {
    const cont = document.getElementById('videoResults');
    cont.querySelector('.empty-state')?.remove();
    const startTime = Date.now();
    if (!isResume) savePendingTask({ kind: 'video', taskId, model, prompt, costInfo });
    const card = el('div', { className: 'video-task-card', id: `task-${taskId}` });
    card.innerHTML = `
        <div class="vtc-header">
            <span class="vtc-model">${model}</span>
            <span class="vtc-timer" id="tm-${taskId}">(耗時 0s)</span>
            <span class="vtc-status ${status?.toLowerCase() || 'pending'}" id="st-${taskId}">${status || 'PENDING'}</span>
        </div>
        <div class="vtc-prompt">${prompt.substring(0, 120)}${prompt.length > 120 ? '...' : ''}</div>
        <div class="vtc-progress"><div class="vtc-progress-bar" id="pb-${taskId}" style="width:5%"></div></div>
        <div id="rv-${taskId}"></div>`;
    cont.insertBefore(card, cont.firstChild);
    pollVideo(taskId, startTime, model, costInfo);
}

function addVideoResult(model, prompt, src, isHistory = false, elapsed = null) {
    const cont = document.getElementById('videoResults');
    cont.querySelector('.empty-state')?.remove();
    const card = el('div', { className: 'video-task-card' });
    card.innerHTML = `
        <div class="vtc-header"><span class="vtc-model">${model}</span>${elapsed ? `<span class="vtc-timer">(耗時 ${elapsed})</span>` : ''}<span class="vtc-status succeeded">SUCCEEDED</span></div>
        <div class="vtc-prompt">${prompt.substring(0, 120)}</div>
        <video class="video-player" controls src="${src}"></video>
        <div class="video-card-actions">
            <a href="${src}" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a>
            <button class="btn btn-ghost btn-sm" onclick="openLightbox('${src}', 'video')">展開預覽</button>
        </div>`;
    cont.insertBefore(card, cont.firstChild);
}

async function pollVideo(taskId, startTime, model, costInfo) {
    let tries = 0;
    const maxTries = 360; // 30 min max (5s * 360) — video-edit/重型任務常超過 15 min
    const poll = async () => {
        tries++;
        // 更新計時器
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const tmEl = document.getElementById(`tm-${taskId}`);
        const elapsedText = elapsed >= 60 ? `${Math.floor(elapsed/60)}m${elapsed%60}s` : `${elapsed}s`;
        if (tmEl) tmEl.textContent = `(耗時 ${elapsedText})`;

        if (tries > maxTries) { updateVTC(taskId, 'TIMEOUT', null, '等待超時'); clearPendingTask(taskId); return; }
        try {
            const res = await fetch(`/api/video/status/${taskId}`, { headers: { 'Authorization': `Bearer ${apiKey}` } });
            const data = await res.json();
            const st = data.status;
            const stEl = document.getElementById(`st-${taskId}`);
            const pbEl = document.getElementById(`pb-${taskId}`);
            const rvEl = document.getElementById(`rv-${taskId}`);

            const isDone = st && ['SUCCEEDED','COMPLETED','SUCCESS','SUCCEED','DONE','FINISHED','completed','success'].includes(st);
            const isFailed = st && ['FAILED','FAIL','FAILURE','ERROR','failed','error'].includes(st);
            if (isDone) {
                if (stEl) { stEl.textContent = 'SUCCEEDED'; stEl.className = 'vtc-status succeeded'; }
                if (pbEl) pbEl.style.width = '100%';
                if (rvEl && data.local_path) {
                    rvEl.innerHTML = `<video class="video-player" controls src="${data.local_path}"></video>
                        <div class="video-card-actions">
                            <a href="${data.local_path}" download class="img-dl">下載影片</a>
                            <button class="btn btn-ghost btn-sm" onclick="openLightbox('${data.local_path}', 'video')">展開預覽</button>
                        </div>`;
                    if (data.actual_prompt) {
                        const extEl = el('div', { className: 'img-actual-prompt' });
                        extEl.textContent = 'Prompt Extend 擴充後：' + data.actual_prompt;
                        rvEl.appendChild(extEl);
                    }
                } else if (rvEl && data.video_url) {
                    rvEl.innerHTML = `<video class="video-player" controls src="${data.video_url}"></video>
                        <div class="video-card-actions">
                            <a href="${data.video_url}" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a>
                            <button class="btn btn-ghost btn-sm" onclick="openLightbox('${data.video_url}', 'video')">展開預覽</button>
                        </div>`;
                    if (data.actual_prompt) {
                        const extEl = el('div', { className: 'img-actual-prompt' });
                        extEl.textContent = 'Prompt Extend 擴充後：' + data.actual_prompt;
                        rvEl.appendChild(extEl);
                    }
                }
                toast('影片生成完成！', 'success');
                addVideoCost(model, costInfo);
                clearPendingTask(taskId);
            } else if (isFailed) {
                const errMsg = data.error_message || 'Unknown';
                const isSchedulerErr = errMsg.toLowerCase().includes('scheduler');
                if (stEl) { stEl.textContent = 'FAILED'; stEl.className = 'vtc-status failed'; }
                if (pbEl) { pbEl.style.width = '100%'; pbEl.style.background = 'var(--red)'; }
                if (rvEl) {
                    const hint = isSchedulerErr
                        ? '<br><span style="color:var(--fg-muted)">DashScope 排程器暫時繁忙，請稍後重新提交任務。</span>'
                        : '';
                    rvEl.innerHTML = `<p style="font-size:0.82rem;color:var(--red)">錯誤：${errMsg}${hint}</p>`;
                }
                toast(isSchedulerErr ? 'DashScope 排程器繁忙，請重新提交' : '影片生成失敗', 'error');
                clearPendingTask(taskId);
            } else {
                if (stEl) { stEl.textContent = st || 'PENDING'; stEl.className = `vtc-status ${(st || 'pending').toLowerCase()}`; }
                // 進度條：前 30s 累積到 20%，之後緩慢增長到最多 90%
                const prog = elapsed < 30
                    ? 5 + (elapsed / 30) * 15
                    : Math.min(20 + ((elapsed - 30) / 600) * 70, 90);
                if (pbEl) pbEl.style.width = `${prog.toFixed(1)}%`;
                setTimeout(poll, 5000);
            }
        } catch (_) { setTimeout(poll, 5000); }
    };
    poll();
}

// ── Upload helpers ────────────────────────────────────────────
// ── Image Edit 多圖管理 ────────────────────────────────────────
function onImgFilesAdd(files) {
    const remaining = imgMaxRef - imgRefFiles.length;
    const toAdd = Array.from(files).slice(0, remaining);
    imgRefFiles = [...imgRefFiles, ...toAdd];
    renderImgThumbs();
    document.getElementById('imgFileInput').value = '';
}

function removeImgFile(idx) {
    imgRefFiles.splice(idx, 1);
    renderImgThumbs();
}

function renderImgThumbs() {
    const grid = document.getElementById('imgThumbGrid');
    const countEl = document.getElementById('imgRefCount');
    const addBtn = document.getElementById('imgAddBtn');
    if (!grid) return;
    grid.innerHTML = imgRefFiles.map((f, i) => `
        <div class="img-thumb">
            <img src="${URL.createObjectURL(f)}" alt="${f.name}">
            <button class="img-thumb-remove" onclick="removeImgFile(${i})">✕</button>
        </div>`).join('');
    if (countEl) countEl.textContent = `${imgRefFiles.length} / ${imgMaxRef} 張`;
    if (addBtn) addBtn.style.display = imgRefFiles.length >= imgMaxRef ? 'none' : '';
}

function previewImg(e, previewId, zoneId) {
    const file = e.target.files[0];
    if (!file) return;
    const preview = document.getElementById(previewId);
    preview.src = URL.createObjectURL(file);
    preview.classList.remove('hidden');
    document.querySelector(`#${zoneId} .upload-zone-icon`)?.classList.add('hidden');
    document.querySelector(`#${zoneId} p`)?.classList.add('hidden');
}

function handleRefUpload(e) {
    // 沒有明確標 max_ref 的模型維持原本「不設限」的行為（9 只是個保守的上界，
    // 沒有實測過的模型不要硬編一個猜測值進來擋人）
    refFiles = [...refFiles, ...Array.from(e.target.files)].slice(0, vidRefLimit('r2v', 99));
    renderRefList();
}
function renderRefList() {
    document.getElementById('refList').innerHTML = refFiles.map((f, i) => `
        <div class="ref-item">
            <span>${f.name}</span>
            <button onclick="removeRef(${i})">✕</button>
        </div>`).join('');
}
function removeRef(i) { refFiles.splice(i, 1); renderRefList(); }


// ── MuleAI Image Edit ──────────────────────────────────────────
function onMuleaiImgFilesAdd(files) {
    const remaining = 9 - muleaiImgRefFiles.length;
    const toAdd = Array.from(files).slice(0, remaining);
    muleaiImgRefFiles = [...muleaiImgRefFiles, ...toAdd];
    renderMuleaiImgThumbs();
    document.getElementById('muleaiImgFileInput').value = '';
}

function removeMuleaiImgFile(idx) {
    muleaiImgRefFiles.splice(idx, 1);
    renderMuleaiImgThumbs();
}

function renderMuleaiImgThumbs() {
    const grid = document.getElementById('muleaiImgThumbGrid');
    const countEl = document.getElementById('muleaiImgRefCount');
    const addBtn = document.getElementById('muleaiImgAddBtn');
    if (!grid) return;
    grid.innerHTML = muleaiImgRefFiles.map((f, i) => `
        <div class="img-thumb">
            <img src="${URL.createObjectURL(f)}" alt="${f.name}">
            <button class="img-thumb-remove" onclick="removeMuleaiImgFile(${i})">✕</button>
        </div>`).join('');
    if (countEl) countEl.textContent = `${muleaiImgRefFiles.length} / 9 張`;
    if (addBtn) addBtn.style.display = muleaiImgRefFiles.length >= 9 ? 'none' : '';
}

// ── Voice (ASR / TTS) ─────────────────────────────────────────
let voiceAsrFile = null;

function onVoiceTaskChange() {
    const t = document.getElementById('voiceTaskType').value;
    populateSelect('voiceModel', models.voice?.[t] || []);
    document.getElementById('voiceAsrUploadSection').style.display = t === 'asr' ? '' : 'none';
    document.getElementById('voiceTtsSettingsSection').style.display = t === 'tts' ? '' : 'none';
    document.getElementById('voiceTtsAdvancedSection').style.display = t === 'tts' ? '' : 'none';
    document.getElementById('voiceAsrPromptPanel').style.display = t === 'asr' ? '' : 'none';
    document.getElementById('voiceTtsPromptPanel').style.display = t === 'tts' ? '' : 'none';
    onVoiceModelChange();
}

function onVoiceModelChange() {
    const t = document.getElementById('voiceTaskType').value;
    updateModelPriceHint('voiceModelPrice', document.getElementById('voiceModel').value);
    if (t !== 'tts') return;
    const model = document.getElementById('voiceModel').value;
    // Gemini TTS 走 /v1/audio/speech，只吃 model/input/voice——instructions 帶了
    // 上游會直接回 400，sample_rate/volume/language_hints 也不支援，所以這些
    // CosyVoice 專屬的進階欄位只在選到 qwen-audio-3.0-tts 系列時才顯示。
    const isGemini = model.startsWith('gemini');
    document.getElementById('voiceTtsAdvancedSection').style.display = isGemini ? 'none' : '';

    // 音色下拉選單依選到的模型重建——qwen 的兩個模型各自只支援自己專屬的音色，
    // 不能混用；3 個 gemini 模型則共用同一組 30 個官方音色。
    const modelInfo = (models.voice?.tts || []).find(m => m.id === model);
    const voices = (modelInfo && modelInfo.voices) || [];
    const voiceSel = document.getElementById('voiceTtsVoice');
    voiceSel.innerHTML = '<option value="">留空 = 預設音色</option>' +
        voices.map(v => `<option value="${v.id}">${v.name} — ${v.desc}</option>`).join('');
}

function onVoiceAsrFileChange(event) {
    const file = event.target.files[0];
    if (!file) return;
    voiceAsrFile = file;
    document.getElementById('voiceAsrFileName').textContent = file.name;
    document.getElementById('voiceAsrLabel').innerHTML = `已選擇：${file.name}`;
    document.getElementById('voiceAsrIcon').textContent = '✅';
    document.getElementById('voiceAsrClearBtn').style.display = '';
}

function clearVoiceAsrFile() {
    voiceAsrFile = null;
    document.getElementById('voiceAsrFileInput').value = '';
    document.getElementById('voiceAsrFileName').textContent = '尚未選擇音檔';
    document.getElementById('voiceAsrLabel').innerHTML = '上傳音檔<br><span style="font-size:11px;color:var(--text-muted)">支援 WAV / MP3 / OGG 等常見格式</span>';
    document.getElementById('voiceAsrIcon').textContent = '🎙';
    document.getElementById('voiceAsrClearBtn').style.display = 'none';
}

function addVoiceResultCard(title) {
    const area = document.getElementById('voiceResults');
    area.querySelector('.empty-state')?.remove();
    const card = el('div', { className: 'voice-result' });
    card.innerHTML = `<div class="voice-result-header"><span>${title}</span></div>`;
    area.insertBefore(card, area.firstChild);
    return card;
}

async function sendVoiceAsr() {
    const model = document.getElementById('voiceModel').value;
    if (!voiceAsrFile) { toast('請先上傳音檔', 'error'); return; }

    const btn = document.getElementById('voiceAsrSendBtn');
    btn.disabled = true;
    showLoading('語音辨識中，請稍候...');
    const startTime = Date.now();
    const isStreaming = model.includes('streaming');

    const card = addVoiceResultCard(`${model}（辨識中…）`);
    const textEl = el('div', { className: 'voice-result-text', textContent: '' });
    card.appendChild(textEl);

    try {
        if (isStreaming) {
            const fd = new FormData();
            fd.append('model', model);
            fd.append('audio', voiceAsrFile);
            const resp = await fetch('/api/voice/asr/stream', { method: 'POST', headers: { 'Authorization': `Bearer ${apiKey}` }, body: fd });
            if (resp.status === 401) { handleLogout(); throw new Error('Unauthorized'); }
            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '', fullText = '';
            while (true) {
                const { done, value } = await reader.read();
                if (done) break;
                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n\n');
                buffer = lines.pop();
                for (const line of lines) {
                    if (!line.startsWith('data:')) continue;
                    const payload = line.slice(5).trim();
                    if (!payload) continue;
                    try {
                        const evt = JSON.parse(payload);
                        if (evt.type === 'error') throw new Error(evt.error || '串流辨識失敗');
                        const delta = evt.text ?? evt.delta ?? '';
                        if (delta) { fullText += delta; textEl.textContent = fullText; }
                    } catch (parseErr) { /* 忽略無法解析的中繼事件 */ }
                }
            }
            card.querySelector('.voice-result-header span').textContent = `${model}（耗時 ${fmtElapsed(Date.now() - startTime)}）`;
            toast('語音辨識完成！', 'success');
            addFixedCost(model);
        } else {
            const fd = new FormData();
            fd.append('model', model);
            fd.append('audio', voiceAsrFile);
            const res = await apiPostForm('/api/voice/asr', fd);
            if (res.success) {
                textEl.textContent = res.text || '（無辨識結果）';
                card.querySelector('.voice-result-header span').textContent = `${model}（耗時 ${fmtElapsed(Date.now() - startTime)}）`;
                toast('語音辨識完成！', 'success');
                addFixedCost(model);
            } else {
                throw new Error(res.error || '辨識失敗');
            }
        }
    } catch (e) {
        card.querySelector('.voice-result-header span').textContent = `${model}（失敗）`;
        textEl.textContent = `錯誤：${e.message}`;
        toast(`錯誤：${e.message}`, 'error');
    }
    hideLoading();
    btn.disabled = false;
}

async function sendVoiceTts() {
    const model        = document.getElementById('voiceModel').value;
    const text         = document.getElementById('voiceTtsText').value.trim();
    const voice        = document.getElementById('voiceTtsVoice').value.trim();
    const format       = document.getElementById('voiceTtsFormat').value;
    const instructions = document.getElementById('voiceTtsInstructions').value.trim();
    const sampleRateRaw = document.getElementById('voiceTtsSampleRate').value;
    const sampleRate   = sampleRateRaw ? parseInt(sampleRateRaw) : null;
    const volume       = parseInt(document.getElementById('voiceTtsVolume').value);
    // language_hints 上游目前只處理陣列第一個值，這裡固定只送一個
    const langHint = document.getElementById('voiceTtsLangHint').value;
    const languageHints = langHint ? [langHint] : [];
    if (!text) { toast('請輸入文字內容', 'error'); return; }

    const btn = document.getElementById('voiceTtsSendBtn');
    btn.disabled = true;
    showLoading('語音合成中，請稍候...');
    const startTime = Date.now();

    try {
        const body = { model, text, voice, format, instructions, volume, language_hints: languageHints };
        if (sampleRate !== null) body.sample_rate = sampleRate;
        const res = await apiPost('/api/voice/tts', body);
        if (res.success && res.audio_url) {
            const elapsed = fmtElapsed(Date.now() - startTime);
            const card = addVoiceResultCard(`${model}（耗時 ${elapsed}）`);
            const audioEl = el('audio', { controls: true });
            audioEl.src = res.audio_url;
            card.appendChild(audioEl);
            const meta = el('div', { className: 'voice-result-meta' });
            meta.innerHTML = `<a href="${res.audio_url}" download>下載音檔</a>`;
            card.appendChild(meta);
            toast('語音合成完成！', 'success');
        } else {
            toast(res.error || '合成失敗', 'error');
        }
    } catch (e) {
        toast(`錯誤：${e.message}`, 'error');
    }
    hideLoading();
    btn.disabled = false;
}

// ── API helpers ───────────────────────────────────────────────
async function apiPost(url, body) {
    const r = await fetch(url, { method: 'POST', headers: authHeader(), body: JSON.stringify(body) });
    if (r.status === 401) { handleLogout(); throw new Error('Unauthorized'); }
    return r.json();
}
async function apiPostForm(url, fd) {
    const headers = { 'Authorization': `Bearer ${apiKey}` };
    const r = await fetch(url, { method: 'POST', headers: headers, body: fd });
    if (r.status === 401) { handleLogout(); throw new Error('Unauthorized'); }
    if (r.status === 413) throw new Error('上傳檔案過大（上限 200MB）');
    const ct = r.headers.get('Content-Type') || '';
    if (ct.includes('application/json')) return r.json();
    // 非 JSON 回應（如 nginx 錯誤頁）
    const text = await r.text();
    throw new Error(`伺服器錯誤 ${r.status}: ${text.slice(0, 120)}`);
}

// ── Toast ─────────────────────────────────────────────────────
function toast(msg, type = 'info') {
    const t = el('div', { className: `toast ${type}`, textContent: msg });
    document.getElementById('toastContainer').appendChild(t);
    setTimeout(() => { t.style.opacity = '0'; t.style.transform = 'translateX(24px)'; t.style.transition = '0.25s ease'; setTimeout(() => t.remove(), 280); }, 3800);
}

// ── Loading with dynamic timer ────────────────────────────────
function showLoading(txt = '處理中...') {
    document.getElementById('loadingText').textContent = txt;
    document.getElementById('loadingTimer').textContent = '';
    document.getElementById('loadingOverlay').classList.remove('hidden');

    // 動態計時
    const startTime = Date.now();
    loadingTimerInterval = setInterval(() => {
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const display = elapsed >= 60 ? `${Math.floor(elapsed/60)}m ${elapsed%60}s` : `${elapsed}s`;
        const timerEl = document.getElementById('loadingTimer');
        if (timerEl) timerEl.textContent = `已等待 ${display}`;
    }, 1000);
}

function hideLoading() {
    document.getElementById('loadingOverlay').classList.add('hidden');
    if (loadingTimerInterval) {
        clearInterval(loadingTimerInterval);
        loadingTimerInterval = null;
    }
}

// ── Tooltip ───────────────────────────────────────────────────
(function initTooltips() {
    let tip = null;

    function createTip(text) {
        tip = document.createElement('div');
        tip.className = 'global-tooltip';
        tip.textContent = text;
        document.body.appendChild(tip);
    }

    function positionTip(target) {
        if (!tip) return;
        const r = target.getBoundingClientRect();
        const tw = tip.offsetWidth;
        const th = tip.offsetHeight;
        let left = r.left + r.width / 2 - tw / 2;
        let top  = r.top - th - 8;
        if (left < 8) left = 8;
        if (left + tw > window.innerWidth - 8) left = window.innerWidth - tw - 8;
        if (top < 8) top = r.bottom + 8;
        tip.style.left = left + 'px';
        tip.style.top  = top  + 'px';
    }

    function removeTip() {
        if (tip) { tip.remove(); tip = null; }
    }

    document.addEventListener('mouseover', e => {
        const el = e.target.closest('[data-tip]');
        if (!el) return;
        removeTip();
        createTip(el.dataset.tip);
        positionTip(el);
    });

    document.addEventListener('mousemove', e => {
        if (!tip) return;
        const el = e.target.closest('[data-tip]');
        if (el) positionTip(el);
        else removeTip();
    });

    document.addEventListener('mouseout', e => {
        if (!e.target.closest('[data-tip]')) removeTip();
    });
})();

// ── Utils ─────────────────────────────────────────────────────
function el(tag, props = {}) {
    return Object.assign(document.createElement(tag), props);
}

// 圖片/影片結果卡片共用的「耗時」顯示格式，跟影片任務輪詢的計時器（pollVideo）
// 保持一致：60 秒內顯示整數秒，超過則顯示 分m秒s
function fmtElapsed(ms) {
    const sec = Math.floor(ms / 1000);
    return sec >= 60 ? `${Math.floor(sec / 60)}m${sec % 60}s` : `${sec}s`;
}


// ── Lightbox ──────────────────────────────────────────────────
function openLightbox(src, type = 'image') {
    const lb  = document.getElementById('lightbox');
    const img = document.getElementById('lightboxImg');
    const vid = document.getElementById('lightboxVideo');
    if (type === 'video') {
        img.style.display = 'none';
        img.src = '';
        vid.src = src;
        vid.style.display = '';
    } else {
        vid.style.display = 'none';
        vid.pause();
        vid.src = '';
        img.src = src;
        img.style.display = '';
    }
    lb.classList.remove('hidden');
    document.body.style.overflow = 'hidden';
}

function closeLightbox() {
    const lb  = document.getElementById('lightbox');
    const vid = document.getElementById('lightboxVideo');
    lb.classList.add('hidden');
    vid.pause();
    vid.src = '';
    document.getElementById('lightboxImg').src = '';
    document.body.style.overflow = '';
}

document.addEventListener('keydown', e => { if (e.key === 'Escape') closeLightbox(); });

// ── MuleAI Generation ───────────────────────────────────────────

async function sendMuleAIVideo() {
    const model      = document.getElementById('muleaiModel').value || 'wan2.7-i2v-spicy';
    const isZImage   = model.includes('z-image');
    const isImgEdit  = model === 'qwen-image-edit-spicy';
    const isFaceSwap = model === 'face-swap';
    const isVideoModel = !isZImage && !isImgEdit && !isFaceSwap;

    const prompt    = document.getElementById('muleaiVidPrompt').value.trim();
    if (!prompt && !isFaceSwap) { toast('請輸入 Prompt', 'error'); return; }

    const negPrompt  = document.getElementById('muleaiVidNegPrompt').value.trim();
    const resolution = document.getElementById('muleaiVidResolution').value;
    const duration   = parseInt(document.getElementById('muleaiVidDuration').value);
    const extend     = document.getElementById('muleaiVidPromptExtend').checked;
    const seedRaw    = document.getElementById('muleaiVidSeed').value.trim();
    const seed       = seedRaw !== '' ? parseInt(seedRaw) : null;

    const fd = new FormData();
    fd.append('model', model);

    if (isFaceSwap) {
        const srcFile  = document.getElementById('muleaiFirstFrameInput').files[0];
        const faceFile = document.getElementById('muleaiFaceImgInput').files[0];
        if (!srcFile)  { toast('請上傳來源圖片', 'error'); return; }
        if (!faceFile) { toast('請上傳換臉參考圖', 'error'); return; }
        fd.append('image', srcFile);
        fd.append('face_image', faceFile);

    } else if (isImgEdit) {
        const srcFile = document.getElementById('muleaiFirstFrameInput').files[0];
        if (!srcFile) { toast('請上傳來源圖片', 'error'); return; }
        fd.append('image', srcFile);
        fd.append('prompt', prompt);
        fd.append('negative_prompt', negPrompt);
        fd.append('prompt_extend', extend);
        if (seed !== null) fd.append('seed', seed);

    } else if (isZImage) {
        const imgRes = document.getElementById('muleaiImgResolution').value;
        fd.append('img_resolution', imgRes);
        fd.append('prompt', prompt);
        fd.append('negative_prompt', negPrompt);
        fd.append('prompt_extend', extend);
        if (seed !== null) fd.append('seed', seed);

    } else {
        // 影片模型
        const firstFrameFile = document.getElementById('muleaiFirstFrameInput').files[0];
        if (!firstFrameFile) { toast('請上傳首幀圖片', 'error'); return; }
        fd.append('image', firstFrameFile);
        fd.append('prompt', prompt);
        fd.append('negative_prompt', negPrompt);
        fd.append('resolution', resolution);
        fd.append('duration', duration);
        fd.append('prompt_extend', extend);
        if (seed !== null) fd.append('seed', seed);
        const audioEnabled = document.getElementById('muleaiAudioEnable')?.checked;
        if (audioEnabled) {
            fd.append('enable_audio', 'true');
            const audioInput = document.getElementById('muleaiAudioInput');
            if (audioInput && audioInput.files[0]) fd.append('audio', audioInput.files[0]);
        }
    }

    const btn = document.getElementById('muleaiVidSendBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> 提交中...';

    try {
        const res = await apiPostForm('/api/muleai/generate', fd);
        if (res.success && res.task_id) {
            const displayPrompt = isFaceSwap ? '換臉任務' : prompt;
            addMuleAIVideoTask(res.task_id, model, displayPrompt, res.status);
            toast('任務已提交，輪詢中...', 'info');
        } else {
            toast(res.error || '提交失敗', 'error');
        }
    } catch (e) {
        toast('錯誤：' + e.message, 'error');
    }
    btn.disabled = false;
    const label = isVideoModel ? '生成影片' : '生成圖片';
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> ' + label;
}

function addMuleAIVideoTask(taskId, model, prompt, status, isResume = false) {
    const cont = document.getElementById('muleaiVideoResults');
    const empty = cont.querySelector('.empty-state');
    if (empty) empty.remove();
    const startTime = Date.now();
    if (!isResume) savePendingTask({ kind: 'muleai', taskId, model, prompt });
    const card = el('div', { className: 'video-task-card', id: 'mtask-' + taskId });
    card.innerHTML = '<div class="vtc-header"><span class="vtc-model">' + model + '</span><span class="vtc-timer" id="mtm-' + taskId + '">(耗時 0s)</span><span class="vtc-status ' + (status ? status.toLowerCase() : 'pending') + '" id="mst-' + taskId + '">' + (status || 'PENDING') + '</span></div><div class="vtc-prompt">' + prompt.substring(0, 120) + '</div><div class="vtc-progress"><div class="vtc-progress-bar" id="mpb-' + taskId + '" style="width:5%"></div></div><div id="mrv-' + taskId + '"></div>';
    cont.insertBefore(card, cont.firstChild);
    pollMuleAIVideo(taskId, startTime, model, prompt);
}

async function pollMuleAIVideo(taskId, startTime, model, promptText) {
    let tries = 0;
    const maxTries = 360; // Up to 30 mins
    const poll = async () => {
        tries++;
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const tmEl = document.getElementById('mtm-' + taskId);
        const elapsedText = elapsed >= 60 ? Math.floor(elapsed/60) + 'm' + (elapsed%60) + 's' : elapsed + 's';
        if (tmEl) tmEl.textContent = '(耗時 ' + elapsedText + ')';

        if (tries > maxTries) { 
            const stEl = document.getElementById('mst-' + taskId);
            if (stEl) { stEl.textContent = 'TIMEOUT'; stEl.className = 'vtc-status failed'; }
            clearPendingTask(taskId);
            return; 
        }
        
        try {
            const res = await fetch(`/api/muleai/status/${model}/${taskId}`, { headers: authHeader() });
            const data = await res.json();
            const st = data.status;
            const stEl = document.getElementById('mst-' + taskId);
            const pbEl = document.getElementById('mpb-' + taskId);
            const rvEl = document.getElementById('mrv-' + taskId);

            if (st && ['SUCCEEDED', 'SUCCESS', 'completed', 'COMPLETED'].includes(st.toUpperCase ? st.toUpperCase() : st)) {
                if (stEl) { stEl.textContent = 'SUCCEEDED'; stEl.className = 'vtc-status succeeded'; }
                if (pbEl) pbEl.style.width = '100%';
                if (rvEl) {
                    if (data.videos && data.videos.length > 0) {
                        const src = data.videos[0];
                        rvEl.innerHTML = '<video class="video-player" controls src="' + src + '"></video><div class="video-card-actions"><a href="' + src + '" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a><button class="btn btn-ghost btn-sm" onclick="openLightbox(\'' + src + '\', \'video\')">展開預覽</button></div>';
                        TaskHistory.save('muleai_video', model, promptText || 'MuleAI Video', src);
                    } else if (data.images && data.images.length > 0) {
                        const src = data.images[0];
                        rvEl.innerHTML = '<img src="' + src + '" alt="Generated Image" class="muleai-img-result" onclick="openLightbox(\'' + src + '\')"><div class="video-card-actions"><a href="' + src + '" download target="_blank" rel="noopener noreferrer" class="img-dl">下載圖片</a></div>';
                        TaskHistory.save('muleai_image', model, promptText || 'MuleAI Image', src);
                    }
                }
                toast('任務完成！', 'success');
                addFixedCost(model);
                clearPendingTask(taskId);
            } else if (st === 'FAILED' || st === 'failed') {
                if (stEl) { stEl.textContent = 'FAILED'; stEl.className = 'vtc-status failed'; }
                if (pbEl) { pbEl.style.width = '100%'; pbEl.style.background = 'var(--red)'; }
                if (rvEl) rvEl.innerHTML = '<p style="font-size:0.82rem;color:var(--red)">錯誤：' + (data.error_message || (data.error ? data.error.detail : '未知錯誤')) + '</p>';
                toast('生成失敗', 'error');
                clearPendingTask(taskId);
            } else {
                if (stEl) { stEl.textContent = st || 'PENDING'; stEl.className = 'vtc-status ' + (st ? st.toLowerCase() : 'pending'); }
                const prog = Math.min(5 + (elapsed / 60) * 80, 90);
                if (pbEl) pbEl.style.width = prog.toFixed(1) + '%';
                setTimeout(poll, 10000); // Check every 10s
            }
        } catch (_) { setTimeout(poll, 10000); }
    };
    poll();
}
