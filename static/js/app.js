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
// w3.0 影片四顆走跟其他 Spicy 模型不同的請求形狀：首幀圖選填、多了畫面比例與
// 智能時長、解析度檔次逐顆不同（pro 系列沒有 720p 以下）。用前綴判斷，之後若
// 再加同家族的型號會自動落進同一條路徑。
function isW3SpicyVideo(modelId) {
    return typeof modelId === 'string' && modelId.startsWith('w3.0-video');
}

function _muleaiMeta(modelId) {
    return (models.muleai || []).find(m => m.id === modelId) || {};
}

// ── w3.0 參考素材 ─────────────────────────────────────────────
// 上限：圖片 10、影片 5、音訊 5；影片總長 ≤15 秒；有影片輸入時「輸入總長 ＋ 輸出
// 時長 ≤30 秒」。最後那條**閘道不擋**（它要先把每支素材抓回來才知道長度），會由
// 上游拒絕——所以在送出前自己用 <video>.duration 算，否則使用者要等任務失敗才知道。
const MULEAI_REF_LIMITS = { image: 10, video: 5, audio: 5 };
const MULEAI_REF_VIDEO_TOTAL_SEC = 15;
const MULEAI_REF_TOTAL_WITH_OUTPUT_SEC = 30;
let muleaiRefFiles = { image: [], video: [], audio: [] };

// 讀本機檔案的長度（秒）。讀不到就回 null，呼叫端一律當成「不確定」而不是 0，
// 否則一個讀不到長度的檔案會讓總長被低估、檢查形同虛設
function probeMediaDuration(file) {
    return new Promise((resolve) => {
        const isVideo = file.type.startsWith('video');
        const el = document.createElement(isVideo ? 'video' : 'audio');
        const url = URL.createObjectURL(file);
        const done = (v) => { URL.revokeObjectURL(url); resolve(v); };
        el.preload = 'metadata';
        el.onloadedmetadata = () => done(Number.isFinite(el.duration) ? el.duration : null);
        el.onerror = () => done(null);
        el.src = url;
    });
}

async function onMuleaiRefAdd(kind, files) {
    const cap = MULEAI_REF_LIMITS[kind];
    const room = cap - muleaiRefFiles[kind].length;
    const toAdd = Array.from(files).slice(0, room);
    if (files.length > room) toast(`${kind === 'image' ? '參考圖' : kind === 'video' ? '參考影片' : '參考音訊'}最多 ${cap} 個`, 'error');
    for (const f of toAdd) {
        f._dur = (kind === 'image') ? 0 : await probeMediaDuration(f);
        muleaiRefFiles[kind].push(f);
    }
    document.getElementById(`muleaiRef${kind === 'image' ? 'Img' : kind === 'video' ? 'Vid' : 'Aud'}Input`).value = '';
    renderMuleaiRefLists();
}

function removeMuleaiRef(kind, idx) { muleaiRefFiles[kind].splice(idx, 1); renderMuleaiRefLists(); }

function renderMuleaiRefLists() {
    const map = { image: ['Img', '10'], video: ['Vid', '5'], audio: ['Aud', '5'] };
    Object.entries(map).forEach(([kind, [tag, cap]]) => {
        const list = document.getElementById(`muleaiRef${tag}List`);
        const cnt = document.getElementById(`muleaiRef${tag}Count`);
        if (!list) return;
        list.innerHTML = muleaiRefFiles[kind].map((f, i) => {
            const d = f._dur == null ? '長度未知' : (kind === 'image' ? '' : `${f._dur.toFixed(1)}s`);
            return `<div class="ref-item"><span>${i + 1}. ${f.name}${d ? ' · ' + d : ''}</span>` +
                   `<button onclick="removeMuleaiRef('${kind}',${i})">✕</button></div>`;
        }).join('');
        if (cnt) cnt.textContent = `${muleaiRefFiles[kind].length} / ${cap}`;
    });
    updateMuleaiRefDurHint();
}

// 回傳 null 代表沒問題，否則回傳要顯示給使用者的錯誤字串
function checkMuleaiRefDurations() {
    const vids = muleaiRefFiles.video;
    if (!vids.length) return null;
    if (vids.some(f => f._dur == null)) return '有參考影片讀不出長度，無法確認是否超過限制';
    const total = vids.reduce((a, f) => a + f._dur, 0);
    if (total > MULEAI_REF_VIDEO_TOTAL_SEC)
        return `參考影片總長 ${total.toFixed(1)} 秒，超過 ${MULEAI_REF_VIDEO_TOTAL_SEC} 秒上限`;
    const out = document.getElementById('muleaiSmartDur')?.checked
        ? null : parseInt(document.getElementById('muleaiVidDuration').value);
    if (out != null && total + out > MULEAI_REF_TOTAL_WITH_OUTPUT_SEC)
        return `參考影片總長 ${total.toFixed(1)} 秒 ＋ 輸出 ${out} 秒 = ${(total + out).toFixed(1)} 秒，超過 ${MULEAI_REF_TOTAL_WITH_OUTPUT_SEC} 秒上限`;
    return null;
}

function updateMuleaiRefDurHint() {
    const el = document.getElementById('muleaiRefDurHint');
    if (!el) return;
    const err = checkMuleaiRefDurations();
    el.textContent = err || '';
    el.style.color = err ? 'var(--danger, #B3574F)' : '';
}

function onMuleaiW3ModeChange() {
    const isRef = document.getElementById('muleaiW3Mode').value === 'reference';
    document.getElementById('muleaiRefSection').style.display = isRef ? '' : 'none';
    document.getElementById('muleaiLastFrameSection').style.display = isRef ? 'none' : '';
    // 首幀圖屬於 keyframe 模式，切到參考模式就收起來（送出時也不會帶）
    document.getElementById('muleaiImgUploadSection').style.display = isRef ? 'none' : '';
    if (isRef) renderMuleaiRefLists();
}

function onMuleaiSmartDurToggle() {
    const on = document.getElementById('muleaiSmartDur').checked;
    const slider = document.getElementById('muleaiVidDuration');
    slider.disabled = on;
    // 智能時長由模型自行決定長度，事前算不出秒數，所以不顯示估價（顯示了會嚴重低估）
    document.getElementById('muleaiVidDurVal').textContent = on ? '由模型決定' : slider.value;
    updateMuleaiRefDurHint();
}

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

    // w3.0：解析度／比例／時長上限逐顆不同，一律由 MODELS 的資料驅動，不要寫死
    const isW3  = isW3SpicyVideo(model);
    const meta  = isW3 ? _muleaiMeta(model) : {};
    const resSel = document.getElementById('muleaiVidResolution');
    if (isW3) {
        const list = meta.resolutions || [];
        resSel.innerHTML = list.map(r => `<option value="${r}">${r}</option>`).join('');
        // 預設選最低的一檔（清單第一個就是最便宜的），不用上游的 1080p 預設——
        // 讓使用者主動往上選，而不是預設就落在較高的檔次
        if (list.length) resSel.value = list[0];
    } else if (isVideoModel) {
        resSel.innerHTML = '<option value="1080P">1080P</option><option value="720P">720P</option>';
    }

    const ratioGroup = document.getElementById('muleaiVidRatioGroup');
    ratioGroup.style.display = isW3 ? '' : 'none';
    if (isW3) {
        const ratios = meta.ratios || [];
        const ratioSel = document.getElementById('muleaiVidRatio');
        ratioSel.innerHTML = ratios.map(r => `<option value="${r}">${r === 'adaptive' ? '自動 (adaptive)' : r}</option>`).join('');
    }

    const durSlider = document.getElementById('muleaiVidDuration');
    if (isVideoModel) {
        durSlider.min = meta.min_dur || 2;
        durSlider.max = meta.max_dur || 15;
        if (+durSlider.value > +durSlider.max) durSlider.value = durSlider.max;
        if (+durSlider.value < +durSlider.min) durSlider.value = durSlider.min;
        document.getElementById('muleaiVidDurVal').textContent = durSlider.value;
    }
    const smartGroup = document.getElementById('muleaiSmartDurGroup');
    smartGroup.style.display = (isW3 && meta.smart_duration) ? '' : 'none';
    if (!isW3 || !meta.smart_duration) {
        const sd = document.getElementById('muleaiSmartDur');
        if (sd) sd.checked = false;
        durSlider.disabled = false;
    }

    // 首幀 / 來源圖上傳區
    document.getElementById('muleaiImgUploadSection').style.display = (isVideoModel || isImgEdit || isFaceSwap) ? '' : 'none';

    const uploadTitle = document.getElementById('muleaiImgUploadTitle');
    if (uploadTitle) {
        if (isFaceSwap)     uploadTitle.textContent = '來源圖片 (必填)';
        else if (isImgEdit) uploadTitle.textContent = '來源圖片 (必填)';
        else if (isW3)      uploadTitle.textContent = '首幀圖片 (選填，不放就是文生影片)';
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
    if (isW3) document.getElementById('muleaiAudioUploadSection').style.display = 'none';

    // 素材模式（首尾幀／參考）只有 w3.0 有；切到別的模型要把這幾區收乾淨，
    // 並且把首幀圖上傳區還原回來（參考模式會把它藏起來）
    document.getElementById('muleaiW3ModeSection').style.display = isW3 ? '' : 'none';
    if (isW3) {
        onMuleaiW3ModeChange();
    } else {
        document.getElementById('muleaiRefSection').style.display = 'none';
        document.getElementById('muleaiLastFrameSection').style.display = 'none';
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
    // w3.0 的配音是純開關（由模型自行產生聲音），沒有「上傳音軌」這回事，
    // 開關打開也不要露出上傳區
    const isW3 = isW3SpicyVideo(document.getElementById('muleaiModel').value);
    document.getElementById('muleaiAudioUploadSection').style.display = (enabled && !isW3) ? '' : 'none';
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
    // 部署閘門：後端會把正式環境還沒部署的模型從 /api/models 拿掉（app.py 的
    // _DEPLOY_GATED_MODELS）。音樂整類都被拿掉時，連任務選項一起藏——留一個
    // 選了之後模型下拉是空白的任務類型，比沒有更糟
    const musicOpt = document.querySelector('#voiceTaskType option[value="music"]');
    if (musicOpt) {
        const hasMusic = (models.voice?.music || []).length > 0;
        musicOpt.hidden = !hasMusic;
        musicOpt.disabled = !hasMusic;
        if (!hasMusic && document.getElementById('voiceTaskType').value === 'music') {
            document.getElementById('voiceTaskType').value = 'asr';
        }
    }
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

// 上傳前把長邊縮到 2048。實測 qwen3-vl-flash 的圖片 token 在 2048×2048 之後就**封頂
// 不再增加**（512²=285、1024²=1053、1536²=2333、2048²=2529，之後 3072² 與 4096² 都是
// 2529），也就是說更大的圖不會被更精細地看，只是讓使用者多等上傳時間——同一組測試圖
// 4096² 的 PNG 是 279K、2048² 只有 111K，真實照片的差距會更大。
// 縮圖在瀏覽器端做，後端與上游都不必改。
const VISION_MAX_EDGE = 2048;

function _downscaleImage(file) {
    return new Promise((resolve) => {
        const fr = new FileReader();
        fr.onload = () => {
            const dataUrl = fr.result;
            const img = new Image();
            img.onload = () => {
                const long = Math.max(img.width, img.height);
                // 本來就在上限內就原樣送出，不要重新編碼——重壓一次只會讓畫質變差
                if (long <= VISION_MAX_EDGE) { resolve({ url: dataUrl, w: img.width, h: img.height, scaled: false }); return; }
                const ratio = VISION_MAX_EDGE / long;
                const w = Math.round(img.width * ratio), h = Math.round(img.height * ratio);
                const cv = document.createElement('canvas');
                cv.width = w; cv.height = h;
                cv.getContext('2d').drawImage(img, 0, 0, w, h);
                // 統一輸出 JPEG：縮圖後的照片用 PNG 反而更大，而 0.92 品質對辨識沒有影響
                resolve({ url: cv.toDataURL('image/jpeg', 0.92), w, h, scaled: true });
            };
            // 讀不出來（壞檔、或瀏覽器不支援的格式）就原樣送出，讓上游去回報錯誤，
            // 不要在這裡把它擋掉——縮圖只是最佳化，不該變成新的失敗點
            img.onerror = () => resolve({ url: dataUrl, scaled: false });
            img.src = dataUrl;
        };
        fr.onerror = () => resolve(null);
        fr.readAsDataURL(file);
    });
}

function onTextVisionUpload(e) {
    const files = Array.from(e.target.files || []);
    Promise.all(files.map(async (f) => {
        const r = await _downscaleImage(f);
        if (!r) return null;
        return { name: f.name, url: r.url, scaled: r.scaled, w: r.w, h: r.h };
    })).then(added => {
        const ok = added.filter(Boolean);
        textVisionImages = [...textVisionImages, ...ok];
        renderTextVisionList();
        const scaled = ok.filter(x => x.scaled);
        if (scaled.length) {
            toast(`已將 ${scaled.length} 張圖縮到長邊 ${VISION_MAX_EDGE}（超過這個尺寸不會提高辨識精細度，只會拉長上傳時間）`);
        }
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

    // no_sampling：後端對這些模型（Claude 系，Bedrock 限制）刻意不送 temperature/
    // top_p，滑桿顯示著只會讓使用者以為調了有效——整組收起來
    document.getElementById('textTempGroup').style.display = modelInfo.no_sampling ? 'none' : '';
    document.getElementById('textTopPGroup').style.display = modelInfo.no_sampling ? 'none' : '';

    // 百煉方言四參數：依 MODELS 旗標分家族顯示（適用清單見 reference §2.3.24）。
    // clear_thinking（GLM）與 preserve_thinking（qwen3.7/3.6 系）是兩顆給不同家族的
    // 參數，各自獨立顯示，不做三態選擇器
    document.getElementById('textThinkingBudgetGroup').style.display = modelInfo.thinking_budget ? '' : 'none';
    document.getElementById('textClearThinkingGroup').style.display = modelInfo.clear_thinking ? '' : 'none';
    document.getElementById('textPreserveThinkingGroup').style.display = modelInfo.preserve_thinking ? '' : 'none';
    document.getElementById('textRepPenaltyGroup').style.display = modelInfo.repetition_penalty ? '' : 'none';

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
// 手機版：收合／展開左側參數欄。桌面版這顆按鈕是隱藏的（CSS 的 media query），
// 所以這裡不需要判斷視窗寬度。
function toggleMobileParams() {
    const collapsed = document.body.classList.toggle('params-collapsed');
    const btn = document.getElementById('mobileParamsToggle');
    if (btn) {
        btn.textContent = collapsed ? '▼ 參數' : '▲ 參數';
        btn.setAttribute('aria-expanded', String(!collapsed));
    }
}

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

// 一次影片生成的預估花費（USD）。送出前的確認與「本次花費」累加共用這一個入口，
// 兩邊才不會各算各的。
//
// ⚠️ 先前是「按次計費的就加 model_price」——那是錯的。閘道的 ali task adaptor 會把
// `seconds` 當成計費倍率乘進額度（見 EstimateBilling()），所以 model_price 其實是
// **每秒基準價**。HappyHorse 一支 10 秒 1080P 實際約 $1.80，先前只累加 $0.02（90 倍）；
// Veo 則因為是 token 型又查不到尺寸表，直接完全沒被計入。
function videoCostFor(modelId, costInfo) {
    const res = costInfo?.resolution, sec = costInfo?.seconds;
    // Seedance 優先用自己驗證過的 token 公式：它的幀數是「秒數 × 24 + 1」，多出來的
    // 那 1 幀是整支影片一次性的開銷，比「每秒單價 × 秒數」精確
    const byTokens = estimateVideoTokenCost(modelId, res, sec);
    if (byTokens) return byTokens;
    const perSec = videoPerSecondPrice(modelId, res, { audio: costInfo?.audio, mode: costInfo?.mode });
    if (perSec != null && sec) return perSec * sec;
    // 都算不出來就不計——寧可少算，也不要顯示一個編出來的數字
    return null;
}

function estimateVideoCost(modelId, costInfo) {
    if (!pricingMap[modelId]) return null;
    return videoCostFor(modelId, costInfo);
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
    if (!pricingMap[modelId]) return;
    const est = videoCostFor(modelId, costInfo);
    if (est) addCost(est);
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

// ── 影片模型的官方每秒單價（USD / 秒）──────────────────────────────────────────
// 影片本來就是按秒計費的，所以 UI 一律顯示「每秒」而不是「每次」。
//
// ⚠️ 這裡放的是**各廠商的官方定價**，不是從平台的計費倍率反推的。理由：倍率是
// 「基準價 × 解析度倍率 × 秒數 × …」層層相乘的結果，要從中還原出每秒單價得先知道
// 基準價對應哪一檔解析度、以及秒數有沒有被乘進去——那是一連串看不見的假設，錯了
// 也不會有任何徵兆。官方定價是可以直接對照查證的單一數字。
//
// 音訊會影響價格的兩家已標在下面。沒有列在這裡的模型（gemini-omni）走 token 計費、
// 且長度由模型自己決定，沒有「每秒」這個概念，維持原本的顯示。
const _VIDEO_SEC_PRICE = {
    // w3.0 影片四顆（Spicy 頁籤）。閘道的 /api/pricing 對這四顆只給 model_price
    // ＝**最便宜那一檔的每秒價**（沒有單位、沒有檔次），照它顯示會變成「$0.05/次」
    // ——一支 5 秒 1080p 實際是 $1.00，差 20 倍。所以價格一律以這裡的官方每秒單價
    // 為準。配音開關不影響價格（與 Veo 相反），故不設 _withAudio。
    // ⚠️ 這是 w3.0-video（CarrotHub），與下面的 wan3.0-video（萬相 3.0）是**不同的
    //    上游路徑**，別名只差一個 an，不要看串了。
    // 前端用的是 -spicy 這組（閘道端的模型重定向）；官方名同時列著，是為了萬一
    // 改回直呼官方名時價格顯示不會落空——兩組價格必須一致，改一邊要改兩邊。
    'w3.0-video-spicy':           { '480p': 0.05,  '720p': 0.10, '1080p': 0.20 },
    'w3.0-video-prime-spicy':     { '480p': 0.068, '720p': 0.14, '1080p': 0.28 },
    'w3.0-video-pro-spicy':       { '1080p': 0.18, '2k': 0.20,   '4k': 0.23 },
    'w3.0-video-prime-pro-spicy': { '1080p': 0.26, '2k': 0.28,   '4k': 0.31 },
    'w3.0-video':           { '480p': 0.05,  '720p': 0.10, '1080p': 0.20 },
    'w3.0-video-prime':     { '480p': 0.068, '720p': 0.14, '1080p': 0.28 },
    'w3.0-video-pro':       { '1080p': 0.18, '2k': 0.20,   '4k': 0.23 },
    'w3.0-video-prime-pro': { '1080p': 0.26, '2k': 0.28,   '4k': 0.31 },
    // 萬相 3.0（all-in-one）：480P $0.05 / 720P $0.10 / 1080P $0.20
    'wan3.0-video':     { '480P': 0.05, '720P': 0.10, '1080P': 0.20 },
    // 萬相 3.0 Prime（高速版）：官方定價 2026-08-25 核對（平台倍率表同源）；
    // 漏列這裡會退回「$0.068/次」的錯誤顯示（那是後台 model_price＝480P 基準每秒價）
    'wan3.0-video-prime': { '480P': 0.068, '720P': 0.14, '1080P': 0.28 },
    // 萬相 2.7 全系列：720P $0.10 / 1080P $0.15
    'wan2.7-t2v':       { '720P': 0.10, '1080P': 0.15 },
    'wan2.7-i2v':       { '720P': 0.10, '1080P': 0.15 },
    'wan2.7-r2v':       { '720P': 0.10, '1080P': 0.15 },
    'wan2.7-videoedit': { '720P': 0.10, '1080P': 0.15 },
    // 萬相 2.6 t2v/r2v：同 2.7
    'wan2.6-t2v':       { '720P': 0.10, '1080P': 0.15 },
    'wan2.6-r2v':       { '720P': 0.10, '1080P': 0.15 },
    // 萬相 2.6 i2v 系列：標示價含音訊，無聲減半
    'wan2.6-i2v':       { '720P': 0.05, '1080P': 0.075, _noAudioHalf: true },
    'wan2.6-i2v-flash': { '720P': 0.05, '1080P': 0.075, _noAudioHalf: true },
    'wan2.6-r2v-flash': { '720P': 0.05, '1080P': 0.075 },
    // 萬相 2.2 動作動畫：固定 720P 輸出，依服務模式（wan-std / wan-pro）不同價
    'wan2.2-animate-move': { _byMode: { 'wan-std': 0.12, 'wan-pro': 0.18 } },
    'wan2.2-animate-mix':  { _byMode: { 'wan-std': 0.18, 'wan-pro': 0.26 } },
    // HappyHorse
    'happyhorse-1.0-t2v':        { '720P': 0.14, '1080P': 0.24 },
    'happyhorse-1.0-i2v':        { '720P': 0.14, '1080P': 0.24 },
    'happyhorse-1.0-r2v':        { '720P': 0.14, '1080P': 0.24 },
    'happyhorse-1.0-video-edit': { '720P': 0.14, '1080P': 0.24 },
    'happyhorse-1.1-t2v':        { '720P': 0.14, '1080P': 0.18 },
    'happyhorse-1.1-i2v':        { '720P': 0.14, '1080P': 0.18 },
    'happyhorse-1.1-r2v':        { '720P': 0.14, '1080P': 0.18 },
    // Veo（Google 官方定價）：標示的是「純影片」價，含配音另計，見 _withAudio
    'veo-3.1-generate-001':      { '720P': 0.20, '1080P': 0.20, '4K': 0.40,
                                   _withAudio: { '720P': 0.40, '1080P': 0.40, '4K': 0.60 } },
    // Fast 這一列的數字是對的，不要改。閘道後台的基準價一度被填成 $0.30/秒（官方的
    // 3 倍），已確認是填錯並改回 $0.10，實收與這裡顯示的一致。
    'veo-3.1-fast-generate-001': { '720P': 0.08, '1080P': 0.10, '4K': 0.25,
                                   _withAudio: { '720P': 0.10, '1080P': 0.12, '4K': 0.30 } },
    // Lite **有**配音檔次（先前這裡註明「不支援配音」是錯的，有聲時實付比顯示的高
    // 66%：720P $0.03→$0.05、1080P $0.05→$0.08）。
    // 沒有 4K 檔次——4K 請求會落到 1080P 的價，所以 4K 直接沿用 1080P 的兩個數字。
    // ⚠️ 閘道端的修正還沒部署（時程未定），在那之前有聲仍按無聲收，所以拿正式站的
    // 實際扣款來對照會發現這裡「偏高」——**那不是錯，不要改回去**。部署後就會一致。
    'veo-3.1-lite-generate-001': { '720P': 0.03, '1080P': 0.05, '4K': 0.05,
                                   _withAudio: { '720P': 0.05, '1080P': 0.08, '4K': 0.08 } },
};

// 解析度標籤各家大小寫不一致（既有影片模型用 480P/720P/1080P，w3.0 用
// 480p/2k/4k），比對前一律轉小寫。fallbackCheapest 只給「下拉選單基準價」用：
// 選單固定拿 720P 當基準，但 w3.0 的 pro 檔次根本沒有 720p，硬比會落空、退回
// 顯示成沒有意義的 token 單價。落空時改用該模型**最便宜的一檔**，並把實際用到
// 的標籤一起回傳——顯示 $0.18/秒（720P）這種該模型不存在的檔次比不顯示更糟。
function _resolveVideoTier(modelId, resolution, fallbackCheapest) {
    const t = _VIDEO_SEC_PRICE[modelId];
    if (!t) return null;
    const keys = Object.keys(t).filter(k => !k.startsWith('_'));
    if (!keys.length) return null;
    const want = String(resolution == null ? '' : resolution).toLowerCase();
    const hit = keys.find(k => k.toLowerCase() === want);
    if (hit) return hit;
    if (!fallbackCheapest) return null;
    return keys.reduce((a, b) => (t[a] <= t[b] ? a : b));
}

// 回傳每秒單價，查不到就回 null（呼叫端會退回原本的顯示方式）
function videoPerSecondPrice(modelId, resolution, opts) {
    const o = opts || {};
    const t = _VIDEO_SEC_PRICE[modelId];
    if (t) {
        if (t._byMode) return t._byMode[o.mode] ?? Object.values(t._byMode)[0];
        const tier = _resolveVideoTier(modelId, resolution, o.fallbackCheapest);
        if (o.audio && t._withAudio && tier != null && t._withAudio[tier] != null) return t._withAudio[tier];
        let v = tier == null ? null : t[tier];
        if (v == null) return null;
        if (t._noAudioHalf && o.audio === false) v = v / 2;
        return v;
    }
    // Seedance 沒有公開的每秒價表，但它的 token 公式我們自己驗證過，換算結果跟
    // 官方每秒單價完全相符（見下方 estimateVideoPerSecond 的註解），所以用算的
    return estimateVideoPerSecond(modelId, resolution);
}

// 每秒單價（USD）。用**整整 24 幀**算，而不是把 seconds=1 丟進 estimateVideoTokenCost()
// ——那條公式的幀數是 `秒數 × 24 + 1`，多出來的那 1 幀是整支影片只算一次的固定開銷，
// 拿去當每秒費率會高估。
//
// 這個算法對得上兩個已知的官方每秒單價（720p）：
//   dreamina-seedance-2.5  1280×720×24/1024 = 21,600 tokens × $10.7/1M = $0.2311/秒 ✓
//   dreamina-seedance-2.0  同上 tokens × $7.0/1M              = $0.1512/秒 ✓
// 兩個都跟 README 記錄的官方數字完全相符，所以這條換算是可信的。
function estimateVideoPerSecond(modelId, resolution) {
    const p = pricingMap[modelId];
    if (!p || p.type !== 'token') return null;
    const dims = _SEEDANCE_DIMS[modelId]?.[resolution];
    if (!dims) return null;
    const tokensPerSec = dims[0] * dims[1] * 24 / 1024;
    const ratio = _SEEDANCE_RES_RATIO[resolution] || 1;
    return tokensPerSec / 1e6 * p.output * ratio;
}

// resolution 省略時用目前選到的解析度＋目前的配音開關（模型旁的即時提示用），
// 傳入固定值則連配音也一併切成固定基準（下拉選單用）——選單是拿來**比較模型**的，
// 每個項目都用同一個基準才比得出高下；先前是用「建立選單當下」的解析度，等於基準
// 隨使用者的操作順序而變。
//
// ⚠️ 配音的基準是**含配音**，不是照當下的開關。理由是配音關閉時選單顯示的是純影片價
// （Veo 標準版 $0.2），但同一行的文案寫「含原生配音」，一行之內自相矛盾；而且各家族
// 表上的基準價含意本來就不同（萬相 2.6 i2v 的 $0.05 是**含**音訊、Veo 的 $0.2 是
// **不含**），並排等於在比不同的東西。統一成含配音之後兩者才是同一個基準。
const PRICE_BASELINE_RESOLUTION = '720P';

// 價格會隨配音變動的模型，標上「含配音」才知道這個數字的基準是什麼；其餘模型
// （Seedance、HappyHorse、萬相 2.7…）配音不影響單價，標了只是雜訊
function _audioAffectsPrice(modelId) {
    const t = _VIDEO_SEC_PRICE[modelId];
    return !!(t && (t._withAudio || t._noAudioHalf));
}

function formatPriceSuffix(modelId, resolution) {
    const p = pricingMap[modelId];
    if (!p) return '';
    // 影片模型優先用官方每秒單價（含按次計費登記的那些——它們的 model_price 其實是
    // 每秒基準價，顯示成「/次」會嚴重低估：例如 HappyHorse 標 $0.02 但 720P 官方是
    // $0.14/秒，一支 5 秒的片子差 35 倍）
    const baseline = resolution != null;
    // Spicy 頁籤的影片模型有自己的解析度選單。先前這裡固定讀影片頁籤的
    // `videoResolution`，Spicy 那邊選什麼都不會反映在價格提示上。
    const resSelId = isW3SpicyVideo(modelId) ? 'muleaiVidResolution' : 'videoResolution';
    const res = resolution || document.getElementById(resSelId)?.value;
    const audio = baseline ? true : document.getElementById('vidAudio')?.checked;
    const perSec = videoPerSecondPrice(modelId, res, {
        audio,
        mode: document.getElementById('videoAnimateMode')?.value,
        fallbackCheapest: baseline,
    });
    if (perSec != null) {
        // 動作動畫固定 720P 輸出、UI 上根本沒有解析度選單，標上解析度只會誤導
        const byMode = !!_VIDEO_SEC_PRICE[modelId]?._byMode;
        if (byMode) return ` ・ 約 $${formatUsd(Number(perSec.toFixed(4)))}/秒`;
        const note = baseline && _audioAffectsPrice(modelId) ? '・含配音' : '';
        // 標籤用實際查到的那一檔，不是要求的那一檔（見 _resolveVideoTier）
        const tier = _resolveVideoTier(modelId, res, baseline) || res;
        return ` ・ 約 $${formatUsd(Number(perSec.toFixed(4)))}/秒（${tier}${note}）`;
    }
    if (p.type === 'fixed') return ` ・ $${formatUsd(p.price)}/次`;
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
            Object.assign(document.createElement('option'), { value: m.id, textContent: `${m.name} — ${m.desc}${formatPriceSuffix(m.id, PRICE_BASELINE_RESOLUTION)}` })
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
    // 兩條路都可用的模型才顯示「送出方式」；只有一條時不給選（免得出現無效的選項）
    const modes = spec.modes || ['size'];
    document.getElementById('imgCustomSizeModeRow').style.display = modes.length > 1 ? '' : 'none';
    if (modes.length === 1) document.getElementById('imgCustomSizeMode').value = modes[0];
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
        return;
    }
    msg.style.color = 'var(--text-muted)';
    // 順便把實際會送出的欄位寫出來，讓使用者看得到兩種寫法的差別
    const sent = customSizeMode() === 'wh'
        ? `width: ${r.w}, height: ${r.h}`
        : `size: "${r.size}"`;
    // 對齊時明確告知，而不是默默改掉使用者輸入的值
    const alignNote = r.aligned ? `（尺寸會對齊到 ${_imgCustomSizeSpec.align} 的倍數）` : '';
    msg.textContent = `實際輸出 ${r.w}×${r.h}${alignNote}　送出 ${sent}`;
}

function customSizeMode() {
    const el = document.getElementById('imgCustomSizeMode');
    const modes = (_imgCustomSizeSpec && _imgCustomSizeSpec.modes) || ['size'];
    return modes.includes(el.value) ? el.value : modes[0];
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

    // ref_strength 滑桿整個收起（2026-08-25 考古定案）：這個參數名從未存在於閘道
    // （平台 repo＋git 全歷史 grep 為 0），送出後靜默消失——滑桿從第一天起就沒有
    // 作用過。先前「僅 Wan 系列支援」的認知源自證據等級混淆（「帶了不會被拒」
    // 被記成「有效」）。平台若日後實作轉發再恢復顯示。
    document.getElementById('imgRefStrengthGroup').style.display = 'none';

    // prompt_extend 僅 T2I 與 qwen-image-2.0 系列（i2i 融合模型）支援，其餘 I2I 圖像編輯模型後端不支援此參數
    document.getElementById('imgPromptExtendGroup').style.display =
        ((t === 't2i' || modelInfo.fusion_edit) && !modelInfo.no_prompt_extend) ? '' : 'none';

    // GPT Image 家族沒有這四顆參數（OpenAI 圖像 API 不存在它們；平台實測 seed/
    // watermark 轉發會 400、negative_prompt/prompt_extend 被靜默丟棄）——控制項
    // 留著只會讓使用者以為填了有作用，整組藏起來並清空值
    const negG = document.getElementById('imgNegGroup');
    negG.style.display = modelInfo.no_negative_prompt ? 'none' : '';
    if (modelInfo.no_negative_prompt) document.getElementById('imageNegPrompt').value = '';
    document.getElementById('imgWatermarkGroup').style.display = modelInfo.no_watermark ? 'none' : '';
    if (modelInfo.no_watermark) document.getElementById('imgWatermark').checked = false;
    document.getElementById('imgSeedGroup').style.display = modelInfo.no_seed ? 'none' : '';
    if (modelInfo.no_seed) document.getElementById('imgSeed').value = '';

    // GPT Image 專屬參數（quality/background/output_format），T2I/I2I 皆適用；
    // moderation 是 generations 專屬參數，edits 端點沒有——I2I 時單獨收起來
    document.getElementById('imgGptParamsSection').style.display = modelInfo.supports_gpt_params ? '' : 'none';
    document.getElementById('imgModerationGroup').style.display = (t === 't2i') ? '' : 'none';

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

    // vedit-specific controls（ratio 改為依模型家族動態顯示，見 syncVidRatio）
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

// 畫面比例：各代行為不同（平台端 2026-08-25 實測確認），不能共用一份下拉——
//   wan3.0：值域 adaptive + 五種比例，未指定時平台會主動下發 adaptive
//   wan2.7 / happyhorse：同值域，但未指定時不送、交給上游預設
//   wan2.6 及更早：不走 ratio（比例含在 size 裡），顯示這個下拉只會誤導
//   其他家族（veo/seedance）：維持原本僅 vedit 顯示的行為
function syncVidRatio() {
    const t     = document.getElementById('videoTaskType').value;
    const model = document.getElementById('videoModel').value || '';
    const group = document.getElementById('vidRatioGroup');
    const sel   = document.getElementById('videoRatio');
    const isWan30  = model.startsWith('wan3.0');
    const isFamily = isWan30 || model.startsWith('wan2.7') || model.startsWith('happyhorse');
    // no_ratio：該接口官方沒有 ratio 參數（wan2.7-i2v／happyhorse i2v／
    // happyhorse-1.0-video-edit，比例跟隨輸入素材），顯式送出會被閘道 422 拒絕
    const noRatio = !!(models.video.find(m => m.id === model && m.type === t) || {}).no_ratio;
    const show = !noRatio && ((isFamily && t !== 'animate') || (t === 'vedit'));
    group.style.display = show ? '' : 'none';
    if (!show) { sel.value = ''; return; }
    const prev = sel.value;
    const opts = isWan30 ? [['adaptive', '自動（依內容）']] : [['', '自動（預設）']];
    ['16:9', '9:16', '1:1', '4:3', '3:4'].forEach(r => opts.push([r, r]));
    sel.innerHTML = opts.map(([v, l]) => `<option value="${v}">${l}</option>`).join('');
    sel.value = opts.some(([v]) => v === prev) ? prev : opts[0][0];
}

// 智能時長（duration=-1，僅萬相 3.0）：模型依內容自行決定長度。
// 開啟時隱藏秒數滑桿、費用提示改成「依實際生成長度計費」——這種情況平台走
// 保證金預扣、完成後按實際秒數結算，事前給任何數字都是假的。
function syncSmartDur() {
    const modelId   = document.getElementById('videoModel').value;
    const taskType  = document.getElementById('videoTaskType').value;
    const modelInfo = models.video.find(m => m.id === modelId) || {};
    const row = document.getElementById('vidSmartDurRow');
    const supported = !!modelInfo.smart_duration && taskType !== 'animate' && !modelInfo.no_duration;
    row.style.display = supported ? '' : 'none';
    if (!supported && document.getElementById('vidSmartDuration').checked) {
        document.getElementById('vidSmartDuration').checked = false;
    }
    onSmartDurToggle();
}

function onSmartDurToggle() {
    const on = document.getElementById('vidSmartDuration').checked &&
               document.getElementById('vidSmartDurRow').style.display !== 'none';
    const modelInfo = models.video.find(m => m.id === document.getElementById('videoModel').value) || {};
    document.getElementById('vidDurationGroup').style.display =
        (on || modelInfo.no_duration || document.getElementById('videoTaskType').value === 'animate') ? 'none' : '';
    const priceEl = document.getElementById('videoModelPrice');
    if (on) priceEl.textContent = '（依實際生成長度計費）';
    else updateModelPriceHint('videoModelPrice', document.getElementById('videoModel').value);
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

    // animate 的 watermark 開關收起（📄官方＋讀碼，reference §2.3.13／P1-1）：
    // wan-animate 是全阿里唯一 watermark 在 input 層的模型，閘道結構沒有該欄位
    // 且 animate 分支硬編 Parameters.Watermark=false——客戶開了也送不出去、無錯誤。
    // 平台修復（P1-1）部署後再恢復顯示。
    document.getElementById('vidWatermarkGroup').style.display =
        (taskType === 'animate') ? 'none' : '';
    if (taskType === 'animate') document.getElementById('vidWatermark').checked = false;

    // 調整時長範圍。no_duration 與 no_resolution 是兩個獨立旗標：gemini-omni 兩者皆無、
    // happyhorse-1.0-video-edit 官方沒有 duration 但有 resolution（720P/1080P）
    document.getElementById('vidDurationGroup').style.display = modelInfo.no_duration ? 'none' : '';
    document.getElementById('vidResolutionGroup').style.display =
        (modelInfo.no_resolution || taskType === 'animate') ? 'none' : '';
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
    // min_dur=0 的視頻編輯（wan2.7-videoedit）：0=保留原長，實際截斷範圍是 [2,max]，
    // 1 是非法值——滑到 1 直接進位到 2，不讓非法值送得出去
    if (minD === 0 && curVal === 1) { curVal = 2; }
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

    // 放在最後：syncSmartDur 會依開關狀態隱藏時長滑桿，必須蓋過上面
    // 依 no_duration 重設 display 的那行，順序反了智能時長開啟時滑桿會跑回來
    // 運鏡模式（shot_type，僅 wan2.6 系 t2v/i2v/r2v）：官方要求搭配 prompt_extend=true
    const shotG = document.getElementById('vidShotTypeGroup');
    if (shotG) {
        shotG.style.display = modelInfo.shot_type ? '' : 'none';
        if (!modelInfo.shot_type) document.getElementById('videoShotType').value = '';
    }
    syncVidRatio();
    syncSmartDur();
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
    // 配音會改變單價（Veo 含配音是純影片的兩倍；萬相 2.6 i2v 關掉配音減半），
    // 所以切換後要重算價格提示
    updateModelPriceHint('videoModelPrice', document.getElementById('videoModel').value);
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
            temperature, max_tokens: maxTokens,
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
        // top_p：勾「模型預設」就整個不送（不要替模型補預設值）
        if (!document.getElementById('textTopPAuto').checked) body.top_p = topP;
        const _tb = document.getElementById('textThinkingBudget').value.trim();
        if (_tb && modelInfo?.thinking_budget) body.thinking_budget = parseInt(_tb);
        const _ct = document.getElementById('textClearThinking').value;
        if (_ct && modelInfo?.clear_thinking) body.clear_thinking = (_ct === 'true');
        const _pt = document.getElementById('textPreserveThinking').value;
        if (_pt && modelInfo?.preserve_thinking) body.preserve_thinking = (_pt === 'true');
        const _rp = document.getElementById('textRepPenalty').value.trim();
        if (_rp && modelInfo?.repetition_penalty) body.repetition_penalty = parseFloat(_rp);
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
                const reqPanel = buildRequestPanel(data.request);
                if (reqPanel) aDiv.appendChild(reqPanel);
            }
        } else {
            const reader  = res.body.getReader();
            const decoder = new TextDecoder();
            let full = '', buf = '', usage = null, reqSummary = null;
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
                        if (d.request) {
                            reqSummary = d.request;   // 串流的第一個事件，最後掛到訊息尾巴
                        } else if (d.reasoning) {
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
            const reqPanel = buildRequestPanel(reqSummary);
            if (reqPanel) aDiv.appendChild(reqPanel);
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
    // 自訂尺寸：不合法就擋在這裡，省下一次必定失敗的呼叫。
    // 依「送出方式」二選一——size 字串，或頂層 width/height 兩個欄位。
    let customWH = null;
    if (size === CUSTOM_SIZE_VALUE) {
        const r = currentCustomSize();
        if (r.error) { toast(r.error, "error"); return; }
        size = r.size;
        if (customSizeMode() === 'wh') customWH = { width: r.w, height: r.h };
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
    const moderation   = gptParamsVisible ? document.getElementById('imgModeration').value : '';

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
            // 選了 width/height 那條路時帶這兩個欄位，後端會據此把 size 拿掉
            if (customWH) Object.assign(body, customWH);
            if (imgSeed !== null) body.seed = imgSeed;
            if (aspectRatio) body.aspect_ratio = aspectRatio;
            if (enableSequential) body.enable_sequential = true;
            if (quality) body.quality = quality;
            if (background) body.background = background;
            if (outputFormat) body.output_format = outputFormat;
            if (moderation) body.moderation = moderation;   // 僅 generations 有這個參數，edits 沒有
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


// ── 「查看實際請求」面板 ──────────────────────────────────────────────
// 讓使用者在 playground 試好參數之後，能直接照抄去接 API——這是這個平台最常被
// 問到的事（文檔站每次寫新模型都要我們手動整理一次呼叫方式）。
// ⚠️ **永遠不顯示 Authorization 的實際內容。** 後端回傳的 auth 欄位固定是
// `Bearer $NENAI_API_KEY`，複製出來的 cURL 也是。使用者知道自己的 key，平台再顯示
// 一次只會多一條外洩管道（截圖、螢幕分享、貼給別人問問題）。
function buildRequestPanel(req) {
    if (!req) return null;
    const url = (req.base_url || '') + (req.endpoint || '');
    const bodyObj = req.body || req.form || null;
    const bodyStr = bodyObj ? JSON.stringify(bodyObj, null, 2) : '';
    const wrap = el('details', { className: 'req-panel' });
    const sum = el('summary');
    sum.textContent = '查看實際請求';
    wrap.appendChild(sum);
    const body = el('div', { className: 'req-body' });
    const meta = el('div', { className: 'req-line' });
    meta.textContent = `${req.method || 'POST'} ${url}`;
    body.appendChild(meta);
    const auth = el('div', { className: 'req-line req-auth' });
    auth.textContent = `Authorization: ${req.auth || 'Bearer $NENAI_API_KEY'}`;
    body.appendChild(auth);
    if (req.note) {
        const note = el('div', { className: 'req-note' });
        note.textContent = req.note;
        body.appendChild(note);
    }
    if (bodyStr) {
        const pre = el('pre', { className: 'req-json' });
        pre.textContent = bodyStr;
        body.appendChild(pre);
    }
    const btn = el('button', { className: 'btn btn-ghost btn-sm' });
    btn.textContent = '複製 cURL';
    btn.onclick = () => {
        const lines = [`curl -X ${req.method || 'POST'} ${url} \\`,
                       `  -H "Authorization: Bearer $NENAI_API_KEY" \\`];
        if (req.form) {
            lines.push('  -H "Content-Type: multipart/form-data" \\');
            Object.entries(req.form).forEach(([k, v]) =>
                lines.push(`  -F ${JSON.stringify(k + '=' + v)} \\`));
            lines.push('  # 檔案欄位請自行加上 -F "image=@your-file.png"');
        } else {
            lines.push('  -H "Content-Type: application/json" \\');
            lines.push(`  -d ${JSON.stringify(bodyStr)}`);
        }
        navigator.clipboard.writeText(lines.join('\n'))
            .then(() => toast('已複製 cURL（金鑰請自行代入 $NENAI_API_KEY）', 'success'))
            .catch(() => toast('複製失敗', 'error'));
    };
    body.appendChild(btn);
    wrap.appendChild(body);
    return wrap;
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
    const reqPanel = buildRequestPanel(res.request);
    if (reqPanel) frag.appendChild(reqPanel);
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
    let duration    = parseInt(document.getElementById('videoDuration').value);
    const _vidInfo  = models.video.find(m => m.id === model && m.type === taskType) || {};
    // no_duration（happyhorse-1.0-video-edit）：官方沒有 duration 參數，送 0 讓後端整個略過；
    // min_dur=0（wan2.7-videoedit）：0=保留原長、截斷範圍 [2,max]，1 是非法值進位到 2
    if (_vidInfo.no_duration) duration = 0;
    else if (_vidInfo.min_dur === 0 && duration === 1) duration = 2;
    const audio         = document.getElementById('vidAudio').checked;
    const vidExtend     = document.getElementById('vidPromptExtend').checked;
    const vidWatermark  = document.getElementById('vidWatermark').checked;
    const vidSeedRaw    = document.getElementById('vidSeed').value.trim();
    const vidSeed       = vidSeedRaw !== '' ? parseInt(vidSeedRaw) : null;
    // 智能時長開啟時送 -1（模型自行決定長度，僅萬相 3.0；費用依實際秒數結算）
    const smartDur = document.getElementById('vidSmartDuration').checked &&
                     document.getElementById('vidSmartDurRow').style.display !== 'none';
    if (smartDur) duration = -1;
    // ratio 只在下拉可見時帶值（各家族值域不同，隱藏時代表該模型不吃這個參數）
    const ratioVal = document.getElementById('vidRatioGroup').style.display !== 'none'
                   ? document.getElementById('videoRatio').value : '';

    if (!prompt && taskType !== 'vedit' && taskType !== 'animate') { toast('請輸入 Prompt', 'error'); return; }

    // 昂貴任務先確認（門檻與理由見 confirmIfExpensive）。放在按鈕鎖定之前，
    // 使用者取消時不需要再把按鈕解鎖。智能時長無法預知秒數，以該模型上限
    // 當最壞情況估——寧可多確認一次，不要低估後放行
    if (!confirmIfExpensive(model, { resolution, seconds: smartDur ? 30 : duration, audio,
                                     mode: document.getElementById('videoAnimateMode')?.value })) return;

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
            if (ratioVal) fd.append('ratio', ratioVal);
            if (vidSeed !== null) fd.append('seed', vidSeed);
            if (audio) {
                const audioFile = document.getElementById('vidT2VAudioInput').files[0];
                if (audioFile) fd.append('audio_file', audioFile);
            }
            const _shot = document.getElementById('videoShotType').value;
            if (_shot && _vidInfo.shot_type) fd.append('shot_type', _shot);
            res = await apiPostForm('/api/video/t2v', fd);

        } else if (taskType === 'i2v') {
            const i2vMode = document.getElementById('videoI2VMode').value;
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('resolution', resolution);
            fd.append('duration', duration); fd.append('i2v_mode', i2vMode);
            fd.append('prompt_extend', vidExtend); fd.append('watermark', vidWatermark);
            if (ratioVal) fd.append('ratio', ratioVal);
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
            const _shot = document.getElementById('videoShotType').value;
            if (_shot && _vidInfo.shot_type) fd.append('shot_type', _shot);
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
            if (ratioVal) fd.append('ratio', ratioVal);
            if (vidSeed !== null) fd.append('seed', vidSeed);
            if (audio) {
                const bgmFile = document.getElementById('vidT2VAudioInput')?.files[0];
                if (bgmFile) fd.append('audio_file', bgmFile);
            }
            refFiles.forEach(f => fd.append('reference_files', f));
            const _shot = document.getElementById('videoShotType').value;
            if (_shot && _vidInfo.shot_type) fd.append('shot_type', _shot);
            res = await apiPostForm('/api/video/r2v', fd);
        }

        if (res.success && res.task_id) {
            addVideoTask(res.task_id, model, prompt, res.status,
                         { resolution, seconds: duration, audio,
                           mode: document.getElementById('videoAnimateMode')?.value },
                         false, res.request);
            toast('任務已提交，輪詢中...', 'info');
        } else if (res.success && res.video_url) {
            addVideoResult(model, prompt, res.local_path || res.video_url, false, fmtElapsed(Date.now() - startTime));
            TaskHistory.save('video', model, prompt, res.local_path || res.video_url);
            toast('影片生成完成！', 'success');
            addVideoCost(model, { resolution, seconds: duration, audio,
                                  mode: document.getElementById('videoAnimateMode')?.value });
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
                addMuleAIVideoTask(t.taskId, t.model, t.prompt || '', '恢復中', true, t.req);
            } else {
                addVideoTask(t.taskId, t.model, t.prompt || '', '恢復中', t.costInfo, true, t.req);
            }
        } catch (e) { console.warn('恢復任務失敗', t.taskId, e); }
    });
    toast(`已恢復 ${alive.length} 個進行中的任務`, 'info');
}

function addVideoTask(taskId, model, prompt, status, costInfo, isResume = false, req = null) {
    const cont = document.getElementById('videoResults');
    cont.querySelector('.empty-state')?.remove();
    const startTime = Date.now();
    if (!isResume) savePendingTask({ kind: 'video', taskId, model, prompt, costInfo, req });
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
    // 請求面板在「送出當下」就掛上去，不必等輪詢完成——影片動輒兩三分鐘，
    // 使用者想照抄參數時沒理由等它跑完。
    const reqPanel = buildRequestPanel(req);
    if (reqPanel) card.appendChild(reqPanel);
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
    document.getElementById('voiceRealtimeSection').style.display = t === 'realtime' ? '' : 'none';
    document.getElementById('voiceRealtimePromptPanel').style.display = t === 'realtime' ? 'flex' : 'none';
    document.getElementById('voiceMusicSection').style.display = t === 'music' ? '' : 'none';
    document.getElementById('voiceMusicPromptPanel').style.display = t === 'music' ? '' : 'none';
    // 即時對話有自己的訊息區，ASR/TTS/音樂 的結果區在這個模式下要收起來
    document.getElementById('voiceResults').style.display = t === 'realtime' ? 'none' : '';
    // 離開即時對話頁面就把連線收掉——WebSocket 連著不會自己斷，而使用者切走之後
    // 看不到任何狀態，麥克風卻還開著
    if (t !== 'realtime') stopRealtime();
    onVoiceModelChange();
}

function onVoiceModelChange() {
    const t = document.getElementById('voiceTaskType').value;
    updateModelPriceHint('voiceModelPrice', document.getElementById('voiceModel').value);
    if (t === 'realtime') {
        const model = document.getElementById('voiceModel').value;
        const info = (models.voice?.realtime || []).find(m => m.id === model);
        // 換模型就把既有連線收掉——連線綁在「開始對話」當下選的模型上，留著會變成
        // 畫面上選 B、實際還連著 A；而且兩個家族的音色與斷句詞彙互不相容，
        // 拿 B 的值對 A 送 session.update 只會收到一串錯誤
        if (rtWs) { stopRealtime(); rtLog('', '已切換模型，請重新按「開始對話」', 'sys'); }
        const sel = document.getElementById('voiceRtVoice');
        sel.innerHTML = (info?.voices || [])
            .map(v => `<option value="${v.id}"${v.id === info.default_voice ? ' selected' : ''}>${v.name} — ${v.desc}</option>`)
            .join('');
        // 「什麼時候算你說完」的選項依模型重建——兩個家族的 turn_detection 詞彙不同
        // （omni 收 semantic_vad，audio-3.0 收 smart_turn），選項清單由後端 MODELS 的
        // turn_modes 提供，都是逐一實測過的合法值
        const tdSel = document.getElementById('voiceRtTurnDetection');
        const modes = info?.turn_modes || ['semantic_vad', 'server_vad', 'none'];
        const prevTd = tdSel.value;
        tdSel.innerHTML = modes.map(m =>
            `<option value="${m}">${_RT_TURN_MODE_LABELS[m] || m}</option>`).join('');
        tdSel.value = modes.includes(prevTd) ? prevTd : modes[0];
        // 純語音模型沒有畫面輸入——實測圖片會被上游靜默忽略（不報錯，模型口頭說
        // 看不到），所以附件鈕直接藏起來，殘留的附件也一併清掉
        document.getElementById('voiceRtAttachBtn').style.display = info?.audio_only ? 'none' : '';
        if (info?.audio_only && rtPendingFrames.length) clearRealtimeFile();
        return;
    }
    if (t === 'music') {
        // 靈感圖片欄只有支援圖片輸入的模型才顯示（lyria-002 帶圖上游會直接報錯）；
        // 切到不支援的模型就把已選的圖清掉，避免靜默夾帶造成 400
        const info = (models.voice?.music || []).find(m => m.id === document.getElementById('voiceModel').value);
        document.getElementById('voiceMusicImageGroup').style.display = info?.image_input ? '' : 'none';
        if (!info?.image_input) clearVoiceMusicFile();
        return;
    }
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

// ── 即時語音對話（realtime）────────────────────────────────────────────────
// 走後端的 /ws/omni 代理，不直連閘道：瀏覽器的 WebSocket 建構子不能帶 header，
// 直連只能把金鑰塞進子協定（openai-insecure-api-key.<key>），那會讓金鑰出現在
// 前端可見的握手參數裡。
//
// 音訊格式（實測）：上行 PCM 16kHz、下行 PCM 24kHz，都是 mono s16le 裸流。
// 下行沒有 wav 檔頭，要自己塞進 AudioBuffer 播放。
const RT_IN_RATE = 16000, RT_OUT_RATE = 24000;

let rtWs = null;            // 與後端代理的連線
let rtMicStream = null;     // getUserMedia 拿到的麥克風
let rtInCtx = null, rtProcessor = null, rtMicSink = null;
let rtOutCtx = null, rtPlayhead = 0, rtSources = [];
let rtAssistantLine = null; // 目前這一輪 AI 逐字稿的 DOM 節點（逐字更新同一行）

// state：'' 未連線／'on' 已連線／'busy' 處理中／'err' 出錯——狀態燈比文字更快讀
function rtSetStatus(text, state = '') {
    const el = document.getElementById('voiceRtStatus');
    if (el) el.textContent = text;
    const dot = document.getElementById('voiceRtDot');
    if (dot) dot.className = 'rt-dot' + (state ? ' ' + state : '');
}

// 對話一律**往下追加**、最新在底並自動捲到底；自己說的靠右、AI 靠左。
// frames 有值時把送出的畫面貼在該則訊息上方——不然使用者只看得到自己打的字，
// 無從確認「這一輪到底有沒有把圖帶上去」。
// kind：'me' | 'ai' | 'sys' | 'err'
function rtLog(who, text, kind = 'ai', frames) {
    const area = document.getElementById('voiceRtMessages');
    document.getElementById('voiceRtEmpty')?.remove();
    const wrap = el('div', { className: `rt-msg ${kind}` });
    if (who) {
        const w = el('div', { className: 'rt-who' });
        w.textContent = who;
        wrap.appendChild(w);
    }
    const bubble = el('div', { className: 'rt-bubble' });
    if (frames && frames.length) {
        const strip = el('div', { className: 'rt-frames' });
        frames.forEach(f => {
            const img = new Image();
            img.src = `data:image/jpeg;base64,${f}`;
            strip.appendChild(img);
        });
        bubble.appendChild(strip);
    }
    const p = el('div');
    p.textContent = text;
    bubble.appendChild(p);
    wrap.appendChild(bubble);
    area.appendChild(wrap);
    rtScrollToBottom();
    return p;
}

// 只有在使用者本來就貼在底部時才自動捲——他往上翻看舊訊息時把畫面拉走很惱人
function rtScrollToBottom(force) {
    const area = document.getElementById('voiceRtMessages');
    if (!area) return;
    const nearBottom = area.scrollHeight - area.scrollTop - area.clientHeight < 120;
    if (force || nearBottom) area.scrollTop = area.scrollHeight;
}

function clearRealtimeLog() {
    const area = document.getElementById('voiceRtMessages');
    area.innerHTML = '<div class="rt-empty" id="voiceRtEmpty"><p>對話內容會顯示在這裡</p></div>';
    rtAssistantLine = null;
}

function toggleRealtimeConnection() {
    if (rtWs) { stopRealtime(); return; }
    startRealtime();
}

function startRealtime() {
    const model = document.getElementById('voiceModel').value;
    const key = apiKey;
    if (!key) { toast('請先登入', 'error'); return; }

    rtSetStatus('連線中…', 'busy');
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    rtWs = new WebSocket(`${proto}://${location.host}/ws/omni?api_key=${encodeURIComponent(key)}&model=${encodeURIComponent(model)}`);

    rtWs.onopen = async () => {
        rtSetStatus('已連線', 'on');
        document.getElementById('voiceRtConnectBtn').innerHTML = '結束對話';
        document.getElementById('voiceRtMicBtn').disabled = false;
        document.getElementById('voiceRtSendBtn').disabled = false;
        rtSendSessionUpdate();
        rtOutCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RT_OUT_RATE });
        rtPlayhead = 0;
        // 這是「語音對話」，按下開始就該能直接講話——不要再逼使用者找第二顆按鈕。
        // 麥克風被拒絕也不當成失敗：打字那條路仍然完整可用
        await toggleRealtimeMic();
    };
    rtWs.onmessage = (e) => rtHandleEvent(JSON.parse(e.data));
    rtWs.onerror = () => rtLog('', '連線發生錯誤', 'err');
    rtWs.onclose = () => { if (rtWs) { rtLog('', '對話已結束', 'sys'); stopRealtime(); } };
}

function rtSendSessionUpdate() {
    if (!rtWs || rtWs.readyState !== WebSocket.OPEN) return;
    const modalities = document.getElementById('voiceRtModalities').value.split(',');
    const td = document.getElementById('voiceRtTurnDetection').value;
    rtWs.send(JSON.stringify({
        type: 'session.update',
        session: {
            modalities,
            voice: document.getElementById('voiceRtVoice').value,
            input_audio_format: 'pcm16',
            output_audio_format: 'pcm',
            instructions: document.getElementById('voiceRtInstructions').value,
            turn_detection: td === 'none' ? null : { type: td },
        },
    }));
    rtUpdateCommitButton();
}

function onRealtimeVoiceChange() { rtSendSessionUpdate(); }

function stopRealtime() {
    const ws = rtWs;
    rtWs = null;                        // 先清空，避免 onclose 再繞回來
    if (ws) { try { ws.close(); } catch (e) {} }
    stopRealtimeMic();
    rtStopPlayback();
    if (rtOutCtx) { try { rtOutCtx.close(); } catch (e) {} rtOutCtx = null; }
    rtSetStatus('尚未連線');
    const btn = document.getElementById('voiceRtConnectBtn');
    if (btn) btn.innerHTML = '開始對話';
    const mic = document.getElementById('voiceRtMicBtn');
    if (mic) { mic.disabled = true; mic.classList.remove('live'); }
    const send = document.getElementById('voiceRtSendBtn');
    if (send) send.disabled = true;
}

// ── 麥克風上行 ────────────────────────────────────────────────────────────
async function toggleRealtimeMic() {
    if (rtMicStream) { stopRealtimeMic(); return; }
    try {
        rtMicStream = await navigator.mediaDevices.getUserMedia({
            audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
        });
    } catch (err) {
        // 不擋流程：打字那條路仍然完整可用，只是要讓使用者知道為什麼不能講話
        rtLog('', '沒有麥克風權限，可以先用打字的', 'sys');
        return;
    }
    // 要求 16kHz，但瀏覽器不保證會照給（Safari 常常給 44.1k/48k），所以下面一律
    // 以 ctx.sampleRate 為準做重取樣，不能假設拿到的就是 16k
    rtInCtx = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: RT_IN_RATE });
    const src = rtInCtx.createMediaStreamSource(rtMicStream);
    rtProcessor = rtInCtx.createScriptProcessor(4096, 1, 1);
    rtProcessor.onaudioprocess = (e) => {
        if (!rtWs || rtWs.readyState !== WebSocket.OPEN) return;
        const pcm = rtFloatToPcm16(e.inputBuffer.getChannelData(0), rtInCtx.sampleRate);
        rtWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: rtBytesToBase64(pcm) }));
    };
    src.connect(rtProcessor);
    // ⚠️ ScriptProcessor 必須連到 destination 才會持續觸發，但直接連過去會把麥克風
    // 原音播出來變成回授。所以中間串一個 gain=0 的節點：節點鏈完整、聲音是靜音的
    rtMicSink = rtInCtx.createGain();
    rtMicSink.gain.value = 0;
    rtProcessor.connect(rtMicSink);
    rtMicSink.connect(rtInCtx.destination);

    const btn = document.getElementById('voiceRtMicBtn');
    btn.classList.add('live');
    btn.title = '關閉麥克風';
    rtSetStatus(rtPendingFrames.length ? '已附上畫面：說完後按「說完了，送出」' : '收音中，直接說話就好', 'on');
}

function stopRealtimeMic() {
    if (rtProcessor) { try { rtProcessor.disconnect(); } catch (e) {} rtProcessor = null; }
    if (rtMicSink) { try { rtMicSink.disconnect(); } catch (e) {} rtMicSink = null; }
    if (rtMicStream) { rtMicStream.getTracks().forEach(t => t.stop()); rtMicStream = null; }
    if (rtInCtx) { try { rtInCtx.close(); } catch (e) {} rtInCtx = null; }
    const btn = document.getElementById('voiceRtMicBtn');
    if (btn) { btn.classList.remove('live'); btn.title = '麥克風'; }
}

// Float32（-1~1）→ 16kHz PCM16。srcRate 不是 16k 時線性重取樣
function rtFloatToPcm16(input, srcRate) {
    let data = input;
    if (srcRate !== RT_IN_RATE) {
        const ratio = srcRate / RT_IN_RATE;
        const outLen = Math.floor(input.length / ratio);
        data = new Float32Array(outLen);
        for (let i = 0; i < outLen; i++) data[i] = input[Math.floor(i * ratio)];
    }
    const out = new Int16Array(data.length);
    for (let i = 0; i < data.length; i++) {
        const s = Math.max(-1, Math.min(1, data[i]));
        out[i] = s < 0 ? s * 0x8000 : s * 0x7FFF;
    }
    return new Uint8Array(out.buffer);
}

function rtBytesToBase64(bytes) {
    let bin = '';
    const CHUNK = 0x8000;   // 一次全部展開會在長音訊上爆掉呼叫堆疊
    for (let i = 0; i < bytes.length; i += CHUNK) {
        bin += String.fromCharCode.apply(null, bytes.subarray(i, i + CHUNK));
    }
    return btoa(bin);
}

// ── 下行播放 ──────────────────────────────────────────────────────────────
function rtPlayChunk(b64) {
    if (!rtOutCtx) return;
    const bin = atob(b64);
    const pcm = new Int16Array(bin.length / 2);
    for (let i = 0; i < pcm.length; i++) {
        pcm[i] = (bin.charCodeAt(i * 2) | (bin.charCodeAt(i * 2 + 1) << 8)) << 16 >> 16;
    }
    const buf = rtOutCtx.createBuffer(1, pcm.length, RT_OUT_RATE);
    const ch = buf.getChannelData(0);
    for (let i = 0; i < pcm.length; i++) ch[i] = pcm[i] / 32768;
    const node = rtOutCtx.createBufferSource();
    node.buffer = buf;
    node.connect(rtOutCtx.destination);
    // 依序接續播放：每塊排在前一塊結束的時間點，否則會全部同時播出變成雜音
    const now = rtOutCtx.currentTime;
    if (rtPlayhead < now) rtPlayhead = now + 0.05;
    node.start(rtPlayhead);
    rtPlayhead += buf.duration;
    rtSources.push(node);
    node.onended = () => { rtSources = rtSources.filter(s => s !== node); };
}

function rtStopPlayback() {
    rtSources.forEach(s => { try { s.stop(); } catch (e) {} });
    rtSources = [];
    rtPlayhead = 0;
}

// ── 事件處理 ──────────────────────────────────────────────────────────────
function rtHandleEvent(ev) {
    switch (ev.type) {
        case 'session.created':
            rtSetStatus('已連線', 'on');
            break;
        case 'input_audio_buffer.speech_started':
            // 使用者插話：立刻停掉正在播的回答，否則兩個人的聲音會疊在一起
            rtStopPlayback();
            rtSetStatus('聽到你在說話…', 'on');
            break;
        case 'input_audio_buffer.speech_stopped':
            rtSetStatus('思考中…', 'busy');
            break;
        case 'conversation.item.input_audio_transcription.completed':
            if (ev.transcript) rtLog('你說的', ev.transcript, 'me');
            break;
        case 'response.audio.delta':
            if (ev.delta) rtPlayChunk(ev.delta);
            break;
        case 'response.audio_transcript.delta':
        case 'response.text.delta':
            if (!rtAssistantLine) rtAssistantLine = rtLog('AI', '', 'ai');
            rtAssistantLine.textContent += ev.delta || '';
            rtScrollToBottom();
            break;
        case 'response.done':
            rtAssistantLine = null;
            // 帶畫面提問時暫時關掉的斷句偵測，答完就還原成使用者選的設定
            if (rtVadOverridden) { rtVadOverridden = false; rtSendSessionUpdate(); }
            rtSetStatus(rtMicStream ? '收音中，直接說話就好' : '已連線', 'on');
            rtApplyUsage(ev.response?.usage);
            break;
        case 'error':
            rtLog('', ev.error?.message || JSON.stringify(ev), 'err');
            rtSetStatus('發生錯誤', 'err');
            break;
    }
}

// realtime 的花費估算。四檔單價全部來自 /api/pricing（後端已把音訊那兩檔一起帶出來），
// 不做人工快照——這四檔會跟著後台走。
//
// ⚠️ 「同時輸出語音時，輸出的文字不計費」這條規則的條件是**這次回應真的產出了音訊
// token**，不是「開了語音模式」。同一個 session 裡某次只回文字的話，那次的文字照常
// 收費。所以要依 audio_tokens 是否 > 0 分支，不能一律當免費。
const _RT_AUDIO_ONLY_OUTPUT_BILLING = new Set([
    'qwen3.5-omni-plus-realtime',
    // 官方計費規則與閘道白名單都包含這三個（2026-08-16 上架）。⚠️ 測試網關實測
    // 當下文字輸出「仍被計費」（用量增幅對帳：qwen-audio-3.0-realtime-flash 一輪
    // 增幅 $0.0003663 = 文字照收的算式，免費版是 $0.0003393）——已回報閘道端，
    // 這裡照**官方規則**估，修好前估算會略低於實收
    'qwen3.5-omni-flash-realtime',
    'qwen-audio-3.0-realtime-plus',
    'qwen-audio-3.0-realtime-flash',
]);

// 「什麼時候算你說完」各選項的顯示文字。哪個模型有哪些選項由 MODELS 的 turn_modes
// 決定（實測：omni 系列收 semantic_vad / server_vad，audio-3.0 系列收 server_vad /
// smart_turn，null 兩邊都收）
const _RT_TURN_MODE_LABELS = {
    semantic_vad: '自動判斷（聽語意）',
    server_vad: '自動判斷（聽停頓）',
    smart_turn: '自動判斷（智慧斷句）',
    none: '我自己按按鈕',
};

function rtApplyUsage(usage) {
    if (!usage) return;
    // 這支上游回的是**複數**的 input_tokens_details / output_tokens_details，
    // 跟 OpenAI 的單數 input_token_details 不同，兩種都收
    const inD = usage.input_tokens_details || usage.input_token_details || {};
    const outD = usage.output_tokens_details || usage.output_token_details || {};
    const inText = inD.text_tokens || 0, inAudio = inD.audio_tokens || 0;
    // 附上圖片/影格時，上游把它們計成 video_tokens（實測一張 480x480 約 225 個），
    // 官方費率是跟文字同一檔。漏掉這塊的話，帶畫面的那幾輪估出來的花費會偏低
    const inVisual = (inD.video_tokens || 0) + (inD.image_tokens || 0);
    const outText = outD.text_tokens || 0, outAudio = outD.audio_tokens || 0;

    const model = document.getElementById('voiceModel').value;
    const p = pricingMap[model];
    let cost = null;
    if (p && p.type === 'token') {
        const textOutFree = outAudio > 0 && _RT_AUDIO_ONLY_OUTPUT_BILLING.has(model);
        cost = (inText + inVisual) / 1e6 * p.input
             + (textOutFree ? 0 : outText / 1e6 * p.output)
             + inAudio / 1e6 * (p.audio_input || p.input)
             + outAudio / 1e6 * (p.audio_output || p.output);
        addCost(cost);
    }
    const tok = (text, audio, visual) => {
        const s = [`${text} 文字`];
        if (audio) s.push(`${audio} 語音`);
        if (visual) s.push(`${visual} 畫面`);
        return s.join('、');
    };
    document.getElementById('voiceRtUsage').textContent =
        `輸入 ${tok(inText, inAudio, inVisual)}　輸出 ${tok(outText, outAudio)}` +
        (cost != null ? `　約 $${formatUsd(Number(cost.toFixed(6)))}` : '');
}

// ── 畫面輸入（圖片／影片）────────────────────────────────────────────────
// ⚠️ 這條路徑的形狀是實測出來的，**不要照 chat 的 image_url 寫法推**。實測結果：
//   - content 裡放 input_image（image_url / image、data URI / 裸 base64）→ 全都不行。
//     最危險的是「裸 base64」那個變體：**不報錯，但模型完全沒看到圖**（回「請提供圖片」）。
//   - content 裡放 input_video + 影格陣列 → 也不行，同樣是收下但看不到。這個變體一度
//     騙過我——單張圖問一次，它答「紅色」剛好對；換三張底色不同的圖再問，三次全都
//     回「白色、圓形」，才看出那是編出來的答案。**驗這種事一定要有會變的對照組。**
//   - 可用的是 `input_image_buffer.append`，但它有前置條件：必須**先送過音訊**，
//     否則回 `Error append image before append audio.`。送一段靜音就夠了。
// 驗證方式：三張底色與形狀都不同的圖各問一次，三次答案完全跟著圖變（藍/圓、綠/方、
// 黃/三角），才判定它真的讀到了。
const RT_MAX_FRAMES = 8;          // 影片取樣上限
let rtPendingFrames = [];         // 待送出的影格（裸 base64 JPEG，不含 data URI 前綴）

function onRealtimeFileChange(event) {
    const file = event.target.files[0];
    if (!file) return;
    const box = document.getElementById('voiceRtAttach');
    const name = document.getElementById('voiceRtAttachName');
    box.style.display = '';
    name.textContent = '處理中…';
    const done = (n) => {
        // 直接把第一張畫面當縮圖，使用者才確定「我附上的是這個」
        document.getElementById('voiceRtAttachThumb').src = `data:image/jpeg;base64,${rtPendingFrames[0]}`;
        name.textContent = n > 1 ? `${file.name}・取樣 ${n} 張畫面` : file.name;
        rtEnterFrameTurnMode();
    };
    if (file.type.startsWith('video/')) {
        rtExtractVideoFrames(file).then(frames => { rtPendingFrames = frames; done(frames.length); })
            .catch(() => { toast('影片讀取失敗', 'error'); clearRealtimeFile(); });
    } else {
        rtImageToJpegBase64(file).then(b64 => { rtPendingFrames = [b64]; done(1); })
            .catch(() => { toast('圖片讀取失敗', 'error'); clearRealtimeFile(); });
    }
}

// 附上畫面的那一輪**必須手動送出**。原因見 rtSendFramesIfAny 的註解：只要 VAD 開著，
// 影格就不會被帶上（而且不報錯）。所以一附上畫面就把這一輪切成手動模式，並且明白
// 告訴使用者要按哪顆按鈕——否則使用者對著麥克風問「這張圖是什麼」，模型會回
// 「我沒看到圖片」，而畫面上沒有任何線索說明為什麼。
function rtEnterFrameTurnMode() {
    rtUpdateCommitButton();
    if (!rtWs || rtWs.readyState !== WebSocket.OPEN) return;
    if (document.getElementById('voiceRtTurnDetection').value !== 'none') {
        rtWs.send(JSON.stringify({ type: 'session.update', session: { turn_detection: null } }));
        rtVadOverridden = true;
    }
    rtSetStatus(rtMicStream ? '已附上畫面：說完後按「說完了，送出」' : '已附上畫面：可以打字送出，或開麥克風說', 'on');
}

// 手動送出鈕出現的兩種情形：使用者自己選了手動模式，或這一輪附了畫面（附畫面時
// 一定要手動——開著 VAD 影格不會被帶上，見 rtSendFramesIfAny）
function rtUpdateCommitButton() {
    const manual = document.getElementById('voiceRtTurnDetection').value === 'none';
    const show = manual || rtPendingFrames.length > 0;
    document.getElementById('voiceRtCommitBtn').style.display = show ? '' : 'none';
    document.getElementById('voiceRtHintText').textContent = rtPendingFrames.length
        ? '附了畫面時，說完要自己按「說完了，送出」'
        : 'Enter 送出，Shift+Enter 換行';
}

function clearRealtimeFile() {
    rtPendingFrames = [];
    document.getElementById('voiceRtFileInput').value = '';
    document.getElementById('voiceRtAttach').style.display = 'none';
    document.getElementById('voiceRtAttachThumb').removeAttribute('src');
    rtUpdateCommitButton();
}

function rtImageToJpegBase64(file) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.onload = () => {
            // 縮到長邊 960，避免一張高解析度圖把單一 WebSocket 訊息撐得過大
            const scale = Math.min(1, 960 / Math.max(img.width, img.height));
            const c = document.createElement('canvas');
            c.width = Math.round(img.width * scale);
            c.height = Math.round(img.height * scale);
            c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
            URL.revokeObjectURL(img.src);
            resolve(c.toDataURL('image/jpeg', 0.85).split(',')[1]);
        };
        img.onerror = reject;
        img.src = URL.createObjectURL(file);
    });
}

// 影片在瀏覽器端取樣成畫面：上游的 image buffer 收的是一張張影格，不是影片檔
function rtExtractVideoFrames(file) {
    return new Promise((resolve, reject) => {
        const video = document.createElement('video');
        video.muted = true;
        video.src = URL.createObjectURL(file);
        video.onerror = reject;
        video.onloadedmetadata = async () => {
            const dur = video.duration || 0;
            const n = Math.max(1, Math.min(RT_MAX_FRAMES, Math.ceil(dur)));
            const scale = Math.min(1, 960 / Math.max(video.videoWidth, video.videoHeight));
            const c = document.createElement('canvas');
            c.width = Math.round(video.videoWidth * scale);
            c.height = Math.round(video.videoHeight * scale);
            const ctx = c.getContext('2d');
            const frames = [];
            for (let i = 0; i < n; i++) {
                const t = dur * (i + 0.5) / n;
                await new Promise(r => { video.onseeked = r; video.currentTime = t; });
                ctx.drawImage(video, 0, 0, c.width, c.height);
                frames.push(c.toDataURL('image/jpeg', 0.85).split(',')[1]);
            }
            URL.revokeObjectURL(video.src);
            resolve(frames);
        };
    });
}

// image buffer 的前置條件：要先送過音訊。送一段靜音把門打開。
//
// ⚠️ 而且**必須先關掉斷句偵測**。開著 semantic_vad 時，那段靜音會被 VAD 判定成
// 「沒有語音」而丟掉，圖片緩衝就不會被帶上——**不會有任何錯誤訊息**，模型會回一個
// 編出來的答案（三張完全不同的圖都回「白色、圓形」）。這個變因是拿 turn_detection
// 開/關兩組各跑三張對照圖才隔離出來的。
let rtVadOverridden = false;

function rtSendFramesIfAny() {
    if (!rtPendingFrames.length) return false;
    if (document.getElementById('voiceRtTurnDetection').value !== 'none') {
        rtWs.send(JSON.stringify({ type: 'session.update', session: { turn_detection: null } }));
        rtVadOverridden = true;
    }
    const silence = new Uint8Array(RT_IN_RATE * 0.6 * 2);   // 0.6 秒的 16kHz 靜音
    rtWs.send(JSON.stringify({ type: 'input_audio_buffer.append', audio: rtBytesToBase64(silence) }));
    rtPendingFrames.forEach(f => rtWs.send(JSON.stringify({ type: 'input_image_buffer.append', image: f })));
    return true;
}

// ── 送出 ──────────────────────────────────────────────────────────────────
function sendRealtimeText() {
    if (!rtWs || rtWs.readyState !== WebSocket.OPEN) { toast('請先開始對話', 'error'); return; }
    const box = document.getElementById('voiceRtText');
    const text = box.value.trim();
    if (!text && !rtPendingFrames.length) return;
    const frames = rtPendingFrames.slice();
    rtSendFramesIfAny();
    rtLog('你', text, 'me', frames);
    if (frames.length) clearRealtimeFile();
    rtWs.send(JSON.stringify({
        type: 'conversation.item.create',
        item: { type: 'message', role: 'user', content: [{ type: 'input_text', text: text || '描述你看到的畫面。' }] },
    }));
    rtWs.send(JSON.stringify({ type: 'response.create' }));
    box.value = '';
    rtSetStatus('思考中…', 'busy');
}

function commitRealtimeAudio() {
    if (!rtWs || rtWs.readyState !== WebSocket.OPEN) return;
    // 語音提問也要帶畫面——先前只有「送出文字」那條路徑會送影格，所以使用者用說的
    // 問「這張圖是什麼」，模型永遠回「我沒看到圖片」
    const frames = rtPendingFrames.slice();
    if (frames.length) {
        rtSendFramesIfAny();
        rtLog('你附上的畫面', '', 'me', frames);
        clearRealtimeFile();
    }
    rtWs.send(JSON.stringify({ type: 'input_audio_buffer.commit' }));
    rtWs.send(JSON.stringify({ type: 'response.create' }));
    rtSetStatus('思考中…', 'busy');
}

function addVoiceResultCard(title) {
    const area = document.getElementById('voiceResults');
    area.querySelector('.empty-state')?.remove();
    const card = el('div', { className: 'voice-result' });
    card.innerHTML = `<div class="voice-result-header"><span>${title}</span></div>`;
    area.insertBefore(card, area.firstChild);
    return card;
}


// ASR 選填參數（語言提示/熱詞/熱詞表 ID）→ 後端約定的表單形狀：
// language_hints 是 JSON 陣列字串（最多 4 個）、vocabulary 是 JSON 物件字串
// （{"詞": 權重}，權重 1~5，50=超級熱詞，未標權重預設 4）、vocabulary_id 純字串
function appendAsrExtraFields(fd) {
    const hints = document.getElementById('voiceAsrLangHints').value.trim();
    if (hints) {
        const arr = hints.split(/[,，]/).map(s => s.trim()).filter(Boolean).slice(0, 4);
        if (arr.length) fd.append('language_hints', JSON.stringify(arr));
    }
    const vocabRaw = document.getElementById('voiceAsrVocab').value.trim();
    if (vocabRaw) {
        const obj = {};
        vocabRaw.split('\n').map(s => s.trim()).filter(Boolean).forEach(line => {
            const m = line.match(/^(.+?)[\s:：]+(\d{1,2})$/);
            if (m) obj[m[1].trim()] = parseInt(m[2]);
            else obj[line] = 4;
        });
        if (Object.keys(obj).length) fd.append('vocabulary', JSON.stringify(obj));
    }
    const vocabId = document.getElementById('voiceAsrVocabId').value.trim();
    if (vocabId) fd.append('vocabulary_id', vocabId);
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
            appendAsrExtraFields(fd);
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
            appendAsrExtraFields(fd);
            const res = await apiPostForm('/api/voice/asr', fd);
            if (res.success) {
                textEl.textContent = res.text || '（無辨識結果）';
                card.querySelector('.voice-result-header span').textContent = `${model}（耗時 ${fmtElapsed(Date.now() - startTime)}）`;
                const reqPanel = buildRequestPanel(res.request);
                if (reqPanel) card.appendChild(reqPanel);
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

// ── 音樂生成（Lyria）──────────────────────────────────────────
let voiceMusicFile = null;

function onVoiceMusicFileChange(event) {
    const file = event.target.files[0];
    if (!file) return;
    voiceMusicFile = file;
    document.getElementById('voiceMusicLabel').innerHTML = `已選擇：${file.name}`;
    document.getElementById('voiceMusicIcon').textContent = '✅';
    document.getElementById('voiceMusicClearBtn').style.display = '';
}

function clearVoiceMusicFile() {
    voiceMusicFile = null;
    const input = document.getElementById('voiceMusicFileInput');
    if (input) input.value = '';
    const label = document.getElementById('voiceMusicLabel');
    if (label) label.innerHTML = '附一張圖片作為音樂靈感<br><span style="font-size:11px;color:var(--text-muted)">選填，模型會依畫面的氛圍作曲</span>';
    const icon = document.getElementById('voiceMusicIcon');
    if (icon) icon.textContent = '🎨';
    const btn = document.getElementById('voiceMusicClearBtn');
    if (btn) btn.style.display = 'none';
}

async function sendVoiceMusic() {
    const model = document.getElementById('voiceModel').value;
    const prompt = document.getElementById('voiceMusicPrompt').value.trim();
    if (!prompt) { toast('請描述你想要的音樂', 'error'); return; }

    const info = (models.voice?.music || []).find(m => m.id === model);
    const btn = document.getElementById('voiceMusicSendBtn');
    btn.disabled = true;
    showLoading(`音樂生成中（${info?.duration_hint || '約 30 秒'}），請稍候...`);
    const startTime = Date.now();

    try {
        const fd = new FormData();
        fd.append('model', model);
        fd.append('prompt', prompt);
        if (voiceMusicFile && info?.image_input) fd.append('image', voiceMusicFile);
        const res = await apiPostForm('/api/music/generate', fd);
        if (res.success && res.audio_url) {
            const card = addVoiceResultCard(`${model}（耗時 ${fmtElapsed(Date.now() - startTime)}）`);
            const audioEl = el('audio', { controls: true });
            audioEl.src = res.audio_url;
            card.appendChild(audioEl);
            // lyria-3 會附曲式/歌詞說明文字，一併呈現
            for (const t of res.texts || []) {
                const p = el('div', { className: 'voice-result-text', textContent: t });
                card.appendChild(p);
            }
            const meta = el('div', { className: 'voice-result-meta' });
            meta.innerHTML = `<a href="${res.audio_url}" download>下載音檔</a>`;
            card.appendChild(meta);
            const reqPanel = buildRequestPanel(res.request);
            if (reqPanel) card.appendChild(reqPanel);
            const p = pricingMap[model];
            if (p && p.type === 'fixed' && p.price) addCost(p.price);
            toast('音樂生成完成！', 'success');
        } else {
            // 錯誤訊息原樣呈現——安全過濾（content_blocked）之類的錯誤，使用者要看到
            // 原文才知道換個寫法就好，吞掉只會剩下「失敗」兩個字
            toast(res.error || '生成失敗', 'error');
        }
    } catch (e) {
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
            const reqPanel = buildRequestPanel(res.request);
            if (reqPanel) card.appendChild(reqPanel);
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

// ── 剪貼簿貼上圖片 ────────────────────────────────────────────
// 統一機制：把剪貼簿裡的圖片用 DataTransfer 塞進原本的 <input type=file> 再派發
// change 事件——完全走「點擊選檔」的既有流程，預覽、狀態、張數上限都由原 handler
// 處理，這裡只負責「路由到哪個輸入」。
//
// 每個分頁的圖片輸入按優先序列出（zone 是判斷可見性與閃提示用的容器）。貼上時取
// 「第一個可見的目標」；單檔輸入已有檔案、且後面還有可見的目標時讓位給下一個——
// 讓「貼首幀、再貼尾幀」這種兩段操作自然成立。AI Canvas 是獨立頁面（canvas.js），
// 不在此處理。
const _PASTE_TARGETS = {
    text:  [{input: 'textVisionInput',       zone: 'textVisionGroup',        label: '對話圖片'}],
    image: [{input: 'imgFileInput',          zone: 'imgUploadSection',       label: '參考圖片'}],
    video: [{input: 'vidFirstFrameInput',    zone: 'vidFirstFrameZone',      label: '首幀圖片'},
            {input: 'vidLastFrameInput',     zone: 'vidLastFrameZone',       label: '尾幀圖片'},
            {input: 'vidEditRefInput',       zone: 'vidEditRefZone',         label: '參考圖片'},
            {input: 'vidRefInput',           zone: 'vidR2VUpload',           label: '參考文件'},
            {input: 'vidAnimateImgInput',    zone: 'vidAnimateImgZone',      label: '人物圖片'}],
    muleai:[{input: 'muleaiFirstFrameInput', zone: 'muleaiImgUploadSection', label: '來源圖片'},
            {input: 'muleaiFaceImgInput',    zone: 'muleaiFaceImgSection',   label: '換臉參考圖'}],
    voice: [{input: 'voiceRtFileInput',      zone: 'voiceRtAttachBtn',       label: '即時對話畫面'},
            {input: 'voiceMusicFileInput',   zone: 'voiceMusicImageGroup',   label: '靈感圖片'}],
};

function _pickPasteTarget(tab) {
    const visible = (_PASTE_TARGETS[tab] || []).filter(t => {
        const z = document.getElementById(t.zone);
        return z && document.getElementById(t.input) && z.offsetParent !== null;
    });
    if (!visible.length) return null;
    // 優先給「還沒有檔案」的單檔輸入；多檔輸入（multiple）本來就是追加，不用讓位
    return visible.find(t => {
        const inp = document.getElementById(t.input);
        return inp.multiple || !inp.files.length;
    }) || visible[0];
}

function _flashPasteZone(zoneId) {
    const z = document.getElementById(zoneId);
    if (!z) return;
    z.classList.remove('paste-flash');
    requestAnimationFrame(() => z.classList.add('paste-flash'));
    setTimeout(() => z.classList.remove('paste-flash'), 1000);
}

document.addEventListener('paste', (e) => {
    const cd = e.clipboardData;
    if (!cd) return;
    const imgs = [...(cd.files || [])].filter(f => f.type.startsWith('image/'));
    if (!imgs.length) return;
    // 還沒登入（主畫面隱藏）就不攔
    const app = document.getElementById('mainApp');
    if (!app || app.classList.contains('hidden')) return;
    // 剪貼簿同時有文字、且焦點在文字欄位時，尊重「貼文字」的意圖不攔——
    // 從網頁複製的內容常常文字與圖片並存，攔下來會把使用者要貼的字吃掉
    const a = document.activeElement;
    if (a && (a.tagName === 'TEXTAREA' || a.tagName === 'INPUT') && cd.getData('text')) return;
    const tab = document.querySelector('.tab-btn.active')?.dataset.tab;
    const target = _pickPasteTarget(tab);
    if (!target) return;   // 這個頁面沒有可收圖的欄位（例如 t2i 模式），安靜略過
    e.preventDefault();
    const input = document.getElementById(target.input);
    const dt = new DataTransfer();
    (input.multiple ? imgs : imgs.slice(0, 1)).forEach(f => dt.items.add(f));
    input.files = dt.files;
    input.dispatchEvent(new Event('change'));
    _flashPasteZone(target.zone);
    toast(`已貼上圖片 → ${target.label}`, 'success');
});

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

    } else if (isW3SpicyVideo(model)) {
        // w3.0：首幀圖選填（沒有就是文生影片）；智能時長送 -1
        const isRefMode = document.getElementById('muleaiW3Mode').value === 'reference';
        if (isRefMode) {
            const durErr = checkMuleaiRefDurations();
            if (durErr) { toast(durErr, 'error'); return; }
            const total = muleaiRefFiles.image.length + muleaiRefFiles.video.length + muleaiRefFiles.audio.length;
            if (!total && !prompt) { toast('參考模式請至少提供一個素材或提示詞', 'error'); return; }
            muleaiRefFiles.image.forEach((f, i) => fd.append(`reference_image_${i + 1}`, f));
            muleaiRefFiles.video.forEach((f, i) => fd.append(`reference_video_${i + 1}`, f));
            muleaiRefFiles.audio.forEach((f, i) => fd.append(`reference_audio_${i + 1}`, f));
        } else {
            const firstFrameFile = document.getElementById('muleaiFirstFrameInput').files[0];
            if (firstFrameFile) fd.append('image', firstFrameFile);
            const lastFrameFile = document.getElementById('muleaiLastFrameInput').files[0];
            if (lastFrameFile) fd.append('last_frame', lastFrameFile);
        }
        fd.append('prompt', prompt);
        fd.append('resolution', resolution);
        fd.append('ratio', document.getElementById('muleaiVidRatio').value);
        fd.append('duration', document.getElementById('muleaiSmartDur')?.checked ? -1 : duration);
        fd.append('prompt_extend', extend);
        if (seed !== null) fd.append('seed', seed);
        if (document.getElementById('muleaiAudioEnable')?.checked) fd.append('enable_audio', 'true');

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
            addMuleAIVideoTask(res.task_id, model, displayPrompt, res.status, false, res.request);
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

function addMuleAIVideoTask(taskId, model, prompt, status, isResume = false, req = null) {
    const cont = document.getElementById('muleaiVideoResults');
    const empty = cont.querySelector('.empty-state');
    if (empty) empty.remove();
    const startTime = Date.now();
    if (!isResume) savePendingTask({ kind: 'muleai', taskId, model, prompt, req });
    const card = el('div', { className: 'video-task-card', id: 'mtask-' + taskId });
    card.innerHTML = '<div class="vtc-header"><span class="vtc-model">' + model + '</span><span class="vtc-timer" id="mtm-' + taskId + '">(耗時 0s)</span><span class="vtc-status ' + (status ? status.toLowerCase() : 'pending') + '" id="mst-' + taskId + '">' + (status || 'PENDING') + '</span></div><div class="vtc-prompt">' + prompt.substring(0, 120) + '</div><div class="vtc-progress"><div class="vtc-progress-bar" id="mpb-' + taskId + '" style="width:5%"></div></div><div id="mrv-' + taskId + '"></div>';
    const reqPanel = buildRequestPanel(req);
    if (reqPanel) card.appendChild(reqPanel);
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
