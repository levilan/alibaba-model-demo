/**
 * Alibaba Cloud AI Testing Platform
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
let apiKey = sessionStorage.getItem('dashscope_api_key') || '';
let muleApiKey = sessionStorage.getItem('muleai_api_key') || '';
let models = { text: [], image: [], video: [], voice: { asr: [], tts: [] }, tts_voices: [], cosyvoice_voices: {}, muleai: [] };
let refFiles = [];
let editRefFiles = [];  // for video editing reference images
let imgRefFiles = [];   // for image edit reference images (up to 9)
let muleaiImgRefFiles = [];
let loadingTimerInterval = null;
let asrFile = null;


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
        card.innerHTML = '<div class="vtc-header"><span class="vtc-model">' + model + '</span><span class="vtc-status succeeded">SUCCEEDED</span></div><div class="vtc-prompt">' + prompt.substring(0, 120) + '</div><video class="video-player" controls src="' + src + '"></video><div style="margin-top:8px"><a href="' + src + '" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a><button class="img-lb-btn" onclick="openLightbox(\'' + src + '\',\'video\')">⛶ 放大</button></div>';
        cont.insertBefore(card, cont.firstChild);
    }
}

function addMuleAIImageResult(model, prompt, src, isHistory = false) {
    const cont = document.getElementById('muleaiVideoResults');
    if (cont) {
        const empty = cont.querySelector('.empty-state');
        if (empty) empty.remove();
        const card = el('div', { className: 'video-task-card' });
        card.innerHTML = '<div class="vtc-header"><span class="vtc-model">' + model + '</span><span class="vtc-status succeeded">SUCCEEDED</span></div><div class="vtc-prompt">' + prompt.substring(0, 120) + '</div><img src="' + src + '" alt="Generated Image" style="max-width:100%;height:auto;border-radius:8px;cursor:zoom-in" onclick="openLightbox(\'' + src + '\')"><div style="margin-top:8px"><a href="' + src + '" download target="_blank" rel="noopener noreferrer" class="img-dl">下載圖片</a></div>';
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

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    if (apiKey) attemptAutoLogin();
    updateOmniVoices(document.getElementById('omniModel')?.value || 'qwen3.5-omni-flash-realtime');

    document.getElementById('apiKeyInput').addEventListener('keydown', e => {
        if (e.key === 'Enter') handleLogin();
    });
    document.getElementById('muleApiKeyInput').addEventListener('keydown', e => {
        if (e.key === 'Enter') handleLogin();
    });
        document.getElementById('textPrompt').addEventListener('keydown', e => {
        if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') sendText();
    });
    const muleaiPrompt = document.getElementById('muleaiPrompt');
    if (muleaiPrompt) {
        muleaiPrompt.addEventListener('keydown', e => {
            if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') sendMuleAIText();
        });
    }
});

async function attemptAutoLogin() {
    try {
        const res = await fetch('/api/models', { headers: authHeader() });
        if (res.ok) {
            models = await res.json();
            showApp();
        } else {
            apiKey = '';
            sessionStorage.removeItem('dashscope_api_key');
        }
    } catch (_) { /* show login */ }
}

// ── Auth ──────────────────────────────────────────────────────
async function handleLogin() {
    const key = document.getElementById('apiKeyInput').value.trim();
    const mKey = document.getElementById('muleApiKeyInput').value.trim();
    const errEl = document.getElementById('loginError');
    errEl.textContent = '';

    if (!key) { errEl.textContent = '請輸入 API Key'; return; }
    if (!key.startsWith('sk-')) { errEl.textContent = 'API Key 格式有誤，須以 sk- 開頭'; return; }

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
            sessionStorage.setItem('dashscope_api_key', key);
            if (mKey) {
                muleApiKey = mKey;
                sessionStorage.setItem('muleai_api_key', mKey);
            } else {
                muleApiKey = '';
                sessionStorage.removeItem('muleai_api_key');
            }
            const mRes = await fetch('/api/models', { headers: authHeader() });
            models = await mRes.json();
            showApp();
        } else {
            errEl.textContent = data.message || '驗證失敗，請確認 API Key';
        }
    } catch (e) {
        errEl.textContent = '網路錯誤，請重試';
    }
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><path d="M15 3h4a2 2 0 012 2v14a2 2 0 01-2 2h-4M10 17l5-5-5-5M15 12H3"/></svg><span>登入</span>';
}

function showApp() { console.log('SHOW APP EXECUTING'); 
    document.getElementById('loginOverlay').style.display = 'none';
    const app = document.getElementById('mainApp');
    app.classList.remove('hidden');
    app.style.display = 'flex';
    const masked = apiKey.slice(0, 6) + '****' + apiKey.slice(-4);
    document.getElementById('apiKeyLabel').textContent = masked;
    try { populateSelectors(); console.log('Populate OK'); TaskHistory.load(); } catch(e) { console.log('POPULATE ERROR:', e.message); toast('UI 載入發生錯誤，請聯絡開發者', 'error'); }
}

function handleLogout() {
    apiKey = '';
    muleApiKey = '';
    sessionStorage.removeItem('dashscope_api_key');
    sessionStorage.removeItem('muleai_api_key');
    location.reload();
}

function authHeader() {
    return { 
        'Authorization': 'Bearer ' + apiKey, 
        'Content-Type': 'application/json',
        'X-NenAI-API-Key': muleApiKey || ''
    };
}

// ── Selectors ─────────────────────────────────────────────────
function onMuleaiAudioToggle(cb) {
    document.getElementById('muleaiAudioUploadArea').style.display = cb.checked ? '' : 'none';
    if (!cb.checked) {
        document.getElementById('muleaiAudioInput').value = '';
        document.getElementById('muleaiAudioUpHint').innerHTML = '上傳音訊（可選）<br><span style="font-size:11px;color:var(--text-muted)">留空由平台自動配音</span>';
    }
}

function onMuleaiAudioFileChange(event) {
    const file = event.target.files[0];
    const hint = document.getElementById('muleaiAudioUpHint');
    if (file) {
        hint.innerHTML = `<strong>${file.name}</strong><br><span style="font-size:11px;color:var(--text-muted)">${(file.size/1024).toFixed(0)} KB</span>`;
    } else {
        hint.innerHTML = '上傳音訊（可選）<br><span style="font-size:11px;color:var(--text-muted)">留空由平台自動配音</span>';
    }
}

function onMuleaiModelChange() {
    const model = document.getElementById('muleaiModel').value;
    const isZImage   = model.includes('z-image');
    const isImgEdit  = model === 'qwen-image-edit-spicy';
    const isFaceSwap = model === 'face-swap';
    const isImageModel = isZImage || isImgEdit || isFaceSwap;

    // 解析度 / 時長控制
    document.getElementById('muleaiVidResGroup').style.display = isImageModel ? 'none' : '';
    document.getElementById('muleaiImgResGroup').style.display = isZImage ? '' : 'none';
    document.getElementById('muleaiVidDurGroup').style.display = isImageModel ? 'none' : '';

    // 來源圖片：影片模型必填、圖像編輯必填、換臉必填；純文生圖(z-image)不需要
    document.getElementById('muleaiImgUploadSection').style.display = (!isZImage) ? '' : 'none';

    // 換臉參考圖：僅 face-swap 顯示
    document.getElementById('muleaiFaceImgSection').style.display = isFaceSwap ? '' : 'none';

    // 配音設定：僅影片模型顯示
    document.getElementById('muleaiAudioSection').style.display = (!isImageModel) ? '' : 'none';
    if (isImageModel) {
        const cb = document.getElementById('muleaiEnableAudio');
        if (cb) { cb.checked = false; document.getElementById('muleaiAudioUploadArea').style.display = 'none'; }
    }

    // Prompt 區：face-swap 不需要
    document.getElementById('muleaiPromptSection').style.display = isFaceSwap ? 'none' : '';
    document.getElementById('muleaiPromptExtendGroup').style.display = isFaceSwap ? 'none' : '';

    // 更新上傳區標題
    const uploadTitle = document.getElementById('muleaiImgUploadTitle');
    if (uploadTitle) {
        if (isImgEdit) uploadTitle.textContent = '來源圖片 (必填)';
        else if (isFaceSwap) uploadTitle.textContent = '來源圖片 (必填)';
        else uploadTitle.textContent = '首幀圖片 (影片必填)';
    }

    const promptInput = document.getElementById('muleaiVidPrompt');
    if (promptInput) {
        if (isZImage) promptInput.placeholder = "描述圖片畫面與細節...";
        else if (isImgEdit) promptInput.placeholder = "描述編輯效果（例：將人物改為紅髮）...";
        else promptInput.placeholder = "描述影片動作與細節...";
    }

    const sendBtn = document.getElementById('muleaiVidSendBtn');
    if (sendBtn) {
        sendBtn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg>\n' + (isImageModel ? '生成圖片' : '生成影片');
    }
}

function populateSelectors() {
    populateSelect('textModel', models.text);
    populateSelect('muleaiModel', models.muleai || []);
    onImgTaskChange();
    onVidTaskChange();
    onMuleaiModelChange();
    populateSelect('asrModel', models.voice?.asr || []);
    populateSelect('ttsModel', models.voice?.tts || []);
    const ttsModelSel = document.getElementById('ttsModel');
    if (ttsModelSel) ttsModelSel.onchange = onTtsModelChange;
    restoreDesignKey();
    onTtsModelChange();
}

let clonedVoices = [];  // { voice_id, target_model, status } 來自後端的複刻音色

function onTtsModelChange() {
    const model = document.getElementById('ttsModel')?.value || '';
    if (model.startsWith('cosyvoice')) {
        loadClonedVoices(model);   // 會在取得清單後呼叫 populateTtsVoices
    } else {
        clonedVoices = [];
        populateTtsVoices();
    }
}

function populateTtsVoices() {
    const sel = document.getElementById('ttsVoice');
    if (!sel) return;
    const model = document.getElementById('ttsModel')?.value || '';
    const isCosy = model.startsWith('cosyvoice');
    const list = isCosy ? ((models.cosyvoice_voices || {})[model] || []) : (models.tts_voices || []);
    const prev = sel.value;
    sel.innerHTML = '';
    list.forEach(v => {
        sel.appendChild(Object.assign(document.createElement('option'), {
            value: v.id,
            textContent: `${v.name}（${v.gender}）— ${v.style}`,
        }));
    });
    // 複刻 / 設計音色（已就緒者可直接合成）
    clonedVoices.forEach(c => {
        const ready = !c.status || c.status === 'OK';
        const isDesign = c.region === 'beijing' || (c.voice_id || '').includes('-vd-');
        const m = (c.model || '').replace('cosyvoice-', '');
        const tag = isDesign ? `🎨 設計${m ? '(' + m + ')' : ''}` : '🎙️ 複刻';
        const opt = Object.assign(document.createElement('option'), {
            value: c.voice_id,
            textContent: `${tag} — ${c.voice_id}${ready ? '' : `（${c.status}）`}`,
        });
        if (!ready) opt.disabled = true;
        sel.appendChild(opt);
    });
    if (prev) sel.value = prev;
    // 切換複刻區塊顯示
    const cloneGroup = document.getElementById('ttsCloneGroup');
    if (cloneGroup) cloneGroup.style.display = isCosy ? '' : 'none';
    renderCloneList();
}

// 設計（北京區）API Key：前端輸入，存 localStorage，透過 X-Design-Api-Key header 帶到北京區操作
function getDesignKey() {
    return (document.getElementById('ttsDesignKey')?.value || '').trim();
}
function saveDesignKey() {
    try { localStorage.setItem('designApiKey', getDesignKey()); } catch (_) {}
}
function restoreDesignKey() {
    try {
        const k = localStorage.getItem('designApiKey');
        const el = document.getElementById('ttsDesignKey');
        if (k && el) el.value = k;
    } catch (_) {}
}
function voiceHeaders(extra) {
    const h = { 'Authorization': `Bearer ${apiKey}`, ...(extra || {}) };
    const dk = getDesignKey();
    if (dk) h['X-Design-Api-Key'] = dk;
    return h;
}

async function loadClonedVoices(model) {
    if (!apiKey) { clonedVoices = []; populateTtsVoices(); return; }
    try {
        const res = await fetch(`/api/voice/voices?target_model=${encodeURIComponent(model)}`, {
            headers: voiceHeaders(),
        });
        const data = await res.json();
        clonedVoices = (res.ok && data.success) ? (data.voices || []) : [];
    } catch (_) {
        clonedVoices = [];
    }
    populateTtsVoices();
}

function renderCloneList() {
    const box = document.getElementById('ttsCloneList');
    if (!box) return;
    if (!clonedVoices.length) {
        box.innerHTML = '<p style="margin:8px 0 0;color:var(--text-muted);font-size:12px">尚無複刻音色</p>';
        return;
    }
    box.innerHTML = '';
    clonedVoices.forEach(c => {
        const ready = !c.status || c.status === 'OK';
        const row = document.createElement('div');
        row.style.cssText = 'display:flex;align-items:center;gap:6px;margin-top:6px;font-size:12px';
        const label = document.createElement('span');
        label.style.cssText = 'flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap';
        label.title = c.voice_id;
        const isDesign = c.region === 'beijing' || (c.voice_id || '').includes('-vd-');
        const icon = !ready ? '⏳' : isDesign ? '🎨' : '🎙️';
        const m = (c.model || '').replace('cosyvoice-', '');
        label.textContent = `${icon} ${m ? '[' + m + '] ' : ''}${c.voice_id}${ready ? '' : `（${c.status}）`}`;
        const del = Object.assign(document.createElement('button'), {
            className: 'btn btn-ghost btn-sm', textContent: '刪除',
        });
        del.onclick = () => deleteClone(c.voice_id);
        row.append(label, del);
        box.appendChild(row);
    });
}

async function deleteClone(voiceId) {
    if (!confirm(`確定刪除複刻音色？\n${voiceId}`)) return;
    try {
        const res = await fetch(`/api/voice/voices/${encodeURIComponent(voiceId)}`, {
            method: 'DELETE',
            headers: voiceHeaders(),
        });
        const data = await res.json();
        if (res.ok && data.success) {
            loadClonedVoices(document.getElementById('ttsModel').value);
        } else {
            alert(data.error || '刪除失敗');
        }
    } catch (err) {
        alert('刪除失敗：' + err.message);
    }
}

let cloneFile = null;
function onCloneFileChange(e) {
    cloneFile = e.target.files[0] || null;
    document.getElementById('ttsCloneFileName').textContent = cloneFile ? cloneFile.name : '尚未選擇檔案';
    document.getElementById('ttsCloneBtn').style.display = cloneFile ? '' : 'none';
    document.getElementById('ttsCloneStatus').textContent = '';
}

async function sendClone() {
    if (!cloneFile) return;
    const model = document.getElementById('ttsModel').value;
    const btn = document.getElementById('ttsCloneBtn');
    const status = document.getElementById('ttsCloneStatus');
    btn.disabled = true;
    status.style.color = 'var(--text-muted)';
    status.textContent = '複刻中…（約需數秒）';
    try {
        const fd = new FormData();
        fd.append('audio', cloneFile);
        fd.append('target_model', model);
        const res = await fetch('/api/voice/clone', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}` },
            body: fd,
        });
        const data = await res.json();
        if (res.ok && data.success) {
            status.style.color = 'var(--success, #16a34a)';
            status.textContent = `✓ 複刻完成：${data.voice_id}`;
            await loadClonedVoices(model);
            document.getElementById('ttsVoice').value = data.voice_id;
        } else {
            status.style.color = 'var(--error, #dc2626)';
            status.textContent = data.error || '複刻失敗';
        }
    } catch (err) {
        status.style.color = 'var(--error, #dc2626)';
        status.textContent = '複刻失敗：' + err.message;
    } finally {
        btn.disabled = false;
    }
}

async function sendDesign() {
    const model = document.getElementById('ttsDesignModel').value;
    const prompt = document.getElementById('ttsDesignPrompt').value.trim();
    const preview = document.getElementById('ttsDesignPreview').value.trim();
    const btn = document.getElementById('ttsDesignBtn');
    const status = document.getElementById('ttsDesignStatus');
    const audio = document.getElementById('ttsDesignPreviewAudio');
    if (!getDesignKey()) { status.style.color = 'var(--error, #dc2626)'; status.textContent = '請先填入設計 API Key（北京區）'; return; }
    if (!prompt) { status.style.color = 'var(--error, #dc2626)'; status.textContent = '請輸入音色描述'; return; }
    if (preview.length < 15) { status.style.color = 'var(--error, #dc2626)'; status.textContent = '試聽文字需 15–200 字（必填）'; return; }
    btn.disabled = true;
    audio.style.display = 'none';
    status.style.color = 'var(--text-muted)';
    status.textContent = '生成中…（約需數秒）';
    try {
        const res = await fetch('/api/voice/design', {
            method: 'POST',
            headers: voiceHeaders({ 'Content-Type': 'application/json' }),
            body: JSON.stringify({ target_model: model, voice_prompt: prompt, preview_text: preview }),
        });
        const data = await res.json();
        if (res.ok && data.success) {
            status.style.color = 'var(--success, #16a34a)';
            status.textContent = `✓ 已生成：${data.voice_id}`;
            if (data.preview_url) {
                audio.src = data.preview_url;
                audio.style.display = '';
            }
            await loadClonedVoices(document.getElementById('ttsDesignModel').value);
            document.getElementById('ttsVoice').value = data.voice_id;
        } else {
            status.style.color = 'var(--error, #dc2626)';
            status.textContent = data.error || '生成失敗';
        }
    } catch (err) {
        status.style.color = 'var(--error, #dc2626)';
        status.textContent = '生成失敗：' + err.message;
    } finally {
        btn.disabled = false;
    }
}

function populateSelect(id, list, filterFn = null) {
    const sel = document.getElementById(id);
    sel.innerHTML = '';
    const filtered = filterFn ? list.filter(filterFn) : list;
    let group = '';
    filtered.forEach(m => {
        if (m.group !== group) {
            sel.appendChild(Object.assign(document.createElement('optgroup'), { label: m.group }));
            group = m.group;
        }
        sel.lastElementChild.appendChild(
            Object.assign(document.createElement('option'), { value: m.id, textContent: `${m.name} — ${m.desc}` })
        );
    });
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

function onImgModelChange() {
    const t = document.getElementById('imageTaskType').value;
    const modelId = document.getElementById('imageModel').value;
    // 同一 model id 可能同時存在 t2i 與 i2i 兩筆資料（如 qwen-image-2.0），需依 type 一併比對避免混淆
    const modelInfo = models.image.find(m => m.id === modelId && m.type === t) || {};

    // 更新尺寸選單
    const sizeEl = document.getElementById('imageSize');
    const currentSize = sizeEl.value;
    const sizes = modelInfo.sizes || ["1024*1024","1280*720","720*1280","1024*768","768*1024"];
    const sizeLabels = {
        "1024*1024": "1024×1024 (1:1)", "1280*720": "1280×720 (16:9)", "720*1280": "720×1280 (9:16)",
        "1024*768": "1024×768 (4:3)", "768*1024": "768×1024 (3:4)",
        "960*1280": "960×1280 (3:4)", "1280*960": "1280×960 (4:3)",
        "960*1696": "960×1696 (9:16)", "1696*960": "1696×960 (16:9)",
    };
    sizeEl.innerHTML = sizes.map(s =>
        `<option value="${s}"${s === currentSize ? ' selected' : ''}>${sizeLabels[s] || s}</option>`
    ).join('');

    // 更新張數上限（i2i 模式下，僅 max_n > 1 的模型如 qwen-image-2.0 系列才顯示張數選擇）
    const maxN = modelInfo.max_n || 4;
    const nSlider = document.getElementById('imgN');
    nSlider.max = maxN;
    if (parseInt(nSlider.value) > maxN) {
        nSlider.value = maxN;
        document.getElementById('imgNVal').textContent = maxN;
    }
    document.getElementById('imgNGroup').style.display = (maxN > 1) ? '' : 'none';

    // ref_strength 僅 Wan 圖像編輯系列支援，qwen-image-2.0 系列無此參數
    document.getElementById('imgRefStrengthGroup').style.display =
        (t === 'i2i' && !modelInfo.no_ref_strength) ? '' : 'none';

    // prompt_extend 僅 T2I 與 qwen-image-2.0 系列（i2i 融合模型）支援，其餘 I2I 圖像編輯模型後端不支援此參數
    document.getElementById('imgPromptExtendGroup').style.display =
        (t === 't2i' || modelInfo.no_ref_strength) ? '' : 'none';

    // 參考圖張數上限（qwen-image-2.0 系列最多 3 張，其餘模型最多 9 張）
    imgMaxRef = modelInfo.max_ref || 9;
    if (imgRefFiles.length > imgMaxRef) imgRefFiles = imgRefFiles.slice(0, imgMaxRef);
    renderImgThumbs();
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
    const modelInfo = models.video.find(m => m.id === modelId) || {};

    // 顯示/隱藏自動配音
    const audioRow = document.getElementById('vidAudioRow');
    audioRow.style.display = modelInfo.audio ? '' : 'none';
    if (!modelInfo.audio) document.getElementById('vidAudio').checked = false;

    // 調整時長範圍
    const dur    = document.getElementById('videoDuration');
    const minD   = modelInfo.min_dur ?? 3;
    const maxD   = modelInfo.max_dur || 10;
    dur.min  = minD;
    dur.max  = maxD;
    dur.step = 1;
    let curVal = parseInt(dur.value);
    if (curVal < minD) { curVal = minD; }
    if (curVal > maxD) { curVal = maxD; }
    dur.value = curVal;
    document.getElementById('durVal').textContent = curVal;
    const rangeEl = document.getElementById('durRange');
    if (rangeEl) rangeEl.textContent = `（${minD} ~ ${maxD} 秒）`;

    // resolution: vedit only supports 720P/1080P
    const resEl = document.getElementById('videoResolution');
    if (taskType === 'vedit') {
        Array.from(resEl.options).forEach(o => {
            o.hidden = (o.value === '480P');
        });
        if (resEl.value === '480P') resEl.value = '720P';
    } else {
        Array.from(resEl.options).forEach(o => { o.hidden = false; });
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
function onEditRefUpload(e) {
    const newFiles = Array.from(e.target.files);
    editRefFiles = [...editRefFiles, ...newFiles].slice(0, 3);
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

// ── Voice ─────────────────────────────────────────────────────
function onVoiceTaskChange() {
    const t = document.getElementById('voiceTaskType').value;
    document.getElementById('voiceAsrPanel').style.display = t === 'asr' ? '' : 'none';
    document.getElementById('voiceTtsPanel').style.display = t === 'tts' ? '' : 'none';
    document.getElementById('voiceAsrMain').style.display  = t === 'asr' ? '' : 'none';
    document.getElementById('voiceTtsMain').style.display  = t === 'tts' ? '' : 'none';
}

function onAsrFileChange(e) {
    const file = e.target.files[0];
    if (!file) return;
    asrFile = file;
    document.getElementById('asrFileName').textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
}

function onAsrDrop(e) {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    asrFile = file;
    document.getElementById('asrFileName').textContent = `${file.name}  (${(file.size / 1024).toFixed(1)} KB)`;
}

async function sendASR() {
    if (!asrFile) { toast('請先上傳音訊檔案', 'error'); return; }
    const model = document.getElementById('asrModel').value;
    const btn   = document.getElementById('asrSendBtn');

    btn.disabled = true;
    btn.textContent = '識別中...';
    document.getElementById('asrResult').classList.add('hidden');

    const fd = new FormData();
    fd.append('audio', asrFile);
    fd.append('model', model);

    try {
        const res = await fetch('/api/voice/asr', {
            method: 'POST',
            headers: { 'Authorization': `Bearer ${apiKey}` },
            body: fd,
        });
        const data = await res.json();
        if (data.success) {
            document.getElementById('asrResultText').textContent = data.text || '（無識別內容）';
            document.getElementById('asrResultMeta').textContent = `模型：${data.model}`;
            document.getElementById('asrResult').classList.remove('hidden');
            toast('識別完成', 'success');
        } else {
            toast(data.error || '識別失敗', 'error');
        }
    } catch (e) {
        toast('網路錯誤：' + e.message, 'error');
    }
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="5 3 19 12 5 21 5 3"/></svg> 開始識別';
}

function copyAsrResult() {
    const text = document.getElementById('asrResultText').textContent;
    navigator.clipboard.writeText(text).then(() => toast('已複製到剪貼板', 'success'));
}

document.addEventListener('DOMContentLoaded', () => {
    const ta = document.getElementById('ttsPrompt');
    if (ta) ta.addEventListener('input', () => {
        document.getElementById('ttsCharCount').textContent = ta.value.length;
    });
});

async function sendTTS() {
    const text  = document.getElementById('ttsPrompt').value.trim();
    if (!text) { toast('請輸入合成文字', 'error'); return; }
    const model  = document.getElementById('ttsModel').value;
    const voice  = document.getElementById('ttsVoice').value;
    const format = document.getElementById('ttsFormat').value;
    const btn    = document.getElementById('ttsSendBtn');

    btn.disabled = true;
    btn.textContent = '合成中...';
    document.getElementById('ttsResult').classList.add('hidden');

    try {
        const res = await fetch('/api/voice/tts', {
            method: 'POST',
            headers: { ...authHeader(), ...(getDesignKey() ? { 'X-Design-Api-Key': getDesignKey() } : {}) },
            body: JSON.stringify({ model, voice, text, format }),
        });
        const data = await res.json();
        if (data.success) {
            const player = document.getElementById('ttsAudioPlayer');
            player.src   = data.audio_url;
            player.load();
            const dl = document.getElementById('ttsDownloadLink');
            dl.href = data.audio_url;
            dl.download = data.audio_url.split('/').pop();
            document.getElementById('ttsResultMeta').textContent =
                `模型：${data.model}  ·  音色：${data.voice}  ·  格式：${format.toUpperCase()}`;
            document.getElementById('ttsResult').classList.remove('hidden');
            toast('語音合成完成', 'success');
        } else {
            toast(data.error || '合成失敗', 'error');
        }
    } catch (e) {
        toast('網路錯誤：' + e.message, 'error');
    }
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"/><path d="M15.54 8.46a5 5 0 010 7.07M19.07 4.93a10 10 0 010 14.14"/></svg> 合成語音';
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

    const btn = document.getElementById('textSendBtn');
    btn.disabled = true;
    document.getElementById('textPrompt').value = '';

    try {
        const body = {
            model, prompt, system_prompt: systemPrompt,
            temperature, top_p: topP, max_tokens: maxTokens,
            presence_penalty: presencePenalty, frequency_penalty: frequencyPenalty,
            stream: useStream,
            enable_thinking: enableThinking && modelInfo?.thinking,
        };
        if (topK > 0) body.top_k = topK;
        if (seed !== null) body.seed = seed;
        if (stop.length > 0) body.stop = stop;

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
                contentDiv.textContent = data.content || '';
                const meta = el('div', { className: 'msg-meta' });
                meta.innerHTML = '<span>' + model + ' (耗時 ' + elapsed + 's)</span><span>' + new Date().toLocaleTimeString() + '</span>';
                aDiv.appendChild(meta);
            }
        } else {
            const reader  = res.body.getReader();
            const decoder = new TextDecoder();
            let full = '', buf = '';
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
                        if (d.content) {
                            full += d.content;
                            contentDiv.textContent = full;
                            output.scrollTop = output.scrollHeight;
                        } else if (d.error) {
                            contentDiv.textContent = '⚠ 錯誤：' + d.error;
                            aDiv.classList.remove('streaming-cursor');
                        }
                    } catch (_) { /* skip */ }
                }
            }
            aDiv.classList.remove('streaming-cursor');
            const elapsed = ((Date.now() - startTime) / 1000).toFixed(2);
            const meta = el('div', { className: 'msg-meta' });
            meta.innerHTML = '<span>' + model + ' (耗時 ' + elapsed + 's)</span><span>' + new Date().toLocaleTimeString() + '</span>';
            aDiv.appendChild(meta);
        }
    } catch (e) {
        contentDiv.textContent = '⚠ 錯誤：' + e.message;
        aDiv.classList.remove('streaming-cursor');
    }
    btn.disabled = false;
}

function clearChat() {
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
    const size     = document.getElementById('imageSize').value;
    const extend      = document.getElementById('imgPromptExtend').checked;
    const watermark   = document.getElementById('imgWatermark').checked;
    const n           = parseInt(document.getElementById('imgN').value) || 1;
    const imgSeedRaw  = document.getElementById('imgSeed').value.trim();
    const imgSeed     = imgSeedRaw !== '' ? parseInt(imgSeedRaw) : null;
    const refStrength = parseFloat(document.getElementById('imgRefStrength').value);

    if (!prompt) { toast('請輸入 Prompt', 'error'); return; }

    const btn = document.getElementById('imageSendBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> 生成中...';
    showLoading('圖片生成中，請稍候...');

    try {
        let res;
        if (taskType === 't2i') {
            const body = { model, prompt, negative_prompt: negPrompt, size, n, prompt_extend: extend, watermark };
            if (imgSeed !== null) body.seed = imgSeed;
            res = await apiPost('/api/image/generate', body);
        } else {
            if (!imgRefFiles.length) { toast('請先上傳至少一張參考圖片', 'error'); hideLoading(); btn.disabled = false; btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg> 生成'; return; }
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('size', size);
            fd.append('watermark', watermark); fd.append('ref_strength', refStrength);
            fd.append('n', n); fd.append('prompt_extend', extend);
            if (imgSeed !== null) fd.append('seed', imgSeed);
            imgRefFiles.forEach((f, i) => fd.append(`image_${i + 1}`, f));
            res = await apiPostForm('/api/image/edit', fd);
        }

        if (res.success && res.images?.length) {
            const gallery = document.getElementById('imageResults');
            gallery.querySelector('.empty-state')?.remove();
            res.images.forEach(img => {
                const src = img.local_path || img.url;
                const card = el('div', { className: 'img-card' });
                card.innerHTML = `
                    <img src="${src}" alt="Generated" loading="lazy" onclick="openLightbox('${src}')">
                    <div class="img-card-footer">
                        <span class="img-model-tag">${res.model}</span>
                        <a href="${src}" download class="img-dl">下載</a>
                    </div>`;
                gallery.insertBefore(card, gallery.firstChild);
            });
            toast(`圖片生成完成！共 ${res.images.length} 張`, 'success');
        } else {
            const errMsg = res.error || '生成失敗';
            toast(errMsg, 'error');
            console.error('Image generation error:', res);
        }
    } catch (e) {
        toast(`錯誤：${e.message}`, 'error');
    }
    hideLoading();
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg> 生成';
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

    const btn = document.getElementById('videoSendBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> 提交中...';

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
                const audioFile = document.getElementById('vidT2VAudioInput')?.files[0];
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
            addVideoTask(res.task_id, model, prompt, res.status);
            toast('任務已提交，輪詢中...', 'info');
        } else if (res.success && res.video_url) {
            addVideoResult(model, prompt, res.local_path || res.video_url);
            TaskHistory.save('video', model, prompt, res.local_path || res.video_url);
            toast('影片生成完成！', 'success');
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

function addVideoTask(taskId, model, prompt, status) {
    const cont = document.getElementById('videoResults');
    cont.querySelector('.empty-state')?.remove();
    const startTime = Date.now();
    const card = el('div', { className: 'video-task-card', id: `task-${taskId}` });
    card.innerHTML = `
        <div class="vtc-header">
            <span class="vtc-model">${model}</span>
            <span class="vtc-status ${status?.toLowerCase() || 'pending'}" id="st-${taskId}">${status || 'PENDING'}</span>
            <span class="vtc-timer" id="tm-${taskId}">0s</span>
        </div>
        <div class="vtc-prompt">${prompt.substring(0, 120)}${prompt.length > 120 ? '...' : ''}</div>
        <div class="vtc-progress"><div class="vtc-progress-bar" id="pb-${taskId}" style="width:5%"></div></div>
        <div id="rv-${taskId}"></div>`;
    cont.insertBefore(card, cont.firstChild);
    pollVideo(taskId, startTime);
}

function addVideoResult(model, prompt, src, isHistory = false) {
    const cont = document.getElementById('videoResults');
    cont.querySelector('.empty-state')?.remove();
    const card = el('div', { className: 'video-task-card' });
    card.innerHTML = `
        <div class="vtc-header"><span class="vtc-model">${model}</span><span class="vtc-status succeeded">SUCCEEDED</span></div>
        <div class="vtc-prompt">${prompt.substring(0, 120)}</div>
        <video class="video-player" controls src="${src}"></video>
        <div style="margin-top:8px"><a href="${src}" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a><button class="img-lb-btn" onclick="openLightbox('${src}','video')">⛶ 放大</button></div>`;
    cont.insertBefore(card, cont.firstChild);
}

async function pollVideo(taskId, startTime) {
    let tries = 0;
    const maxTries = 180; // 15 min max (5s * 180)
    const poll = async () => {
        tries++;
        // 更新計時器
        const elapsed = Math.floor((Date.now() - startTime) / 1000);
        const tmEl = document.getElementById(`tm-${taskId}`);
        if (tmEl) tmEl.textContent = elapsed >= 60 ? `${Math.floor(elapsed/60)}m${elapsed%60}s` : `${elapsed}s`;

        if (tries > maxTries) { updateVTC(taskId, 'TIMEOUT', null, '等待超時'); return; }
        try {
            const res = await fetch(`/api/video/status/${taskId}`, { headers: { 'Authorization': `Bearer ${apiKey}` } });
            const data = await res.json();
            const st = data.status;
            const stEl = document.getElementById(`st-${taskId}`);
            const pbEl = document.getElementById(`pb-${taskId}`);
            const rvEl = document.getElementById(`rv-${taskId}`);

            if (st === 'SUCCEEDED') {
                if (stEl) { stEl.textContent = 'SUCCEEDED'; stEl.className = 'vtc-status succeeded'; }
                if (pbEl) pbEl.style.width = '100%';
                if (rvEl && data.local_path) {
                    rvEl.innerHTML = `<video class="video-player" controls src="${data.local_path}"></video>
                        <div style="margin-top:8px"><a href="${data.local_path}" download class="img-dl">下載影片</a><button class="img-lb-btn" onclick="openLightbox('${data.local_path}','video')">⛶ 放大</button></div>`;
                }
                toast('影片生成完成！', 'success');
            } else if (st === 'FAILED') {
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
    refFiles = [...refFiles, ...Array.from(e.target.files)];
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

// ── API helpers ───────────────────────────────────────────────
async function apiPost(url, body) {
    const r = await fetch(url, { method: 'POST', headers: authHeader(), body: JSON.stringify(body) });
    if (r.status === 401) { if(url.includes('muleai')) throw new Error('MuleAI API Key 無效或未提供'); handleLogout(); throw new Error('Unauthorized'); }
    return r.json();
}
async function apiPostForm(url, fd) {
    const headers = { 'Authorization': `Bearer ${apiKey}` };
    if (typeof muleApiKey !== 'undefined' && muleApiKey) {
        headers['X-NenAI-API-Key'] = muleApiKey;
    }
    const r = await fetch(url, { method: 'POST', headers: headers, body: fd });
    if (r.status === 401) { if(url.includes('muleai')) throw new Error('NenAI API Key 無效或未提供'); handleLogout(); throw new Error('Unauthorized'); }
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


// ── MuleAI Generation ───────────────────────────────────────────

async function sendMuleAIVideo() {
    const model = document.getElementById('muleaiModel').value || 'wan2.7-i2v-spicy';
    const isZImage   = model.includes('z-image');
    const isImgEdit  = model === 'qwen-image-edit-spicy';
    const isFaceSwap = model === 'face-swap';
    const isImageModel = isZImage || isImgEdit || isFaceSwap;

    const prompt    = document.getElementById('muleaiVidPrompt').value.trim();
    const negPrompt = document.getElementById('muleaiVidNegPrompt').value.trim();
    const extend    = document.getElementById('muleaiVidPromptExtend').checked;
    const seedRaw   = document.getElementById('muleaiVidSeed').value.trim();
    const seed      = seedRaw !== '' ? parseInt(seedRaw) : null;

    if (!isFaceSwap && !prompt) { toast('請輸入 Prompt', 'error'); return; }

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
        // video
        const srcFile = document.getElementById('muleaiFirstFrameInput').files[0];
        if (!srcFile) { toast('請上傳首幀圖片', 'error'); return; }
        fd.append('image', srcFile);
        fd.append('prompt', prompt);
        fd.append('negative_prompt', negPrompt);
        fd.append('resolution', document.getElementById('muleaiVidResolution').value);
        fd.append('duration', parseInt(document.getElementById('muleaiVidDuration').value));
        fd.append('prompt_extend', extend);
        if (seed !== null) fd.append('seed', seed);
        const enableAudio = document.getElementById('muleaiEnableAudio')?.checked || false;
        fd.append('enable_audio', enableAudio);
        if (enableAudio) {
            const audioFile = document.getElementById('muleaiAudioInput').files[0];
            if (audioFile) fd.append('audio', audioFile);
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
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> ' + (isImageModel ? '生成圖片' : '生成影片');
}

function addMuleAIVideoTask(taskId, model, prompt, status) {
    const cont = document.getElementById('muleaiVideoResults');
    const empty = cont.querySelector('.empty-state');
    if (empty) empty.remove();
    const startTime = Date.now();
    const card = el('div', { className: 'video-task-card', id: 'mtask-' + taskId });
    card.innerHTML = '<div class="vtc-header"><span class="vtc-model">' + model + '</span><span class="vtc-status ' + (status ? status.toLowerCase() : 'pending') + '" id="mst-' + taskId + '">' + (status || 'PENDING') + '</span><span class="vtc-timer" id="mtm-' + taskId + '">0s</span></div><div class="vtc-prompt">' + prompt.substring(0, 120) + '</div><div class="vtc-progress"><div class="vtc-progress-bar" id="mpb-' + taskId + '" style="width:5%"></div></div><div id="mrv-' + taskId + '"></div>';
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
        if (tmEl) tmEl.textContent = elapsed >= 60 ? Math.floor(elapsed/60) + 'm' + (elapsed%60) + 's' : elapsed + 's';

        if (tries > maxTries) { 
            const stEl = document.getElementById('mst-' + taskId);
            if (stEl) { stEl.textContent = 'TIMEOUT'; stEl.className = 'vtc-status failed'; }
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
                        rvEl.innerHTML = '<video class="video-player" controls src="' + src + '"></video><div style="margin-top:8px"><a href="' + src + '" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a><button class="img-lb-btn" onclick="openLightbox(\'' + src + '\',\'video\')">⛶ 放大</button></div>';
                    } else if (data.images && data.images.length > 0) {
                        const src = data.images[0];
                        rvEl.innerHTML = `<img src="${src}" alt="Generated Image" style="max-width:100%;height:auto;border-radius:8px;cursor:zoom-in" onclick="openLightbox('${src}')"><div style="margin-top:8px"><a href="${src}" download target="_blank" rel="noopener noreferrer" class="img-dl">下載圖片</a></div>`;
                    }
                }
                toast('任務完成！', 'success');
            } else if (st === 'FAILED' || st === 'failed') {
                if (stEl) { stEl.textContent = 'FAILED'; stEl.className = 'vtc-status failed'; }
                if (pbEl) { pbEl.style.width = '100%'; pbEl.style.background = 'var(--red)'; }
                if (rvEl) rvEl.innerHTML = '<p style="font-size:0.82rem;color:var(--red)">錯誤：' + (data.error_message || (data.error ? data.error.detail : '未知錯誤')) + '</p>';
                toast('生成失敗', 'error');
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
