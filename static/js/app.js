/**
 * NenAI Testing Platform
 * Frontend JS — API Key auth, SSE streaming, polling
 */

// ── State ─────────────────────────────────────────────────────
let apiKey = sessionStorage.getItem('nenai_api_key') || '';
let models = { text: [], image: [], video: [], muleai: [] };
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
                else if (item.type === 'muleai_video') addMuleAIVideoResult(item.model, item.prompt, item.url, true);
                else if (item.type === 'muleai_image') addMuleAIImageResult(item.model, item.prompt, item.url, true);
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
        if (!isHistory) TaskHistory.save('muleai_video', model, prompt, src);
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
        if (!isHistory) TaskHistory.save('muleai_image', model, prompt, src);
    }
}

// ── Init ──────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
    if (apiKey) attemptAutoLogin();

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

// ── Auth ──────────────────────────────────────────────────────
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
        } else {
            errEl.textContent = data.message || '驗證失敗，請確認 API Key';
        }
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
    try { populateSelectors(); TaskHistory.load(); } catch(e) { toast('UI 載入發生錯誤，請聯絡開發者', 'error'); }
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
    onImgTaskChange();
    onVidTaskChange();
    onMuleaiModelChange();
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
function onImgTaskChange() {
    const t = document.getElementById('imageTaskType').value;
    populateSelect('imageModel', models.image, m => m.type === t);
    document.getElementById('imgUploadSection').classList.toggle('hidden', t !== 'i2i');
    document.getElementById('imgNGroup').style.display = (t === 't2i') ? '' : 'none';
    document.getElementById('imgRefStrengthGroup').style.display = (t === 'i2i') ? '' : 'none';
    if (t !== 'i2i') { imgRefFiles = []; renderImgThumbs(); }
    onImgModelChange();
}

function onImgModelChange() {
    const modelId = document.getElementById('imageModel').value;
    const modelInfo = models.image.find(m => m.id === modelId) || {};

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

    // 更新張數上限
    const maxN = modelInfo.max_n || 4;
    const nSlider = document.getElementById('imgN');
    nSlider.max = maxN;
    if (parseInt(nSlider.value) > maxN) {
        nSlider.value = maxN;
        document.getElementById('imgNVal').textContent = maxN;
    }
}

// ── Video 任務/模型切換 ────────────────────────────────────────
function onVidTaskChange() {
    const t = document.getElementById('videoTaskType').value;
    populateSelect('videoModel', models.video, m => m.type === t);

    document.getElementById('vidI2VUpload').classList.toggle('hidden', t !== 'i2v');
    document.getElementById('vidR2VUpload').classList.toggle('hidden', t !== 'r2v');
    document.getElementById('vidEditUpload').classList.toggle('hidden', t !== 'vedit');

    // vedit-specific controls
    document.getElementById('vidRatioGroup').style.display = (t === 'vedit') ? '' : 'none';
    document.getElementById('vidAudioSettingGroup').style.display = (t === 'vedit') ? '' : 'none';

    // i2v-specific controls
    document.getElementById('vidI2VModeGroup').style.display = (t === 'i2v') ? '' : 'none';

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


// ── Omni Realtime ─────────────────────────────────────────────
let omniAudioContext;
let omniMicrophone;
let omniProcessor;
let omniWebSocket;
let outAudioCtx;
let nextPlayTime = 0;

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
                "input_audio_transcription": {
                    "model": null
                },
                "turn_detection": {
                    "type": "server_vad",
                    "threshold": 0.2,
                    "prefix_padding_ms": 300,
                    "silence_duration_ms": 800
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
    const roleColor = role === 'User' ? 'var(--primary-color)' : role === 'LLM' ? '#00c853' : '#757575';
    area.innerHTML += `<div style="margin-bottom: 5px;"><strong style="color:${roleColor}">[${role}]</strong> ${text}</div>`;
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

    if (!prompt && taskType !== 'vedit') { toast('請輸入 Prompt', 'error'); return; }

    const btn = document.getElementById('videoSendBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> 提交中...';

    try {
        let res;
        if (taskType === 't2v') {
            const body = { model, prompt, negative_prompt: negPrompt, resolution, duration, audio,
                           prompt_extend: vidExtend, watermark: vidWatermark };
            if (vidSeed !== null) body.seed = vidSeed;
            res = await apiPost('/api/video/t2v', body);

        } else if (taskType === 'i2v') {
            const i2vMode = document.getElementById('videoI2VMode').value;
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('resolution', resolution);
            fd.append('duration', duration); fd.append('i2v_mode', i2vMode);
            fd.append('prompt_extend', vidExtend); fd.append('watermark', vidWatermark);
            if (vidSeed !== null) fd.append('seed', vidSeed);

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

        } else {
            // r2v
            if (!refFiles.length) { toast('請上傳參考文件', 'error'); btn.disabled = false; btn.innerHTML = _vidBtnHTML(); return; }
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('resolution', resolution); fd.append('duration', duration);
            fd.append('prompt_extend', vidExtend); fd.append('watermark', vidWatermark);
            if (vidSeed !== null) fd.append('seed', vidSeed);
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
        <div class="video-card-actions">
            <a href="${src}" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a>
            <button class="btn btn-ghost btn-sm" onclick="openLightbox('${src}', 'video')">展開預覽</button>
        </div>`;
    cont.insertBefore(card, cont.firstChild);
}

async function pollVideo(taskId, startTime) {
    let tries = 0;
    const maxTries = 360; // 30 min max (5s * 360) — video-edit/重型任務常超過 15 min
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
                } else if (rvEl && data.video_url) {
                    rvEl.innerHTML = `<video class="video-player" controls src="${data.video_url}"></video>
                        <div class="video-card-actions">
                            <a href="${data.video_url}" download target="_blank" rel="noopener noreferrer" class="img-dl">下載影片</a>
                            <button class="btn btn-ghost btn-sm" onclick="openLightbox('${data.video_url}', 'video')">展開預覽</button>
                        </div>`;
                }
                toast('影片生成完成！', 'success');
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
    const remaining = 9 - imgRefFiles.length;
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
    if (countEl) countEl.textContent = `${imgRefFiles.length} / 9 張`;
    if (addBtn) addBtn.style.display = imgRefFiles.length >= 9 ? 'none' : '';
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

            if (st === 'SUCCEEDED' || st === 'completed') {
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
