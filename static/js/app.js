/**
 * Alibaba Cloud AI Testing Platform
 * Frontend JS — API Key auth, SSE streaming, polling
 */

// ── State ─────────────────────────────────────────────────────
let apiKey = sessionStorage.getItem('dashscope_api_key') || '';
let models = { text: [], image: [], video: [] };
let refFiles = [];
let loadingTimerInterval = null;

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
            sessionStorage.removeItem('dashscope_api_key');
        }
    } catch (_) { /* show login */ }
}

// ── Auth ──────────────────────────────────────────────────────
async function handleLogin() {
    const key = document.getElementById('apiKeyInput').value.trim();
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
    populateSelectors();
}

function handleLogout() {
    apiKey = '';
    sessionStorage.removeItem('dashscope_api_key');
    location.reload();
}

function authHeader() {
    return { 'Authorization': `Bearer ${apiKey}`, 'Content-Type': 'application/json' };
}

// ── Selectors ─────────────────────────────────────────────────
function populateSelectors() {
    populateSelect('textModel', models.text);
    onImgTaskChange();
    onVidTaskChange();
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
    onVidModelChange();
}

function onVidModelChange() {
    const modelId = document.getElementById('videoModel').value;
    const modelInfo = models.video.find(m => m.id === modelId) || {};

    // 顯示/隱藏自動配音
    const audioRow = document.getElementById('vidAudioRow');
    audioRow.style.display = modelInfo.audio ? '' : 'none';
    if (!modelInfo.audio) document.getElementById('vidAudio').checked = false;

    // 調整時長範圍
    const dur = document.getElementById('videoDuration');
    const minD = modelInfo.min_dur || 3;
    const maxD = modelInfo.max_dur || 10;
    dur.min = minD;
    dur.max = maxD;
    if (parseInt(dur.value) < minD) { dur.value = minD; document.getElementById('durVal').textContent = minD; }
    if (parseInt(dur.value) > maxD) { dur.value = maxD; document.getElementById('durVal').textContent = maxD; }
}

// ── Tab ───────────────────────────────────────────────────────
function switchTab(tab) {
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-content').forEach(s => s.classList.remove('active'));
    document.querySelector(`[data-tab="${tab}"]`).classList.add('active');
    document.getElementById(`tab-${tab}`).classList.add('active');
}

// ── Text Generation ───────────────────────────────────────────
async function sendText() {
    const prompt = document.getElementById('textPrompt').value.trim();
    if (!prompt) { toast('請輸入提示詞', 'error'); return; }

    const model         = document.getElementById('textModel').value;
    const systemPrompt  = document.getElementById('textSystemPrompt').value;
    const temperature   = parseFloat(document.getElementById('textTemperature').value);
    const maxTokens     = parseInt(document.getElementById('textMaxTokens').value);
    const enableThinking= document.getElementById('textThinking').checked;
    const modelInfo     = models.text.find(m => m.id === model);

    const output = document.getElementById('textOutput');
    output.querySelector('.empty-state')?.remove();

    const uDiv = el('div', { className: 'chat-message user', textContent: prompt });
    output.appendChild(uDiv);

    const aDiv = el('div', { className: 'chat-message assistant streaming-cursor' });
    output.appendChild(aDiv);
    output.scrollTop = output.scrollHeight;

    const btn = document.getElementById('textSendBtn');
    btn.disabled = true;
    document.getElementById('textPrompt').value = '';

    try {
        const res = await fetch('/api/text/generate', {
            method: 'POST',
            headers: authHeader(),
            body: JSON.stringify({
                model, prompt, system_prompt: systemPrompt,
                temperature, max_tokens: maxTokens,
                enable_thinking: enableThinking && modelInfo?.thinking,
            }),
        });

        const reader  = res.body.getReader();
        const decoder = new TextDecoder();
        let full = '', buf = '';

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
                        aDiv.textContent = full;
                        output.scrollTop = output.scrollHeight;
                    } else if (d.error) {
                        aDiv.textContent = `⚠ 錯誤：${d.error}`;
                        aDiv.classList.remove('streaming-cursor');
                    } else if (d.done) {
                        aDiv.classList.remove('streaming-cursor');
                        const meta = el('div', { className: 'msg-meta' });
                        meta.innerHTML = `<span>${model}</span><span>${new Date().toLocaleTimeString()}</span>`;
                        aDiv.appendChild(meta);
                    }
                } catch (_) { /* skip */ }
            }
        }
        aDiv.classList.remove('streaming-cursor');
    } catch (e) {
        aDiv.textContent = `⚠ 錯誤：${e.message}`;
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
    const extend   = document.getElementById('imgPromptExtend').checked;
    const n        = parseInt(document.getElementById('imgN').value) || 1;

    if (!prompt) { toast('請輸入 Prompt', 'error'); return; }

    const btn = document.getElementById('imageSendBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> 生成中...';
    showLoading('圖片生成中，請稍候...');

    try {
        let res;
        if (taskType === 't2i') {
            res = await apiPost('/api/image/generate', { model, prompt, negative_prompt: negPrompt, size, n, prompt_extend: extend });
        } else {
            const fileInput = document.getElementById('imgFileInput');
            if (!fileInput.files.length) { toast('請先上傳圖片', 'error'); hideLoading(); btn.disabled = false; btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="M21 15l-5-5L5 21"/></svg> 生成'; return; }
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('size', size);
            fd.append('image', fileInput.files[0]);
            res = await apiPostForm('/api/image/edit', fd);
        }

        if (res.success && res.images?.length) {
            const gallery = document.getElementById('imageResults');
            gallery.querySelector('.empty-state')?.remove();
            res.images.forEach(img => {
                const src = img.local_path || img.url;
                const card = el('div', { className: 'img-card' });
                card.innerHTML = `
                    <img src="${src}" alt="Generated" loading="lazy">
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
    const audio     = document.getElementById('vidAudio').checked;

    if (!prompt) { toast('請輸入 Prompt', 'error'); return; }

    const btn = document.getElementById('videoSendBtn');
    btn.disabled = true;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/></svg> 提交中...';

    try {
        let res;
        if (taskType === 't2v') {
            res = await apiPost('/api/video/t2v', { model, prompt, negative_prompt: negPrompt, resolution, duration, audio });
        } else if (taskType === 'i2v') {
            const fi = document.getElementById('vidImgInput');
            if (!fi.files.length) { toast('請上傳首幀圖片', 'error'); btn.disabled = false; btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> 生成'; return; }
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('negative_prompt', negPrompt); fd.append('resolution', resolution); fd.append('duration', duration);
            fd.append('image', fi.files[0]);
            res = await apiPostForm('/api/video/i2v', fd);
        } else {
            if (!refFiles.length) { toast('請上傳參考文件', 'error'); btn.disabled = false; btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> 生成'; return; }
            const fd = new FormData();
            fd.append('model', model); fd.append('prompt', prompt);
            fd.append('resolution', resolution); fd.append('duration', duration);
            refFiles.forEach(f => fd.append('reference_files', f));
            res = await apiPostForm('/api/video/r2v', fd);
        }

        if (res.success && res.task_id) {
            addVideoTask(res.task_id, model, prompt, res.status);
            toast('任務已提交，輪詢中...', 'info');
        } else if (res.success && res.video_url) {
            addVideoResult(model, prompt, res.local_path || res.video_url);
            toast('影片生成完成！', 'success');
        } else {
            toast(res.error || '生成失敗', 'error');
            console.error('Video generation error:', res);
        }
    } catch (e) {
        toast(`錯誤：${e.message}`, 'error');
    }
    btn.disabled = false;
    btn.innerHTML = '<svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polygon points="23 7 16 12 23 17 23 7"/><rect x="1" y="5" width="15" height="14" rx="2"/></svg> 生成';
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

function addVideoResult(model, prompt, src) {
    const cont = document.getElementById('videoResults');
    cont.querySelector('.empty-state')?.remove();
    const card = el('div', { className: 'video-task-card' });
    card.innerHTML = `
        <div class="vtc-header"><span class="vtc-model">${model}</span><span class="vtc-status succeeded">SUCCEEDED</span></div>
        <div class="vtc-prompt">${prompt.substring(0, 120)}</div>
        <video class="video-player" controls src="${src}"></video>
        <div style="margin-top:8px"><a href="${src}" download class="img-dl">下載影片</a></div>`;
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
                        <div style="margin-top:8px"><a href="${data.local_path}" download class="img-dl">下載影片</a></div>`;
                }
                toast('影片生成完成！', 'success');
            } else if (st === 'FAILED') {
                if (stEl) { stEl.textContent = 'FAILED'; stEl.className = 'vtc-status failed'; }
                if (pbEl) pbEl.style.width = '100%'; pbEl && (pbEl.style.background = 'var(--red)');
                if (rvEl) rvEl.innerHTML = `<p style="font-size:0.82rem;color:var(--red)">錯誤：${data.error_message || 'Unknown'}</p>`;
                toast('影片生成失敗', 'error');
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

// ── API helpers ───────────────────────────────────────────────
async function apiPost(url, body) {
    const r = await fetch(url, { method: 'POST', headers: authHeader(), body: JSON.stringify(body) });
    if (r.status === 401) { handleLogout(); throw new Error('Unauthorized'); }
    return r.json();
}
async function apiPostForm(url, fd) {
    const r = await fetch(url, { method: 'POST', headers: { 'Authorization': `Bearer ${apiKey}` }, body: fd });
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

// ── Utils ─────────────────────────────────────────────────────
function el(tag, props = {}) {
    return Object.assign(document.createElement(tag), props);
}
