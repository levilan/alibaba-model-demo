// AI Canvas — 節點式畫布，讓使用者用拖拉連線的方式組合平台上的圖片/影片/圖像編輯模型
//
// 節點的表單控制項（model/prompt/按鈕/預覽）全部用真正的 HTML DOM 元素蓋在
// LiteGraph 畫布節點上方（隨畫布縮放/平移同步移動），而不是用 LiteGraph 內建
// 畫布繪製的 widget —— 原生 widget 無法呈現可下載的大圖預覽、也不好用。
// 節點本身（標題列、輸入/輸出插槽、連線）仍交給 LiteGraph 處理。
(function () {
    const apiKey = sessionStorage.getItem('nenai_api_key') || '';
    if (!apiKey) {
        document.getElementById('canvasLoginGate').style.display = 'flex';
        return;
    }
    document.getElementById('canvasApp').style.display = '';

    let MODELS = { text: [], image: [], video: [], muleai: [] };
    let graph, lgCanvas;
    const domLayer = document.getElementById('canvasApp');

    // ── API helpers ─────────────────────────────────────────────
    async function apiFetch(url, opts = {}) {
        const headers = Object.assign({ Authorization: `Bearer ${apiKey}` }, opts.headers || {});
        if (opts.body && typeof opts.body === 'string' && !headers['Content-Type']) {
            headers['Content-Type'] = 'application/json';
        }
        const res = await fetch(url, Object.assign({}, opts, { headers }));
        if (res.status === 401) {
            sessionStorage.removeItem('nenai_api_key');
            location.href = '/';
            throw new Error('Unauthorized');
        }
        return res;
    }

    async function fetchAsBlob(url) {
        if (!url) throw new Error('缺少來源檔案網址');
        const isRemote = /^https?:\/\//i.test(url) && !url.startsWith(location.origin);
        const target = isRemote ? `/api/proxy/fetch?url=${encodeURIComponent(url)}` : url;
        const res = await apiFetch(target);
        if (!res.ok) throw new Error('無法取得來源檔案');
        return await res.blob();
    }

    function blobToDataUri(blob) {
        return new Promise((resolve, reject) => {
            const reader = new FileReader();
            reader.onload = () => resolve(reader.result);
            reader.onerror = reject;
            reader.readAsDataURL(blob);
        });
    }

    async function pollVideoTask(taskId, { intervalMs = 3000, maxTries = 200 } = {}) {
        for (let i = 0; i < maxTries; i++) {
            const res = await apiFetch(`/api/video/status/${taskId}`);
            const data = await res.json();
            const st = (data.status || '').toUpperCase();
            if (['SUCCEEDED', 'COMPLETED', 'SUCCESS', 'SUCCEED', 'DONE', 'FINISHED'].includes(st)) return data;
            if (['FAILED', 'FAIL', 'FAILURE', 'ERROR'].includes(st)) throw new Error(data.error_message || '生成失敗');
            await new Promise(r => setTimeout(r, intervalMs));
        }
        throw new Error('等待逾時');
    }

    function getModelsFor(category, type) {
        const list = MODELS[category] || [];
        return type ? list.filter(m => m.type === type) : list;
    }

    function sizesForModel(category, modelId) {
        const m = (MODELS[category] || []).find(x => x.id === modelId);
        return (m && m.sizes) || ['1024*1024', '1280*720', '720*1280'];
    }

    function showToast(msg) {
        let el = document.getElementById('cvToast');
        if (!el) {
            el = document.createElement('div');
            el.id = 'cvToast';
            el.className = 'canvas-toast';
            domLayer.appendChild(el);
        }
        el.textContent = msg;
        el.style.display = 'block';
        clearTimeout(el._t);
        el._t = setTimeout(() => { el.style.display = 'none'; }, 5000);
    }

    // ── DOM panel helpers（節點上蓋的真實 HTML 表單） ────────────────
    function el(tag, cls, html) {
        const e = document.createElement(tag);
        if (cls) e.className = cls;
        if (html != null) e.innerHTML = html;
        return e;
    }

    function buildSelect(values, current, onChange) {
        const sel = el('select');
        sel.innerHTML = values.map(v => `<option value="${v}"${v === current ? ' selected' : ''}>${v}</option>`).join('');
        sel.addEventListener('mousedown', (e) => e.stopPropagation());
        sel.addEventListener('change', () => onChange(sel.value));
        return sel;
    }

    function attachDomPanel(node, panel) {
        panel.className = 'cv-node-panel';
        panel.addEventListener('mousedown', (e) => e.stopPropagation());
        panel.addEventListener('wheel', (e) => e.stopPropagation());
        domLayer.appendChild(panel);
        node._domPanel = panel;
    }

    // LiteGraph 的座標轉換公式（來自 DragAndScale.convertOffsetToCanvas）：
    // canvasPixel = (graphPos + ds.offset) * ds.scale —— ctx 是先 scale 再 translate。
    // node.pos 是節點「主體」（title 列下方）的左上角，title 往上額外佔用 NODE_TITLE_HEIGHT。
    function positionAllPanels() {
        if (!graph || !lgCanvas) return;
        const canvasEl = document.getElementById('litegraphCanvas');
        const rect = canvasEl.getBoundingClientRect();
        const scale = lgCanvas.ds.scale || 1;
        const offset = lgCanvas.ds.offset || [0, 0];
        graph._nodes.forEach(node => {
            const panel = node._domPanel;
            if (!panel) return;
            const collapsed = node.flags && node.flags.collapsed;
            if (collapsed) { panel.style.display = 'none'; return; }
            panel.style.display = '';
            const screenX = rect.left + (node.pos[0] + offset[0]) * scale;
            const screenY = rect.top + (node.pos[1] + offset[1]) * scale;
            panel.style.left = screenX + 'px';
            panel.style.top = screenY + 'px';
            panel.style.width = node.size[0] + 'px';
            panel.style.transform = `scale(${scale})`;
        });
        requestAnimationFrame(positionAllPanels);
    }

    function buildPreview(node, kind) {
        const box = el('div', 'cv-preview');
        box.innerHTML = '<span class="cv-empty">尚未生成</span>';
        node._previewBox = box;
        return box;
    }

    function setPreviewEmpty(node, text) {
        node._previewBox.innerHTML = `<span class="cv-empty">${text}</span>`;
    }

    function setPreviewImage(node, url) {
        node._previewBox.innerHTML = '';
        const img = el('img');
        img.src = url;
        node._previewBox.appendChild(img);
        const dl = el('a', 'cv-dl-btn', '⬇ 下載');
        dl.href = url; dl.download = 'image.png'; dl.target = '_blank';
        dl.addEventListener('mousedown', (e) => e.stopPropagation());
        node._previewBox.appendChild(dl);
    }

    function setPreviewVideo(node, url) {
        node._previewBox.innerHTML = '';
        const video = el('video');
        video.src = url; video.controls = true;
        node._previewBox.appendChild(video);
        const dl = el('a', 'cv-dl-btn', '⬇ 下載');
        dl.href = url; dl.download = 'video.mp4'; dl.target = '_blank';
        dl.addEventListener('mousedown', (e) => e.stopPropagation());
        node._previewBox.appendChild(dl);
    }

    // ── Node: Text（手動輸入 prompt；若連接「圖片」輸入，可改用圖片分析結果） ──
    function TextPromptNode() {
        this.addInput('image', 'image');
        this.addOutput('text', 'string');
        this.properties = { text: '', status: '' };
        this.size = [300, 230];
        this.color = '#3d3320'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <label>Prompt</label>
            <textarea placeholder="輸入文字…"></textarea>
            <button class="cv-generate cv-analyze-btn" style="display:none;margin-top:8px">🔍 分析已連接的圖片</button>
            <div class="cv-status"></div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.text = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        this.analyzeBtn = panel.querySelector('.cv-analyze-btn');
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());
    }
    TextPromptNode.title = '文字 Text';
    TextPromptNode.prototype.onExecute = function () {
        this.setOutputData(0, this.properties.text);
    };
    TextPromptNode.prototype.onConnectionsChange = function (type) {
        if (type !== LiteGraph.INPUT || !this.analyzeBtn) return;
        const hasImage = !!this.getInputNode(0);
        this.analyzeBtn.style.display = hasImage ? '' : 'none';
    };
    TextPromptNode.prototype.analyzeImage = async function () {
        const imgUrl = this.getInputData(0);
        if (!imgUrl) { showToast('請先連接一張圖片'); return; }
        this.statusEl.textContent = '分析中…';
        try {
            const blob = await fetchAsBlob(imgUrl);
            const dataUri = await blobToDataUri(blob);
            const res = await apiFetch('/api/text/analyze_image', {
                method: 'POST',
                body: JSON.stringify({ prompt: this.properties.text || '請用一句話描述這張圖片的內容。', image_data_uri: dataUri }),
            });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error((data.error && (data.error.message || data.error)) || '分析失敗');
            this.properties.text = data.content || '';
            this.textarea.value = this.properties.text;
            this.statusEl.textContent = '完成';
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            showToast('圖片分析失敗：' + e.message);
        }
    };
    TextPromptNode.prototype.onRemoved = function () {
        if (this._domPanel) { this._domPanel.remove(); this._domPanel = null; }
    };

    // ── Node: Image（文生圖，t2i 模型） ─────────────────────────
    function ImageGenNode() {
        this.addInput('prompt', 'string');
        this.addOutput('image', 'image');
        const models = getModelsFor('image', 't2i');
        this.properties = { model: (models[0] && models[0].id) || '', prompt: '', size: '1024*1024', status: '' };
        this.imageUrl = null;
        this.size = [320, 470];
        this.color = '#1f3a2e'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <label>模型</label>
            <div class="cv-select-slot"></div>
            <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
            <textarea placeholder="輸入文字…"></textarea>
            <label>尺寸</label>
            <div class="cv-size-slot"></div>
            <button class="cv-generate">▶ 生成圖片</button>
            <div class="cv-status"></div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => {
            this.properties.model = v;
            const sizes = sizesForModel('image', v);
            this._rebuildSizeSelect(sizes);
        });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);
        this._rebuildSizeSelect(sizesForModel('image', this.properties.model));

        panel.appendChild(buildPreview(this));
    }
    ImageGenNode.title = '圖片 Image';
    ImageGenNode.prototype._rebuildSizeSelect = function (sizes) {
        const slot = this._domPanel.querySelector('.cv-size-slot');
        if (!sizes.includes(this.properties.size)) this.properties.size = sizes[0];
        slot.innerHTML = '';
        this.sizeSelect = buildSelect(sizes, this.properties.size, (v) => { this.properties.size = v; });
        slot.appendChild(this.sizeSelect);
    };
    ImageGenNode.prototype.onExecute = function () {
        this.setOutputData(0, this.imageUrl);
    };
    ImageGenNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        this.statusEl.textContent = '生成中…';
        setPreviewEmpty(this, '生成中…');
        try {
            const res = await apiFetch('/api/image/generate', {
                method: 'POST',
                body: JSON.stringify({ model: this.properties.model, prompt, size: this.properties.size, n: 1 }),
            });
            const data = await res.json();
            if (!res.ok || !data.images || !data.images.length) throw new Error((data.error && (data.error.message || data.error)) || '生成失敗');
            this.imageUrl = data.images[0].local_path || data.images[0].url;
            this.statusEl.textContent = '完成';
            setPreviewImage(this, this.imageUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('圖片生成失敗：' + e.message);
        }
    };
    ImageGenNode.prototype.onRemoved = function () {
        if (this._domPanel) { this._domPanel.remove(); this._domPanel = null; }
    };

    // ── Node: Video（t2v / i2v，依 first_frame 是否連線自動切換） ──
    function VideoGenNode() {
        this.addInput('prompt', 'string');
        this.addInput('first_frame', 'image');
        this.addInput('last_frame', 'image');
        this.addOutput('video', 'video');
        const models = getModelsFor('video', 't2v');
        this.properties = { model: (models[0] && models[0].id) || '', prompt: '', resolution: '720P', duration: 5, status: '' };
        this.videoUrl = null;
        this.size = [320, 500];
        this.color = '#1f2f3a'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <label>模型 <span class="cv-hint cv-mode-hint">（文生影片）</span></label>
            <div class="cv-select-slot"></div>
            <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
            <textarea placeholder="輸入文字…"></textarea>
            <label>解析度</label>
            <div class="cv-res-slot"></div>
            <label>時長（秒）<span class="cv-dur-val">5</span></label>
            <input type="range" class="cv-dur-slider" min="2" max="15" step="1" value="5">
            <button class="cv-generate">▶ 生成影片</button>
            <div class="cv-status"></div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        this.modeHintEl = panel.querySelector('.cv-mode-hint');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => { this.properties.model = v; });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        this.resSelect = buildSelect(['480P', '720P', '1080P'], this.properties.resolution, (v) => { this.properties.resolution = v; });
        panel.querySelector('.cv-res-slot').appendChild(this.resSelect);

        this.durSlider = panel.querySelector('.cv-dur-slider');
        this.durValEl = panel.querySelector('.cv-dur-val');
        this.durSlider.addEventListener('input', () => {
            this.properties.duration = parseInt(this.durSlider.value);
            this.durValEl.textContent = this.durSlider.value;
        });

        panel.appendChild(buildPreview(this));
    }
    VideoGenNode.title = '影片 Video';
    VideoGenNode.prototype.onExecute = function () {
        this.setOutputData(0, this.videoUrl);
    };
    VideoGenNode.prototype.onConnectionsChange = function (type) {
        if (type !== LiteGraph.INPUT || !this.modelSelect) return;
        const hasFirstFrame = !!this.getInputNode(1);
        const list = getModelsFor('video', hasFirstFrame ? 'i2v' : 't2v');
        const values = list.map(m => m.id);
        if (!values.includes(this.properties.model)) this.properties.model = values[0] || '';
        this.modelSelect.innerHTML = values.map(v => `<option value="${v}"${v === this.properties.model ? ' selected' : ''}>${v}</option>`).join('');
        this.modeHintEl.textContent = hasFirstFrame ? '（圖生影片 / 參考圖）' : '（文生影片）';
    };
    VideoGenNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        const firstFrameUrl = this.getInputData(1);
        const lastFrameUrl = this.getInputData(2);
        this.statusEl.textContent = '送出中…';
        setPreviewEmpty(this, '送出中…');
        try {
            const fd = new FormData();
            fd.append('model', this.properties.model);
            fd.append('prompt', prompt);
            fd.append('resolution', this.properties.resolution);
            fd.append('duration', String(this.properties.duration));
            let endpoint = '/api/video/t2v';
            if (firstFrameUrl) {
                endpoint = '/api/video/i2v';
                fd.append('i2v_mode', lastFrameUrl ? 'first_last_frame' : 'first_frame');
                fd.append('first_frame', await fetchAsBlob(firstFrameUrl), 'first_frame.png');
                if (lastFrameUrl) fd.append('last_frame', await fetchAsBlob(lastFrameUrl), 'last_frame.png');
            }
            const res = await apiFetch(endpoint, { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error((data.error && (data.error.message || data.error)) || '任務建立失敗');
            this.statusEl.textContent = '生成中…';
            setPreviewEmpty(this, '生成中…（可能需要 1～數分鐘）');
            const result = await pollVideoTask(data.task_id);
            this.videoUrl = result.local_path || result.video_url;
            this.statusEl.textContent = '完成';
            setPreviewVideo(this, this.videoUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('影片生成失敗：' + e.message);
        }
    };
    VideoGenNode.prototype.onRemoved = function () {
        if (this._domPanel) { this._domPanel.remove(); this._domPanel = null; }
    };

    // ── Node: Editing（i2i 圖像編輯，需連接一張輸入圖片） ───────────
    function ImageEditNode() {
        this.addInput('image', 'image');
        this.addInput('prompt', 'string');
        this.addOutput('image', 'image');
        const models = getModelsFor('image', 'i2i');
        this.properties = { model: (models[0] && models[0].id) || '', prompt: '', size: '1024*1024', status: '' };
        this.imageUrl = null;
        this.size = [320, 470];
        this.color = '#3a2340'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <label>模型</label>
            <div class="cv-select-slot"></div>
            <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
            <textarea placeholder="輸入編輯指示…"></textarea>
            <button class="cv-generate">▶ 編輯圖片</button>
            <div class="cv-status"></div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => { this.properties.model = v; });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        panel.appendChild(buildPreview(this));
    }
    ImageEditNode.title = '圖像編輯 Editing';
    ImageEditNode.prototype.onExecute = function () {
        this.setOutputData(0, this.imageUrl);
    };
    ImageEditNode.prototype.generate = async function () {
        const srcImage = this.getInputData(0);
        if (!srcImage) { showToast('請先連接一張來源圖片'); return; }
        const promptIn = this.getInputData(1);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        this.statusEl.textContent = '生成中…';
        setPreviewEmpty(this, '生成中…');
        try {
            const fd = new FormData();
            fd.append('model', this.properties.model);
            fd.append('prompt', prompt);
            fd.append('size', this.properties.size);
            fd.append('n', '1');
            fd.append('image_1', await fetchAsBlob(srcImage), 'source.png');
            const res = await apiFetch('/api/image/edit', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.images || !data.images.length) throw new Error((data.error && (data.error.message || data.error)) || '生成失敗');
            this.imageUrl = data.images[0].local_path || data.images[0].url;
            this.statusEl.textContent = '完成';
            setPreviewImage(this, this.imageUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('圖像編輯失敗：' + e.message);
        }
    };
    ImageEditNode.prototype.onRemoved = function () {
        if (this._domPanel) { this._domPanel.remove(); this._domPanel = null; }
    };

    // ── Node: Audio（尚未有可用的 TTS 後端，先提供停用佔位節點） ───
    function AudioPlaceholderNode() {
        this.addInput('text', 'string');
        this.addOutput('audio', 'audio');
        this.size = [300, 110];
        this.color = '#333'; this.bgcolor = '#2a2a2a';
        const panel = el('div');
        panel.innerHTML = `<div class="cv-status" style="margin-top:4px">平台目前沒有可用的 TTS 後端，此節點尚未支援</div>`;
        attachDomPanel(this, panel);
    }
    AudioPlaceholderNode.title = '語音 Audio（尚未支援）';
    AudioPlaceholderNode.prototype.onRemoved = function () {
        if (this._domPanel) { this._domPanel.remove(); this._domPanel = null; }
    };

    function registerNodeTypes() {
        LiteGraph.registerNodeType('nenai/text', TextPromptNode);
        LiteGraph.registerNodeType('nenai/image', ImageGenNode);
        LiteGraph.registerNodeType('nenai/video', VideoGenNode);
        LiteGraph.registerNodeType('nenai/edit', ImageEditNode);
        LiteGraph.registerNodeType('nenai/audio', AudioPlaceholderNode);
    }

    // ── Canvas/graph bootstrap ───────────────────────────────────
    function resizeCanvasEl() {
        const canvasEl = document.getElementById('litegraphCanvas');
        canvasEl.width = window.innerWidth;
        canvasEl.height = window.innerHeight - 52;
        if (lgCanvas) lgCanvas.resize();
    }

    function updateZoomLabel() {
        document.getElementById('zoomLabel').textContent = Math.round((lgCanvas.ds.scale || 1) * 100) + '%';
    }

    function fitView() {
        const nodes = graph._nodes;
        if (!nodes.length) return;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        nodes.forEach(n => {
            minX = Math.min(minX, n.pos[0]);
            minY = Math.min(minY, n.pos[1]);
            maxX = Math.max(maxX, n.pos[0] + n.size[0]);
            maxY = Math.max(maxY, n.pos[1] + n.size[1]);
        });
        const w = Math.max(maxX - minX, 1), h = Math.max(maxY - minY, 1);
        const cw = lgCanvas.canvas.width, ch = lgCanvas.canvas.height;
        const scale = Math.min(cw / (w + 160), ch / (h + 160), 1.2);
        lgCanvas.ds.scale = scale;
        lgCanvas.ds.offset[0] = (cw - w * scale) / 2 / scale - minX;
        lgCanvas.ds.offset[1] = (ch - h * scale) / 2 / scale - minY;
        lgCanvas.setDirty(true, true);
        updateZoomLabel();
    }

    const NODE_MENU_TYPES = { text: 'nenai/text', image: 'nenai/image', video: 'nenai/video', edit: 'nenai/edit', audio: 'nenai/audio' };

    function wireToolbar() {
        const addMenu = document.getElementById('addNodeMenu');
        document.getElementById('btnAddNode').addEventListener('click', (e) => {
            e.stopPropagation();
            addMenu.style.display = addMenu.style.display === 'none' ? 'block' : 'none';
        });
        document.addEventListener('click', (e) => {
            if (!addMenu.contains(e.target) && e.target.id !== 'btnAddNode') addMenu.style.display = 'none';
        });
        addMenu.querySelectorAll('button[data-type]').forEach(btn => {
            if (btn.disabled) return;
            btn.addEventListener('click', () => {
                const type = NODE_MENU_TYPES[btn.dataset.type];
                if (!type) return;
                const node = LiteGraph.createNode(type);
                const canvasEl = document.getElementById('litegraphCanvas');
                const cx = (canvasEl.width / 2) / lgCanvas.ds.scale - lgCanvas.ds.offset[0];
                const cy = (canvasEl.height / 2) / lgCanvas.ds.scale - lgCanvas.ds.offset[1];
                node.pos = [cx - node.size[0] / 2, cy - node.size[1] / 2];
                graph.add(node);
                addMenu.style.display = 'none';
            });
        });

        document.getElementById('btnFitView').addEventListener('click', fitView);
        document.getElementById('zoomIn').addEventListener('click', () => {
            lgCanvas.ds.scale = Math.min((lgCanvas.ds.scale || 1) * 1.2, 4);
            lgCanvas.setDirty(true, true);
            updateZoomLabel();
        });
        document.getElementById('zoomOut').addEventListener('click', () => {
            lgCanvas.ds.scale = Math.max((lgCanvas.ds.scale || 1) / 1.2, 0.1);
            lgCanvas.setDirty(true, true);
            updateZoomLabel();
        });
        document.getElementById('zoomReset').addEventListener('click', () => {
            lgCanvas.ds.scale = 1;
            lgCanvas.ds.offset = [0, 0];
            lgCanvas.setDirty(true, true);
            updateZoomLabel();
        });
    }

    async function loadModels() {
        const res = await apiFetch('/api/models');
        if (!res.ok) throw new Error('無法載入模型清單');
        MODELS = await res.json();
    }

    async function init() {
        try {
            await loadModels();
        } catch (e) {
            showToast('模型清單載入失敗：' + e.message);
        }
        registerNodeTypes();

        graph = new LGraph();
        lgCanvas = new LGraphCanvas('#litegraphCanvas', graph);
        lgCanvas.background_image = null;
        lgCanvas.render_canvas_border = false;

        resizeCanvasEl();
        window.addEventListener('resize', resizeCanvasEl);

        graph.start();
        wireToolbar();
        updateZoomLabel();
        requestAnimationFrame(positionAllPanels);
        setInterval(updateZoomLabel, 500);
    }

    init();
})();
