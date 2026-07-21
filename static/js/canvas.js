// AI Canvas — 節點式畫布，讓使用者用拖拉連線的方式組合平台上的圖片/影片/圖像編輯模型
(function () {
    const apiKey = sessionStorage.getItem('nenai_api_key') || '';
    if (!apiKey) {
        document.getElementById('canvasLoginGate').style.display = 'flex';
        return;
    }
    document.getElementById('canvasApp').style.display = '';

    let MODELS = { text: [], image: [], video: [], muleai: [] };
    let graph, lgCanvas;

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
            document.getElementById('canvasApp').appendChild(el);
        }
        el.textContent = msg;
        el.style.display = 'block';
        clearTimeout(el._t);
        el._t = setTimeout(() => { el.style.display = 'none'; }, 5000);
    }

    function ensureImageLoaded(node, url) {
        if (!url) { node._previewImg = null; node._previewUrl = null; return; }
        if (node._previewUrl === url && node._previewImg) return;
        node._previewUrl = url;
        const img = new Image();
        img.onload = () => { node._previewImg = img; node.setDirtyCanvas(true, true); };
        img.onerror = () => { node._previewImg = null; };
        img.src = url;
    }

    function drawPreviewBox(node, ctx, statusText) {
        const ph = 130, py = node.size[1] - ph - 6, px = 6, pw = node.size[0] - 12;
        ctx.fillStyle = '#111';
        ctx.fillRect(px, py, pw, ph);
        if (node._previewImg) {
            const img = node._previewImg;
            const scale = Math.min(pw / img.width, ph / img.height);
            const dw = img.width * scale, dh = img.height * scale;
            ctx.drawImage(img, px + (pw - dw) / 2, py + (ph - dh) / 2, dw, dh);
        } else {
            ctx.fillStyle = '#666';
            ctx.font = '12px sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(statusText || '尚未生成', px + pw / 2, py + ph / 2);
            ctx.textAlign = 'left';
        }
    }

    // ── Node: Text（純輸入來源，供其他節點的 prompt 使用） ─────────
    function TextPromptNode() {
        this.addOutput('text', 'string');
        this.properties = { text: '' };
        this.addWidget('text', 'prompt', '', (v) => { this.properties.text = v; }, { multiline: true });
        this.size = [260, 130];
    }
    TextPromptNode.title = '文字 Text';
    TextPromptNode.prototype.onExecute = function () {
        this.setOutputData(0, this.properties.text);
    };

    // ── Node: Image（文生圖，t2i 模型） ─────────────────────────
    function ImageGenNode() {
        this.addInput('prompt', 'string');
        this.addOutput('image', 'image');
        const models = getModelsFor('image', 't2i');
        this.properties = {
            model: (models[0] && models[0].id) || '',
            prompt: '', size: '1024*1024', status: '尚未生成',
        };
        this.imageUrl = null;
        this.modelWidget = this.addWidget('combo', 'model', this.properties.model, (v) => {
            this.properties.model = v;
            const sizes = sizesForModel('image', v);
            this.sizeWidget.options.values = sizes;
            if (!sizes.includes(this.properties.size)) {
                this.properties.size = sizes[0];
                this.sizeWidget.value = sizes[0];
            }
        }, { values: models.map(m => m.id) });
        this.addWidget('text', 'prompt', '', (v) => { this.properties.prompt = v; }, { multiline: true });
        this.sizeWidget = this.addWidget('combo', 'size', this.properties.size, (v) => { this.properties.size = v; },
            { values: sizesForModel('image', this.properties.model) });
        this.addWidget('button', '▶ 生成圖片', null, () => this.generate());
        this.size = [280, 320];
    }
    ImageGenNode.title = '圖片 Image';
    ImageGenNode.prototype.onExecute = function () {
        this.setOutputData(0, this.imageUrl);
    };
    ImageGenNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        this.properties.status = '生成中…';
        this.setDirtyCanvas(true);
        try {
            const res = await apiFetch('/api/image/generate', {
                method: 'POST',
                body: JSON.stringify({ model: this.properties.model, prompt, size: this.properties.size, n: 1 }),
            });
            const data = await res.json();
            if (!res.ok || !data.images || !data.images.length) throw new Error((data.error && (data.error.message || data.error)) || '生成失敗');
            this.imageUrl = data.images[0].local_path || data.images[0].url;
            this.properties.status = '完成';
            ensureImageLoaded(this, this.imageUrl);
        } catch (e) {
            this.properties.status = '錯誤：' + e.message;
            showToast('圖片生成失敗：' + e.message);
        }
        this.setDirtyCanvas(true);
    };
    ImageGenNode.prototype.onDrawForeground = function (ctx) {
        if (this.flags.collapsed) return;
        drawPreviewBox(this, ctx, this.properties.status);
    };
    ImageGenNode.prototype.onDblClick = function () {
        if (this.imageUrl) window.open(this.imageUrl, '_blank');
    };

    // ── Node: Video（t2v / i2v，依 first_frame 是否連線自動切換） ──
    function VideoGenNode() {
        this.addInput('prompt', 'string');
        this.addInput('first_frame', 'image');
        this.addInput('last_frame', 'image');
        this.addOutput('video', 'video');
        const models = getModelsFor('video', 't2v');
        this.properties = {
            model: (models[0] && models[0].id) || '',
            prompt: '', resolution: '720P', duration: 5, status: '尚未生成',
        };
        this.videoUrl = null;
        this.modelWidget = this.addWidget('combo', 'model', this.properties.model,
            (v) => { this.properties.model = v; }, { values: models.map(m => m.id) });
        this.addWidget('text', 'prompt', '', (v) => { this.properties.prompt = v; }, { multiline: true });
        this.addWidget('combo', 'resolution', this.properties.resolution,
            (v) => { this.properties.resolution = v; }, { values: ['480P', '720P', '1080P'] });
        this.addWidget('slider', 'duration', this.properties.duration,
            (v) => { this.properties.duration = Math.round(v); }, { min: 2, max: 15, step: 1 });
        this.addWidget('button', '▶ 生成影片', null, () => this.generate());
        this.size = [280, 340];
    }
    VideoGenNode.title = '影片 Video';
    VideoGenNode.prototype.onExecute = function () {
        this.setOutputData(0, this.videoUrl);
    };
    VideoGenNode.prototype.onConnectionsChange = function (type) {
        if (type !== LiteGraph.INPUT || !this.modelWidget) return;
        const hasFirstFrame = !!this.getInputNode(1);
        const list = getModelsFor('video', hasFirstFrame ? 'i2v' : 't2v');
        this.modelWidget.options.values = list.map(m => m.id);
        if (!list.find(m => m.id === this.properties.model)) {
            this.properties.model = (list[0] && list[0].id) || '';
            this.modelWidget.value = this.properties.model;
        }
    };
    VideoGenNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        const firstFrameUrl = this.getInputData(1);
        const lastFrameUrl = this.getInputData(2);
        this.properties.status = '送出中…';
        this.setDirtyCanvas(true);
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
            this.properties.status = '生成中…';
            this.setDirtyCanvas(true);
            const result = await pollVideoTask(data.task_id);
            this.videoUrl = result.local_path || result.video_url;
            this.properties.status = '完成';
        } catch (e) {
            this.properties.status = '錯誤：' + e.message;
            showToast('影片生成失敗：' + e.message);
        }
        this.setDirtyCanvas(true);
    };
    VideoGenNode.prototype.onDrawForeground = function (ctx) {
        if (this.flags.collapsed) return;
        const ph = 100, py = this.size[1] - ph - 6, px = 6, pw = this.size[0] - 12;
        ctx.fillStyle = '#111';
        ctx.fillRect(px, py, pw, ph);
        ctx.fillStyle = this.videoUrl ? '#4ade80' : '#666';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'center';
        ctx.fillText(this.videoUrl ? '🎬 影片已生成（雙擊播放）' : (this.properties.status || '尚未生成'), px + pw / 2, py + ph / 2);
        ctx.textAlign = 'left';
    };
    VideoGenNode.prototype.onDblClick = function () {
        if (this.videoUrl) window.open(this.videoUrl, '_blank');
    };

    // ── Node: Editing（i2i 圖像編輯，需連接一張輸入圖片） ───────────
    function ImageEditNode() {
        this.addInput('image', 'image');
        this.addInput('prompt', 'string');
        this.addOutput('image', 'image');
        const models = getModelsFor('image', 'i2i');
        this.properties = {
            model: (models[0] && models[0].id) || '',
            prompt: '', size: '1024*1024', status: '尚未生成',
        };
        this.imageUrl = null;
        this.modelWidget = this.addWidget('combo', 'model', this.properties.model,
            (v) => { this.properties.model = v; }, { values: models.map(m => m.id) });
        this.addWidget('text', 'prompt', '', (v) => { this.properties.prompt = v; }, { multiline: true });
        this.addWidget('button', '▶ 編輯圖片', null, () => this.generate());
        this.size = [280, 320];
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
        this.properties.status = '生成中…';
        this.setDirtyCanvas(true);
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
            this.properties.status = '完成';
            ensureImageLoaded(this, this.imageUrl);
        } catch (e) {
            this.properties.status = '錯誤：' + e.message;
            showToast('圖像編輯失敗：' + e.message);
        }
        this.setDirtyCanvas(true);
    };
    ImageEditNode.prototype.onDrawForeground = function (ctx) {
        if (this.flags.collapsed) return;
        drawPreviewBox(this, ctx, this.properties.status);
    };
    ImageEditNode.prototype.onDblClick = function () {
        if (this.imageUrl) window.open(this.imageUrl, '_blank');
    };

    // ── Node: Audio（尚未有可用的 TTS 後端，先提供停用佔位節點） ───
    function AudioPlaceholderNode() {
        this.addInput('text', 'string');
        this.addOutput('audio', 'audio');
        this.addWidget('text', 'text', '', () => {}, { multiline: true });
        this.size = [260, 130];
    }
    AudioPlaceholderNode.title = '語音 Audio（尚未支援）';
    AudioPlaceholderNode.prototype.onDrawForeground = function (ctx) {
        if (this.flags.collapsed) return;
        ctx.fillStyle = '#666';
        ctx.font = '11px sans-serif';
        ctx.fillText('平台目前沒有可用的 TTS 後端', 8, this.size[1] - 14);
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
        const scale = Math.min(cw / (w + 160), ch / (h + 160), 1.5);
        lgCanvas.ds.scale = scale;
        lgCanvas.ds.offset[0] = -minX * scale + (cw - w * scale) / 2;
        lgCanvas.ds.offset[1] = -minY * scale + (ch - h * scale) / 2;
        lgCanvas.setDirty(true, true);
        updateZoomLabel();
    }

    const NODE_MENU_TYPES = { text: 'nenai/text', image: 'nenai/image', video: 'nenai/video', edit: 'nenai/edit' };

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
                const cx = (canvasEl.width / 2 - lgCanvas.ds.offset[0]) / lgCanvas.ds.scale;
                const cy = (canvasEl.height / 2 - lgCanvas.ds.offset[1]) / lgCanvas.ds.scale;
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

        // 每隔一段時間同步縮放百分比顯示（使用者用滑鼠滾輪縮放時也會更新）
        setInterval(updateZoomLabel, 500);
    }

    init();
})();
