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

    // 這個版本的 LiteGraph 選取狀態存在 lgCanvas.selected_nodes 這個字典裡（用
    // node.id 當 key），不是 node.selected 這個布林屬性——實測過才發現的，直接
    // 讀 node.selected 永遠是 undefined。
    function isNodeSelected(node) {
        return !!(lgCanvas.selected_nodes && lgCanvas.selected_nodes[node.id]);
    }

    function attachDomPanel(node, panel) {
        panel.className = 'cv-node-panel';
        panel.addEventListener('mousedown', (e) => {
            e.stopPropagation();
            // DOM 面板蓋在畫布上方，滑鼠事件傳不到 LiteGraph 原生的點擊選取邏輯，
            // 所以點擊節點本體時要自己呼叫官方選取 API（這樣空白處點擊取消選取、
            // Delete 鍵刪除節點等原生行為才會維持正常運作）
            lgCanvas.selectNode(node, false);
            lgCanvas.setDirty(true, true);
        });
        panel.addEventListener('wheel', (e) => e.stopPropagation());
        domLayer.appendChild(panel);
        node._domPanel = panel;
    }

    // ── 節點「外框裝飾」：右上角刪除鈕、每個輸出插槽旁的「+」快速新增關聯節點鈕 ──
    // （原生 LiteGraph 的連線插槽很小、不好抓，這裡提供更明顯的點擊入口）
    const NODE_TYPE_LABELS = {
        text: { type: 'nenai/text', label: '📄　文字 Text' },
        camera_angle: { type: 'nenai/camera_angle', label: '📐　相機角度 Camera Angle' },
        load_image: { type: 'nenai/load_image', label: '📁　上傳圖片 Load Image' },
        image: { type: 'nenai/image', label: '🖼️　圖片 Image' },
        video: { type: 'nenai/video', label: '🎬　影片 Video' },
        edit: { type: 'nenai/edit', label: '✂️　圖像編輯 Editing' },
        audio: { type: 'nenai/audio', label: '🎵　語音 Audio（尚未支援）', disabled: true },
        muleai: { type: 'nenai/muleai', label: '🌶️　MuleAI Spicy' },
    };

    function connectToFirstCompatibleInput(sourceNode, outSlot, targetNode) {
        const outType = sourceNode.outputs[outSlot].type;
        const inputs = targetNode.inputs || [];
        for (let i = 0; i < inputs.length; i++) {
            if (inputs[i].type === outType) { sourceNode.connect(outSlot, targetNode, i); return true; }
        }
        if (inputs.length) { sourceNode.connect(outSlot, targetNode, 0); return true; }
        return false;
    }

    let _quickAddMenu = null;
    let _quickAddMenuDocHandler = null;
    function closeQuickAddMenu() {
        if (_quickAddMenu) { _quickAddMenu.remove(); _quickAddMenu = null; }
        if (_quickAddMenuDocHandler) { document.removeEventListener('click', _quickAddMenuDocHandler); _quickAddMenuDocHandler = null; }
    }
    // sourceNode/outSlot 為 null 時（例如在空白畫布右鍵），選擇的節點會直接
    // 建立在點擊位置，不會自動連線——用於取代原生右鍵選單（已被關閉）
    function openQuickAddMenu(sourceNode, outSlot, screenX, screenY) {
        closeQuickAddMenu();
        const menu = el('div', 'add-node-menu');
        menu.style.left = screenX + 'px';
        menu.style.top = screenY + 'px';
        menu.style.display = 'block';
        menu.innerHTML = `<div class="add-node-menu-title">${sourceNode ? '新增關聯節點' : '新增節點'}</div>` +
            Object.values(NODE_TYPE_LABELS).map(t =>
                `<button data-type="${t.type}"${t.disabled ? ' class="disabled" disabled' : ''}>${t.label}</button>`
            ).join('');
        menu.addEventListener('mousedown', (e) => e.stopPropagation());
        domLayer.appendChild(menu);
        menu.querySelectorAll('button[data-type]:not(.disabled)').forEach(btn => {
            btn.addEventListener('click', () => {
                const newNode = LiteGraph.createNode(btn.dataset.type);
                if (sourceNode) {
                    newNode.pos = [sourceNode.pos[0] + sourceNode.size[0] + 90, sourceNode.pos[1]];
                    graph.add(newNode);
                    connectToFirstCompatibleInput(sourceNode, outSlot, newNode);
                } else {
                    const canvasEl = document.getElementById('litegraphCanvas');
                    const rect = canvasEl.getBoundingClientRect();
                    const scale = lgCanvas.ds.scale || 1;
                    const offset = lgCanvas.ds.offset || [0, 0];
                    newNode.pos = [(screenX - rect.left) / scale - offset[0], (screenY - rect.top) / scale - offset[1]];
                    graph.add(newNode);
                }
                selectNodeOnly(newNode);
                closeQuickAddMenu();
            });
        });
        _quickAddMenu = menu;
        _quickAddMenuDocHandler = (e) => {
            if (_quickAddMenu && !_quickAddMenu.contains(e.target)) { closeQuickAddMenu(); }
        };
        setTimeout(() => document.addEventListener('click', _quickAddMenuDocHandler), 0);
    }

    function attachNodeChrome(node) {
        const closeBtn = el('button', 'cv-close-btn', '✕');
        closeBtn.title = '刪除節點';
        closeBtn.addEventListener('mousedown', (e) => e.stopPropagation());
        closeBtn.addEventListener('click', (e) => { e.stopPropagation(); graph.remove(node); });
        domLayer.appendChild(closeBtn);
        node._closeBtn = closeBtn;

        node._addBtns = (node.outputs || []).map((out, slot) => {
            const btn = el('button', 'cv-add-link-btn', '+');
            btn.title = '新增關聯節點';
            btn.addEventListener('mousedown', (e) => e.stopPropagation());
            btn.addEventListener('click', (e) => {
                e.stopPropagation();
                const r = btn.getBoundingClientRect();
                openQuickAddMenu(node, slot, r.right + 6, r.top - 6);
            });
            domLayer.appendChild(btn);
            return btn;
        });
    }

    // 讓生成類節點預設只顯示圖片/影片本身：表單控制（.cv-controls）移出節點面板，
    // 變成一個獨立的浮動面板，只在節點被選中時才出現，且用固定像素大小顯示
    // （不隨畫布縮放），這樣不管畫布縮到多小，面板文字都維持可讀——這是實際比對
    // 參考產品後發現的關鍵差異：它的設定面板永遠是固定大小，只有位置跟著節點移動。
    function wireConfigOverlay(node, panel) {
        const controls = panel.querySelector('.cv-controls');
        if (!controls) return;
        controls.remove();
        controls.classList.add('cv-config-overlay');
        controls.addEventListener('mousedown', (e) => e.stopPropagation());
        controls.addEventListener('wheel', (e) => e.stopPropagation());
        domLayer.appendChild(controls);
        node._configOverlay = controls;
    }

    function selectNodeOnly(node) {
        lgCanvas.selectNode(node, false);
        lgCanvas.setDirty(true, true);
    }

    function sharedOnRemoved() {
        if (this._domPanel) { this._domPanel.remove(); this._domPanel = null; }
        if (this._configOverlay) { this._configOverlay.remove(); this._configOverlay = null; }
        if (this._closeBtn) { this._closeBtn.remove(); this._closeBtn = null; }
        (this._addBtns || []).forEach(b => b.remove());
        this._addBtns = [];
    }

    // LiteGraph 的座標轉換公式（來自 DragAndScale.convertOffsetToCanvas）：
    // canvasPixel = (graphPos + ds.offset) * ds.scale —— ctx 是先 scale 再 translate。
    // node.pos 是節點「主體」（title 列下方）的左上角，title 往上額外佔用 NODE_TITLE_HEIGHT。
    //
    // 重要：LiteGraph 會在節點主體最上方、依插槽數量畫出原生的輸入/輸出連線圓點
    // （每格高 NODE_SLOT_HEIGHT）。如果 DOM 面板從 node.pos[1] 就開始蓋，會把這些
    // 圓點完全蓋住、連拖曳連線都抓不到——所以面板必須往下留出「插槽區」的高度，
    // 同時把 node.size[1] 一併加大，讓節點的底色範圍完整包住插槽區 + 面板內容。
    function socketZoneHeight(node) {
        const rows = Math.max((node.inputs || []).length, (node.outputs || []).length, 1);
        return rows * (LiteGraph.NODE_SLOT_HEIGHT || 20) + 14;
    }

    function positionAllPanels() {
        if (!graph || !lgCanvas) return;
        const canvasEl = document.getElementById('litegraphCanvas');
        const rect = canvasEl.getBoundingClientRect();
        const scale = lgCanvas.ds.scale || 1;
        const offset = lgCanvas.ds.offset || [0, 0];
        const titleH = LiteGraph.NODE_TITLE_HEIGHT || 30;
        const toScreen = (gx, gy) => [rect.left + (gx + offset[0]) * scale, rect.top + (gy + offset[1]) * scale];
        graph._nodes.forEach(node => {
            const collapsed = node.flags && node.flags.collapsed;
            const zoneH = socketZoneHeight(node);
            const panel = node._domPanel;
            if (panel) {
                // 面板高度用實際渲染出來的內容量測，不要用固定猜測值——不然遇到
                // 長寬比不同的圖片、變長的錯誤訊息、動態新增的參考圖插槽等內容
                // 比預期高時，畫面會超出節點自己畫的深色底框（超出框框）
                panel.style.width = node.size[0] + 'px';
                node.size[1] = (panel.offsetHeight || node._contentHeight || 200) + zoneH;
                if (collapsed) {
                    panel.style.display = 'none';
                } else {
                    panel.style.display = '';
                    const [sx, sy] = toScreen(node.pos[0], node.pos[1] + zoneH);
                    panel.style.left = sx + 'px';
                    panel.style.top = sy + 'px';
                    panel.style.transform = `scale(${scale})`;
                }
            } else if (node._contentHeight) {
                node.size[1] = node._contentHeight + zoneH;
            }
            if (node._closeBtn) {
                const [sx, sy] = toScreen(node.pos[0] + node.size[0], node.pos[1] - titleH);
                node._closeBtn.style.left = (sx - 22 * scale) + 'px';
                node._closeBtn.style.top = (sy + 4 * scale) + 'px';
                node._closeBtn.style.transform = `scale(${scale})`;
            }
            (node._addBtns || []).forEach((btn, slot) => {
                if (collapsed) { btn.style.display = 'none'; return; }
                btn.style.display = '';
                const p = node.getConnectionPos(false, slot);
                const [sx, sy] = toScreen(p[0], p[1]);
                // 往右偏移一段距離，避免蓋住原生連線圓點（那個點仍保留給滑鼠拖曳連線用）
                btn.style.left = (sx + 18 * scale) + 'px';
                btn.style.top = (sy - 11 * scale) + 'px';
                btn.style.transform = `scale(${scale})`;
            });
            // 設定浮層固定像素大小、不套用縮放 transform，只在節點被選中時顯示，
            // 貼在節點下方——這樣不管畫布縮到多小，面板文字都維持可讀
            if (node._configOverlay) {
                const showConfig = isNodeSelected(node) && !collapsed;
                node._configOverlay.style.display = showConfig ? '' : 'none';
                if (showConfig && panel) {
                    const [sx, sy] = toScreen(node.pos[0], node.pos[1] + zoneH);
                    const bodyScreenH = (panel.offsetHeight || 0) * scale;
                    node._configOverlay.style.left = sx + 'px';
                    node._configOverlay.style.top = (sy + bodyScreenH + 10) + 'px';
                }
            }
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

    // ── Lightbox：點擊預覽圖/影片放大顯示 ─────────────────────────
    const lightboxEl = document.getElementById('cvLightbox');
    const lightboxBody = lightboxEl.querySelector('.cv-lightbox-body');
    function openLightbox(kind, url) {
        lightboxBody.innerHTML = '';
        const media = el(kind === 'video' ? 'video' : 'img');
        media.src = url;
        if (kind === 'video') { media.controls = true; media.autoplay = true; }
        lightboxBody.appendChild(media);
        lightboxEl.style.display = 'flex';
    }
    function closeLightbox() {
        lightboxEl.style.display = 'none';
        lightboxBody.innerHTML = '';
    }
    lightboxEl.querySelector('.cv-lightbox-close').addEventListener('click', closeLightbox);
    lightboxEl.addEventListener('click', (e) => { if (e.target === lightboxEl) closeLightbox(); });
    document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeLightbox(); });

    function setPreviewImage(node, url) {
        node._previewBox.innerHTML = '';
        const img = el('img');
        img.src = url;
        img.addEventListener('mousedown', (e) => e.stopPropagation());
        img.addEventListener('click', () => openLightbox('image', url));
        node._previewBox.appendChild(img);
        const dl = el('a', 'cv-dl-btn', '⬇ 下載');
        dl.href = url; dl.download = 'image.png'; dl.target = '_blank';
        dl.addEventListener('mousedown', (e) => e.stopPropagation());
        node._previewBox.appendChild(dl);
    }

    function setPreviewVideo(node, url) {
        // 影片本身有原生播放控制列，不能整個蓋 click 監聽（會跟播放/拖曳衝突），
        // 改用獨立的「⤢ 放大」按鈕開燈箱
        node._previewBox.innerHTML = '';
        const video = el('video');
        video.src = url; video.controls = true;
        video.addEventListener('mousedown', (e) => e.stopPropagation());
        node._previewBox.appendChild(video);
        const zoom = el('button', 'cv-zoom-btn', '⤢');
        zoom.title = '放大預覽';
        zoom.addEventListener('mousedown', (e) => e.stopPropagation());
        zoom.addEventListener('click', () => openLightbox('video', url));
        node._previewBox.appendChild(zoom);
        const dl = el('a', 'cv-dl-btn', '⬇ 下載');
        dl.href = url; dl.download = 'video.mp4'; dl.target = '_blank';
        dl.addEventListener('mousedown', (e) => e.stopPropagation());
        node._previewBox.appendChild(dl);
    }

    // ── Node: Text（可手動輸入當純 prompt 來源，也可選模型做真正的文字生成；
    //              若連接「圖片」輸入，改為顯示「分析圖片」按鈕呼叫圖片分析） ──
    function TextPromptNode() {
        this.addInput('image', 'image');
        this.addOutput('text', 'string');
        const models = getModelsFor('text');
        this.properties = { model: (models[0] && models[0].id) || '', text: '', status: '' };
        this.generatedText = null;
        this._contentHeight = 340;
        this.size = [300, 340];
        this.color = '#3d3320'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <label>模型<span class="cv-hint">（不生成也可直接當純文字輸出）</span></label>
            <div class="cv-select-slot"></div>
            <label>Prompt</label>
            <textarea placeholder="輸入文字…"></textarea>
            <button class="cv-generate cv-gen-text-btn">▶ 生成文字</button>
            <button class="cv-generate cv-analyze-btn" style="display:none">🔍 分析已連接的圖片</button>
            <div class="cv-status"></div>
            <div class="cv-output-box" style="display:none"></div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => {
            this.properties.text = this.textarea.value;
            this.generatedText = null;
            this.outputBox.style.display = 'none';
        });
        this.statusEl = panel.querySelector('.cv-status');
        this.outputBox = panel.querySelector('.cv-output-box');
        panel.querySelector('.cv-gen-text-btn').addEventListener('click', () => this.generateText());
        this.analyzeBtn = panel.querySelector('.cv-analyze-btn');
        this.analyzeBtn.addEventListener('click', () => this.analyzeImage());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => { this.properties.model = v; });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);
        attachNodeChrome(this);
    }
    TextPromptNode.title = '文字 Text';
    TextPromptNode.prototype.onExecute = function () {
        this.setOutputData(0, this.generatedText != null ? this.generatedText : this.properties.text);
    };
    TextPromptNode.prototype.onConnectionsChange = function (type) {
        if (type !== LiteGraph.INPUT || !this.analyzeBtn) return;
        const hasImage = !!this.getInputNode(0);
        this.analyzeBtn.style.display = hasImage ? '' : 'none';
    };
    TextPromptNode.prototype.generateText = async function () {
        const prompt = this.properties.text;
        if (!prompt) { showToast('請輸入文字'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        this.statusEl.textContent = '生成中…';
        try {
            const res = await apiFetch('/api/text/generate', {
                method: 'POST',
                body: JSON.stringify({ model: this.properties.model, prompt, stream: false }),
            });
            const data = await res.json();
            if (!res.ok) throw new Error(data.detail || '生成失敗');
            this.generatedText = data.content || '';
            this.outputBox.textContent = this.generatedText;
            this.outputBox.style.display = '';
            this.statusEl.textContent = '完成（輸出已改為生成結果）';
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            showToast('文字生成失敗：' + e.message);
        }
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
            this.generatedText = data.content || '';
            this.textarea.value = this.generatedText;
            this.properties.text = this.generatedText;
            this.outputBox.textContent = this.generatedText;
            this.outputBox.style.display = '';
            this.statusEl.textContent = '完成';
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            showToast('圖片分析失敗：' + e.message);
        }
    };
    TextPromptNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Camera Angle（多角度相機控制，仿 ComfyUI-qwenmultiangle 的拖曳式
    // 相機角度控制介面，輸出符合 Qwen-Image-Edit 多角度 LoRA 慣例格式的 prompt，
    // 例如 "<sks> front view eye-level shot medium shot"——這裡只產生格式化好
    // 的提示詞，NenAI 平台目前沒有對應的 LoRA 管道，效果仍取決於下游圖片編輯
    // 模型是否認得這種寫法）用 2D SVG 取代原專案的 Three.js 3D 場景，互動邏輯
    // （atan2 反推角度、環形/弧形限制拖曳範圍）比照原專案的 CameraWidget.ts ──
    function _classifyAzimuth(deg) {
        const table = [
            [22.5, 'front view'], [67.5, 'front-right quarter view'], [112.5, 'right side view'],
            [157.5, 'back-right quarter view'], [202.5, 'back view'], [247.5, 'back-left quarter view'],
            [292.5, 'left side view'], [337.5, 'front-left quarter view'], [360.1, 'front view'],
        ];
        for (const [max, label] of table) if (deg < max) return label;
        return 'front view';
    }
    function _classifyElevation(deg) {
        if (deg < -15) return 'low-angle shot';
        if (deg <= 15) return 'eye-level shot';
        if (deg <= 45) return 'elevated shot';
        return 'high-angle shot';
    }
    function _classifyZoom(z) {
        if (z < 2) return 'wide shot';
        if (z <= 6) return 'medium shot';
        return 'close-up';
    }
    // 橢圓形方位環（模擬俯視的水平軌道）
    const CAM_RING = { cx: 130, cy: 148, rx: 104, ry: 26 };
    // 仰角弧（左側，-30°~60° 對應弧上 200°~100° 的掃角）——圓心/半徑要讓整段弧
    // 都落在 viewBox（260x200）範圍內，否則 SVG 預設會裁切掉超出範圍的部分，
    // 導致弧線跟控制點大部分時間畫面外看不到也點不到
    const CAM_ARC = { cx: 75, cy: 140, r: 68, angMin: -30, angMax: 60, sweepMin: 200, sweepMax: 100 };
    function _azimuthPoint(deg) {
        const rad = deg * Math.PI / 180;
        return [CAM_RING.cx + CAM_RING.rx * Math.sin(rad), CAM_RING.cy + CAM_RING.ry * Math.cos(rad)];
    }
    function _elevationPoint(deg) {
        const t = (deg - CAM_ARC.angMin) / (CAM_ARC.angMax - CAM_ARC.angMin);
        const sweep = (CAM_ARC.sweepMin + t * (CAM_ARC.sweepMax - CAM_ARC.sweepMin)) * Math.PI / 180;
        return [CAM_ARC.cx + CAM_ARC.r * Math.cos(sweep), CAM_ARC.cy - CAM_ARC.r * Math.sin(sweep)];
    }
    function _svgPoint(svgEl, clientX, clientY) {
        const rect = svgEl.getBoundingClientRect();
        const vb = svgEl.viewBox.baseVal;
        return [
            (clientX - rect.left) / rect.width * vb.width + vb.x,
            (clientY - rect.top) / rect.height * vb.height + vb.y,
        ];
    }
    function _attachDrag(dotEl, svgEl, onMove) {
        dotEl.addEventListener('mousedown', (e) => {
            e.stopPropagation(); e.preventDefault();
            const move = (ev) => { const [x, y] = _svgPoint(svgEl, ev.clientX, ev.clientY); onMove(x, y); };
            const up = () => { document.removeEventListener('mousemove', move); document.removeEventListener('mouseup', up); };
            document.addEventListener('mousemove', move);
            document.addEventListener('mouseup', up);
        });
    }
    function CameraAngleNode() {
        this.addInput('image', 'image');
        this.addOutput('prompt', 'string');
        // 把接進來的圖片原樣傳出去，這樣同一條 Load Image/圖片節點可以一路接到
        // 下游節點的「參考圖」，不用另外拉一條線——不然使用者容易誤以為接了
        // image 輸入就會自動把圖傳給下游，結果下游其實完全沒收到參考圖
        this.addOutput('image', 'image');
        this.properties = { horizontal: 0, vertical: 0, zoom: 5 };
        this._contentHeight = 470;
        this.size = [300, 470];
        this.color = '#2a3a3d'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-cam-row"><label>horizontal_angle</label><input type="number" class="cv-cam-h" min="0" max="360" step="1" value="0"></div>
            <div class="cv-cam-row"><label>vertical_angle</label><input type="number" class="cv-cam-v" min="-30" max="60" step="1" value="0"></div>
            <div class="cv-cam-row"><label>zoom</label><input type="number" class="cv-cam-z" min="0" max="10" step="0.1" value="5"></div>
            <div class="cv-hint">拖曳圖上的控制點調整相機角度</div>
            <div class="cv-cam-widget">
                <svg class="cv-cam-svg" viewBox="0 0 260 200" preserveAspectRatio="xMidYMid meet">
                    <ellipse class="cv-cam-ring" cx="${CAM_RING.cx}" cy="${CAM_RING.cy}" rx="${CAM_RING.rx}" ry="${CAM_RING.ry}"></ellipse>
                    <path class="cv-cam-arc"></path>
                    <circle class="cv-cam-az-dot" r="9"></circle>
                    <circle class="cv-cam-el-dot" r="8"></circle>
                </svg>
                <div class="cv-cam-card"><img class="cv-cam-card-img" style="display:none"></div>
            </div>
            <div class="cv-cam-readout">
                <span class="cv-cam-ro-h">HORIZONTAL<br><b>0°</b></span>
                <span class="cv-cam-ro-v">VERTICAL<br><b>0°</b></span>
                <span class="cv-cam-ro-z">ZOOM<br><b>5.0</b></span>
            </div>
            <label>輸出 Prompt</label>
            <div class="cv-output-box"></div>`;
        attachDomPanel(this, panel);

        this.hInput = panel.querySelector('.cv-cam-h');
        this.vInput = panel.querySelector('.cv-cam-v');
        this.zInput = panel.querySelector('.cv-cam-z');
        this.outputBox = panel.querySelector('.cv-output-box');
        this.svgEl = panel.querySelector('.cv-cam-svg');
        this.arcPath = panel.querySelector('.cv-cam-arc');
        this.azDot = panel.querySelector('.cv-cam-az-dot');
        this.elDot = panel.querySelector('.cv-cam-el-dot');
        this.cardEl = panel.querySelector('.cv-cam-card');
        this.cardImg = panel.querySelector('.cv-cam-card-img');
        this.roH = panel.querySelector('.cv-cam-ro-h b');
        this.roV = panel.querySelector('.cv-cam-ro-v b');
        this.roZ = panel.querySelector('.cv-cam-ro-z b');

        const [ax, ay] = _elevationPoint(CAM_ARC.angMin);
        const [bx, by] = _elevationPoint(CAM_ARC.angMax);
        this.arcPath.setAttribute('d', `M ${ax} ${ay} A ${CAM_ARC.r} ${CAM_ARC.r} 0 0 1 ${bx} ${by}`);

        const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
        const refresh = () => {
            const { horizontal, vertical, zoom } = this.properties;
            const [azx, azy] = _azimuthPoint(horizontal);
            this.azDot.setAttribute('cx', azx); this.azDot.setAttribute('cy', azy);
            const [elx, ely] = _elevationPoint(vertical);
            this.elDot.setAttribute('cx', elx); this.elDot.setAttribute('cy', ely);
            const tilt = clamp((horizontal <= 180 ? horizontal : horizontal - 360) * 0.5, -70, 70);
            const scale = 0.65 + (zoom / 10) * 0.6;
            this.cardEl.style.transform = `translate(-50%,-50%) perspective(500px) rotateY(${tilt}deg) scale(${scale})`;
            this.hInput.value = Math.round(horizontal);
            this.vInput.value = Math.round(vertical);
            this.zInput.value = zoom.toFixed(1);
            this.roH.textContent = Math.round(horizontal) + '°';
            this.roV.textContent = Math.round(vertical) + '°';
            this.roZ.textContent = zoom.toFixed(1);
            this.outputBox.textContent = this._buildPrompt();
        };
        this.refresh = refresh;

        this.hInput.addEventListener('mousedown', (e) => e.stopPropagation());
        this.vInput.addEventListener('mousedown', (e) => e.stopPropagation());
        this.zInput.addEventListener('mousedown', (e) => e.stopPropagation());
        this.hInput.addEventListener('change', () => { this.properties.horizontal = clamp(parseFloat(this.hInput.value) || 0, 0, 360); refresh(); });
        this.vInput.addEventListener('change', () => { this.properties.vertical = clamp(parseFloat(this.vInput.value) || 0, -30, 60); refresh(); });
        this.zInput.addEventListener('change', () => { this.properties.zoom = clamp(parseFloat(this.zInput.value) || 0, 0, 10); refresh(); });

        _attachDrag(this.azDot, this.svgEl, (x, y) => {
            const nx = (x - CAM_RING.cx) / CAM_RING.rx;
            const ny = (y - CAM_RING.cy) / CAM_RING.ry;
            let deg = Math.atan2(nx, ny) * 180 / Math.PI;
            if (deg < 0) deg += 360;
            this.properties.horizontal = deg;
            refresh();
        });
        _attachDrag(this.elDot, this.svgEl, (x, y) => {
            let ang = Math.atan2(-(y - CAM_ARC.cy), x - CAM_ARC.cx) * 180 / Math.PI;
            if (ang < 0) ang += 360;
            const lo = Math.min(CAM_ARC.sweepMin, CAM_ARC.sweepMax);
            const hi = Math.max(CAM_ARC.sweepMin, CAM_ARC.sweepMax);
            ang = clamp(ang, lo, hi);
            const t = (ang - CAM_ARC.sweepMin) / (CAM_ARC.sweepMax - CAM_ARC.sweepMin);
            this.properties.vertical = clamp(CAM_ARC.angMin + t * (CAM_ARC.angMax - CAM_ARC.angMin), -30, 60);
            refresh();
        });

        refresh();
        attachNodeChrome(this);
    }
    CameraAngleNode.title = '相機角度 Camera Angle';
    CameraAngleNode.prototype._buildPrompt = function () {
        const { horizontal, vertical, zoom } = this.properties;
        return `<sks> ${_classifyAzimuth(horizontal)} ${_classifyElevation(vertical)} ${_classifyZoom(zoom)}`;
    };
    CameraAngleNode.prototype.onExecute = function () {
        const imgUrl = this.getInputData(0);
        if (imgUrl && imgUrl !== this._lastImgUrl) {
            this._lastImgUrl = imgUrl;
            this.cardImg.src = imgUrl;
            this.cardImg.style.display = '';
        }
        this.setOutputData(0, this._buildPrompt());
        this.setOutputData(1, imgUrl || null);
    };
    CameraAngleNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Load Image（直接上傳本機圖片，不經過模型生成，作為其他節點的
    // 圖片輸入來源，例如接到圖片編輯節點的「參考圖」或影片節點的 first_frame）──
    function LoadImageNode() {
        this.addOutput('image', 'image');
        this.imageUrl = null;
        this._contentHeight = 300;
        // 寬度需與 .cv-config-overlay 的固定 300px 對齊，否則設定浮層會跟節點本體寬度對不齊
        this.size = [300, 300];
        this.color = '#3a2f1f'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>上傳圖片</label>
                <input type="file" class="cv-load-file" accept="image/*" style="display:none">
                <button class="cv-generate cv-load-btn">📁 選擇檔案</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.fileInput = panel.querySelector('.cv-load-file');
        this.statusEl = panel.querySelector('.cv-status');
        const loadBtn = panel.querySelector('.cv-load-btn');
        loadBtn.addEventListener('mousedown', (e) => e.stopPropagation());
        loadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('mousedown', (e) => e.stopPropagation());
        this.fileInput.addEventListener('change', () => this._onFile());

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
    }
    LoadImageNode.title = '上傳圖片 Load Image';
    LoadImageNode.prototype._onFile = function () {
        const file = this.fileInput.files[0];
        if (!file) return;
        if (this.imageUrl && this.imageUrl.startsWith('blob:')) URL.revokeObjectURL(this.imageUrl);
        this.imageUrl = URL.createObjectURL(file);
        setPreviewImage(this, this.imageUrl);
        this.statusEl.textContent = '已載入：' + file.name;
    };
    LoadImageNode.prototype.onExecute = function () {
        this.setOutputData(0, this.imageUrl);
    };
    LoadImageNode.prototype.onRemoved = function () {
        if (this.imageUrl && this.imageUrl.startsWith('blob:')) URL.revokeObjectURL(this.imageUrl);
        sharedOnRemoved.call(this);
    };

    // ── Node: Image（文生圖，t2i 模型） ─────────────────────────
    // 圖片節點依「是否連接參考圖」自動切換 t2i（純文生圖）/ i2i（拿參考圖做
    // 圖像生成，實際呼叫 /api/image/edit）——這樣使用者可以直接把一個圖片節點
    // 的輸出拉線接到另一個圖片節點的「參考圖」輸入，做「用圖像生成圖像」。
    function ImageGenNode() {
        this.addInput('prompt', 'string');
        this.addInput('參考圖', 'image');
        this.addOutput('image', 'image');
        const models = getModelsFor('image', 't2i');
        this.properties = { model: (models[0] && models[0].id) || '', prompt: '', size: '1024*1024', status: '' };
        this.imageUrl = null;
        this._contentHeight = 470;
        this.size = [320, 470];
        this.color = '#1f3a2e'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>模型 <span class="cv-hint cv-mode-hint">（文生圖）</span></label>
                <div class="cv-select-slot"></div>
                <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
                <textarea placeholder="輸入文字…"></textarea>
                <label>尺寸</label>
                <div class="cv-size-slot"></div>
                <button class="cv-generate">▶ 生成圖片</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        this.modeHintEl = panel.querySelector('.cv-mode-hint');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => {
            this.properties.model = v;
            const sizes = sizesForModel('image', v);
            this._rebuildSizeSelect(sizes);
        });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);
        this._rebuildSizeSelect(sizesForModel('image', this.properties.model));

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
    }
    ImageGenNode.title = '圖片 Image';
    ImageGenNode.prototype._rebuildSizeSelect = function (sizes) {
        const slot = this._domPanel.querySelector('.cv-size-slot');
        if (!sizes.includes(this.properties.size)) this.properties.size = sizes[0];
        slot.innerHTML = '';
        this.sizeSelect = buildSelect(sizes, this.properties.size, (v) => { this.properties.size = v; });
        slot.appendChild(this.sizeSelect);
    };
    ImageGenNode.prototype._detectMode = function () {
        return this.getInputNode(1) ? 'i2i' : 't2i';
    };
    ImageGenNode.prototype.onExecute = function () {
        this.setOutputData(0, this.imageUrl);
    };
    ImageGenNode.prototype.onConnectionsChange = function (type) {
        if (type !== LiteGraph.INPUT || !this.modelSelect) return;
        const mode = this._detectMode();
        const list = getModelsFor('image', mode);
        const values = list.map(m => m.id);
        if (!values.includes(this.properties.model)) this.properties.model = values[0] || '';
        this.modelSelect.innerHTML = values.map(v => `<option value="${v}"${v === this.properties.model ? ' selected' : ''}>${v}</option>`).join('');
        this._rebuildSizeSelect(sizesForModel('image', this.properties.model));
        this.modeHintEl.textContent = mode === 'i2i' ? '（參考圖生成圖像）' : '（文生圖）';
    };
    ImageGenNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        const mode = this._detectMode();
        this.statusEl.textContent = '生成中…';
        setPreviewEmpty(this, '生成中…');
        try {
            let res;
            if (mode === 'i2i') {
                const refUrl = this.getInputData(1);
                if (!refUrl) throw new Error('參考圖節點尚未生成完成，請先按上游圖片節點的「生成圖片」');
                const fd = new FormData();
                fd.append('model', this.properties.model);
                fd.append('prompt', prompt);
                fd.append('size', this.properties.size);
                fd.append('n', '1');
                fd.append('image_1', await fetchAsBlob(refUrl), 'ref.png');
                res = await apiFetch('/api/image/edit', { method: 'POST', body: fd });
            } else {
                res = await apiFetch('/api/image/generate', {
                    method: 'POST',
                    body: JSON.stringify({ model: this.properties.model, prompt, size: this.properties.size, n: 1 }),
                });
            }
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
    ImageGenNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Video（t2v / i2v / r2v，依連接的圖片組合自動切換） ──
    // r2v（參考生影片）可以連接多張參考圖到同一個節點：初始給 2 個「參考圖」
    // 輸入插槽，並提供「+ 新增參考圖輸入」按鈕可再加到最多 6 張。
    const VIDEO_MAX_REF_SLOTS = 6;
    function VideoGenNode() {
        this.addInput('prompt', 'string');
        this.addInput('first_frame', 'image');
        this.addInput('last_frame', 'image');
        this.addOutput('video', 'video');
        this.refSlots = [];
        this._addRefSlot();
        this._addRefSlot();
        const models = getModelsFor('video', 't2v');
        this.properties = { model: (models[0] && models[0].id) || '', prompt: '', resolution: '720P', duration: 5, status: '' };
        this.videoUrl = null;
        this._contentHeight = 560;
        this.size = [320, 560];
        this.color = '#1f2f3a'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>模型 <span class="cv-hint cv-mode-hint">（文生影片）</span></label>
                <div class="cv-select-slot"></div>
                <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
                <textarea placeholder="輸入文字…"></textarea>
                <label>解析度</label>
                <div class="cv-res-slot"></div>
                <label>時長（秒）<span class="cv-dur-val">5</span></label>
                <input type="range" class="cv-dur-slider" min="2" max="15" step="1" value="5">
                <button class="cv-add-ref-btn">+ 新增參考圖輸入</button>
                <button class="cv-generate cv-submit-btn">▶ 生成影片</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        this.modeHintEl = panel.querySelector('.cv-mode-hint');
        panel.querySelector('.cv-add-ref-btn').addEventListener('click', () => this._addRefSlot());
        panel.querySelector('.cv-submit-btn').addEventListener('click', () => this.generate());

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
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
    }
    VideoGenNode.title = '影片 Video';
    VideoGenNode.prototype._addRefSlot = function () {
        if (this.refSlots.length >= VIDEO_MAX_REF_SLOTS) { showToast(`最多 ${VIDEO_MAX_REF_SLOTS} 張參考圖`); return; }
        this.addInput('參考圖 ' + (this.refSlots.length + 1), 'image');
        this.refSlots.push(this.inputs.length - 1);
    };
    // 模式判定要用「有沒有連線」（結構性、graph topology），不能用「有沒有資料」
    // （getInputData 要等上游節點生成完畢並執行過 onExecute 才會有值）——否則
    // 剛接上參考圖但還沒按生成時，會誤判成沒有參考圖而退回 i2v/first_frame。
    VideoGenNode.prototype._detectMode = function () {
        const hasRef = this.refSlots.some(i => !!this.getInputNode(i));
        if (hasRef) return 'r2v';
        if (this.getInputNode(1)) return 'i2v';
        return 't2v';
    };
    VideoGenNode.prototype.onExecute = function () {
        this.setOutputData(0, this.videoUrl);
    };
    VideoGenNode.prototype.onConnectionsChange = function (type) {
        if (type !== LiteGraph.INPUT || !this.modelSelect) return;
        const mode = this._detectMode();
        const list = getModelsFor('video', mode);
        const values = list.map(m => m.id);
        if (!values.includes(this.properties.model)) this.properties.model = values[0] || '';
        this.modelSelect.innerHTML = values.map(v => `<option value="${v}"${v === this.properties.model ? ' selected' : ''}>${v}</option>`).join('');
        const firstFrameAlsoConnected = mode === 'r2v' && !!this.getInputNode(1);
        this.modeHintEl.textContent = mode === 'r2v'
            ? (firstFrameAlsoConnected ? '（參考生影片 / 多圖，first_frame 將被忽略）' : '（參考生影片 / 多圖）')
            : mode === 'i2v' ? '（圖生影片）' : '（文生影片）';
    };
    VideoGenNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        const mode = this._detectMode();
        if (mode === 'r2v' && this.getInputNode(1)) {
            showToast('已連接參考圖，first_frame 的圖片將被忽略（r2v 與 i2v 是互斥的兩種生成模式）');
        }
        this.statusEl.textContent = '送出中…';
        setPreviewEmpty(this, '送出中…');
        try {
            const fd = new FormData();
            fd.append('model', this.properties.model);
            fd.append('prompt', prompt);
            fd.append('resolution', this.properties.resolution);
            fd.append('duration', String(this.properties.duration));
            let endpoint = '/api/video/t2v';
            if (mode === 'r2v') {
                endpoint = '/api/video/r2v';
                const refUrls = this.refSlots.map(i => this.getInputData(i)).filter(Boolean);
                if (!refUrls.length) throw new Error('參考圖節點尚未生成完成，請先按上游圖片節點的「生成圖片」');
                for (const url of refUrls) {
                    fd.append('reference_files', await fetchAsBlob(url), 'ref.png');
                }
            } else if (mode === 'i2v') {
                endpoint = '/api/video/i2v';
                const firstFrameUrl = this.getInputData(1);
                const lastFrameUrl = this.getInputData(2);
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
    VideoGenNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Editing（i2i 圖像編輯，需連接一張輸入圖片） ───────────
    function ImageEditNode() {
        this.addInput('image', 'image');
        this.addInput('prompt', 'string');
        this.addOutput('image', 'image');
        const models = getModelsFor('image', 'i2i');
        this.properties = { model: (models[0] && models[0].id) || '', prompt: '', size: '1024*1024', status: '' };
        this.imageUrl = null;
        this._contentHeight = 470;
        this.size = [320, 470];
        this.color = '#3a2340'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>模型</label>
                <div class="cv-select-slot"></div>
                <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
                <textarea placeholder="輸入編輯指示…"></textarea>
                <button class="cv-generate">▶ 編輯圖片</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => { this.properties.model = v; });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
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
    ImageEditNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: MuleAI（wan2.7-i2v-spicy / z-image-spicy / qwen-image-edit-spicy /
    //              face-swap 四個模型共用一個節點，走 /api/muleai/generate + 獨立的
    //              /api/muleai/status/{model}/{task_id} 輪詢——參數與必填輸入差異很大，
    //              靠 model select 動態切換要顯示的欄位、輸入插槽名稱與輸出型別） ──
    const MULEAI_VIDEO_MODEL = 'wan2.7-i2v-spicy';
    const MULEAI_T2I_MODEL = 'z-image-spicy';
    const MULEAI_FACESWAP_MODEL = 'face-swap';

    function buildMuleaiModelSelect(current, onChange) {
        const models = getModelsFor('muleai');
        const groups = {};
        models.forEach(m => { (groups[m.group] = groups[m.group] || []).push(m); });
        const sel = el('select');
        sel.innerHTML = Object.keys(groups).map(g =>
            `<optgroup label="${g}">` +
            groups[g].map(m => `<option value="${m.id}"${m.id === current ? ' selected' : ''}>${m.name}</option>`).join('') +
            `</optgroup>`
        ).join('');
        sel.addEventListener('mousedown', (e) => e.stopPropagation());
        sel.addEventListener('change', () => onChange(sel.value));
        return sel;
    }

    async function pollMuleaiTask(model, taskId, { intervalMs = 4000, maxTries = 300 } = {}) {
        for (let i = 0; i < maxTries; i++) {
            const res = await apiFetch(`/api/muleai/status/${model}/${taskId}`);
            const data = await res.json();
            const st = (data.status || '').toUpperCase();
            if (['SUCCEEDED', 'COMPLETED', 'SUCCESS'].includes(st)) return data;
            if (['FAILED', 'FAIL', 'FAILURE', 'ERROR'].includes(st)) throw new Error(data.error_message || '生成失敗');
            await new Promise(r => setTimeout(r, intervalMs));
        }
        throw new Error('等待逾時');
    }

    function MuleAiGenNode() {
        this.addInput('image', 'image');
        this.addInput('換臉參考圖', 'image');
        this.addInput('prompt', 'string');
        this.addOutput('output', 'image');
        const models = getModelsFor('muleai');
        this.properties = {
            model: (models[0] && models[0].id) || MULEAI_VIDEO_MODEL,
            prompt: '', resolution: '1080P', imgResolution: '1024*1536', duration: 5, status: '',
        };
        this.resultUrl = null;
        this._contentHeight = 560;
        this.size = [320, 560];
        this.color = '#4a1f2e'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>模型 <span class="cv-hint cv-mode-hint"></span></label>
                <div class="cv-select-slot"></div>
                <div class="cv-mu-prompt-group">
                    <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
                    <textarea placeholder="輸入文字…"></textarea>
                </div>
                <div class="cv-mu-res-group">
                    <label>解析度</label>
                    <div class="cv-res-slot"></div>
                </div>
                <div class="cv-mu-imgres-group">
                    <label>圖片尺寸</label>
                    <div class="cv-imgres-slot"></div>
                </div>
                <div class="cv-mu-dur-group">
                    <label>時長（秒）<span class="cv-dur-val">5</span></label>
                    <input type="range" class="cv-dur-slider" min="2" max="15" step="1" value="5">
                </div>
                <button class="cv-generate">▶ 生成</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.promptGroup = panel.querySelector('.cv-mu-prompt-group');
        this.resGroup = panel.querySelector('.cv-mu-res-group');
        this.imgResGroup = panel.querySelector('.cv-mu-imgres-group');
        this.durGroup = panel.querySelector('.cv-mu-dur-group');
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        this.modeHintEl = panel.querySelector('.cv-mode-hint');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildMuleaiModelSelect(this.properties.model, (v) => {
            this.properties.model = v;
            this._syncUiForModel();
        });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        this.resSelect = buildSelect(['1080P', '720P'], this.properties.resolution, (v) => { this.properties.resolution = v; });
        panel.querySelector('.cv-res-slot').appendChild(this.resSelect);

        this.imgResSelect = buildSelect(['1024*1536', '1536*1024', '1024*1024'], this.properties.imgResolution, (v) => { this.properties.imgResolution = v; });
        panel.querySelector('.cv-imgres-slot').appendChild(this.imgResSelect);

        this.durSlider = panel.querySelector('.cv-dur-slider');
        this.durValEl = panel.querySelector('.cv-dur-val');
        this.durSlider.addEventListener('input', () => {
            this.properties.duration = parseInt(this.durSlider.value);
            this.durValEl.textContent = this.durSlider.value;
        });

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
        this._syncUiForModel();
    }
    MuleAiGenNode.title = 'MuleAI Spicy';
    MuleAiGenNode.prototype._isVideo = function () { return this.properties.model === MULEAI_VIDEO_MODEL; };
    MuleAiGenNode.prototype._isFaceSwap = function () { return this.properties.model === MULEAI_FACESWAP_MODEL; };
    MuleAiGenNode.prototype._isT2i = function () { return this.properties.model === MULEAI_T2I_MODEL; };
    MuleAiGenNode.prototype._needsImage = function () { return this.properties.model !== MULEAI_T2I_MODEL; };
    MuleAiGenNode.prototype._syncUiForModel = function () {
        const isVideo = this._isVideo(), isFaceSwap = this._isFaceSwap(), isT2i = this._isT2i();
        this.promptGroup.style.display = isFaceSwap ? 'none' : '';
        this.resGroup.style.display = isVideo ? '' : 'none';
        this.imgResGroup.style.display = isT2i ? '' : 'none';
        this.durGroup.style.display = isVideo ? '' : 'none';
        this.inputs[0].name = isVideo ? '首幀圖片' : '來源圖';
        this.modeHintEl.textContent =
            isVideo ? '（圖生影片 + 配音）' : isFaceSwap ? '（來源圖 + 換臉參考圖）' : isT2i ? '（純文生圖）' : '（來源圖 + Prompt）';
        const outType = isVideo ? 'video' : 'image';
        if (this.outputs[0].type !== outType) {
            this.disconnectOutput(0);
            this.outputs[0].type = outType;
            this.outputs[0].name = outType;
        }
        this.resultUrl = null;
        setPreviewEmpty(this, '尚未生成');
        // LiteGraph 節點主體（插槽名稱等）畫在會被快取的背景層，直接改屬性不會自動
        // 觸發重繪，要手動標記兩層都要重畫——這裡踩到跟選取狀態同一類的坑
        lgCanvas.setDirty(true, true);
    };
    MuleAiGenNode.prototype.onExecute = function () {
        this.setOutputData(0, this.resultUrl);
    };
    MuleAiGenNode.prototype.generate = async function () {
        const model = this.properties.model;
        const isVideo = this._isVideo(), isFaceSwap = this._isFaceSwap(), needsImage = this._needsImage();
        const promptIn = this.getInputData(2);
        const prompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        if (!isFaceSwap && !prompt) { showToast('請輸入 prompt'); return; }
        let imageBlob = null, faceBlob = null;
        if (needsImage) {
            const imgUrl = this.getInputData(0);
            if (!imgUrl) { showToast('請先連接一張來源圖片'); return; }
            imageBlob = await fetchAsBlob(imgUrl);
        }
        if (isFaceSwap) {
            const faceUrl = this.getInputData(1);
            if (!faceUrl) { showToast('請先連接換臉參考圖'); return; }
            faceBlob = await fetchAsBlob(faceUrl);
        }
        this.statusEl.textContent = '送出中…';
        setPreviewEmpty(this, '送出中…');
        try {
            const fd = new FormData();
            fd.append('model', model);
            if (!isFaceSwap) fd.append('prompt', prompt);
            if (isVideo) {
                fd.append('resolution', this.properties.resolution);
                fd.append('duration', String(this.properties.duration));
            }
            if (this._isT2i()) fd.append('img_resolution', this.properties.imgResolution);
            if (imageBlob) fd.append('image', imageBlob, 'image.png');
            if (faceBlob) fd.append('face_image', faceBlob, 'face.png');
            const res = await apiFetch('/api/muleai/generate', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error((data.error && (data.error.message || data.error)) || '任務建立失敗');
            this.statusEl.textContent = '生成中…';
            setPreviewEmpty(this, '生成中…（可能需要 1～數分鐘）');
            const result = await pollMuleaiTask(model, data.task_id);
            const kind = isVideo ? 'video' : 'image';
            const url = kind === 'video' ? (result.videos && result.videos[0]) : (result.images && result.images[0]);
            if (!url) throw new Error('生成完成但沒有取得結果檔案');
            this.resultUrl = url;
            this.statusEl.textContent = '完成';
            if (kind === 'video') setPreviewVideo(this, url); else setPreviewImage(this, url);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('MuleAI 生成失敗：' + e.message);
        }
    };
    MuleAiGenNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Audio（尚未有可用的 TTS 後端，先提供停用佔位節點） ───
    function AudioPlaceholderNode() {
        this.addInput('text', 'string');
        this.addOutput('audio', 'audio');
        this._contentHeight = 110;
        this.size = [300, 110];
        this.color = '#333'; this.bgcolor = '#2a2a2a';
        const panel = el('div');
        panel.innerHTML = `<div class="cv-status" style="margin-top:4px">平台目前沒有可用的 TTS 後端，此節點尚未支援</div>`;
        attachDomPanel(this, panel);
        attachNodeChrome(this);
    }
    AudioPlaceholderNode.title = '語音 Audio（尚未支援）';
    AudioPlaceholderNode.prototype.onRemoved = sharedOnRemoved;

    function registerNodeTypes() {
        LiteGraph.registerNodeType('nenai/text', TextPromptNode);
        LiteGraph.registerNodeType('nenai/camera_angle', CameraAngleNode);
        LiteGraph.registerNodeType('nenai/load_image', LoadImageNode);
        LiteGraph.registerNodeType('nenai/image', ImageGenNode);
        LiteGraph.registerNodeType('nenai/video', VideoGenNode);
        LiteGraph.registerNodeType('nenai/edit', ImageEditNode);
        LiteGraph.registerNodeType('nenai/audio', AudioPlaceholderNode);
        LiteGraph.registerNodeType('nenai/muleai', MuleAiGenNode);
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

    // ── 連線動態流動效果：沿每條連線的貝茲曲線畫幾個發光的移動光點，呈現
    // 資料流動的感覺。LGraphCanvas 建構時會自動用 requestAnimationFrame
    // 持續重繪（startRendering），此 hook 每一幀都會被呼叫，ctx 此時仍在
    // 已經套用 pan/zoom 的畫布座標系底下（和 node.pos 同一個座標空間）。
    function cubicBezierPoint(p1, c1, c2, p2, t) {
        const mt = 1 - t;
        const a = mt * mt * mt, b = 3 * mt * mt * t, c = 3 * mt * t * t, d = t * t * t;
        return [
            a * p1[0] + b * c1[0] + c * c2[0] + d * p2[0],
            a * p1[1] + b * c1[1] + c * c2[1] + d * p2[1],
        ];
    }
    function drawFlowingLinks(ctx) {
        if (!graph) return;
        const t = (Date.now() % 2600) / 2600;
        for (const id in graph.links) {
            const link = graph.links[id];
            if (!link) continue;
            const originNode = graph.getNodeById(link.origin_id);
            const targetNode = graph.getNodeById(link.target_id);
            if (!originNode || !targetNode) continue;
            const p1 = originNode.getConnectionPos(false, link.origin_slot);
            const p2 = targetNode.getConnectionPos(true, link.target_slot);
            const dist = Math.max(Math.abs(p2[0] - p1[0]) * 0.5, 40);
            const c1 = [p1[0] + dist, p1[1]];
            const c2 = [p2[0] - dist, p2[1]];
            const color = (LGraphCanvas.link_type_colors[link.type] || '#8fd3ff');
            for (let k = 0; k < 2; k++) {
                const lt = (t + k * 0.5) % 1;
                const [x, y] = cubicBezierPoint(p1, c1, c2, p2, lt);
                ctx.save();
                ctx.beginPath();
                ctx.arc(x, y, 3.2, 0, Math.PI * 2);
                ctx.fillStyle = color;
                ctx.shadowColor = color;
                ctx.shadowBlur = 10;
                ctx.globalAlpha = 0.55 + 0.45 * Math.sin(lt * Math.PI);
                ctx.fill();
                ctx.restore();
            }
        }
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

    const NODE_MENU_TYPES = { text: 'nenai/text', camera_angle: 'nenai/camera_angle', load_image: 'nenai/load_image', image: 'nenai/image', video: 'nenai/video', edit: 'nenai/edit', audio: 'nenai/audio', muleai: 'nenai/muleai' };

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
                selectNodeOnly(node);
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
        // litegraph.js 內建打包了約 200 個範例節點（basic/math/audio/graphics/
        // network... 等），會出現在「新增節點」搜尋選單與拖線放空白處的選單裡，
        // 跟這個平台的 AI 生成節點完全無關；先清空原生登記表，只留下我們自己
        // 註冊的 5 種節點。
        LiteGraph.registered_node_types = {};
        registerNodeTypes();
        // 標題列改細一點、字放小，讓它讀起來更像截圖參考裡那種淡淡的標籤，
        // 不是一個很搶眼的色塊——真正的視覺重點放在下面的圖片預覽
        LiteGraph.NODE_TITLE_HEIGHT = 24;
        LiteGraph.NODE_TEXT_SIZE = 12;
        // 依資料型別上色連線，方便一眼看出文字/圖片/影片/音訊的連線關係
        Object.assign(LGraphCanvas.link_type_colors, {
            string: '#facc15', image: '#4f8cff', video: '#4ade80', audio: '#f87171',
        });

        graph = new LGraph();
        lgCanvas = new LGraphCanvas('#litegraphCanvas', graph);
        // 細緻的暗色點狀網格背景，取代 litegraph 預設的方格網格圖
        lgCanvas.background_image =
            "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='28' height='28'%3E" +
            "%3Ccircle cx='2' cy='2' r='1.4' fill='%233a3a3d'/%3E%3C/svg%3E";
        lgCanvas.render_canvas_border = false;
        lgCanvas.links_render_mode = LiteGraph.SPLINE_LINK;
        lgCanvas.onDrawForeground = drawFlowingLinks;
        // litegraph 原生的畫布右鍵選單（Add Node / Add Group）跟這個平台的節點選單是
        // 兩套獨立系統：Add Group 建出來的原生 Group 框沒有我們自訂的關閉鈕，選不到
        // 也刪不掉，會卡在畫面上——直接關閉原生右鍵選單，改成右鍵開啟自訂的新增節點選單
        lgCanvas.getCanvasMenuOptions = () => null;
        document.getElementById('litegraphCanvas').addEventListener('contextmenu', (e) => {
            e.preventDefault();
            openQuickAddMenu(null, null, e.clientX, e.clientY);
        });

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
