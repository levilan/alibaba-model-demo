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

    let MODELS = { text: [], image: [], video: [], muleai: [], voice: { asr: [], tts: [] } };
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

    // MODELS.voice 跟其他分類不一樣，是巢狀的 {asr:[...], tts:[...]}，不能直接
    // 套用假設「MODELS[category] 本身就是陣列」的 getModelsFor()。
    function getVoiceTtsModels() {
        return (MODELS.voice && MODELS.voice.tts) || [];
    }

    // qwen 的兩個 TTS 模型各自只支援自己專屬的音色清單，不能混用；3 個 gemini
    // 模型則共用同一組官方音色（見 app.py 的 _GEMINI_TTS_VOICES）。
    function voicesForTtsModel(modelId) {
        const m = getVoiceTtsModels().find(x => x.id === modelId);
        return (m && m.voices) || [];
    }

    // ── 專案存檔還原共用小工具 ────────────────────────────────────
    // 動態新增的「參考圖 N」插槽不是 LiteGraph 原生 properties 的一部分，配置
    // 還原後 this.inputs 陣列本身雖然會自動復原，但我們自己追蹤插槽索引用的
    // this.refSlots 陣列不會——直接從還原後的 inputs 名稱規律反推回來即可，
    // 不需要額外序列化。
    function _collectRefSlots(node) {
        const slots = [];
        (node.inputs || []).forEach((inp, i) => { if (/^參考圖 \d+$/.test(inp.name)) slots.push(i); });
        return slots;
    }
    // 生成結果的圖片/影片網址存在各節點自己的實例欄位（this.imageUrl/videoUrl），
    // 不是 LiteGraph 原生 properties 的一部分，必須靠 onSerialize/onConfigure
    // 手動存取還原；上傳圖片產生的 blob: 網址則刻意不做這件事——分頁重新整理後
    // 瀏覽器記憶體裡的 Blob 資料已經不存在，存下去也是無效網址。
    function _restoreGenResult(node, cv) {
        if (!cv) return;
        // 組圖模式（enable_sequential）存的是多張圖片網址；優先判斷這個，否則落回單張圖片
        if (cv.imageUrls && cv.imageUrls.length) {
            node.imageUrls = cv.imageUrls; node.imageUrl = cv.imageUrls[0];
            setPreviewImageGallery(node, cv.imageUrls); if (node.statusEl) node.statusEl.textContent = '完成';
        } else if (cv.imageUrl) {
            node.imageUrl = cv.imageUrl; setPreviewImage(node, cv.imageUrl); if (node.statusEl) node.statusEl.textContent = '完成';
        }
        if (cv.videoUrl) { node.videoUrl = cv.videoUrl; setPreviewVideo(node, cv.videoUrl); if (node.statusEl) node.statusEl.textContent = '完成'; }
        if (cv.audioUrl) { node.audioUrl = cv.audioUrl; setPreviewAudio(node, cv.audioUrl); if (node.statusEl) node.statusEl.textContent = '完成'; }
    }
    // 之前發生過使用者把 prompt 節點接上後，textarea 仍顯示舊的手動輸入文字，
    // 誤以為連線生效、生成時卻其實還是用手動文字（因為看不出「目前真正會送出
    // 的內容」）——這裡讓 textarea 在連線時即時鏡射上游輸出、鎖成唯讀，斷線
    // 後才還原成可編輯，讓「現在用的是連接文字還是手動輸入」一眼就能分辨。
    function _syncPromptTextarea(node, textarea, inputSlot) {
        const connected = !!node.getInputNode(inputSlot);
        if (connected) {
            const val = node.getInputData(inputSlot);
            const text = (val != null && val !== '') ? String(val) : '（等待上游節點輸出…）';
            if (textarea.value !== text) textarea.value = text;
            if (!textarea.disabled) {
                textarea.disabled = true;
                textarea.classList.add('cv-textarea-linked');
            }
        } else if (textarea.disabled) {
            textarea.disabled = false;
            textarea.classList.remove('cv-textarea-linked');
            textarea.value = node.properties.prompt != null ? node.properties.prompt : (node.properties.text || '');
        }
    }
    // 相機角度節點只要透過任何一個輸入插槽接上（不限定 prompt 插槽，接參考圖/
    // first_frame/來源圖等都算），就自動把它的 prompt 輸出當前綴帶入，不用另外
    // 手動拉一條 prompt 連線——避免使用者以為「接了圖就會自動帶 prompt」卻其實
    // 沒有的落差。用 duck-typing 找「有 _buildPrompt 方法」的上游節點（目前只
    // 有 CameraAngleNode 有），這樣未來如果有其他節點想比照辦理也不用改這裡。
    function _autoCameraAnglePrefix(node, promptSlot) {
        const promptSrc = promptSlot != null ? node.getInputNode(promptSlot) : null;
        for (let i = 0; i < (node.inputs || []).length; i++) {
            if (i === promptSlot) continue;
            const src = node.getInputNode(i);
            if (src && src !== promptSrc && typeof src._buildPrompt === 'function') return src._buildPrompt();
        }
        return '';
    }
    function _combinePrompt(node, promptSlot, basePrompt) {
        const prefix = _autoCameraAnglePrefix(node, promptSlot);
        return prefix ? `${prefix} ${basePrompt || ''}`.trim() : basePrompt;
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

    // 跟 buildSelect 一樣，但選項需要「值跟顯示文字不同」時使用（例如空字串值要顯示成「auto（自動）」）
    function buildLabeledSelect(options, current, onChange) {
        const sel = el('select');
        sel.innerHTML = options.map(([v, label]) => `<option value="${v}"${v === current ? ' selected' : ''}>${label}</option>`).join('');
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
        text: { type: 'nenai/text', label: '文字 Text' },
        camera_angle: { type: 'nenai/camera_angle', label: '相機角度 Camera Angle' },
        load_image: { type: 'nenai/load_image', label: '上傳圖片 Load Image' },
        image: { type: 'nenai/image', label: '圖片 Image' },
        video: { type: 'nenai/video', label: '影片 Video' },
        video_edit: { type: 'nenai/video_edit', label: '影片編輯 Video Edit' },
        video_animate: { type: 'nenai/video_animate', label: '動作動畫 Animate' },
        edit: { type: 'nenai/edit', label: '圖像編輯 Editing' },
        audio: { type: 'nenai/audio', label: '語音 TTS' },
        muleai: { type: 'nenai/muleai', label: 'MuleAI Spicy' },
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
                // 用 try/finally 確保無論建立節點/連線過程中有沒有出錯，選單一定會關掉——
                // 之前發生過中途拋例外導致 closeQuickAddMenu() 沒被執行、選單卡在畫面上
                try {
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
                } catch (err) {
                    console.error('新增節點失敗', err);
                    showToast('新增節點失敗：' + err.message);
                } finally {
                    closeQuickAddMenu();
                }
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
                    // 從左上角縮放（預設是置中縮放，元素越寬，縮放時往內縮的偏移量越明顯，
                    // 導致節點本體跟固定尺寸的設定浮層在非 100% 縮放時對不齊）
                    panel.style.transformOrigin = 'top left';
                    panel.style.transform = `scale(${scale})`;
                }
            } else if (node._contentHeight) {
                node.size[1] = node._contentHeight + zoneH;
            }
            if (node._closeBtn) {
                const [sx, sy] = toScreen(node.pos[0] + node.size[0], node.pos[1] - titleH);
                node._closeBtn.style.left = (sx - 22 * scale) + 'px';
                node._closeBtn.style.top = (sy + 4 * scale) + 'px';
                node._closeBtn.style.transformOrigin = 'top left';
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
                btn.style.transformOrigin = 'top left';
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
        _clearProgressTimer(node);
        node._previewBox.innerHTML = `<span class="cv-empty">${text}</span>`;
    }

    function _clearProgressTimer(node) {
        if (node._progressTimer) { clearInterval(node._progressTimer); node._progressTimer = null; }
        // 每個 setPreview* 函式開頭都會呼叫這個，順便把組圖模式專用的 grid 排版
        // class 清掉，避免切回單張圖片/影片/音檔時排版殘留成網格樣式
        node._previewBox.classList.remove('cv-preview-grid');
    }

    // 圖片/影片生成都是「送出後輪詢」，後端沒有真正的百分比可回報——用指數趨緩
    // 曲線模擬一個持續前進、但夾在 96% 不會提早衝到 100% 的假進度（時間常數用
    // estimateSec 的 0.6 倍抓，讓進度大約在預估時間附近落在七八成），配合 CSS
    // 跑動的漸層與呼吸光暈，讓使用者在等待 1~3 分鐘的影片生成時能感覺「還在動」
    // 而不是卡住。同一次生成流程若有「送出中→生成中」兩階段，用 updateProgress-
    // Label() 只換文字、不重置已經跑到一半的進度與計時，避免視覺上倒退
    function setPreviewProgress(node, label, estimateSec) {
        _clearProgressTimer(node);
        const box = node._previewBox;
        box.innerHTML = `
            <div class="cv-progress">
                <div class="cv-progress-label"></div>
                <div class="cv-progress-track"><div class="cv-progress-fill"></div></div>
                <div class="cv-progress-meta"><span class="cv-progress-pct"></span><span class="cv-progress-time"></span></div>
            </div>`;
        node._progressLabelEl = box.querySelector('.cv-progress-label');
        node._progressLabelEl.textContent = label;
        const fill = box.querySelector('.cv-progress-fill');
        const pctEl = box.querySelector('.cv-progress-pct');
        const timeEl = box.querySelector('.cv-progress-time');
        const start = Date.now();
        const tau = Math.max(3, estimateSec * 0.6) * 1000;
        const tick = () => {
            const elapsedMs = Date.now() - start;
            const pct = Math.min(96, Math.round((1 - Math.exp(-elapsedMs / tau)) * 96));
            fill.style.width = pct + '%';
            pctEl.textContent = pct + '%';
            timeEl.textContent = Math.round(elapsedMs / 1000) + 's';
        };
        tick();
        node._progressTimer = setInterval(tick, 200);
    }

    function updateProgressLabel(node, label) {
        if (node._progressLabelEl) node._progressLabelEl.textContent = label;
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
        _clearProgressTimer(node);
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

    // 萬相 2.7「組圖模式」（enable_sequential）一次會回多張連貫圖片，用簡單的
    // 兩欄網格排版全部顯示，每張各自可放大/下載，而不是只顯示第一張。
    function setPreviewImageGallery(node, urls) {
        _clearProgressTimer(node);
        node._previewBox.innerHTML = '';
        node._previewBox.classList.add('cv-preview-grid');
        urls.forEach((url, i) => {
            const item = el('div', 'cv-gallery-item');
            const img = el('img');
            img.src = url;
            img.addEventListener('mousedown', (e) => e.stopPropagation());
            img.addEventListener('click', () => openLightbox('image', url));
            item.appendChild(img);
            const dl = el('a', 'cv-dl-btn cv-gallery-dl', '⬇');
            dl.href = url; dl.download = `image_${i + 1}.png`; dl.target = '_blank';
            dl.addEventListener('mousedown', (e) => e.stopPropagation());
            item.appendChild(dl);
            node._previewBox.appendChild(item);
        });
    }

    function setPreviewVideo(node, url) {
        // 影片本身有原生播放控制列，不能整個蓋 click 監聽（會跟播放/拖曳衝突），
        // 改用獨立的「⤢ 放大」按鈕開燈箱
        _clearProgressTimer(node);
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

    function setPreviewAudio(node, url) {
        _clearProgressTimer(node);
        node._previewBox.innerHTML = '';
        const audio = el('audio');
        audio.src = url; audio.controls = true;
        audio.addEventListener('mousedown', (e) => e.stopPropagation());
        node._previewBox.appendChild(audio);
        const dl = el('a', 'cv-dl-btn', '⬇ 下載');
        dl.href = url; dl.download = 'audio.mp3'; dl.target = '_blank';
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
            <button class="cv-generate cv-analyze-btn" style="display:none">分析已連接的圖片</button>
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
            this.setOutputData(0, this.generatedText);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            showToast('文字生成失敗：' + e.message);
        }
    };
    TextPromptNode.prototype.analyzeImage = async function () {
        const imgUrl = this.getInputData(0, true);
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
            this.setOutputData(0, this.generatedText);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            showToast('圖片分析失敗：' + e.message);
        }
    };
    TextPromptNode.prototype.onSerialize = function (o) {
        o.cv = { generatedText: this.generatedText || null };
    };
    TextPromptNode.prototype.onConfigure = function (o) {
        this.textarea.value = this.properties.text || '';
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        const cv = o.cv || {};
        if (cv.generatedText) {
            this.generatedText = cv.generatedText;
            this.outputBox.textContent = cv.generatedText;
            this.outputBox.style.display = '';
            this.statusEl.textContent = '完成（輸出已改為生成結果）';
        }
    };
    TextPromptNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Camera Angle（仿 ComfyUI-qwenmultiangle 的拖曳式相機角度控制介面。
    // 比照原專案設計：視覺化調整方位角/仰角/縮放後，輸出符合該專案 LoRA 慣例
    // 格式的 prompt，例如 "<sks> front view eye-level shot medium shot"，可直
    // 接接到下游圖片節點的 prompt 輸入做 i2i 生成；同時把輸入圖片原樣傳出，方
    // 便同一條線也接到「參考圖」輸入。注意：NenAI 平台沒有對應的 Multiple-
    // Angles-LoRA，一般模型不認得 <sks> 這種觸發詞，效果不會跟原專案一樣精準）
    // 用 2D SVG 取代原專案的 Three.js 3D 場景，互動邏輯（atan2 反推角度、環形/
    // 弧形限制拖曳範圍）比照原專案的 CameraWidget.ts ─────────────────────
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
        // 把接進來的圖片原樣傳出去，這樣同一條線可以同時接到下游節點的 prompt
        // 跟「參考圖」兩個輸入，做 i2i 生成
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
            const prompt = this._buildPrompt();
            this.outputBox.textContent = prompt;
            // 立刻寫入輸出資料，不等 LiteGraph 下一次執行迴圈才呼叫 onExecute——
            // 否則使用者調完角度馬上按生成，下游節點可能讀到還沒更新的舊資料
            this.setOutputData(0, prompt);
            this.setOutputData(1, this.getInputData(0) || null);
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
    // horizontal/vertical/zoom 都存在 this.properties，LiteGraph 還原設定時會自動
    // 覆蓋回去，呼叫 refresh() 就能一次把 SVG 控制點、卡片旋轉、數值輸入框、
    // prompt 文字全部重新同步，不需要額外的序列化資料
    CameraAngleNode.prototype.onConfigure = function () { this.refresh(); };
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
                <button class="cv-generate cv-load-btn">選擇檔案</button>
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
        // 立刻寫入輸出資料，不等 LiteGraph 下一次執行迴圈才呼叫 onExecute
        this.setOutputData(0, this.imageUrl);
    };
    LoadImageNode.prototype.onExecute = function () {
        this.setOutputData(0, this.imageUrl);
    };
    // 上傳的圖片是瀏覽器記憶體裡的 Blob，重新整理分頁後就不存在了，網址存下去
    // 也是無效的——還原專案時只能提示使用者重新選擇檔案
    LoadImageNode.prototype.onConfigure = function () {
        this.statusEl.textContent = '請重新選擇圖片檔案（瀏覽器重新整理後上傳檔案不會保留）';
    };
    LoadImageNode.prototype.onRemoved = function () {
        if (this.imageUrl && this.imageUrl.startsWith('blob:')) URL.revokeObjectURL(this.imageUrl);
        sharedOnRemoved.call(this);
    };

    // ── Node: Image（文生圖，t2i 模型） ─────────────────────────
    // 圖片節點依「是否連接參考圖」自動切換 t2i（純文生圖）/ i2i（拿參考圖做
    // 圖像生成，實際呼叫 /api/image/edit）——這樣使用者可以直接把一個圖片節點
    // 的輸出拉線接到另一個圖片節點的「參考圖」輸入，做「用圖像生成圖像」。
    // 可用「+ 新增參考圖輸入」加到最多 6 張（後端 qwen-image-2.0 融合模型最多
    // 吃 3 張、其餘模型最多 9 張，UI 統一給 6 張上限）。
    const IMAGE_MAX_REF_SLOTS = 6;
    function ImageGenNode() {
        this.addInput('prompt', 'string');
        this.addInput('參考圖 1', 'image');
        this.addOutput('image', 'image');
        this.refSlots = [1];
        const models = getModelsFor('image', 't2i');
        this.properties = {
            model: (models[0] && models[0].id) || '', prompt: '', size: '1024*1024', status: '',
            aspect_ratio: '', enable_sequential: false, seq_n: 4,
            quality: '', background: '', output_format: '',
        };
        this.imageUrl = null;
        this.imageUrls = null;
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
                <div class="cv-cam-prefix-hint" style="display:none"></div>
                <div class="cv-size-group">
                    <label>尺寸</label>
                    <div class="cv-size-slot"></div>
                </div>
                <div class="cv-ar-group" style="display:none">
                    <label>圖片比例<span class="cv-hint">（Gemini 專用，靠 prompt 文字模擬控制）</span></label>
                    <div class="cv-ar-slot"></div>
                </div>
                <label class="cv-check-row cv-seq-row" style="display:none"><input type="checkbox" class="cv-seq-check"> 組圖模式（一次生成連貫故事圖組）</label>
                <div class="cv-seq-n-group" style="display:none">
                    <label>最大張數（實際張數由模型決定）<span class="cv-dur-val cv-seq-n-val">4</span></label>
                    <input type="range" class="cv-seq-n-slider" min="1" max="12" step="1" value="4">
                </div>
                <div class="cv-gpt-group" style="display:none">
                    <label>品質 (quality)</label>
                    <div class="cv-quality-slot"></div>
                    <label>背景 (background)</label>
                    <div class="cv-bg-slot"></div>
                    <label>輸出格式 (output_format)</label>
                    <div class="cv-fmt-slot"></div>
                </div>
                <button class="cv-add-ref-btn">+ 新增參考圖輸入</button>
                <button class="cv-generate">▶ 生成圖片</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        this.modeHintEl = panel.querySelector('.cv-mode-hint');
        this.camHintEl = panel.querySelector('.cv-cam-prefix-hint');
        this.seqCheck = panel.querySelector('.cv-seq-check');
        this.seqCheck.addEventListener('mousedown', (e) => e.stopPropagation());
        this.seqCheck.addEventListener('change', () => {
            // wireConfigOverlay() 執行後 .cv-controls（含 .cv-seq-n-group）已經搬到
            // this._configOverlay，不再是 panel 的子節點，這裡不能用建構子當時的
            // panel 變數查詢，要用當下真正裝著表單的容器
            this.properties.enable_sequential = this.seqCheck.checked;
            const container = this._configOverlay || this._domPanel;
            container.querySelector('.cv-seq-n-group').style.display = this.seqCheck.checked ? '' : 'none';
        });
        this.seqNSlider = panel.querySelector('.cv-seq-n-slider');
        this.seqNValEl = panel.querySelector('.cv-seq-n-val');
        this.seqNSlider.addEventListener('input', () => {
            this.properties.seq_n = parseInt(this.seqNSlider.value);
            this.seqNValEl.textContent = this.seqNSlider.value;
        });
        panel.querySelector('.cv-add-ref-btn').addEventListener('click', () => this._addRefSlot());
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => {
            this.properties.model = v;
            const sizes = sizesForModel('image', v);
            this._rebuildSizeSelect(sizes);
            this._syncModelExtras();
        });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);
        this._rebuildSizeSelect(sizesForModel('image', this.properties.model));

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
        this._syncModelExtras();
    }
    ImageGenNode.title = '圖片 Image';
    ImageGenNode.prototype._addRefSlot = function () {
        if (this.refSlots.length >= IMAGE_MAX_REF_SLOTS) { showToast(`最多 ${IMAGE_MAX_REF_SLOTS} 張參考圖`); return; }
        const w = this.size[0];
        this.addInput('參考圖 ' + (this.refSlots.length + 1), 'image');
        this.refSlots.push(this.inputs.length - 1);
        // addInput() 內部會呼叫 setSize(computeSize())，用原生插槽文字寬度覆蓋掉
        // 我們自訂的節點寬度，導致節點框框變窄變形——加完插槽後把寬度改回來
        this.size[0] = w;
        lgCanvas.setDirty(true, true);
    };
    ImageGenNode.prototype._rebuildSizeSelect = function (sizes) {
        // wireConfigOverlay() 執行後，.cv-size-slot 所在的表單控制區塊已經搬到
        // this._configOverlay（獨立浮層），不再是 this._domPanel 的子節點——
        // 建構子第一次呼叫此方法時 wireConfigOverlay 還沒跑，_configOverlay 尚
        // 不存在，才需要用 _domPanel 當備援
        const container = this._configOverlay || this._domPanel;
        const slot = container.querySelector('.cv-size-slot');
        if (!sizes.includes(this.properties.size)) this.properties.size = sizes[0];
        slot.innerHTML = '';
        this.sizeSelect = buildSelect(sizes, this.properties.size, (v) => { this.properties.size = v; });
        slot.appendChild(this.sizeSelect);
    };
    // Gemini 的「圖片比例」、萬相 2.7 的「組圖模式」、GPT Image 的 quality/
    // background/output_format 都是依目前選到的模型（且部分僅限 T2I）才顯示，
    // 集中在這裡統一處理可見性與選單重建，供建構子/模型切換/連線變化/還原共用。
    ImageGenNode.prototype._syncModelExtras = function () {
        const mode = this._detectMode();
        const modelInfo = getModelsFor('image', mode).find(m => m.id === this.properties.model) || {};
        const container = this._configOverlay || this._domPanel;

        // Gemini 圖片模型不支援 size 參數，隱藏尺寸選單（跟主測試台一致）
        container.querySelector('.cv-size-group').style.display = modelInfo.no_size ? 'none' : '';

        const arGroup = container.querySelector('.cv-ar-group');
        const aspectRatios = (mode === 't2i' && modelInfo.aspect_ratios) || [];
        arGroup.style.display = aspectRatios.length ? '' : 'none';
        if (aspectRatios.length) {
            if (!aspectRatios.includes(this.properties.aspect_ratio)) this.properties.aspect_ratio = aspectRatios[0];
            const slot = container.querySelector('.cv-ar-slot');
            slot.innerHTML = '';
            this.arSelect = buildSelect(aspectRatios, this.properties.aspect_ratio, (v) => { this.properties.aspect_ratio = v; });
            slot.appendChild(this.arSelect);
        }

        const supportsSeq = mode === 't2i' && !!modelInfo.supports_sequential;
        container.querySelector('.cv-seq-row').style.display = supportsSeq ? '' : 'none';
        if (!supportsSeq) this.properties.enable_sequential = false;
        this.seqCheck.checked = this.properties.enable_sequential;
        container.querySelector('.cv-seq-n-group').style.display =
            (supportsSeq && this.properties.enable_sequential) ? '' : 'none';

        const gptGroup = container.querySelector('.cv-gpt-group');
        gptGroup.style.display = modelInfo.supports_gpt_params ? '' : 'none';
        if (modelInfo.supports_gpt_params) {
            const qSlot = container.querySelector('.cv-quality-slot');
            qSlot.innerHTML = '';
            this.qualitySelect = buildLabeledSelect(
                [['', 'auto（自動）'], ['low', 'low'], ['medium', 'medium'], ['high', 'high']],
                this.properties.quality, (v) => { this.properties.quality = v; });
            qSlot.appendChild(this.qualitySelect);
            const bgSlot = container.querySelector('.cv-bg-slot');
            bgSlot.innerHTML = '';
            this.backgroundSelect = buildLabeledSelect(
                [['', 'auto（自動）'], ['opaque', 'opaque（不透明）'], ['transparent', 'transparent（透明）']],
                this.properties.background, (v) => { this.properties.background = v; });
            bgSlot.appendChild(this.backgroundSelect);
            const fmtSlot = container.querySelector('.cv-fmt-slot');
            fmtSlot.innerHTML = '';
            this.outputFormatSelect = buildLabeledSelect(
                [['', '預設'], ['png', 'PNG'], ['jpeg', 'JPEG'], ['webp', 'WEBP']],
                this.properties.output_format, (v) => { this.properties.output_format = v; });
            fmtSlot.appendChild(this.outputFormatSelect);
        }
    };
    ImageGenNode.prototype._detectMode = function () {
        return this.refSlots.some(i => !!this.getInputNode(i)) ? 'i2i' : 't2i';
    };
    ImageGenNode.prototype.onExecute = function () {
        _syncPromptTextarea(this, this.textarea, 0);
        const camPrefix = _autoCameraAnglePrefix(this, 0);
        this.camHintEl.style.display = camPrefix ? '' : 'none';
        if (camPrefix) this.camHintEl.textContent = '將自動附加相機角度 prompt：' + camPrefix;
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
        this._syncModelExtras();
        this.modeHintEl.textContent = mode === 'i2i' ? '（參考圖生成圖像）' : '（文生圖）';
    };
    ImageGenNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0, true);
        const basePrompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        const prompt = _combinePrompt(this, 0, basePrompt);
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        const mode = this._detectMode();
        this.statusEl.textContent = '生成中…';
        setPreviewProgress(this, '生成中…', 20);
        try {
            let res;
            if (mode === 'i2i') {
                const refUrls = this.refSlots.map(i => this.getInputData(i, true)).filter(Boolean);
                if (!refUrls.length) throw new Error('參考圖節點尚未生成完成，請先按上游圖片節點的「生成圖片」');
                const fd = new FormData();
                fd.append('model', this.properties.model);
                fd.append('prompt', prompt);
                fd.append('size', this.properties.size);
                fd.append('n', '1');
                if (this.properties.quality) fd.append('quality', this.properties.quality);
                if (this.properties.background) fd.append('background', this.properties.background);
                if (this.properties.output_format) fd.append('output_format', this.properties.output_format);
                for (let i = 0; i < refUrls.length; i++) {
                    fd.append(`image_${i + 1}`, await fetchAsBlob(refUrls[i]), `ref${i + 1}.png`);
                }
                res = await apiFetch('/api/image/edit', { method: 'POST', body: fd });
            } else {
                const body = { model: this.properties.model, prompt, size: this.properties.size, n: 1 };
                if (this.properties.aspect_ratio) body.aspect_ratio = this.properties.aspect_ratio;
                if (this.properties.enable_sequential) {
                    body.enable_sequential = true;
                    body.n = this.properties.seq_n;
                }
                if (this.properties.quality) body.quality = this.properties.quality;
                if (this.properties.background) body.background = this.properties.background;
                if (this.properties.output_format) body.output_format = this.properties.output_format;
                res = await apiFetch('/api/image/generate', { method: 'POST', body: JSON.stringify(body) });
            }
            const data = await res.json();
            if (!res.ok || !data.images || !data.images.length) throw new Error((data.error && (data.error.message || data.error)) || '生成失敗');
            const urls = data.images.map(img => img.local_path || img.url);
            this.imageUrl = urls[0];
            this.imageUrls = urls.length > 1 ? urls : null;
            this.statusEl.textContent = urls.length > 1 ? `完成（共 ${urls.length} 張）` : '完成';
            if (urls.length > 1) setPreviewImageGallery(this, urls); else setPreviewImage(this, this.imageUrl);
            // 輸出插槽固定是單張圖片，組圖模式下只往下游傳第一張，其餘的張只能在
            // 節點內的圖庫預覽/下載，這是既有 image 輸出型別（單一字串網址）的限制
            this.setOutputData(0, this.imageUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('圖片生成失敗：' + e.message);
        }
    };
    ImageGenNode.prototype.onSerialize = function (o) {
        o.cv = { imageUrl: this.imageUrl || null, imageUrls: this.imageUrls || null };
    };
    ImageGenNode.prototype.onConfigure = function (o) {
        this.textarea.value = this.properties.prompt || '';
        this.refSlots = _collectRefSlots(this);
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        this._rebuildSizeSelect(sizesForModel('image', this.properties.model));
        this._syncModelExtras();
        if (this.seqNSlider) { this.seqNSlider.value = this.properties.seq_n; this.seqNValEl.textContent = this.properties.seq_n; }
        _restoreGenResult(this, o.cv);
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
        const w = this.size[0];
        this.addInput('參考圖 ' + (this.refSlots.length + 1), 'image');
        this.refSlots.push(this.inputs.length - 1);
        // addInput() 內部會呼叫 setSize(computeSize())，用原生插槽文字寬度覆蓋掉
        // 我們自訂的節點寬度，導致節點框框變窄變形——加完插槽後把寬度改回來
        this.size[0] = w;
        // 強制立即重繪：insert 後畫布可能要等下一輪動畫幀才重新計算新插槽的實際
        // 螢幕座標，這段空窗期如果使用者剛好開始拖線，滑鼠命中判定可能用到還沒
        // 更新的舊座標，導致連線拖拉失敗
        lgCanvas.setDirty(true, true);
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
        _syncPromptTextarea(this, this.textarea, 0);
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
        const promptIn = this.getInputData(0, true);
        const basePrompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        const prompt = _combinePrompt(this, 0, basePrompt);
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        const mode = this._detectMode();
        if (mode === 'r2v' && this.getInputNode(1)) {
            showToast('已連接參考圖，first_frame 的圖片將被忽略（r2v 與 i2v 是互斥的兩種生成模式）');
        }
        this.statusEl.textContent = '送出中…';
        setPreviewProgress(this, '送出中…', 90);
        try {
            const fd = new FormData();
            fd.append('model', this.properties.model);
            fd.append('prompt', prompt);
            fd.append('resolution', this.properties.resolution);
            fd.append('duration', String(this.properties.duration));
            let endpoint = '/api/video/t2v';
            if (mode === 'r2v') {
                endpoint = '/api/video/r2v';
                const refUrls = this.refSlots.map(i => this.getInputData(i, true)).filter(Boolean);
                if (!refUrls.length) throw new Error('參考圖節點尚未生成完成，請先按上游圖片節點的「生成圖片」');
                for (const url of refUrls) {
                    fd.append('reference_files', await fetchAsBlob(url), 'ref.png');
                }
            } else if (mode === 'i2v') {
                endpoint = '/api/video/i2v';
                const firstFrameUrl = this.getInputData(1, true);
                const lastFrameUrl = this.getInputData(2, true);
                fd.append('i2v_mode', lastFrameUrl ? 'first_last_frame' : 'first_frame');
                fd.append('first_frame', await fetchAsBlob(firstFrameUrl), 'first_frame.png');
                if (lastFrameUrl) fd.append('last_frame', await fetchAsBlob(lastFrameUrl), 'last_frame.png');
            }
            const res = await apiFetch(endpoint, { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error((data.error && (data.error.message || data.error)) || '任務建立失敗');
            this.statusEl.textContent = '生成中…';
            updateProgressLabel(this, '生成中…（可能需要 1～數分鐘）');
            const result = await pollVideoTask(data.task_id);
            this.videoUrl = result.local_path || result.video_url;
            this.statusEl.textContent = '完成';
            setPreviewVideo(this, this.videoUrl);
            this.setOutputData(0, this.videoUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('影片生成失敗：' + e.message);
        }
    };
    VideoGenNode.prototype.onSerialize = function (o) {
        o.cv = { videoUrl: this.videoUrl || null };
    };
    VideoGenNode.prototype.onConfigure = function (o) {
        this.textarea.value = this.properties.prompt || '';
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        if (this.resSelect) this.resSelect.value = this.properties.resolution;
        if (this.durSlider) { this.durSlider.value = this.properties.duration; this.durValEl.textContent = this.properties.duration; }
        this.refSlots = _collectRefSlots(this);
        _restoreGenResult(this, o.cv);
    };
    VideoGenNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Video Edit（wan2.7-videoedit，文字/參考圖驅動編輯既有影片）──────
    // 來源影片可以接 video 型別輸入（例如接影片節點的輸出），也可以直接在節點
    // 內上傳本機影片檔案——沒有連線時就用上傳的檔案。
    const VEDIT_MAX_REF_SLOTS = 3;
    function VideoEditNode() {
        this.addInput('prompt', 'string');
        this.addInput('video', 'video');
        this.addOutput('video', 'video');
        this.refSlots = [];
        this.localVideoUrl = null;
        const models = getModelsFor('video', 'vedit');
        this.properties = {
            model: (models[0] && models[0].id) || '', prompt: '', resolution: '1080P',
            ratio: '', audioSetting: 'auto', duration: 0, status: '',
        };
        this.videoUrl = null;
        this._contentHeight = 620;
        this.size = [320, 620];
        this.color = '#1f2f3a'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>模型</label>
                <div class="cv-select-slot"></div>
                <label>Prompt<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
                <textarea placeholder="輸入編輯指示…"></textarea>
                <label>來源影片<span class="cv-hint">（未連接 video 輸入時使用）</span></label>
                <input type="file" class="cv-vedit-file" accept="video/*" style="display:none">
                <button class="cv-add-ref-btn cv-vedit-upload-btn">選擇影片檔案</button>
                <label>畫面比例</label>
                <div class="cv-ratio-slot"></div>
                <label>音訊設定</label>
                <div class="cv-audio-slot"></div>
                <button class="cv-add-ref-btn cv-add-ref-vedit-btn">+ 新增參考圖輸入</button>
                <button class="cv-generate cv-submit-btn">▶ 編輯影片</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        this.fileInput = panel.querySelector('.cv-vedit-file');
        this.uploadBtn = panel.querySelector('.cv-vedit-upload-btn');
        this.uploadBtn.addEventListener('mousedown', (e) => e.stopPropagation());
        this.uploadBtn.addEventListener('click', () => this.fileInput.click());
        this.fileInput.addEventListener('mousedown', (e) => e.stopPropagation());
        this.fileInput.addEventListener('change', () => this._onFile());
        panel.querySelector('.cv-add-ref-vedit-btn').addEventListener('click', () => this._addRefSlot());
        panel.querySelector('.cv-submit-btn').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => { this.properties.model = v; });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        this.ratioSelect = buildSelect(['', '16:9', '9:16', '1:1', '4:3', '3:4'], this.properties.ratio, (v) => { this.properties.ratio = v; });
        panel.querySelector('.cv-ratio-slot').appendChild(this.ratioSelect);

        this.audioSelect = buildSelect(['auto', 'origin'], this.properties.audioSetting, (v) => { this.properties.audioSetting = v; });
        panel.querySelector('.cv-audio-slot').appendChild(this.audioSelect);

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
    }
    VideoEditNode.title = '影片編輯 Video Edit';
    VideoEditNode.prototype._addRefSlot = function () {
        if (this.refSlots.length >= VEDIT_MAX_REF_SLOTS) { showToast(`最多 ${VEDIT_MAX_REF_SLOTS} 張參考圖`); return; }
        const w = this.size[0];
        this.addInput('參考圖 ' + (this.refSlots.length + 1), 'image');
        this.refSlots.push(this.inputs.length - 1);
        this.size[0] = w;
        lgCanvas.setDirty(true, true);
    };
    VideoEditNode.prototype._onFile = function () {
        const file = this.fileInput.files[0];
        if (!file) return;
        if (this.localVideoUrl && this.localVideoUrl.startsWith('blob:')) URL.revokeObjectURL(this.localVideoUrl);
        this.localVideoUrl = URL.createObjectURL(file);
        this.statusEl.textContent = '已選擇：' + file.name;
    };
    VideoEditNode.prototype.onExecute = function () {
        _syncPromptTextarea(this, this.textarea, 0);
        this.setOutputData(0, this.videoUrl);
    };
    VideoEditNode.prototype.generate = async function () {
        const promptIn = this.getInputData(0, true);
        const basePrompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        const prompt = _combinePrompt(this, 0, basePrompt);
        const videoUrl = this.getInputData(1, true) || this.localVideoUrl;
        if (!videoUrl) { showToast('請上傳或連接一段來源影片'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        this.statusEl.textContent = '送出中…';
        setPreviewProgress(this, '送出中…', 90);
        try {
            const fd = new FormData();
            fd.append('model', this.properties.model);
            fd.append('prompt', prompt || '');
            fd.append('resolution', this.properties.resolution);
            fd.append('audio_setting', this.properties.audioSetting);
            if (this.properties.ratio) fd.append('ratio', this.properties.ratio);
            fd.append('video', await fetchAsBlob(videoUrl), 'source.mp4');
            const refUrls = this.refSlots.map(i => this.getInputData(i, true)).filter(Boolean);
            for (let i = 0; i < refUrls.length; i++) {
                fd.append(`reference_image_${i + 1}`, await fetchAsBlob(refUrls[i]), `ref${i + 1}.png`);
            }
            const res = await apiFetch('/api/video/vedit', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error((data.error && (data.error.message || data.error)) || '任務建立失敗');
            this.statusEl.textContent = '生成中…';
            updateProgressLabel(this, '生成中…（可能需要 1～數分鐘）');
            const result = await pollVideoTask(data.task_id);
            this.videoUrl = result.local_path || result.video_url;
            this.statusEl.textContent = '完成';
            setPreviewVideo(this, this.videoUrl);
            this.setOutputData(0, this.videoUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('影片編輯失敗：' + e.message);
        }
    };
    VideoEditNode.prototype.onSerialize = function (o) {
        o.cv = { videoUrl: this.videoUrl || null };
    };
    VideoEditNode.prototype.onConfigure = function (o) {
        this.textarea.value = this.properties.prompt || '';
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        if (this.ratioSelect) this.ratioSelect.value = this.properties.ratio;
        if (this.audioSelect) this.audioSelect.value = this.properties.audioSetting;
        this.refSlots = _collectRefSlots(this);
        _restoreGenResult(this, o.cv);
        // 本機上傳的來源影片是 blob:，重新整理後不會保留，需要使用者重新選擇
        if (!this.getInputNode(1) && !this.localVideoUrl) {
            this.statusEl.textContent = '若使用本機上傳來源影片，重新整理後需重新選擇檔案';
        }
    };
    VideoEditNode.prototype.onRemoved = function () {
        if (this.localVideoUrl && this.localVideoUrl.startsWith('blob:')) URL.revokeObjectURL(this.localVideoUrl);
        sharedOnRemoved.call(this);
    };

    // ── Node: Animate（wan2.2-animate-mix 視頻換人 / wan2.2-animate-move 圖生動作）──
    // 把人物圖片套進參考影片：mix 保留原場景與動作、換掉人物；move 把參考影片的
    // 動作/表情遷移到人物圖片上。沒有 prompt，靠 mode + check_image 控制。
    function VideoAnimateNode() {
        this.addInput('人物圖片', 'image');
        this.addInput('參考影片', 'video');
        this.addOutput('video', 'video');
        const models = getModelsFor('video', 'animate');
        this.properties = { model: (models[0] && models[0].id) || '', mode: 'wan-std', checkImage: true, status: '' };
        this.videoUrl = null;
        this._contentHeight = 420;
        this.size = [300, 420];
        this.color = '#1f2f3a'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>模型</label>
                <div class="cv-select-slot"></div>
                <label>服務模式</label>
                <div class="cv-mode-slot"></div>
                <label class="cv-check-row"><input type="checkbox" class="cv-check-image" checked> 圖片預檢查</label>
                <button class="cv-generate cv-submit-btn">▶ 生成動畫</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.statusEl = panel.querySelector('.cv-status');
        this.checkImageEl = panel.querySelector('.cv-check-image');
        this.checkImageEl.addEventListener('mousedown', (e) => e.stopPropagation());
        this.checkImageEl.addEventListener('change', () => { this.properties.checkImage = this.checkImageEl.checked; });
        panel.querySelector('.cv-submit-btn').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => { this.properties.model = v; });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        this.modeSelect = buildSelect(['wan-std', 'wan-pro'], this.properties.mode, (v) => { this.properties.mode = v; });
        panel.querySelector('.cv-mode-slot').appendChild(this.modeSelect);

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
    }
    VideoAnimateNode.title = '動作動畫 Animate';
    VideoAnimateNode.prototype.onExecute = function () {
        this.setOutputData(0, this.videoUrl);
    };
    VideoAnimateNode.prototype.generate = async function () {
        const imgUrl = this.getInputData(0, true);
        const vidUrl = this.getInputData(1, true);
        if (!imgUrl) { showToast('請先連接人物圖片'); return; }
        if (!vidUrl) { showToast('請先連接參考影片'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        this.statusEl.textContent = '送出中…';
        setPreviewProgress(this, '送出中…', 90);
        try {
            const fd = new FormData();
            fd.append('model', this.properties.model);
            fd.append('mode', this.properties.mode);
            fd.append('check_image', this.properties.checkImage);
            fd.append('image', await fetchAsBlob(imgUrl), 'person.png');
            fd.append('video', await fetchAsBlob(vidUrl), 'ref.mp4');
            const res = await apiFetch('/api/video/animate', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error((data.error && (data.error.message || data.error)) || '任務建立失敗');
            this.statusEl.textContent = '生成中…';
            updateProgressLabel(this, '生成中…（可能需要 1～數分鐘）');
            const result = await pollVideoTask(data.task_id);
            this.videoUrl = result.local_path || result.video_url;
            this.statusEl.textContent = '完成';
            setPreviewVideo(this, this.videoUrl);
            this.setOutputData(0, this.videoUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('動作動畫生成失敗：' + e.message);
        }
    };
    VideoAnimateNode.prototype.onSerialize = function (o) {
        o.cv = { videoUrl: this.videoUrl || null };
    };
    VideoAnimateNode.prototype.onConfigure = function (o) {
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        if (this.modeSelect) this.modeSelect.value = this.properties.mode;
        if (this.checkImageEl) this.checkImageEl.checked = !!this.properties.checkImage;
        _restoreGenResult(this, o.cv);
    };
    VideoAnimateNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Editing（i2i 圖像編輯，需連接一張輸入圖片） ───────────
    function ImageEditNode() {
        this.addInput('image', 'image');
        this.addInput('prompt', 'string');
        this.addOutput('image', 'image');
        const models = getModelsFor('image', 'i2i');
        this.properties = {
            model: (models[0] && models[0].id) || '', prompt: '', size: '1024*1024', status: '',
            quality: '', background: '', output_format: '',
        };
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
                <div class="cv-gpt-group" style="display:none">
                    <label>品質 (quality)</label>
                    <div class="cv-quality-slot"></div>
                    <label>背景 (background)</label>
                    <div class="cv-bg-slot"></div>
                    <label>輸出格式 (output_format)</label>
                    <div class="cv-fmt-slot"></div>
                </div>
                <button class="cv-generate">▶ 編輯圖片</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.prompt = this.textarea.value; });
        this.statusEl = panel.querySelector('.cv-status');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => {
            this.properties.model = v;
            this._syncModelExtras();
        });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
        this._syncModelExtras();
    }
    ImageEditNode.title = '圖像編輯 Editing';
    // GPT Image（gpt-image-2/1.5）額外支援 quality/background/output_format 三個
    // OpenAI 標準參數，其他模型（萬相/千問）沒有，靠 supports_gpt_params 判斷顯示
    ImageEditNode.prototype._syncModelExtras = function () {
        const modelInfo = getModelsFor('image', 'i2i').find(m => m.id === this.properties.model) || {};
        const container = this._configOverlay || this._domPanel;
        const gptGroup = container.querySelector('.cv-gpt-group');
        gptGroup.style.display = modelInfo.supports_gpt_params ? '' : 'none';
        if (modelInfo.supports_gpt_params) {
            const qSlot = container.querySelector('.cv-quality-slot');
            qSlot.innerHTML = '';
            this.qualitySelect = buildLabeledSelect(
                [['', 'auto（自動）'], ['low', 'low'], ['medium', 'medium'], ['high', 'high']],
                this.properties.quality, (v) => { this.properties.quality = v; });
            qSlot.appendChild(this.qualitySelect);
            const bgSlot = container.querySelector('.cv-bg-slot');
            bgSlot.innerHTML = '';
            this.backgroundSelect = buildLabeledSelect(
                [['', 'auto（自動）'], ['opaque', 'opaque（不透明）'], ['transparent', 'transparent（透明）']],
                this.properties.background, (v) => { this.properties.background = v; });
            bgSlot.appendChild(this.backgroundSelect);
            const fmtSlot = container.querySelector('.cv-fmt-slot');
            fmtSlot.innerHTML = '';
            this.outputFormatSelect = buildLabeledSelect(
                [['', '預設'], ['png', 'PNG'], ['jpeg', 'JPEG'], ['webp', 'WEBP']],
                this.properties.output_format, (v) => { this.properties.output_format = v; });
            fmtSlot.appendChild(this.outputFormatSelect);
        }
    };
    ImageEditNode.prototype.onExecute = function () {
        _syncPromptTextarea(this, this.textarea, 1);
        this.setOutputData(0, this.imageUrl);
    };
    ImageEditNode.prototype.generate = async function () {
        const srcImage = this.getInputData(0, true);
        if (!srcImage) { showToast('請先連接一張來源圖片'); return; }
        const promptIn = this.getInputData(1, true);
        const basePrompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        const prompt = _combinePrompt(this, 1, basePrompt);
        if (!prompt) { showToast('請輸入 prompt'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        this.statusEl.textContent = '生成中…';
        setPreviewProgress(this, '生成中…', 20);
        try {
            const fd = new FormData();
            fd.append('model', this.properties.model);
            fd.append('prompt', prompt);
            fd.append('size', this.properties.size);
            fd.append('n', '1');
            if (this.properties.quality) fd.append('quality', this.properties.quality);
            if (this.properties.background) fd.append('background', this.properties.background);
            if (this.properties.output_format) fd.append('output_format', this.properties.output_format);
            fd.append('image_1', await fetchAsBlob(srcImage), 'source.png');
            const res = await apiFetch('/api/image/edit', { method: 'POST', body: fd });
            const data = await res.json();
            if (!res.ok || !data.images || !data.images.length) throw new Error((data.error && (data.error.message || data.error)) || '生成失敗');
            this.imageUrl = data.images[0].local_path || data.images[0].url;
            this.statusEl.textContent = '完成';
            setPreviewImage(this, this.imageUrl);
            this.setOutputData(0, this.imageUrl);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('圖像編輯失敗：' + e.message);
        }
    };
    ImageEditNode.prototype.onSerialize = function (o) {
        o.cv = { imageUrl: this.imageUrl || null };
    };
    ImageEditNode.prototype.onConfigure = function (o) {
        this.textarea.value = this.properties.prompt || '';
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        this._syncModelExtras();
        _restoreGenResult(this, o.cv);
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
        if (!this._isFaceSwap()) _syncPromptTextarea(this, this.textarea, 2);
        this.setOutputData(0, this.resultUrl);
    };
    MuleAiGenNode.prototype.generate = async function () {
        const model = this.properties.model;
        const isVideo = this._isVideo(), isFaceSwap = this._isFaceSwap(), needsImage = this._needsImage();
        const promptIn = this.getInputData(2, true);
        const basePrompt = (promptIn != null && promptIn !== '') ? promptIn : this.properties.prompt;
        const prompt = isFaceSwap ? basePrompt : _combinePrompt(this, 2, basePrompt);
        if (!isFaceSwap && !prompt) { showToast('請輸入 prompt'); return; }
        let imageBlob = null, faceBlob = null;
        if (needsImage) {
            const imgUrl = this.getInputData(0, true);
            if (!imgUrl) { showToast('請先連接一張來源圖片'); return; }
            imageBlob = await fetchAsBlob(imgUrl);
        }
        if (isFaceSwap) {
            const faceUrl = this.getInputData(1, true);
            if (!faceUrl) { showToast('請先連接換臉參考圖'); return; }
            faceBlob = await fetchAsBlob(faceUrl);
        }
        this.statusEl.textContent = '送出中…';
        setPreviewProgress(this, '送出中…', isVideo ? 90 : 20);
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
            updateProgressLabel(this, isVideo ? '生成中…（可能需要 1～數分鐘）' : '生成中…');
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
    MuleAiGenNode.prototype.onSerialize = function (o) {
        o.cv = { resultUrl: this.resultUrl || null };
    };
    MuleAiGenNode.prototype.onConfigure = function (o) {
        this.textarea.value = this.properties.prompt || '';
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        if (this.resSelect) this.resSelect.value = this.properties.resolution;
        if (this.imgResSelect) this.imgResSelect.value = this.properties.imgResolution;
        if (this.durSlider) { this.durSlider.value = this.properties.duration; this.durValEl.textContent = this.properties.duration; }
        // _syncUiForModel() 會把 resultUrl 重置為 null 並清空預覽，必須先呼叫
        // 校正插槽名稱/顯示區塊，再把還原的結果蓋回去
        this._syncUiForModel();
        const cv = o.cv || {};
        if (cv.resultUrl) {
            this.resultUrl = cv.resultUrl;
            this.statusEl.textContent = '完成';
            if (this._isVideo()) setPreviewVideo(this, cv.resultUrl); else setPreviewImage(this, cv.resultUrl);
        }
    };
    MuleAiGenNode.prototype.onRemoved = sharedOnRemoved;

    // ── Node: Audio / TTS（語音合成，呼叫 /api/voice/tts）──────────
    // qwen-audio-3.0-tts 系列（CosyVoice v3）支援 instructions/sample_rate/
    // volume/language_hints 這組進階參數；gemini-*-tts 系列走另一條上游
    // endpoint，只吃 model/input/voice，帶了 instructions 會被上游拒絕
    // （400），所以進階參數區塊要依選到的模型動態顯示/隱藏，跟主測試台
    // 語音分頁（static/js/app.js 的 onVoiceModelChange）的規則一致。
    function TtsGenNode() {
        this.addInput('text', 'string');
        this.addOutput('audio', 'audio');
        const models = getVoiceTtsModels();
        this.properties = {
            model: (models[0] && models[0].id) || '', text: '', voice: '', format: 'mp3',
            instructions: '', sample_rate: '', volume: 50, language_hints: 'zh', status: '',
        };
        this.audioUrl = null;
        this._contentHeight = 460;
        this.size = [320, 460];
        this.color = '#2e2a1f'; this.bgcolor = '#2a2a2a';

        const panel = el('div');
        panel.innerHTML = `
            <div class="cv-controls">
                <label>模型</label>
                <div class="cv-select-slot"></div>
                <label>文字內容<span class="cv-hint">（若連接文字節點會優先使用其輸出）</span></label>
                <textarea placeholder="輸入要合成的文字…"></textarea>
                <label>音色 (voice)</label>
                <div class="cv-voice-slot"></div>
                <label>輸出格式</label>
                <div class="cv-format-slot"></div>
                <div class="cv-adv-group">
                    <label>語氣風格描述 (instructions)</label>
                    <textarea class="cv-instructions" rows="2" placeholder="例如：聲音成熟低沉、語速偏慢"></textarea>
                    <label>取樣率 (sample_rate)</label>
                    <div class="cv-samplerate-slot"></div>
                    <label>音量 (volume) <span class="cv-volume-val">50</span></label>
                    <input type="range" class="cv-volume-slider" min="0" max="100" step="1" value="50">
                    <label>語言提示 (language_hints)</label>
                    <div class="cv-langhint-slot"></div>
                </div>
                <button class="cv-generate">▶ 生成語音</button>
                <div class="cv-status"></div>
            </div>`;
        attachDomPanel(this, panel);
        this.textarea = panel.querySelector('textarea');
        this.textarea.addEventListener('input', () => { this.properties.text = this.textarea.value; });
        this.voiceSlot = panel.querySelector('.cv-voice-slot');
        this.advGroup = panel.querySelector('.cv-adv-group');
        this.instructionsInput = panel.querySelector('.cv-instructions');
        this.instructionsInput.value = this.properties.instructions;
        this.instructionsInput.addEventListener('input', () => { this.properties.instructions = this.instructionsInput.value; });
        this.volumeSlider = panel.querySelector('.cv-volume-slider');
        this.volumeValEl = panel.querySelector('.cv-volume-val');
        this.volumeSlider.addEventListener('input', () => {
            this.properties.volume = parseInt(this.volumeSlider.value);
            this.volumeValEl.textContent = this.volumeSlider.value;
        });
        this.statusEl = panel.querySelector('.cv-status');
        panel.querySelector('.cv-generate').addEventListener('click', () => this.generate());

        this.modelSelect = buildSelect(models.map(m => m.id), this.properties.model, (v) => {
            this.properties.model = v;
            this._syncUiForModel();
        });
        panel.querySelector('.cv-select-slot').appendChild(this.modelSelect);

        this.formatSelect = buildSelect(['mp3', 'wav', 'opus', 'flac'], this.properties.format, (v) => { this.properties.format = v; });
        panel.querySelector('.cv-format-slot').appendChild(this.formatSelect);

        this.sampleRateSelect = buildSelect(['', '16000', '22050', '24000', '44100'], this.properties.sample_rate, (v) => { this.properties.sample_rate = v; });
        panel.querySelector('.cv-samplerate-slot').appendChild(this.sampleRateSelect);

        this.langHintSelect = buildSelect(
            ['', 'zh', 'en', 'fr', 'de', 'ja', 'ko', 'ru', 'pt', 'th', 'id', 'vi', 'es', 'it', 'ms', 'fil', 'ar'],
            this.properties.language_hints, (v) => { this.properties.language_hints = v; }
        );
        panel.querySelector('.cv-langhint-slot').appendChild(this.langHintSelect);

        panel.appendChild(buildPreview(this));
        wireConfigOverlay(this, panel);
        attachNodeChrome(this);
        this._syncUiForModel();
    }
    TtsGenNode.title = '語音 TTS';
    TtsGenNode.prototype._rebuildVoiceSelect = function () {
        const voices = voicesForTtsModel(this.properties.model);
        if (!voices.some(v => v.id === this.properties.voice)) this.properties.voice = '';
        this.voiceSlot.innerHTML = '';
        const sel = el('select');
        sel.innerHTML = '<option value="">留空 = 預設音色</option>' +
            voices.map(v => `<option value="${v.id}"${v.id === this.properties.voice ? ' selected' : ''}>${v.name} — ${v.desc}</option>`).join('');
        sel.addEventListener('mousedown', (e) => e.stopPropagation());
        sel.addEventListener('change', () => { this.properties.voice = sel.value; });
        this.voiceSlot.appendChild(sel);
        this.voiceSelect = sel;
    };
    TtsGenNode.prototype._syncUiForModel = function () {
        const isGemini = (this.properties.model || '').startsWith('gemini');
        this.advGroup.style.display = isGemini ? 'none' : '';
        this._rebuildVoiceSelect();
    };
    TtsGenNode.prototype.onExecute = function () {
        _syncPromptTextarea(this, this.textarea, 0);
        this.setOutputData(0, this.audioUrl);
    };
    TtsGenNode.prototype.generate = async function () {
        const textIn = this.getInputData(0, true);
        const text = (textIn != null && textIn !== '') ? String(textIn) : this.properties.text;
        if (!text) { showToast('請輸入文字內容'); return; }
        if (!this.properties.model) { showToast('請選擇模型'); return; }
        const isGemini = this.properties.model.startsWith('gemini');
        this.statusEl.textContent = '生成中…';
        setPreviewProgress(this, '生成中…', 8);
        try {
            const body = { model: this.properties.model, text, format: this.properties.format };
            if (this.properties.voice) body.voice = this.properties.voice;
            if (!isGemini) {
                if (this.properties.instructions) body.instructions = this.properties.instructions;
                if (this.properties.sample_rate) body.sample_rate = parseInt(this.properties.sample_rate);
                if (this.properties.volume != null) body.volume = this.properties.volume;
                if (this.properties.language_hints) body.language_hints = [this.properties.language_hints];
            }
            const res = await apiFetch('/api/voice/tts', { method: 'POST', body: JSON.stringify(body) });
            const data = await res.json();
            if (!res.ok || !data.success) throw new Error(data.error || '合成失敗');
            this.audioUrl = data.audio_url;
            this.statusEl.textContent = '完成';
            setPreviewAudio(this, data.audio_url);
            this.setOutputData(0, data.audio_url);
        } catch (e) {
            this.statusEl.textContent = '錯誤：' + e.message;
            setPreviewEmpty(this, '生成失敗');
            showToast('語音合成失敗：' + e.message);
        }
    };
    TtsGenNode.prototype.onSerialize = function (o) {
        o.cv = { audioUrl: this.audioUrl || null };
    };
    TtsGenNode.prototype.onConfigure = function (o) {
        this.textarea.value = this.properties.text || '';
        this.instructionsInput.value = this.properties.instructions || '';
        if (this.modelSelect) this.modelSelect.value = this.properties.model;
        if (this.formatSelect) this.formatSelect.value = this.properties.format;
        if (this.sampleRateSelect) this.sampleRateSelect.value = this.properties.sample_rate || '';
        if (this.langHintSelect) this.langHintSelect.value = this.properties.language_hints || '';
        if (this.volumeSlider) { this.volumeSlider.value = this.properties.volume; this.volumeValEl.textContent = this.properties.volume; }
        this._syncUiForModel();
        _restoreGenResult(this, o.cv);
    };
    TtsGenNode.prototype.onRemoved = sharedOnRemoved;

    function registerNodeTypes() {
        LiteGraph.registerNodeType('nenai/text', TextPromptNode);
        LiteGraph.registerNodeType('nenai/camera_angle', CameraAngleNode);
        LiteGraph.registerNodeType('nenai/load_image', LoadImageNode);
        LiteGraph.registerNodeType('nenai/image', ImageGenNode);
        LiteGraph.registerNodeType('nenai/video', VideoGenNode);
        LiteGraph.registerNodeType('nenai/video_edit', VideoEditNode);
        LiteGraph.registerNodeType('nenai/video_animate', VideoAnimateNode);
        LiteGraph.registerNodeType('nenai/edit', ImageEditNode);
        LiteGraph.registerNodeType('nenai/audio', TtsGenNode);
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

    const NODE_MENU_TYPES = { text: 'nenai/text', camera_angle: 'nenai/camera_angle', load_image: 'nenai/load_image', image: 'nenai/image', video: 'nenai/video', video_edit: 'nenai/video_edit', video_animate: 'nenai/video_animate', edit: 'nenai/edit', audio: 'nenai/audio', muleai: 'nenai/muleai' };

    // ── 範本庫：一鍵套用常見組合，省去手動拉線 ─────────────────────
    // 每個範本只描述「節點類型 + 相對座標 + 要接的線」，實際節點是套用當下用
    // LiteGraph.createNode()/graph.add()/node.connect() 即時生成，不是還原一份
    // 手刻的序列化 JSON——這樣範本永遠跟目前的節點實作（預設參數、插槽順序）同步，
    // 不會因為節點程式碼改版就跟著過期失真。
    const CANVAS_TEMPLATES = [
        {
            id: 'text-to-image', name: '文字 → 圖片生成',
            desc: '用文字節點寫/生成 Prompt，接到圖片生成節點',
            nodes: [{ type: 'text', pos: [0, 0] }, { type: 'image', pos: [420, -60] }],
            edges: [[0, 0, 1, 0]],
        },
        {
            id: 'image-to-edit', name: '圖片生成 → 圖像編輯',
            desc: '生成一張圖後，直接接到編輯節點做二次調整',
            nodes: [{ type: 'image', pos: [0, 0] }, { type: 'edit', pos: [420, 0] }],
            edges: [[0, 0, 1, 0]],
        },
        {
            id: 'text-to-voice', name: '文字腳本 → 語音配音',
            desc: '用文字節點寫腳本，接到 TTS 節點配音',
            nodes: [{ type: 'text', pos: [0, 0] }, { type: 'audio', pos: [420, 0] }],
            edges: [[0, 0, 1, 0]],
        },
        {
            id: 'text-fanout', name: '文字 → 圖片 + 語音（雙路輸出）',
            desc: '同一段文字同時拿去生成配圖，也拿去配音',
            nodes: [{ type: 'text', pos: [0, 120] }, { type: 'image', pos: [440, -40] }, { type: 'audio', pos: [440, 340] }],
            edges: [[0, 0, 1, 0], [0, 0, 2, 0]],
        },
        {
            id: 'image-to-video', name: '上傳圖片 → 圖生影片',
            desc: '上傳一張圖片當首幀，接到影片節點生成 i2v',
            nodes: [{ type: 'load_image', pos: [0, 0] }, { type: 'video', pos: [420, 0] }],
            edges: [[0, 0, 1, 1]],
        },
    ];
    let _templatePlaceOffset = 0; // 每套用一次範本就往右下偏移，避免疊在前一個範本上面

    function applyTemplate(tpl) {
        const canvasEl = document.getElementById('litegraphCanvas');
        const baseX = (canvasEl.width / 2) / lgCanvas.ds.scale - lgCanvas.ds.offset[0] - 340 + _templatePlaceOffset;
        const baseY = (canvasEl.height / 2) / lgCanvas.ds.scale - lgCanvas.ds.offset[1] - 200 + _templatePlaceOffset;
        _templatePlaceOffset += 60;

        const createdNodes = tpl.nodes.map(spec => {
            const type = NODE_MENU_TYPES[spec.type];
            const node = LiteGraph.createNode(type);
            node.pos = [baseX + spec.pos[0], baseY + spec.pos[1]];
            graph.add(node);
            return node;
        });
        (tpl.edges || []).forEach(([fromIdx, fromSlot, toIdx, toSlot]) => {
            createdNodes[fromIdx].connect(fromSlot, createdNodes[toIdx], toSlot);
        });
        selectNodeOnly(createdNodes[0]);
        lgCanvas.setDirty(true, true);
        showToast(`已套用範本：${tpl.name}`);
    }

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

        const templatesMenu = document.getElementById('templatesMenu');
        CANVAS_TEMPLATES.forEach(tpl => {
            const btn = document.createElement('button');
            btn.innerHTML = `${tpl.name}<span class="tpl-desc">${tpl.desc}</span>`;
            btn.addEventListener('click', () => {
                applyTemplate(tpl);
                templatesMenu.style.display = 'none';
            });
            templatesMenu.appendChild(btn);
        });
        document.getElementById('btnTemplates').addEventListener('click', (e) => {
            e.stopPropagation();
            templatesMenu.style.display = templatesMenu.style.display === 'none' ? 'block' : 'none';
        });
        document.addEventListener('click', (e) => {
            if (!templatesMenu.contains(e.target) && e.target.id !== 'btnTemplates') templatesMenu.style.display = 'none';
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

        document.getElementById('btnExportCanvas').addEventListener('click', exportCanvasToFile);
        document.getElementById('btnImportCanvas').addEventListener('click', () => {
            document.getElementById('canvasImportInput').click();
        });
        document.getElementById('canvasImportInput').addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (file) importCanvasFromFile(file);
            e.target.value = '';
        });
        document.getElementById('btnClearCanvas').addEventListener('click', () => {
            if (!confirm('確定要清空整個畫布嗎？此操作無法復原。')) return;
            graph.clear();
            localStorage.removeItem(CANVAS_STORAGE_KEY);
            _lastSavedJson = null;
            document.getElementById('canvasSaveStatus').textContent = '';
        });
    }

    // ── 專案存檔（本機瀏覽器自動存檔 + 匯出/匯入 JSON 檔案）──────────────
    // 只做「定期輪詢比對」而不是掛在每個可能的變動來源上，是因為這個平台的節點
    // 表單控制（textarea/select/拖曳角度等）都是純 DOM 事件直接改 this.properties，
    // 沒有經過 LiteGraph 的 graph.beforeChange()/afterChange() 通知機制，沒有一個
    // 單一事件可以可靠地涵蓋所有變動來源。
    const CANVAS_STORAGE_KEY = 'nenai_canvas_autosave_v1';
    let _lastSavedJson = null;

    function serializeCanvasPayload() {
        return {
            title: document.getElementById('canvasTitle').value || 'Untitled Canvas',
            savedAt: Date.now(),
            data: graph.serialize(),
        };
    }

    function autosaveTick() {
        try {
            const payload = serializeCanvasPayload();
            const json = JSON.stringify(payload);
            if (json === _lastSavedJson) return;
            localStorage.setItem(CANVAS_STORAGE_KEY, json);
            _lastSavedJson = json;
            const t = new Date(payload.savedAt);
            const hh = String(t.getHours()).padStart(2, '0');
            const mm = String(t.getMinutes()).padStart(2, '0');
            const statusEl = document.getElementById('canvasSaveStatus');
            if (statusEl) statusEl.textContent = `已自動儲存於本機瀏覽器 ${hh}:${mm}（上傳的圖片/影片檔案需重新選擇）`;
        } catch (e) {
            console.warn('Canvas autosave failed:', e);
        }
    }

    function restoreFromStorageIfAny() {
        let json;
        try {
            json = localStorage.getItem(CANVAS_STORAGE_KEY);
        } catch (e) { return; }
        if (!json) return;
        try {
            const payload = JSON.parse(json);
            if (payload.title) document.getElementById('canvasTitle').value = payload.title;
            graph.configure(payload.data);
            _lastSavedJson = json;
        } catch (e) {
            console.warn('Canvas restore failed:', e);
        }
    }

    function exportCanvasToFile() {
        const payload = serializeCanvasPayload();
        const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        const d = new Date(payload.savedAt);
        const stamp = `${d.getFullYear()}${String(d.getMonth() + 1).padStart(2, '0')}${String(d.getDate()).padStart(2, '0')}_${String(d.getHours()).padStart(2, '0')}${String(d.getMinutes()).padStart(2, '0')}`;
        a.href = url;
        a.download = `${payload.title || 'canvas'}_${stamp}.json`;
        a.click();
        URL.revokeObjectURL(url);
        showToast('已匯出畫布 JSON 檔案');
    }

    function importCanvasFromFile(file) {
        const reader = new FileReader();
        reader.onload = () => {
            try {
                const parsed = JSON.parse(reader.result);
                const graphData = parsed.data || parsed;
                if (parsed.title) document.getElementById('canvasTitle').value = parsed.title;
                graph.configure(graphData);
                showToast('已匯入畫布');
            } catch (e) {
                showToast('匯入失敗：檔案格式不正確');
            }
        };
        reader.readAsText(file);
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
        // litegraph 原生的右鍵/插槽選單（畫布空白處的 Add Node/Add Group、節點插槽上
        // 的 Rename Slot/Remove Slot 等）跟這個平台的節點選單是兩套獨立系統，且插槽
        // 選單不只在真正右鍵時觸發，快速點擊（litegraph 內部判定為 pointer_is_double）
        // 時在 mousedown 當下就會直接跳出來——跟拉線操作手感接近，很容易誤觸，蓋住
        // 畫面卡住互動。processContextMenu 是這一切的唯一入口（不論右鍵或快速點擊都
        // 會呼叫它），直接整個關閉，一律改用下面的自訂選單。
        lgCanvas.getCanvasMenuOptions = () => null;
        lgCanvas.processContextMenu = () => null;
        document.getElementById('litegraphCanvas').addEventListener('contextmenu', (e) => {
            e.preventDefault();
            openQuickAddMenu(null, null, e.clientX, e.clientY);
        });

        resizeCanvasEl();
        window.addEventListener('resize', resizeCanvasEl);

        restoreFromStorageIfAny();

        graph.start();
        wireToolbar();
        updateZoomLabel();
        requestAnimationFrame(positionAllPanels);
        setInterval(updateZoomLabel, 500);
        setInterval(autosaveTick, 3000);
        window.addEventListener('beforeunload', autosaveTick);
    }

    init();
})();
