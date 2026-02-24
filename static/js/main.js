/**
 * MedGemma Trauma Analysis — Glassmorphism UI
 * New: CT thumbnail previews on file select + in results sidebar
 */

document.addEventListener('DOMContentLoaded', () => {

    let currentSessionId = null;
    let qaStreaming      = false;
    let stagedFiles      = null;
    let previewUrls      = [];   // stores FileReader data URLs in order

    const el = {
        fileInput:        document.getElementById('fileInput'),
        dropTarget:       document.getElementById('dropTarget'),
        fileCountBadge:   document.getElementById('fileCountBadge'),
        fileCountText:    document.getElementById('fileCountText'),
        ctThumbnails:     document.getElementById('ctThumbnails'),
        ctResultsStrip:   document.getElementById('ctResultsStrip'),
        fileValidation:   document.getElementById('fileValidation'),
        analyzeBtn:       document.getElementById('analyzeBtn'),
        analyzeHint:      document.getElementById('analyzeHint'),

        vitHR:            document.getElementById('vitHR'),
        vitBP:            document.getElementById('vitBP'),
        vitGCS:           document.getElementById('vitGCS'),

        loadingOverlay:   document.getElementById('loadingOverlay'),
        errorBanner:      document.getElementById('errorBanner'),
        errorMessage:     document.getElementById('errorMessage'),
        errorDismiss:     document.getElementById('errorDismiss'),

        resultsContainer: document.getElementById('resultsContainer'),
        patientIdDisplay: document.getElementById('patientIdDisplay'),

        riskBanner:       document.getElementById('riskBanner'),
        riskValue:        document.getElementById('riskValue'),
        riskVolume:       document.getElementById('riskVolume'),
        riskEAST:         document.getElementById('riskEAST'),

        triageRows:       document.getElementById('triageRows'),
        volume:           document.getElementById('volume'),
        pixels:           document.getElementById('pixels'),
        severity:         document.getElementById('severity'),
        organs:           document.getElementById('organs'),
        injuryPattern:    document.getElementById('injuryPattern'),
        diffList:         document.getElementById('diffList'),
        llmReport:        document.getElementById('llmReport'),
        copyReportBtn:    document.getElementById('copyReportBtn'),

        chatHistory:      document.getElementById('chatHistory'),
        typingIndicator:  document.getElementById('typingIndicator'),
        qaInput:          document.getElementById('qaInput'),
        qaSubmit:         document.getElementById('qaSubmit'),
        resetBtn:         document.getElementById('resetBtn'),
    };

    // ── File staging ───────────────────────────────────────────────────
    el.fileInput.addEventListener('change', (e) => stageFiles(e.target.files));

    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(evt =>
        el.dropTarget.addEventListener(evt, (e) => { e.preventDefault(); e.stopPropagation(); }, false)
    );

    el.dropTarget.addEventListener('dragover',  () => el.dropTarget.classList.add('dragover'));
    el.dropTarget.addEventListener('dragleave', () => el.dropTarget.classList.remove('dragover'));
    el.dropTarget.addEventListener('drop', (e) => {
        el.dropTarget.classList.remove('dragover');
        stageFiles(e.dataTransfer.files);
    });

    function stageFiles(files) {
        if (!files || files.length === 0) return;

        const valid   = Array.from(files).filter(f => f.type.startsWith('image/'));
        const invalid = files.length - valid.length;

        if (invalid > 0) showValidation(`${invalid} unsupported file(s) skipped — PNG/JPG only`);
        else hideValidation();

        if (valid.length === 0) {
            el.analyzeBtn.disabled = true;
            el.analyzeHint.textContent = 'No valid image files selected';
            return;
        }

        stagedFiles  = valid;
        previewUrls  = new Array(valid.length).fill(null);

        const n = valid.length;
        el.fileCountText.textContent = `${n} file${n !== 1 ? 's' : ''} staged`;
        el.fileCountBadge.classList.remove('hidden');
        el.analyzeBtn.disabled = false;
        el.analyzeHint.textContent = `${n} CT slice${n !== 1 ? 's' : ''} ready`;

        // Generate thumbnails in upload zone
        el.ctThumbnails.innerHTML = '';
        valid.forEach((file, i) => {
            const reader = new FileReader();
            reader.onload = (e) => {
                previewUrls[i] = e.target.result;
                const img = document.createElement('img');
                img.src = e.target.result;
                img.className = 'ct-thumb';
                img.title = file.name;
                el.ctThumbnails.appendChild(img);
            };
            reader.readAsDataURL(file);
        });
        el.ctThumbnails.classList.remove('hidden');
    }

    // ── Error / validation banners ─────────────────────────────────────
    function showError(msg) {
        el.errorMessage.textContent = msg;
        el.errorBanner.classList.remove('hidden');
        el.errorBanner.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }

    function hideError() { el.errorBanner.classList.add('hidden'); }

    el.errorDismiss.addEventListener('click', hideError);

    function showValidation(msg) {
        el.fileValidation.textContent = msg;
        el.fileValidation.classList.remove('hidden');
    }

    function hideValidation() { el.fileValidation.classList.add('hidden'); }

    // ── Loading steps ──────────────────────────────────────────────────
    const STEP_IDS = ['step1','step2','step3','step4','step5'];
    let stepTimer = null, currentStep = 0;

    function startLoadingSteps() {
        currentStep = 0;
        STEP_IDS.forEach(id => { const s = document.getElementById(id); if (s) s.className = 'step'; });
        const first = document.getElementById(STEP_IDS[0]);
        if (first) first.classList.add('active');
        stepTimer = setInterval(() => {
            if (currentStep < STEP_IDS.length - 1) {
                const prev = document.getElementById(STEP_IDS[currentStep]);
                if (prev) prev.className = 'step done';
                currentStep++;
                const next = document.getElementById(STEP_IDS[currentStep]);
                if (next) next.classList.add('active');
            }
        }, 3500);
    }

    function stopLoadingSteps() {
        clearInterval(stepTimer);
        STEP_IDS.forEach(id => { const s = document.getElementById(id); if (s) s.className = 'step done'; });
    }

    // ── Analyze ────────────────────────────────────────────────────────
    el.analyzeBtn.addEventListener('click', async () => {
        if (!stagedFiles || stagedFiles.length === 0) return;
        hideError();
        await runAnalysis(stagedFiles);
    });

    async function runAnalysis(files) {
        el.loadingOverlay.classList.add('active');
        el.resultsContainer.classList.add('hidden');
        startLoadingSteps();

        const formData = new FormData();
        files.forEach(f => formData.append('files', f));
        if (el.vitHR.value.trim())  formData.append('hr',  el.vitHR.value.trim());
        if (el.vitBP.value.trim())  formData.append('bp',  el.vitBP.value.trim());
        if (el.vitGCS.value.trim()) formData.append('gcs', el.vitGCS.value.trim());

        try {
            const resp = await fetch('/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            stopLoadingSteps();
            el.loadingOverlay.classList.remove('active');
            if (data.success) renderResults(data.result);
            else throw new Error(data.error || 'Unknown error occurred');
        } catch (err) {
            stopLoadingSteps();
            el.loadingOverlay.classList.remove('active');
            showError(`Analysis failed: ${err.message}`);
        }
    }

    // ── Render results ─────────────────────────────────────────────────
    function renderResults(result) {
        currentSessionId = result.session_id;
        el.resultsContainer.classList.remove('hidden');
        el.resultsContainer.scrollIntoView({ behavior: 'smooth' });

        el.patientIdDisplay.textContent = result.patient_id || 'CASE-UNKNOWN';

        // Risk banner
        const quant       = result.quantification || {};
        const riskLevel   = (quant.risk_level || 'UNKNOWN').toUpperCase();
        const volumeML    = (quant.volume_ml || 0).toFixed(1);
        const rec         = quant.recommendation || '--';

        el.riskBanner.className    = `risk-banner ${riskLevel.toLowerCase()}`;
        el.riskValue.textContent   = riskLevel;
        el.riskVolume.textContent  = `${volumeML} mL`;
        el.riskEAST.textContent    = rec;

        // CT images in results sidebar
        el.ctResultsStrip.innerHTML = '';
        const scores = result.triage?.per_slice_scores || [];
        previewUrls.forEach((src, i) => {
            if (!src) return;
            const suspicious = (scores[i] || 0) >= 0.25;
            const img = document.createElement('img');
            img.src = src;
            img.className = `ct-result-thumb${suspicious ? ' suspicious' : ''}`;
            img.title = `Slice ${i + 1} — ${suspicious ? 'Suspicious' : 'Clear'}`;
            el.ctResultsStrip.appendChild(img);
        });

        // Triage rows
        el.triageRows.innerHTML = '';
        scores.forEach((score, i) => {
            const pct        = Math.round(score * 100);
            const suspicious = score >= 0.25;
            const cls        = suspicious ? 'suspicious' : 'clear';
            const label      = suspicious ? 'Suspicious' : 'Clear';

            const row = document.createElement('div');
            row.className = 'triage-row';
            row.innerHTML = `
                <span class="triage-slice-label">SL-${String(i + 1).padStart(2, '0')}</span>
                <div class="triage-bar-wrap">
                    <div class="triage-bar-bg">
                        <div class="triage-bar-fill ${cls}" style="width:${pct}%"></div>
                    </div>
                    <span class="triage-pct">${pct}%</span>
                </div>
                <span class="triage-status ${cls}">${label}</span>
            `;
            el.triageRows.appendChild(row);
        });

        // Quantification
        el.volume.textContent = `${volumeML} mL`;
        el.pixels.textContent = (quant.num_voxels || 0).toLocaleString();

        // Visual findings
        const vf = result.visual_findings || {};
        el.severity.textContent      = (vf.severity_estimate || '--').toUpperCase();
        const organs = vf.organs_involved || [];
        el.organs.textContent        = organs.length ? organs.join(', ') : 'None identified';
        el.injuryPattern.textContent = vf.injury_pattern || '--';

        // Differential
        el.diffList.innerHTML = '';
        const diffs = vf.differential_diagnosis || [];
        if (diffs.length) {
            diffs.forEach(d => {
                const li = document.createElement('li');
                li.textContent = d;
                el.diffList.appendChild(li);
            });
        } else {
            const li = document.createElement('li');
            li.textContent = 'No differential available.';
            el.diffList.appendChild(li);
        }

        // Report
        el.llmReport.textContent = result.report || '--';

        // Reset chat
        el.chatHistory.innerHTML = `
            <div class="message ai">
                <div class="msg-avatar"><i class="fas fa-robot"></i></div>
                <div class="msg-body">
                    <div class="msg-role">MedGemma</div>
                    <div class="msg-text">Scan analyzed. Ask me anything about the findings or treatment options.</div>
                </div>
            </div>`;
    }

    // ── Copy report ────────────────────────────────────────────────────
    el.copyReportBtn.addEventListener('click', async () => {
        try {
            await navigator.clipboard.writeText(el.llmReport.textContent);
            el.copyReportBtn.innerHTML = '<i class="fas fa-check"></i> Copied';
            el.copyReportBtn.classList.add('copied');
            setTimeout(() => {
                el.copyReportBtn.innerHTML = '<i class="fas fa-copy"></i> Copy';
                el.copyReportBtn.classList.remove('copied');
            }, 2000);
        } catch {
            showError('Clipboard write blocked — please select and copy manually.');
        }
    });

    // ── Q&A ────────────────────────────────────────────────────────────
    el.qaSubmit.addEventListener('click', submitQuestion);
    el.qaInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); submitQuestion(); }
    });

    function submitQuestion() {
        if (qaStreaming) return;
        const q = el.qaInput.value.trim();
        if (!q) return;
        if (!currentSessionId) { showError('Please analyze a scan first.'); return; }

        appendMessage('You', q, 'user');
        el.qaInput.value = '';
        const aiTextEl = appendMessage('MedGemma', '', 'ai');
        el.qaSubmit.disabled = true;
        el.typingIndicator.classList.remove('hidden');
        qaStreaming = true;

        const url = `/qa-stream?session_id=${encodeURIComponent(currentSessionId)}&q=${encodeURIComponent(q)}`;
        const evtSource = new EventSource(url);

        evtSource.onmessage = (e) => {
            if (e.data === '[DONE]') {
                evtSource.close();
                el.qaSubmit.disabled = false;
                el.typingIndicator.classList.add('hidden');
                qaStreaming = false;
                return;
            }
            el.typingIndicator.classList.add('hidden');
            aiTextEl.textContent += e.data;
            scrollChat();
        };

        evtSource.onerror = () => {
            evtSource.close();
            el.qaSubmit.disabled = false;
            el.typingIndicator.classList.add('hidden');
            qaStreaming = false;
            aiTextEl.textContent += ' [Connection Error]';
        };
    }

    function appendMessage(role, text, type) {
        const isAI = type === 'ai';
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;

        const avatar = document.createElement('div');
        avatar.className = 'msg-avatar';
        avatar.innerHTML = isAI ? '<i class="fas fa-robot"></i>' : '<i class="fas fa-user"></i>';

        const body = document.createElement('div');
        body.className = 'msg-body';

        const roleDiv = document.createElement('div');
        roleDiv.className = 'msg-role';
        roleDiv.textContent = role;

        const textDiv = document.createElement('div');
        textDiv.className = 'msg-text';
        textDiv.textContent = text;

        body.appendChild(roleDiv);
        body.appendChild(textDiv);
        msgDiv.appendChild(avatar);
        msgDiv.appendChild(body);
        el.chatHistory.appendChild(msgDiv);
        scrollChat();
        return textDiv;
    }

    function scrollChat() {
        el.chatHistory.scrollTop = el.chatHistory.scrollHeight;
    }

    // ── Reset ──────────────────────────────────────────────────────────
    el.resetBtn.addEventListener('click', resetApp);

    function resetApp() {
        el.fileInput.value   = '';
        stagedFiles          = null;
        previewUrls          = [];
        currentSessionId     = null;

        el.fileCountBadge.classList.add('hidden');
        el.ctThumbnails.innerHTML = '';
        el.ctThumbnails.classList.add('hidden');
        el.analyzeBtn.disabled        = true;
        el.analyzeHint.textContent    = 'Select CT slices to begin';

        el.vitHR.value  = '';
        el.vitBP.value  = '';
        el.vitGCS.value = '';

        el.resultsContainer.classList.add('hidden');
        hideError();
        hideValidation();
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});
