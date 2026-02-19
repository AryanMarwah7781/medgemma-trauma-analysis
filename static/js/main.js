/**
 * MedGemma Trauma Analysis - Main Logic
 * Handles file uploads, vital signs, results rendering, and Q&A streaming.
 */

 document.addEventListener('DOMContentLoaded', () => {
    // State
    let currentSessionId = null;
    let qaStreaming = false;

    // DOM Elements
    const elements = {
        fileInput: document.getElementById('fileInput'),
        uploadZone: document.getElementById('uploadZone'),
        vitalsToggle: document.getElementById('vitalsToggle'),
        vitalsPanel: document.getElementById('vitalsPanel'),
        loadingOverlay: document.getElementById('loadingOverlay'),
        resultsContainer: document.getElementById('resultsContainer'),
        
        // Inputs
        vitHR: document.getElementById('vitHR'),
        vitBP: document.getElementById('vitBP'),
        vitGCS: document.getElementById('vitGCS'),

        // Results
        patientIdDisplay: document.getElementById('patientIdDisplay'),
        triageBadges: document.getElementById('triageBadges'),
        volume: document.getElementById('volume'),
        risk: document.getElementById('risk'),
        pixels: document.getElementById('pixels'),
        severity: document.getElementById('severity'),
        organs: document.getElementById('organs'),
        injuryPattern: document.getElementById('injuryPattern'),
        diffList: document.getElementById('diffList'),
        recommendation: document.getElementById('recommendation'),
        llmReport: document.getElementById('llmReport'),
        
        // Chat
        chatContainer: document.getElementById('chatContainer'),
        chatHistory: document.getElementById('chatHistory'),
        qaInput: document.getElementById('qaInput'),
        qaSubmit: document.getElementById('qaSubmit'),
        resetBtn: document.getElementById('resetBtn')
    };

    // --- Event Listeners ---

    // File Upload
    elements.fileInput.addEventListener('change', async (e) => {
        const files = e.target.files;
        if (files && files.length > 0) await runAnalysis(files);
    });

    // Drag & Drop
    ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
        elements.uploadZone.addEventListener(eventName, preventDefaults, false);
    });

    function preventDefaults(e) {
        e.preventDefault();
        e.stopPropagation();
    }

    elements.uploadZone.addEventListener('dragover', () => elements.uploadZone.classList.add('dragover'));
    elements.uploadZone.addEventListener('dragleave', () => elements.uploadZone.classList.remove('dragover'));
    elements.uploadZone.addEventListener('drop', async (e) => {
        elements.uploadZone.classList.remove('dragover');
        const files = e.dataTransfer.files;
        if (files.length > 0) await runAnalysis(files);
    });

    // Vitals Toggle
    elements.vitalsToggle.addEventListener('click', () => {
        elements.vitalsPanel.classList.toggle('open');
        const isOpen = elements.vitalsPanel.classList.contains('open');
        elements.vitalsToggle.innerHTML = isOpen 
            ? '<i class="fas fa-minus"></i> Hide Vitals' 
            : '<i class="fas fa-plus"></i> Add Patient Vitals';
    });

    // Q&A
    elements.qaSubmit.addEventListener('click', submitQuestion);
    elements.qaInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            submitQuestion();
        }
    });

    // Reset
    if(elements.resetBtn) {
        elements.resetBtn.addEventListener('click', resetApp);
    }

    // --- Core Functions ---

    async function runAnalysis(files) {
        // Show loading
        elements.loadingOverlay.classList.add('active');
        elements.resultsContainer.style.display = 'none';

        const formData = new FormData();
        Array.from(files).forEach(f => formData.append('files', f));

        // Add optional vitals
        if (elements.vitHR.value) formData.append('hr', elements.vitHR.value.trim());
        if (elements.vitBP.value) formData.append('bp', elements.vitBP.value.trim());
        if (elements.vitGCS.value) formData.append('gcs', elements.vitGCS.value.trim());

        try {
            const response = await fetch('/upload', { method: 'POST', body: formData });
            const data = await response.json();

            if (data.success) {
                renderResults(data.result);
            } else {
                throw new Error(data.error || 'Unknown error occurred');
            }
        } catch (err) {
            alert(`Error: ${err.message}`);
            elements.loadingOverlay.classList.remove('active');
        }
    }

    function renderResults(result) {
        currentSessionId = result.session_id;

        // Hide upload, show results
        elements.loadingOverlay.classList.remove('active');
        elements.resultsContainer.style.display = 'block';
        
        // Scroll to results
        elements.resultsContainer.scrollIntoView({ behavior: 'smooth' });

        // Populate Data
        elements.patientIdDisplay.textContent = result.patient_id || 'Unknown';

        // Triage
        elements.triageBadges.innerHTML = '';
        const scores = result.triage.per_slice_scores || [];
        scores.forEach((score, i) => {
            const badge = document.createElement('span');
            const pct = Math.round(score * 100);
            const isSuspicious = score >= 0.25;
            badge.className = `badge ${isSuspicious ? 'bg-danger text-white' : 'bg-success text-white'}`;
            // Custom styling handled by CSS classes or inline solely for specific status logic
            badge.style.backgroundColor = isSuspicious ? 'rgba(239, 68, 68, 0.2)' : 'rgba(16, 185, 129, 0.2)';
            badge.style.color = isSuspicious ? '#ef4444' : '#10b981';
            badge.style.border = `1px solid ${isSuspicious ? '#ef4444' : '#10b981'}`;
            badge.style.marginRight = '8px';
            badge.textContent = `Slice ${i + 1}: ${pct}%`;
            elements.triageBadges.appendChild(badge);
        });

        // Quantification
        const quant = result.quantification || {};
        elements.volume.textContent = (quant.volume_ml || 0).toFixed(2) + ' mL';
        elements.pixels.textContent = (quant.num_voxels || 0).toLocaleString();
        
        const riskLevel = (quant.risk_level || 'UNKNOWN').toUpperCase();
        elements.risk.textContent = riskLevel;
        
        // Risk Coloring
        elements.risk.className = 'stat-value'; // reset
        if (riskLevel === 'HIGH') elements.risk.classList.add('text-danger');
        else if (riskLevel === 'MODERATE') elements.risk.classList.add('text-warning');
        else elements.risk.classList.add('text-success');

        // Visual Findings
        const vf = result.visual_findings || {};
        elements.severity.textContent = (vf.severity_estimate || '--').toUpperCase();
        
        const organs = vf.organs_involved || [];
        elements.organs.textContent = organs.length ? organs.join(', ') : 'None identified';
        elements.injuryPattern.textContent = vf.injury_pattern || '--';

        // Differential Diagnosis
        elements.diffList.innerHTML = '';
        const diffs = vf.differential_diagnosis || [];
        if (diffs.length) {
            diffs.forEach(d => {
                const li = document.createElement('li');
                li.textContent = d;
                li.style.marginBottom = '6px';
                li.innerHTML = `<span style="color:var(--primary); margin-right:8px;">•</span>${d}`;
                elements.diffList.appendChild(li);
            });
        } else {
            elements.diffList.innerHTML = '<li class="text-neutral">No differential diagnosis available.</li>';
        }

        // Recommendation
        elements.recommendation.textContent = quant.recommendation || 'See full report below.';

        // Report
        elements.llmReport.textContent = result.report || '--';

        // Reset Chat
        elements.chatHistory.innerHTML = '';
    }

    function submitQuestion() {
        if (qaStreaming) return;
        
        const q = elements.qaInput.value.trim();
        if (!q) return;
        
        if (!currentSessionId) {
             alert('Please upload and analyze a scan first.');
             return;
        }

        // Add User Message
        appendMessage('You', q, 'user');
        elements.qaInput.value = '';

        // Add Placeholder AI Message
        const aiMsgTextEl = appendMessage('MedGemma', '', 'ai');
        
        elements.qaSubmit.disabled = true;
        qaStreaming = true;

        // Stream Response
        const url = `/qa-stream?session_id=${encodeURIComponent(currentSessionId)}&q=${encodeURIComponent(q)}`;
        const evtSource = new EventSource(url);

        evtSource.onmessage = (e) => {
            if (e.data === '[DONE]') {
                evtSource.close();
                elements.qaSubmit.disabled = false;
                qaStreaming = false;
                return;
            }
            aiMsgTextEl.textContent += e.data;
            scrollToBottom();
        };

        evtSource.onerror = () => {
            evtSource.close();
            elements.qaSubmit.disabled = false;
            qaStreaming = false;
            aiMsgTextEl.textContent += " [Connection Error]";
        };
    }

    function appendMessage(role, text, type) {
        const msgDiv = document.createElement('div');
        msgDiv.className = `message ${type}`;
        
        const roleDiv = document.createElement('div');
        roleDiv.style.fontWeight = 'bold';
        roleDiv.style.fontSize = '0.75rem';
        roleDiv.style.marginBottom = '4px';
        roleDiv.style.opacity = '0.7';
        roleDiv.textContent = role;
        
        const textDiv = document.createElement('div');
        textDiv.textContent = text;
        
        msgDiv.appendChild(roleDiv);
        msgDiv.appendChild(textDiv);
        
        elements.chatHistory.appendChild(msgDiv);
        scrollToBottom();
        
        return textDiv;
    }

    function scrollToBottom() {
        elements.chatHistory.scrollTop = elements.chatHistory.scrollHeight;
    }

    function resetApp() {
        elements.fileInput.value = '';
        currentSessionId = null;
        elements.resultsContainer.style.display = 'none';
        
        // Reset inputs
        elements.vitHR.value = '';
        elements.vitBP.value = '';
        elements.vitGCS.value = '';
        
        window.scrollTo({ top: 0, behavior: 'smooth' });
    }
});
