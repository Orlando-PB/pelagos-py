let currentDiagGen = null;
export let isDiagActive = false;
let isRegenerating = false;

export function setupPlotView() {
    document.getElementById('btn-diag-regen').addEventListener('click', async () => {
        isRegenerating = true;
        const btn = document.getElementById('btn-diag-regen');
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="lucide-spin"></i> Regenerating...`;
        lucide.createIcons();
        
        document.getElementById('diag-img').style.opacity = '0.4';
        document.getElementById('diag-img').style.transition = 'opacity 0.2s';
        
        await fetch('/api/diagnostic/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'regenerate', parameters: getDiagParams() })
        });
    });

    document.getElementById('btn-diag-continue').addEventListener('click', async () => {
        const btn = document.getElementById('btn-diag-continue');
        btn.disabled = true;
        btn.innerHTML = `<i data-lucide="loader-2" class="lucide-spin"></i> Continuing...`;
        lucide.createIcons();

        await fetch('/api/diagnostic/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'continue', parameters: getDiagParams() })
        });
    });

    document.getElementById('btn-diag-cancel').addEventListener('click', async () => {
        await fetch('/api/cancel', { method: 'POST' });
        closeDiagnosticUI();
    });
}

export function handleDiagnosticSync(diagData) {
    if (diagData.status === 'paused') {
        const data = diagData.data;
        if (currentDiagGen !== data.generation_id) {
            isRegenerating = false;
            
            document.getElementById('diagnostic-overlay').style.display = 'flex';
            document.getElementById('diag-step-name').textContent = data.step_name;
            
            const img = document.getElementById('diag-img');
            img.src = 'data:image/png;base64,' + data.plot_b64;
            img.style.opacity = '1';
            
            currentDiagGen = data.generation_id;
            renderDiagParams(data.parameters);
            
            const btnRegen = document.getElementById('btn-diag-regen');
            btnRegen.disabled = false;
            btnRegen.innerHTML = '<i data-lucide="refresh-cw"></i> Regenerate';
            
            const btnContinue = document.getElementById('btn-diag-continue');
            btnContinue.disabled = false;
            btnContinue.innerHTML = '<i data-lucide="check"></i> Continue';
            
            lucide.createIcons();
            isDiagActive = true;
        }
    } else if (diagData.status === 'running' && isDiagActive && !isRegenerating) {
        closeDiagnosticUI();
    }
}

export function closeDiagnosticUI() {
    document.getElementById('diagnostic-overlay').style.display = 'none';
    isDiagActive = false;
    isRegenerating = false;
    currentDiagGen = null;
    document.getElementById('diag-img').style.opacity = '1';
}

function renderDiagParams(params) {
    const container = document.getElementById('diag-params');
    let html = '';
    for (const [key, val] of Object.entries(params)) {
        const displayVal = Array.isArray(val) ? val.join(', ') : val;
        html += `
        <div style="margin-bottom: 12px;">
            <label style="display: block; font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">${key}</label>
            <input type="text" data-diag-key="${key}" value="${displayVal}" style="width: 100%; padding: 8px; border: 1px solid var(--border-colour); outline: none;">
        </div>`;
    }
    container.innerHTML = html;
}

function getDiagParams() {
    const inputs = document.querySelectorAll('#diag-params input');
    const params = {};
    inputs.forEach(inp => {
        const key = inp.dataset.diagKey;
        let val = inp.value;
        if (val.includes(',')) {
            params[key] = val.split(',').map(v => {
                const num = Number(v.trim());
                return isNaN(num) ? v.trim() : num;
            });
        } else {
            const num = Number(val);
            params[key] = isNaN(num) ? val : num;
        }
    });
    return params;
}