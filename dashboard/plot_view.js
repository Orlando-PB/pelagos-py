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

        const params = getDiagParams();
        const stepName = document.getElementById('diag-step-name').textContent;

        await fetch('/api/diagnostic/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: 'continue', parameters: params })
        });

        if (window.updateStepParameters) {
            window.updateStepParameters(stepName, params);
        }
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
            
            // Generate HTML string first, then set innerHTML once
            const container = document.getElementById('diag-params');
            container.innerHTML = renderDiagParamsRecursive(data.parameters);
            
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

/**
 * Recursive renderer to handle nested settings (like qc_handling_settings)
 */
function renderDiagParamsRecursive(params, parentPath = '') {
    let html = '';
    const entries = Object.entries(params || {});

    // Handle empty objects (like a QC test with no params)
    if (entries.length === 0 && parentPath !== '') {
        return `<div style="font-size: 10px; color: #94a3b8; font-style: italic; margin-bottom: 8px;">(No parameters)</div>`;
    }

    for (const [key, val] of entries) {
        const currentPath = parentPath ? `${parentPath}.${key}` : key;

        if (val !== null && typeof val === 'object' && !Array.isArray(val)) {
            // It's a dictionary - wrap in a container and recurse
            html += `
            <div style="margin-bottom: 16px; padding-left: 8px; border-left: 2px solid var(--border-colour);">
                <label style="display: block; font-size: 9px; font-weight: 800; color: var(--google-blue); text-transform: uppercase; margin-bottom: 6px;">${key}</label>
                ${renderDiagParamsRecursive(val, currentPath)}
            </div>`;
        } else {
            // It's a simple value - render the input
            const displayVal = Array.isArray(val) ? val.join(', ') : val;
            html += `
            <div style="margin-bottom: 12px;">
                <label style="display: block; font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; margin-bottom: 4px;">${key}</label>
                <input type="text" data-diag-path="${currentPath}" value="${displayVal}" style="width: 100%; padding: 8px; border: 1px solid var(--border-colour); outline: none; border-radius: 4px;">
            </div>`;
        }
    }
    return html;
}

function getDiagParams() {
    const inputs = document.querySelectorAll('#diag-params input');
    const params = {};

    inputs.forEach(inp => {
        const path = inp.dataset.diagPath.split('.');
        let val = inp.value;

        // Convert types
        if (val.includes(',') && !isNaN(parseFloat(val.split(',')[0]))) {
            val = val.split(',').map(v => {
                const num = Number(v.trim());
                return isNaN(num) ? v.trim() : num;
            });
        } else {
            const num = Number(val);
            if (!isNaN(num) && val.trim() !== '') {
                val = num;
            } else if (val.toLowerCase() === 'true') {
                val = true;
            } else if (val.toLowerCase() === 'false') {
                val = false;
            }
        }

        // Rebuild nested object
        let current = params;
        for (let i = 0; i < path.length; i++) {
            const key = path[i];
            if (i === path.length - 1) {
                current[key] = val;
            } else {
                current[key] = current[key] || {};
                current = current[key];
            }
        }
    });

    return params;
}