// steps.js
// This file handles the UI configuration, HTML generation, and event listeners for pipeline steps.

const uiConfig = {
    // These steps cannot be moved or deleted
    fixedSteps: ["Load OG1", "Derive CTD", "Find Profiles", "Data Export"],
    
    // These steps will NOT show the "New parameter..." box
    noCustomParams: ["Load OG1", "Data Export", "Derive CTD", "Find Profiles", "Find Profile Direction"],
    
    colours: {
        load: 'var(--google-blue)',
        qc: 'var(--google-red)',
        export: 'var(--google-green)',
        default: 'var(--google-yellow)'
    }
};

function renderParams(params, basePath, availableQcList, stepName) {
    let html = '';
    for (const [key, val] of Object.entries(params || {})) {
        const path = `${basePath}.${key}`;
        
        if (key === 'qc_settings' && typeof val === 'object') {
            html += `<div class="form-row" style="align-items:flex-start;"><label>QC Settings</label><div style="flex:1; border-left:2px solid var(--border-colour); padding-left:12px;">`;
            for (const [qcTest, qcParams] of Object.entries(val || {})) {
                html += `<div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:4px; font-weight:600; font-size:12px;">${qcTest} <button class="danger btn-del-param" data-path="${path}.${qcTest}"><i data-lucide="x"></i></button></div>`;
                html += `<div style="margin-bottom:12px;">${renderParams(qcParams, `${path}.${qcTest}`, availableQcList, stepName)}</div>`;
            }
            html += `<div style="display:flex; gap:8px;"><select class="sel-add-qc" data-path="${path}"><option value="">Add Test...</option>${availableQcList.map(t=>`<option value="${t}">${t}</option>`).join('')}</select></div>`;
            html += `</div></div>`;
        } else if (val !== null && typeof val === 'object' && !Array.isArray(val)) {
            html += `<div class="form-row" style="align-items:flex-start;"><label>${key}</label><div style="flex:1; border-left:2px solid var(--border-colour); padding-left:12px;">${renderParams(val, path, availableQcList, stepName)}</div></div>`;
        } else {
            const display = Array.isArray(val) ? val.join(', ') : val;
            const isPath = key.toLowerCase().includes('path') || key.toLowerCase().includes('file');
            
            html += `<div class="form-row">
                <label>${key}</label>
                <input type="text" data-path="${path}" value='${display}'>
                ${isPath ? `<button class="btn-browse" data-path="${path}" style="padding: 6px; flex-shrink: 0;"><i data-lucide="folder-search"></i> Browse</button>` : ''}
            </div>`;
        }
    }
    
    // Conditionally hide the Add Parameter box based on our config
    if (!uiConfig.noCustomParams.includes(stepName)) {
        html += `<div class="form-row" style="margin-top:8px;"><input type="text" class="new-key-input" placeholder="New parameter..." style="max-width:140px;"><button class="btn-add-param" data-path="${basePath}">Add</button></div>`;
    }
    return html;
}

function generateStepsHTML(configData, activeStepIndex, availableStepsList, availableQcList) {
    let html = `
        <div class="section-header"><i data-lucide="settings"></i> Pipeline Settings</div>
        <div class="config-block">
            <div class="form-row"><label>Pipeline Name</label><input type="text" data-path="pipeline.name" value="${configData.pipeline.name || ''}"></div>
            <div class="form-row"><label>Description</label><textarea data-path="pipeline.description">${configData.pipeline.description || ''}</textarea></div>
            <div class="form-row"><label>Visualisation</label><input type="checkbox" data-path="pipeline.visualisation" ${configData.pipeline.visualisation ? 'checked' : ''}></div>
        </div>
        <div class="section-header"><i data-lucide="activity"></i> Processing Steps</div>
    `;

    configData.steps.forEach((step, idx) => {
        const isFixed = uiConfig.fixedSteps.includes(step.name);
        const isActive = activeStepIndex === idx;
        const colour = step.name.includes('Load') ? uiConfig.colours.load : step.name.includes('QC') ? uiConfig.colours.qc : step.name.includes('Export') ? uiConfig.colours.export : uiConfig.colours.default;

        html += `
            <div class="step-row ${isActive ? 'active-step' : ''}" data-index="${idx}" ${!isFixed ? 'draggable="true"' : ''}>
                <div class="step-header">
                    ${isFixed ? `<i data-lucide="lock" style="color:var(--text-muted); width:14px;"></i>` : `<div class="drag-handle"><i data-lucide="grip-vertical"></i></div>`}
                    <div class="step-number" style="background-color: ${colour}">${idx + 1}</div>
                    <div class="step-title">${step.name}</div>
                    ${!isFixed ? `<button class="danger btn-del-step" data-index="${idx}"><i data-lucide="trash-2"></i></button>` : ''}
                </div>
                <div class="step-body">
                    ${renderParams(step.parameters || {}, `steps.${idx}.parameters`, availableQcList, step.name)}
                    <div class="form-row" style="margin-top:16px; padding-top:16px; border-top:1px solid var(--border-colour)">
                        <label>Diagnostics</label><input type="checkbox" data-path="steps.${idx}.diagnostics" ${step.diagnostics ? 'checked' : ''}>
                    </div>
                </div>
            </div>
        `;
    });

    html += `
        <div class="add-step-wrapper">
            <div style="position: relative;">
                <select id="select-new-step" class="add-step-select">
                    <option value="" disabled selected>+ Add Processing Step</option>
                    ${availableStepsList.map(s => `<option value="${s}">${s}</option>`).join('')}
                </select>
            </div>
        </div>
    `;

    return html;
}

// All event listeners for the steps are now managed here.
function attachStepListeners(context) {
    const { 
        stepsList, configData, syncYaml, parseAndRenderUI, 
        getNested, setNested, highlightYamlStep, setActiveStepIndex
    } = context;

    // Standard Inputs
    stepsList.querySelectorAll('input[type="text"]:not(.new-key-input), textarea').forEach(el => {
        el.addEventListener('change', e => {
            let val = e.target.value;
            if (val === 'true') val = true; else if (val === 'false') val = false; else if (!isNaN(val) && val !== '') val = Number(val);
            setNested(configData, e.target.dataset.path, val); syncYaml();
        });
    });

    // Checkboxes
    stepsList.querySelectorAll('input[type="checkbox"]').forEach(el => {
        el.addEventListener('change', e => { setNested(configData, e.target.dataset.path, e.target.checked); syncYaml(); });
    });

    // Accordion / Highlighting
    stepsList.querySelectorAll('.step-header').forEach(el => {
        el.addEventListener('click', e => {
            if (['INPUT','BUTTON','SELECT','I'].includes(e.target.tagName)) return;
            const newIndex = parseInt(el.parentElement.dataset.index);
            setActiveStepIndex(newIndex);
            parseAndRenderUI();
            highlightYamlStep(newIndex);
        });
    });

    // Drag and Drop
    stepsList.querySelectorAll('.step-row[draggable="true"]').forEach(row => {
        row.addEventListener('dragstart', e => { 
            e.dataTransfer.setData('text/plain', row.dataset.index); // required for Firefox
            row.classList.add('dragging'); 
        });
        row.addEventListener('dragend', () => row.classList.remove('dragging'));
        row.addEventListener('dragover', e => e.preventDefault());
        row.addEventListener('drop', e => {
            e.preventDefault();
            const sourceIdx = parseInt(e.dataTransfer.getData('text/plain') || document.querySelector('.dragging').dataset.index);
            const targetIdx = parseInt(row.dataset.index);
            if (uiConfig.fixedSteps.includes(configData.steps[targetIdx].name)) return; // Protect fixed steps
            
            const item = configData.steps.splice(sourceIdx, 1)[0];
            configData.steps.splice(targetIdx, 0, item);
            syncYaml(); parseAndRenderUI();
        });
    });

    // Browse Button Integration
    stepsList.querySelectorAll('.btn-browse').forEach(btn => {
        btn.addEventListener('click', async e => {
            e.preventDefault();
            try {
                const res = await fetch('/api/browse');
                const data = await res.json();
                if (data.path) {
                    setNested(configData, btn.dataset.path, data.path);
                    syncYaml(); 
                    parseAndRenderUI();
                }
            } catch (err) {
                console.error("Failed to browse files:", err);
            }
        });
    });

    // Deletes and Adds
    stepsList.querySelectorAll('.btn-del-step').forEach(btn => btn.addEventListener('click', () => { 
        configData.steps.splice(btn.dataset.index, 1); syncYaml(); parseAndRenderUI(); 
    }));
    
    stepsList.querySelectorAll('.btn-del-param').forEach(btn => btn.addEventListener('click', () => { 
        const keys = btn.dataset.path.split('.'); const last = keys.pop(); 
        const obj = getNested(configData, keys.join('.')); delete obj[last]; 
        syncYaml(); parseAndRenderUI(); 
    }));
    
    stepsList.querySelectorAll('.sel-add-qc').forEach(sel => sel.addEventListener('change', e => { 
        if (e.target.value) { setNested(configData, `${e.target.dataset.path}.${e.target.value}`, {}); syncYaml(); parseAndRenderUI(); } 
    }));
    
    stepsList.querySelectorAll('.btn-add-param').forEach(btn => {
        btn.addEventListener('click', e => {
            const key = e.target.parentElement.querySelector('.new-key-input').value.trim();
            if (!key) return;
            let obj = getNested(configData, btn.dataset.path) || {};
            obj[key] = ""; setNested(configData, btn.dataset.path, obj); syncYaml(); parseAndRenderUI();
        });
    });

    const newStepSel = document.getElementById('select-new-step');
    if (newStepSel) {
        newStepSel.addEventListener('change', e => {
            if (!e.target.value) return;
            configData.steps.push({ name: e.target.value, parameters: e.target.value === 'Apply QC' ? { qc_settings: {} } : {}, diagnostics: false });
            syncYaml(); parseAndRenderUI();
        });
    }
}