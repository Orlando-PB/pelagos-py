const uiConfig = {
    colours: {
        load: 'var(--google-blue)',
        qc: 'var(--google-red)',
        export: 'var(--google-green)',
        adjustment: '#0ea5e9',
        default: 'var(--google-yellow)'
    },
    adjustments: [
        "Salinity Adjustment", "Chla Deep Correction", "Chla Quenching Correction", 
        "BBP from Beta", "Isolate BBP Spikes", "Derive Oxygen"
    ]
};

function renderParams(params, basePath, availableQcList) {
    let html = '';
    const entries = Object.entries(params || {});
    
    for (const [key, val] of entries) {
        const path = `${basePath}.${key}`;
        
        if (key === 'qc_settings' && typeof val === 'object' && val !== null) {
            html += `
            <div class="form-row" style="align-items:flex-start;">
                <label>QC Settings</label>
                <div style="flex:1; border-left:2px solid var(--border-colour); padding-left:16px; min-height: 20px;">`;
            
            for (const [qcTest, qcParams] of Object.entries(val)) {
                const safeQcParams = qcParams || {};
                const isDiag = safeQcParams.diagnostics === true;
                
                html += `
                <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px; background:#f1f5f9; padding:6px 10px; border-radius:var(--radius);">
                    <span style="font-weight:700; color:#1e293b; font-size:11px;">${qcTest}</span>
                    <div style="display:flex; align-items:center; gap:12px;">
                        <div style="display:flex; align-items:center; gap:6px;" title="Toggle specific diagnostics for this test">
                            <i data-lucide="line-chart" style="width:14px; height:14px; color:var(--text-muted);"></i>
                            <label class="switch" style="transform: scale(0.75); transform-origin: right center;">
                                <input type="checkbox" class="qc-diag-toggle" data-path="${path}.${qcTest}.diagnostics" ${isDiag ? 'checked' : ''}>
                                <span class="slider"></span>
                            </label>
                        </div>
                        <button class="danger btn-del-param" data-path="${path}.${qcTest}" style="padding:4px; border-radius:var(--radius);" title="Remove this QC test"><i data-lucide="x" style="width:14px; height:14px;"></i></button>
                    </div>
                </div>`;
                
                const { diagnostics, ...pureParams } = safeQcParams;
                const nestedHtml = renderParams(pureParams, `${path}.${qcTest}`, availableQcList);
                if (nestedHtml) {
                    html += `<div style="margin-bottom:16px; padding-left:8px;">${nestedHtml}</div>`;
                }
            }
            
            html += `
                    <div style="display:flex; gap:8px; margin-top: 8px;">
                        <select class="sel-add-qc" data-path="${path}" style="font-size: 11px; padding: 4px 8px; border-radius:var(--radius);">
                            <option value="">+ Add QC Test...</option>
                            ${availableQcList.map(t => `<option value="${t}">${t}</option>`).join('')}
                        </select>
                    </div>
                </div>
            </div>`;
        } else if (val !== null && typeof val === 'object' && !Array.isArray(val)) {
            html += `<div class="form-row" style="align-items:flex-start;"><label>${key}</label><div style="flex:1; border-left:2px solid var(--border-colour); padding-left:16px;">${renderParams(val, path, availableQcList)}</div></div>`;
        } else {
            const display = Array.isArray(val) ? val.join(', ') : (val === null ? '' : val);
            const isPath = key.toLowerCase().includes('path') || key.toLowerCase().includes('file');
            
            html += `
            <div class="form-row">
                <label>${key}</label>
                <input type="text" data-path="${path}" value='${display}'>
                ${isPath ? `<button class="btn-browse" data-path="${path}" style="padding: 10px; flex-shrink: 0;"><i data-lucide="folder-search"></i> Browse</button>` : ''}
            </div>`;
        }
    }
    return html;
}

function generateStepsHTML(configData, activeStepIndex, availableQcList) {
    let html = `
        <div style="padding: 20px; border-bottom: 1px solid var(--border-colour); background: white;">
            <div style="display: flex; align-items: center; gap: 12px; margin-bottom: 20px;">
                <div style="padding: 8px; background: #eff6ff; border-radius: var(--radius); color: var(--google-blue);">
                    <i data-lucide="settings"></i>
                </div>
                <h2 style="font-size: 16px; font-weight: 700; color: #1e293b;">Pipeline Configuration</h2>
            </div>
            <div style="display: grid; gap: 16px;">
                <div>
                    <label style="display: block; font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Pipeline Name</label>
                    <input type="text" data-path="pipeline.name" value="${configData.pipeline.name || ''}" style="width: 100%;">
                </div>
                <div>
                    <label style="display: block; font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px; margin-bottom: 8px;">Description</label>
                    <textarea data-path="pipeline.description" style="width: 100%; height: 60px; resize: none;">${configData.pipeline.description || ''}</textarea>
                </div>
                <div style="display: flex; align-items: center; gap: 12px; padding-top: 4px;">
                    <label class="switch">
                        <input type="checkbox" data-path="pipeline.visualisation" ${configData.pipeline.visualisation ? 'checked' : ''}>
                        <span class="slider"></span>
                    </label>
                    <span style="font-size: 11px; font-weight: 700; color: #475569; text-transform: uppercase; letter-spacing: 0.5px;">Enable Visualisation</span>
                </div>
            </div>
        </div>

        <div style="display: flex; align-items: center; justify-content: space-between; padding: 16px 20px; background: #f8fafc; border-bottom: 1px solid var(--border-colour);">
            <div style="display: flex; align-items: center; gap: 12px;">
                <div style="padding: 8px; background: #f0fdf4; border-radius: var(--radius); color: var(--google-green);">
                    <i data-lucide="activity"></i>
                </div>
                <h2 style="font-size: 16px; font-weight: 700; color: #1e293b;">Processing Steps</h2>
            </div>
            <span style="font-size: 10px; font-weight: 700; color: #94a3b8; background: white; border: 1px solid var(--border-colour); padding: 6px 12px; border-radius: var(--radius); text-transform: uppercase; letter-spacing: 1px;">
                ${configData.steps.filter(s => s.enabled).length} Enabled
            </span>
        </div>
    `;

    configData.steps.forEach((step, idx) => {
        const isActive = activeStepIndex === idx;
        
        let colour = uiConfig.colours.default;
        if (step.name.includes('Load')) colour = uiConfig.colours.load;
        else if (step.name.includes('QC')) colour = uiConfig.colours.qc;
        else if (step.name.includes('Export')) colour = uiConfig.colours.export;
        else if (uiConfig.adjustments.includes(step.name)) colour = uiConfig.colours.adjustment;

        html += `
            <div class="step-row ${isActive && step.enabled ? 'active-step' : ''} ${!step.enabled ? 'disabled-step' : ''}" data-index="${idx}">
                <div class="step-header">
                    <div class="step-icon-wrapper">
                        <div class="step-number" style="background-color: ${colour}">${idx + 1}</div>
                        <div class="step-title">${step.name}</div>
                    </div>
                    <label class="switch" onclick="event.stopPropagation()">
                        <input type="checkbox" class="step-toggle" data-index="${idx}" ${step.enabled ? 'checked' : ''}>
                        <span class="slider"></span>
                    </label>
                </div>
                <div class="step-body">
                    ${renderParams(step.parameters || {}, `steps.${idx}.parameters`, availableQcList)}
                    <div style="margin-top: 16px; padding-top: 16px; border-top: 1px solid var(--border-colour); display: flex; align-items: center; gap: 12px;">
                        <label class="switch">
                            <input type="checkbox" data-path="steps.${idx}.diagnostics" ${step.diagnostics ? 'checked' : ''}>
                            <span class="slider"></span>
                        </label>
                        <span style="font-size: 10px; font-weight: 700; color: #64748b; text-transform: uppercase; letter-spacing: 0.5px;">Enable Step Diagnostics</span>
                    </div>
                </div>
            </div>
        `;
    });

    return html;
}

function attachStepListeners(context) {
    const { 
        stepsList, configData, syncYaml, parseAndRenderUI, 
        getNested, setNested, highlightYamlStep, setActiveStepIndex
    } = context;

    stepsList.querySelectorAll('input[type="text"], textarea').forEach(el => {
        el.addEventListener('change', e => {
            let val = e.target.value;
            if (val === 'true') val = true; else if (val === 'false') val = false; else if (!isNaN(val) && val !== '') val = Number(val);
            setNested(configData, e.target.dataset.path, val); 
            
            const parts = e.target.dataset.path.split('.');
            if (parts[0] === 'steps' && configData.steps[parts[1]]?.name === "Load OG1") {
                if (typeof window.handleAutoOutputPath === 'function') {
                    window.handleAutoOutputPath(val);
                }
            }
            
            syncYaml();
            parseAndRenderUI();
        });
    });

    stepsList.querySelectorAll('input[type="checkbox"]:not(.step-toggle)').forEach(el => {
        el.addEventListener('change', e => { setNested(configData, e.target.dataset.path, e.target.checked); syncYaml(); });
    });

    stepsList.querySelectorAll('.step-toggle').forEach(el => {
        el.addEventListener('change', e => {
            const idx = parseInt(e.target.dataset.index);
            configData.steps[idx].enabled = e.target.checked;
            syncYaml(); 
            parseAndRenderUI();
        });
    });

    stepsList.querySelectorAll('.step-header').forEach(el => {
        el.addEventListener('click', e => {
            if (['INPUT','BUTTON','SELECT','I', 'LABEL', 'SPAN'].includes(e.target.tagName)) return;
            const newIndex = parseInt(el.parentElement.dataset.index);
            if (configData.steps[newIndex].enabled) {
                setActiveStepIndex(newIndex);
                parseAndRenderUI();
                highlightYamlStep(newIndex);
            }
        });
    });

    stepsList.querySelectorAll('.btn-browse').forEach(btn => {
        btn.addEventListener('click', async e => {
            e.preventDefault();
            try {
                const res = await fetch('/api/browse');
                const data = await res.json();
                if (data.path) {
                    setNested(configData, btn.dataset.path, data.path);
                    
                    const parts = btn.dataset.path.split('.');
                    if (parts[0] === 'steps' && configData.steps[parts[1]]?.name === "Load OG1") {
                        if (typeof window.handleAutoOutputPath === 'function') {
                            window.handleAutoOutputPath(data.path);
                        }
                    }
                    
                    syncYaml(); 
                    parseAndRenderUI();
                }
            } catch (err) {}
        });
    });
    
    stepsList.querySelectorAll('.btn-del-param').forEach(btn => btn.addEventListener('click', () => { 
        const keys = btn.dataset.path.split('.'); const last = keys.pop(); 
        const obj = getNested(configData, keys.join('.')); delete obj[last]; 
        syncYaml(); parseAndRenderUI(); 
    }));
    
    stepsList.querySelectorAll('.sel-add-qc').forEach(sel => sel.addEventListener('change', e => { 
        if (e.target.value) { setNested(configData, `${e.target.dataset.path}.${e.target.value}`, {}); syncYaml(); parseAndRenderUI(); } 
    }));

    stepsList.querySelectorAll('.qc-diag-toggle').forEach(el => {
        el.addEventListener('change', e => {
            setNested(configData, e.target.dataset.path, e.target.checked);
            syncYaml();
        });
    });
}