/**
 * SYNTHESIS-ZERO · Master Workstation Controller
 * Page 1: Original QR Device Synchronization Stage
 * Page 2: Minimalist Two-Half Studio (Single Vertical Divider, Plain White Canvas)
 */

// Global State
let studioScene, studioCamera, studioRenderer, studioControls;
let atomMeshGroup, bondMeshGroup, unitCellBoxGroup;
let isRotating = true;
let isDeviceConnected = false;
let currentMaterial = null;
let activeRedesignPayload = null;
let lastKnownScanId = 0;

// High-contrast color palette optimized for pure white background
const ELEMENT_PALETTE = {
    'C': 0x1e293b,   // Slate black
    'H': 0x94a3b8,   // Subtle grey
    'O': 0x059669,   // Vivid emerald
    'N': 0x2563eb,   // Royal sapphire
    'Al': 0x0284c7,  // Cerulean blue
    'Fe': 0xd97706,  // Amber rust
    'Cu': 0xb45309,  // Copper orange
    'Ti': 0x4f46e5,  // Deep indigo
    'Si': 0x0d9488,  // Teal
    'default': 0x64748b
};

document.addEventListener('DOMContentLoaded', () => {
    initThreeStudio();
    bindUIEvents();
    loadConnectionInfo();
    checkCurrentScanState();

    // Polling intervals for companion phone
    setInterval(checkDeviceConnection, 1200);
    setInterval(checkForPhoneSync, 1500);
});

// --- 1. THREE.JS 3D CANVAS (PURE WHITE BACKGROUND) ---
function initThreeStudio() {
    const canvas = document.getElementById('threeStudioCanvas');
    const stage = document.getElementById('canvasStage');
    if (!canvas || !stage) return;

    studioScene = new THREE.Scene();
    studioScene.background = new THREE.Color(0xffffff); // Pure white background

    studioCamera = new THREE.PerspectiveCamera(45, stage.clientWidth / stage.clientHeight, 0.1, 100);
    studioCamera.position.set(0, 0, 8.5);

    studioRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
    studioRenderer.setSize(stage.clientWidth, stage.clientHeight);
    studioRenderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    studioRenderer.setClearColor(0xffffff, 1);

    studioControls = new THREE.OrbitControls(studioCamera, studioRenderer.domElement);
    studioControls.enableDamping = true;
    studioControls.dampingFactor = 0.05;
    studioControls.maxDistance = 25;
    studioControls.minDistance = 2;

    // Clean crisp studio lighting on white
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.95);
    studioScene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0xffffff, 0.6);
    dirLight1.position.set(10, 15, 10);
    studioScene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xe2e8f0, 0.4);
    dirLight2.position.set(-10, -10, -10);
    studioScene.add(dirLight2);

    atomMeshGroup = new THREE.Group();
    bondMeshGroup = new THREE.Group();
    unitCellBoxGroup = new THREE.Group();

    studioScene.add(atomMeshGroup);
    studioScene.add(bondMeshGroup);
    studioScene.add(unitCellBoxGroup);

    window.addEventListener('resize', onStudioResize);
    animateStudio();
}

function onStudioResize() {
    const stage = document.getElementById('canvasStage');
    if (!stage || !studioRenderer || !studioCamera) return;
    const width = stage.clientWidth;
    const height = stage.clientHeight;
    if (width === 0 || height === 0) return;
    studioCamera.aspect = width / height;
    studioCamera.updateProjectionMatrix();
    studioRenderer.setSize(width, height);
}

function animateStudio() {
    requestAnimationFrame(animateStudio);
    if (isRotating && atomMeshGroup) {
        atomMeshGroup.rotation.y += 0.005;
        bondMeshGroup.rotation.y += 0.005;
        unitCellBoxGroup.rotation.y += 0.005;
    }
    if (studioControls) studioControls.update();
    if (studioRenderer && studioScene && studioCamera) {
        studioRenderer.render(studioScene, studioCamera);
    }
}

function render3DStructure(structure) {
    if (!structure || !atomMeshGroup || !bondMeshGroup) return;

    // Clear previous meshes
    while (atomMeshGroup.children.length > 0) atomMeshGroup.remove(atomMeshGroup.children[0]);
    while (bondMeshGroup.children.length > 0) bondMeshGroup.remove(bondMeshGroup.children[0]);
    while (unitCellBoxGroup.children.length > 0) unitCellBoxGroup.remove(unitCellBoxGroup.children[0]);

    const atoms = structure.atoms || [];
    const bonds = structure.bonds || [];
    if (atoms.length === 0) return;

    // Compute bounding center to center the molecule at origin
    let minX = Infinity, maxX = -Infinity;
    let minY = Infinity, maxY = -Infinity;
    let minZ = Infinity, maxZ = -Infinity;

    atoms.forEach(a => {
        minX = Math.min(minX, a.x); maxX = Math.max(maxX, a.x);
        minY = Math.min(minY, a.y); maxY = Math.max(maxY, a.y);
        minZ = Math.min(minZ, a.z); maxZ = Math.max(maxZ, a.z);
    });

    const centerX = (minX + maxX) / 2;
    const centerY = (minY + maxY) / 2;
    const centerZ = (minZ + maxZ) / 2;

    const span = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1.0);
    const scaleFactor = 4.2 / span;

    const atomPositions = [];

    // Render Atomic Spheres
    atoms.forEach(atom => {
        const x = (atom.x - centerX) * scaleFactor;
        const y = (atom.y - centerY) * scaleFactor;
        const z = (atom.z - centerZ) * scaleFactor;
        atomPositions.push(new THREE.Vector3(x, y, z));

        const radius = Math.max(0.18, (atom.radius || 0.8) * 0.32 * Math.min(scaleFactor * 1.5, 1.2));
        const sphereGeo = new THREE.SphereGeometry(radius, 28, 28);
        const colorHex = ELEMENT_PALETTE[atom.element] || ELEMENT_PALETTE.default;

        const sphereMat = new THREE.MeshStandardMaterial({
            color: colorHex,
            roughness: 0.25,
            metalness: 0.15,
        });

        const mesh = new THREE.Mesh(sphereGeo, sphereMat);
        mesh.position.set(x, y, z);
        atomMeshGroup.add(mesh);
    });

    // Render Covalent Bonds (Clean subtle cylinders)
    const bondMat = new THREE.MeshStandardMaterial({
        color: 0xcbd5e1,
        roughness: 0.4,
        metalness: 0.1,
    });

    bonds.forEach(bond => {
        const p1 = atomPositions[bond.source];
        const p2 = atomPositions[bond.target];
        if (!p1 || !p2) return;

        const dist = p1.distanceTo(p2);
        const cylGeo = new THREE.CylinderGeometry(0.045, 0.045, dist, 12);
        const mesh = new THREE.Mesh(cylGeo, bondMat);

        const midpoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        mesh.position.copy(midpoint);
        mesh.quaternion.setFromUnitVectors(new THREE.Vector3(0, 1, 0), new THREE.Vector3().subVectors(p2, p1).normalize());

        bondMeshGroup.add(mesh);
    });

    setText('hudAtomCount', `${atoms.length} Atoms · ${bonds.length} Bonds`);
    onStudioResize();
}

// --- 2. SCREEN ROUTING & WORKFLOW STATES ---

function showScreen(screen) {
    const homePage = document.getElementById('homePage');
    const studioPage = document.getElementById('studioPage');

    if (screen === 'studio') {
        homePage.style.display = 'none';
        studioPage.style.display = 'block';
        setTimeout(onStudioResize, 60);
    } else {
        studioPage.style.display = 'none';
        homePage.style.display = 'block';
    }
}

function showStudioWaitingState() {
    const waitingBox = document.getElementById('studioWaitingBox');
    const splitBox = document.getElementById('studioSplitBox');
    if (waitingBox) waitingBox.style.display = 'flex';
    if (splitBox) splitBox.style.display = 'none';
    currentMaterial = null;
    activeRedesignPayload = null;
}

function showStudioActiveState(data) {
    const waitingBox = document.getElementById('studioWaitingBox');
    const splitBox = document.getElementById('studioSplitBox');

    if (waitingBox) waitingBox.style.display = 'none';
    if (splitBox) splitBox.style.display = 'flex';

    currentMaterial = data;
    lastKnownScanId = data.scan_id || lastKnownScanId;

    // Automatically transition to studio view
    showScreen('studio');

    // Left Half: Photos & Schematic
    const photoImg = document.getElementById('capturedProductImg');
    const bpImg = document.getElementById('cadBlueprintImg');

    if (data.captured_image_uri && photoImg) {
        photoImg.src = data.captured_image_uri;
        photoImg.style.display = 'block';
    }
    if (data.schematic_uri && bpImg) {
        bpImg.src = data.schematic_uri;
    }
    switchToPhotoView();

    // Right Half: Results & Specs
    setText('hudCategory', data.top_category ? `Vision: ${data.top_category}` : 'Analyzed Material');
    setText('hudMaterialName', data.product_name || 'Physical Product');
    setText('valFormulaDisplay', data.formula || '—');
    setText('valDensityDisplay', data.density_g_cm3 ? `${data.density_g_cm3} g/cm³` : '—');

    // Reset Redesign metrics until Redesign is triggered
    setText('valBaselineEnergy', 'Ready to evaluate');
    setText('valRelaxedEnergy', '—');
    setText('valEnergyGain', '—');
    setText('valMeanDisplacement', '—');
    setText('valMaxDisplacement', '—');
    setText('valUncertainty', '—');
    document.getElementById('structureToggleRow').style.display = 'none';

    // Top 5 Probabilities
    const top5Container = document.getElementById('valTop5List');
    if (top5Container) {
        if (data.top5_predictions && data.top5_predictions.length > 0) {
            top5Container.innerHTML = data.top5_predictions.map(p => `
                <span class="vision-tag">${p.category} <strong>${p.probability}%</strong></span>
            `).join('');
            document.getElementById('visionTop5Strip').style.display = 'flex';
        } else {
            document.getElementById('visionTop5Strip').style.display = 'none';
        }
    }

    // 3D Lattice
    if (data.structure_3d) {
        render3DStructure(data.structure_3d);
    }
    setTimeout(onStudioResize, 60);
}

function switchToPhotoView() {
    const btnPhoto = document.getElementById('btnViewPhoto');
    const btnBp = document.getElementById('btnViewBlueprint');
    const photoImg = document.getElementById('capturedProductImg');
    const bpImg = document.getElementById('cadBlueprintImg');

    btnPhoto?.classList.add('active');
    btnBp?.classList.remove('active');

    if (photoImg) photoImg.style.display = 'block';
    if (bpImg) bpImg.style.display = 'none';
}

function switchToBlueprintView() {
    const btnPhoto = document.getElementById('btnViewPhoto');
    const btnBp = document.getElementById('btnViewBlueprint');
    const photoImg = document.getElementById('capturedProductImg');
    const bpImg = document.getElementById('cadBlueprintImg');

    btnBp?.classList.add('active');
    btnPhoto?.classList.remove('active');

    if (bpImg && bpImg.getAttribute('src')) {
        bpImg.style.display = 'block';
        if (photoImg) photoImg.style.display = 'none';
    }
}

// --- 3. EVENT BINDINGS & REDESIGN PIPELINE ---
function bindUIEvents() {
    // Page 1 Navigation
    document.getElementById('btnContinueToStudio')?.addEventListener('click', () => {
        showScreen('studio');
    });

    document.getElementById('btnSkipDirect')?.addEventListener('click', () => {
        showScreen('studio');
    });

    document.getElementById('btnNavReconnect')?.addEventListener('click', () => {
        showScreen('home');
    });

    // Simulate phone connection for test/preview
    document.getElementById('btnSimulatePhone')?.addEventListener('click', async () => {
        try {
            await fetch('/api/device_simulate_connect', { method: 'POST' });
            setDeviceConnectedUI(true, 'Apple iPhone (Simulated)');
        } catch (e) {
            setDeviceConnectedUI(true, 'Test Device');
        }
    });

    // View Toggles: Optical Image vs Physical Sketch
    document.getElementById('btnViewPhoto')?.addEventListener('click', switchToPhotoView);
    document.getElementById('btnViewBlueprint')?.addEventListener('click', switchToBlueprintView);

    // 3D Controls
    document.getElementById('btnToggleSpin')?.addEventListener('click', () => {
        isRotating = !isRotating;
        document.getElementById('btnToggleSpin').textContent = `Spin: ${isRotating ? 'On' : 'Off'}`;
    });

    document.getElementById('btnResetCamera')?.addEventListener('click', () => {
        if (studioCamera) studioCamera.position.set(0, 0, 8.5);
        if (studioControls) studioControls.reset();
        if (atomMeshGroup) atomMeshGroup.rotation.set(0, 0, 0);
        if (bondMeshGroup) bondMeshGroup.rotation.set(0, 0, 0);
    });

    // Single-Click Autonomous Neural Redesign
    const btnRedesign = document.getElementById('btnTriggerRedesign');
    if (btnRedesign) {
        btnRedesign.addEventListener('click', async () => {
            btnRedesign.classList.add('processing');
            btnRedesign.innerHTML = `
                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5" style="animation: spin 1s linear infinite;"><path d="M21 12a9 9 0 1 1-6.219-8.56"/></svg>
                <span>Relaxing Molecular Coordinates...</span>
            `;

            try {
                const response = await fetch('/api/redesign_product', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({})
                });
                const data = await response.json();
                activeRedesignPayload = data;

                // Update physical sketch with relaxed projection
                const bpImg = document.getElementById('cadBlueprintImg');
                if (bpImg && data.schematic_uri) {
                    bpImg.src = data.schematic_uri;
                }

                // Update plain borderless metrics
                const metrics = data.neural_metrics || {};
                setText('valBaselineEnergy', `${metrics.formation_energy_baseline_eV ?? '--'} eV/atom`);
                setText('valRelaxedEnergy', `${metrics.formation_energy_relaxed_eV ?? '--'} eV/atom`);
                setText('valEnergyGain', `${metrics.energy_delta_eV ?? '--'} eV/atom`);
                setText('valUncertainty', `± ${metrics.prediction_uncertainty ?? '--'} eV`);
                setText('valMeanDisplacement', `${metrics.mean_atomic_displacement_A ?? '--'} Å`);
                setText('valMaxDisplacement', `${metrics.max_atomic_displacement_A ?? '--'} Å`);

                const model = data.redesigned_model;
                if (model) {
                    setText('hudMaterialName', model.name);
                    setText('valFormulaDisplay', model.formula);
                    setText('valDensityDisplay', `${model.calculated_density_g_cm3} g/cm³`);
                    render3DStructure(model.structure_3d);
                }

                // Enable Baseline vs AI Rearranged toggle
                document.getElementById('structureToggleRow').style.display = 'flex';
                document.getElementById('btnShowRedesign').classList.add('active');
                document.getElementById('btnShowBaseline').classList.remove('active');

            } catch (err) {
                console.error('Redesign execution error:', err);
            } finally {
                btnRedesign.classList.remove('processing');
                btnRedesign.innerHTML = `
                    <svg width="20" height="20" viewBox="0 0 24 24" fill="currentColor">
                        <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"></polygon>
                    </svg>
                    <span>Redesign Material</span>
                `;
            }
        });
    }

    // Baseline vs Relaxed Structure Switcher
    document.getElementById('btnShowBaseline')?.addEventListener('click', () => {
        document.getElementById('btnShowBaseline').classList.add('active');
        document.getElementById('btnShowRedesign').classList.remove('active');
        if (activeRedesignPayload?.baseline_model?.structure_3d) {
            render3DStructure(activeRedesignPayload.baseline_model.structure_3d);
        } else if (currentMaterial?.structure_3d) {
            render3DStructure(currentMaterial.structure_3d);
        }
    });

    document.getElementById('btnShowRedesign')?.addEventListener('click', () => {
        document.getElementById('btnShowRedesign').classList.add('active');
        document.getElementById('btnShowBaseline').classList.remove('active');
        if (activeRedesignPayload?.redesigned_model?.structure_3d) {
            render3DStructure(activeRedesignPayload.redesigned_model.structure_3d);
        }
    });

    // Dropzone & File Ingestion
    const dropZone = document.getElementById('desktopDropZone');
    const fileInput = document.getElementById('desktopFileInput');

    if (dropZone && fileInput) {
        dropZone.addEventListener('click', () => fileInput.click());
        dropZone.addEventListener('dragover', e => e.preventDefault());
        dropZone.addEventListener('drop', e => {
            e.preventDefault();
            if (e.dataTransfer.files.length > 0) {
                uploadMaterialPhoto(e.dataTransfer.files[0]);
            }
        });
        fileInput.addEventListener('change', e => {
            if (e.target.files.length > 0) {
                uploadMaterialPhoto(e.target.files[0]);
            }
        });
    }
}

// --- 4. PHONE SYNC & API SERVICES ---

async function checkCurrentScanState() {
    try {
        const data = await fetch('/api/latest_scan').then(r => r.json());
        if (data && data.scan_id && data.captured_image_uri) {
            showStudioActiveState(data);
        } else {
            showStudioWaitingState();
        }
    } catch (e) {
        showStudioWaitingState();
    }
}

async function uploadMaterialPhoto(file) {
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/api/scan_material', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        showStudioActiveState(data);
    } catch (err) {
        console.error('Photo upload failed', err);
    }
}

async function checkForPhoneSync() {
    try {
        const status = await fetch('/api/status').then(r => r.json());
        if (status.latest_scan_id && status.latest_scan_id > lastKnownScanId) {
            const newScan = await fetch('/api/latest_scan').then(r => r.json());
            lastKnownScanId = newScan.scan_id;
            showStudioActiveState(newScan);
        }
        if (status.device_connected && !isDeviceConnected) {
            setDeviceConnectedUI(true, status.device_name);
        }
    } catch (err) {
        // Silent catch
    }
}

async function loadConnectionInfo() {
    try {
        const info = await fetch('/api/connect_info').then(r => r.json());
        setText('homeDirectUrl', info.mobile_url);
        if (info.qr_image) {
            document.getElementById('mainQrCodeImg').src = info.qr_image;
        }
        if (info.device_connected) {
            setDeviceConnectedUI(true, info.device_name);
        }
    } catch (e) {
        // Fallback
    }
}

async function checkDeviceConnection() {
    try {
        const status = await fetch('/api/device_status').then(r => r.json());
        if (status.connected && !isDeviceConnected) {
            setDeviceConnectedUI(true, status.device_name);
        } else if (!status.connected && isDeviceConnected) {
            setDeviceConnectedUI(false);
        }
    } catch (e) {
        // Silent catch
    }
}

function setDeviceConnectedUI(connected, deviceName = 'Companion Phone') {
    isDeviceConnected = connected;
    const qrStage = document.getElementById('qrStageBox');
    const connectedStage = document.getElementById('connectedStageBox');
    const label = document.getElementById('detectedPhoneLabel');

    if (connected) {
        if (qrStage) qrStage.style.display = 'none';
        if (connectedStage) connectedStage.style.display = 'flex';
        if (label) label.textContent = deviceName;
    } else {
        if (qrStage) qrStage.style.display = 'flex';
        if (connectedStage) connectedStage.style.display = 'none';
    }
}

function setText(id, val) {
    const el = document.getElementById(id);
    if (el) el.textContent = val;
}
