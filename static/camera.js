/**
 * SYNTHESIS-ZERO · Mobile Camera Portal Logic
 * Real-time camera capture, 3D WebGL crystal renderer, and desktop synchronization.
 */

let scene, camera, renderer, controls;
let atomMeshGroup, bondMeshGroup, unitCellBoxGroup;
let isRotating = true;
let currentMaterial = null;

document.addEventListener('DOMContentLoaded', () => {
    initThreeJS();
    initCamera();
    bindEvents();
    loadInitialMaterial();
    pingDeviceConnection();
    setInterval(pingDeviceConnection, 15000);
});

async function pingDeviceConnection() {
    try {
        await fetch('/api/device_ping', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ user_agent: navigator.userAgent })
        });
    } catch (e) {
        // silent ping
    }
}

function initThreeJS() {
    const canvas = document.getElementById('mobileCanvas3D');
    const container = canvas.parentElement;

    scene = new THREE.Scene();
    camera = new THREE.PerspectiveCamera(45, container.clientWidth / container.clientHeight, 0.1, 100);
    camera.position.set(0, 0, 9);

    renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: true });
    renderer.setSize(container.clientWidth, container.clientHeight);
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));

    controls = new THREE.OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.05;
    controls.enablePan = false;

    // Ambient & Directional Lights
    const ambientLight = new THREE.AmbientLight(0xffffff, 0.7);
    scene.add(ambientLight);

    const dirLight1 = new THREE.DirectionalLight(0x38bdf8, 1.2);
    dirLight1.position.set(10, 15, 10);
    scene.add(dirLight1);

    const dirLight2 = new THREE.DirectionalLight(0xffffff, 0.8);
    dirLight2.position.set(-10, -10, -10);
    scene.add(dirLight2);

    atomMeshGroup = new THREE.Group();
    bondMeshGroup = new THREE.Group();
    unitCellBoxGroup = new THREE.Group();

    scene.add(atomMeshGroup);
    scene.add(bondMeshGroup);
    scene.add(unitCellBoxGroup);

    window.addEventListener('resize', onWindowResize);
    animate();
}

function onWindowResize() {
    const canvas = document.getElementById('mobileCanvas3D');
    const container = canvas.parentElement;
    camera.aspect = container.clientWidth / container.clientHeight;
    camera.updateProjectionMatrix();
    renderer.setSize(container.clientWidth, container.clientHeight);
}

function animate() {
    requestAnimationFrame(animate);
    if (isRotating && atomMeshGroup) {
        atomMeshGroup.rotation.y += 0.008;
        bondMeshGroup.rotation.y += 0.008;
        unitCellBoxGroup.rotation.y += 0.008;
    }
    controls.update();
    renderer.render(scene, camera);
}

// Camera initialization
let videoTrack = null;
async function initCamera() {
    const video = document.getElementById('cameraStream');
    try {
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: { ideal: 'environment' } },
            audio: false
        });
        video.srcObject = stream;
        video.style.display = 'block';
        videoTrack = stream.getVideoTracks()[0];
    } catch (err) {
        console.warn('Direct camera feed unavailable; fallback to file picker mode.', err);
        const vf = document.getElementById('viewfinder');
        if (vf) vf.innerHTML = '<span class="viewfinder-hint">Tap "Pick Photo" to capture material</span>';
    }
}

function bindEvents() {
    // Snap photo
    document.getElementById('btnSnapPhoto').addEventListener('click', captureAndScan);

    // File picker
    document.getElementById('filePicker').addEventListener('change', event => {
        const file = event.target.files[0];
        if (file) uploadImageFile(file);
    });

    // 3D Controls
    document.getElementById('btnToggleRotate').addEventListener('click', () => {
        isRotating = !isRotating;
        document.getElementById('btnToggleRotate').textContent = `Spin: ${isRotating ? 'On' : 'Off'}`;
    });

    document.getElementById('btnResetView').addEventListener('click', () => {
        camera.position.set(0, 0, 9);
        controls.reset();
        atomMeshGroup.rotation.set(0, 0, 0);
        bondMeshGroup.rotation.set(0, 0, 0);
        unitCellBoxGroup.rotation.set(0, 0, 0);
    });

    // Quick chips
    document.querySelectorAll('.chip').forEach(chip => {
        chip.addEventListener('click', async () => {
            document.querySelectorAll('.chip').forEach(c => c.classList.remove('active'));
            chip.classList.add('active');
            const matKey = chip.getAttribute('data-mat');
            await scanByHint(matKey);
        });
    });
}

async function captureAndScan() {
    const video = document.getElementById('cameraStream');
    if (!video || !video.videoWidth) {
        document.getElementById('filePicker').click();
        return;
    }

    const canvas = document.createElement('canvas');
    canvas.width = Math.min(video.videoWidth, 800);
    canvas.height = Math.min(video.videoHeight, 800);
    const ctx = canvas.getContext('2d');
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    const dataUrl = canvas.toDataURL('image/jpeg', 0.85);

    showLoader('Analyzing camera snapshot…', 'MCRSA Neural Network extracting optical features');
    try {
        const response = await fetch('/api/scan_material', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_base64: dataUrl })
        });
        const data = await response.json();
        updateUIWithMaterial(data);
    } catch (err) {
        console.error('Scan failed', err);
        alert('Material scan failed. Please check server connection.');
    } finally {
        hideLoader();
    }
}

async function uploadImageFile(file) {
    showLoader('Uploading image…', 'Extracting lattice and elemental fingerprint');
    const formData = new FormData();
    formData.append('image', file);

    try {
        const response = await fetch('/api/scan_material', {
            method: 'POST',
            body: formData
        });
        const data = await response.json();
        updateUIWithMaterial(data);
    } catch (err) {
        console.error('Upload failed', err);
    } finally {
        hideLoader();
    }
}

async function scanByHint(hint) {
    showLoader('Reconstructing structure…', `Generating 3D coordinates for ${hint}`);
    try {
        const response = await fetch('/api/scan_material', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ hint })
        });
        const data = await response.json();
        updateUIWithMaterial(data);
    } catch (err) {
        console.error('Hint lookup failed', err);
    } finally {
        hideLoader();
    }
}

async function loadInitialMaterial() {
    try {
        const data = await fetch('/api/latest_scan').then(r => r.json());
        updateUIWithMaterial(data);
    } catch (err) {
        console.error('Failed to load initial material', err);
    }
}

function updateUIWithMaterial(data) {
    if (!data) return;
    currentMaterial = data;

    // Update Text Fields
    document.getElementById('matName').textContent = data.name || data.formula;
    document.getElementById('matFormula').textContent = `Formula: ${data.formula} · ${data.crystal_system || ''}`;
    document.getElementById('valCategory').textContent = data.category || 'Compound';
    document.getElementById('valStability').textContent = data.properties?.stability || 'Stable';
    document.getElementById('valDensity').textContent = `${data.properties?.density_g_cm3 ?? '—'} g/cm³`;
    document.getElementById('valConductivity').textContent = data.properties?.electrical_conductivity || '—';
    document.getElementById('valReaction').textContent = data.synthesis_recipe?.reaction_eq || 'Standard synthesis';

    // Reconstruct 3D Scene
    render3DStructure(data.structure_3d);
}

function render3DStructure(structure) {
    if (!structure) return;

    // Clear existing meshes
    while (atomMeshGroup.children.length > 0) {
        const obj = atomMeshGroup.children[0];
        atomMeshGroup.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
    }
    while (bondMeshGroup.children.length > 0) {
        const obj = bondMeshGroup.children[0];
        bondMeshGroup.remove(obj);
        if (obj.geometry) obj.geometry.dispose();
        if (obj.material) obj.material.dispose();
    }

    const atoms = structure.atoms || [];
    const bonds = structure.bonds || [];

    // Compute bounding center to auto-center the molecule
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
    const maxSpan = Math.max(maxX - minX, maxY - minY, maxZ - minZ, 1.0);
    const scaleFactor = 3.6 / maxSpan;

    // Render Atoms as Shiny Spheres
    const sphereGeo = new THREE.SphereGeometry(1, 24, 24);

    atoms.forEach((atom, idx) => {
        const colorHex = atom.color || '#38bdf8';
        const mat = new THREE.MeshPhongMaterial({
            color: new THREE.Color(colorHex),
            shininess: 90,
            specular: 0x444444,
        });
        const sphere = new THREE.Mesh(sphereGeo, mat);
        const radius = (atom.radius || 0.8) * 0.42 * Math.min(scaleFactor, 1.2);
        sphere.scale.set(radius, radius, radius);

        sphere.position.set(
            (atom.x - centerX) * scaleFactor,
            (atom.y - centerY) * scaleFactor,
            (atom.z - centerZ) * scaleFactor
        );
        atomMeshGroup.add(sphere);
    });

    // Render Bonds as Cylinders
    bonds.forEach(bond => {
        const a1 = atoms[bond.source];
        const a2 = atoms[bond.target];
        if (!a1 || !a2) return;

        const p1 = new THREE.Vector3(
            (a1.x - centerX) * scaleFactor,
            (a1.y - centerY) * scaleFactor,
            (a1.z - centerZ) * scaleFactor
        );
        const p2 = new THREE.Vector3(
            (a2.x - centerX) * scaleFactor,
            (a2.y - centerY) * scaleFactor,
            (a2.z - centerZ) * scaleFactor
        );

        const distance = p1.distanceTo(p2);
        const cylinderGeo = new THREE.CylinderGeometry(0.06 * Math.min(scaleFactor, 1.2), 0.06 * Math.min(scaleFactor, 1.2), distance, 12);
        const cylinderMat = new THREE.MeshPhongMaterial({
            color: 0x94a3b8,
            shininess: 60,
        });

        const cylinder = new THREE.Mesh(cylinderGeo, cylinderMat);
        const midPoint = new THREE.Vector3().addVectors(p1, p2).multiplyScalar(0.5);
        cylinder.position.copy(midPoint);

        const direction = new THREE.Vector3().subVectors(p2, p1).normalize();
        const orientation = new THREE.Quaternion().setFromUnitVectors(new THREE.Vector3(0, 1, 0), direction);
        cylinder.setRotationFromQuaternion(orientation);

        bondMeshGroup.add(cylinder);
    });
}

function showLoader(title, sub) {
    const loader = document.getElementById('scanLoader');
    if (loader) {
        document.getElementById('loaderTitle').textContent = title || 'Scanning…';
        document.getElementById('loaderSub').textContent = sub || 'Please wait';
        loader.style.display = 'flex';
    }
}

function hideLoader() {
    const loader = document.getElementById('scanLoader');
    if (loader) loader.style.display = 'none';
}
