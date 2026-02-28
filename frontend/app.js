/* ═══════════════════════════════════════════════════════════════════════════
   VeriSight 2.0 — Frontend Logic
   ═══════════════════════════════════════════════════════════════════════════ */

const API = "http://localhost:8000";
const CIRC = 2 * Math.PI * 52;       // Score ring circumference ≈ 326.73
const CIRC_AVG = 2 * Math.PI * 42;   // Avg ring circumference ≈ 263.89

/* ── Navigation ────────────────────────────────────────────────────────── */
const pages = document.querySelectorAll(".page");
const navBtns = document.querySelectorAll(".nav-btn");

navBtns.forEach(btn => {
    btn.addEventListener("click", () => {
        const target = btn.dataset.page;
        pages.forEach(p => p.classList.toggle("active", p.id === `page-${target}`));
        navBtns.forEach(b => b.classList.toggle("active", b === btn));
        if (target === "analytics") loadAnalytics();
        if (target === "queue") loadQueue();
    });
});

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE 1: INVESTIGATION
   ═══════════════════════════════════════════════════════════════════════════ */
const dropZone = document.getElementById("drop-zone");
const fileInput = document.getElementById("file-input");
const browseBtn = document.getElementById("browse-btn");
const previewWrap = document.getElementById("preview-wrap");
const previewImg = document.getElementById("preview-img");
const previewName = document.getElementById("preview-name");
const removeBtn = document.getElementById("remove-btn");
const analyzeBtn = document.getElementById("analyze-btn");
const scanProgress = document.getElementById("scan-progress");
const scanFill = document.getElementById("scan-fill");
const scanStep = document.getElementById("scan-step");
const resultsEmpty = document.getElementById("results-empty");
const resultsPanel = document.getElementById("results-panel");
const errorState = document.getElementById("error-state");
const errorText = document.getElementById("error-text");
const retryBtn = document.getElementById("retry-btn");
const heatmapToggle = document.getElementById("heatmap-toggle");
const heatmapView = document.getElementById("heatmap-view");
const heatmapImg = document.getElementById("heatmap-img");
const heatmapWrap = document.getElementById("heatmap-toggle-wrap");

let selectedFile = null;
let lastResult = null;

// File handling
browseBtn.addEventListener("click", () => fileInput.click());
dropZone.addEventListener("click", e => { if (e.target !== browseBtn) fileInput.click(); });
fileInput.addEventListener("change", () => { if (fileInput.files.length) handleFile(fileInput.files[0]); });

dropZone.addEventListener("dragover", e => { e.preventDefault(); dropZone.classList.add("drag-over"); });
dropZone.addEventListener("dragleave", () => dropZone.classList.remove("drag-over"));
dropZone.addEventListener("drop", e => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) handleFile(e.dataTransfer.files[0]);
});

function handleFile(file) {
    if (!["image/jpeg", "image/png", "image/jpg"].includes(file.type)) return alert("Please select a JPG, JPEG or PNG image.");
    if (file.size > 10 * 1024 * 1024) return alert("File exceeds 10 MB limit.");
    selectedFile = file;
    previewImg.src = URL.createObjectURL(file);
    previewName.textContent = file.name;
    previewWrap.classList.remove("hidden");
    dropZone.classList.add("hidden");
    analyzeBtn.disabled = false;
    resultsPanel.classList.add("hidden");
    resultsEmpty.classList.remove("hidden");
    errorState.classList.add("hidden");
    heatmapWrap.classList.add("hidden");
}

removeBtn.addEventListener("click", resetUpload);
function resetUpload() {
    selectedFile = null;
    lastResult = null;
    fileInput.value = "";
    previewWrap.classList.add("hidden");
    dropZone.classList.remove("hidden");
    analyzeBtn.disabled = true;
    heatmapWrap.classList.add("hidden");
}

// Scanning progress animation
const SCAN_STEPS = [
    { pct: 10, msg: "Uploading image…" },
    { pct: 20, msg: "Initializing OCR engine…" },
    { pct: 35, msg: "Analyzing expiry region…" },
    { pct: 50, msg: "Running ELA forensics…" },
    { pct: 65, msg: "Computing FFT patterns…" },
    { pct: 80, msg: "ViT synthetic detection…" },
    { pct: 92, msg: "Fusing risk signals…" },
];
let scanTimer = null;

function startScanAnim() {
    scanProgress.classList.remove("hidden");
    scanFill.style.width = "0%";
    scanStep.textContent = "Initializing pipeline…";
    let idx = 0;
    scanTimer = setInterval(() => {
        if (idx < SCAN_STEPS.length) {
            scanFill.style.width = SCAN_STEPS[idx].pct + "%";
            scanStep.textContent = SCAN_STEPS[idx].msg;
            idx++;
        }
    }, 800);
}
function stopScanAnim() {
    clearInterval(scanTimer);
    scanFill.style.width = "100%";
    scanStep.textContent = "Analysis complete";
    setTimeout(() => scanProgress.classList.add("hidden"), 600);
}

// Analyze
analyzeBtn.addEventListener("click", runAnalysis);
retryBtn.addEventListener("click", () => { if (selectedFile) runAnalysis(); });

async function runAnalysis() {
    if (!selectedFile) return;
    resultsEmpty.classList.add("hidden");
    resultsPanel.classList.add("hidden");
    errorState.classList.add("hidden");
    analyzeBtn.disabled = true;
    startScanAnim();

    const fd = new FormData();
    fd.append("file", selectedFile);

    try {
        const ctrl = new AbortController();
        const timer = setTimeout(() => ctrl.abort(), 5 * 60 * 1000);
        let res;
        try {
            res = await fetch(`${API}/predict`, {
                method: "POST",
                body: fd,
                signal: ctrl.signal,
            });
        } catch (networkErr) {
            // fetch() itself threw — network-level failure (server down, CORS blocked, DNS, etc.)
            clearTimeout(timer);
            throw new Error(
                "Cannot reach the backend at " + API +
                ". Make sure the FastAPI server is running (uvicorn api:app --reload --port 8000)."
            );
        }
        clearTimeout(timer);

        if (!res.ok) {
            const b = await res.json().catch(() => ({}));
            throw new Error(b.detail || `Server returned error ${res.status}`);
        }

        const data = await res.json();
        lastResult = data;
        stopScanAnim();
        renderResults(data);
        resultsPanel.classList.remove("hidden");
    } catch (err) {
        stopScanAnim();
        console.error("Analysis error:", err);
        if (err.name === "AbortError") {
            errorText.textContent = "Request timed out after 5 minutes. The ML pipeline may be overloaded.";
        } else {
            errorText.textContent = err.message || "Unexpected error — check the browser console.";
        }
        errorState.classList.remove("hidden");
    }
    analyzeBtn.disabled = false;
}

/* ── Render Investigation Results ──────────────────────────────────────── */
function renderResults(d) {
    const score = d.score ?? 0;
    const decision = d.decision ?? "—";
    const isHigh = score >= 60;
    const isMed = score >= 30 && score < 60;

    // Verdict card
    const vc = document.getElementById("verdict-card");
    vc.className = "verdict-card " + (isHigh ? "high-risk" : isMed ? "medium-risk" : "low-risk");

    document.getElementById("verdict-risk-label").textContent =
        isHigh ? "🔴 HIGH FRAUD RISK" : isMed ? "🟡 MEDIUM RISK" : "🟢 LOW RISK";

    // Ring
    const ringFill = document.getElementById("ring-fill");
    ringFill.style.strokeDashoffset = CIRC - (score / 100) * CIRC;

    animateCounter(document.getElementById("score-value"), score);

    // Meta
    document.getElementById("verdict-decision").textContent = decision;
    document.getElementById("confidence-badge").textContent = `Confidence: ${Math.round((d.fusion_confidence ?? 0) * 100)}%`;
    document.getElementById("file-label").textContent = d.filename;

    // Timeline
    const deliveryDate = "2026-12-31"; // from metadata
    document.getElementById("tl-delivery").textContent = formatDate(deliveryDate);
    const expiryRaw = d.expiry_text || "";
    document.getElementById("tl-expiry").textContent = expiryRaw || "Not detected";

    const tlAlert = document.getElementById("tl-alert");
    const timeline = d.timeline || "";
    const connector = document.getElementById("tl-connector");
    if (timeline.toLowerCase().includes("expired before")) {
        tlAlert.classList.remove("hidden");
        document.getElementById("tl-alert-text").textContent = "⚠ Logical Inconsistency Detected";
        connector.style.background = "var(--red)";
        document.querySelector(".tl-dot.expiry").style.background = "var(--red)";
    } else {
        tlAlert.classList.add("hidden");
        connector.style.background = "var(--border)";
        document.querySelector(".tl-dot.expiry").style.background = "var(--amber)";
    }

    // Signals
    const signalsGrid = document.getElementById("signals-grid");
    const allSignals = [
        { name: "Synthetic Pattern", key: "Synthetic Pattern", icon: "🧠" },
        { name: "Digit Inconsistency", key: "Compression Artifact", icon: "🔢" },
        { name: "Texture Irregularity", key: "Texture Irregularity", icon: "🔍" },
        { name: "Ink Pattern Variation", key: "fft_high", icon: "🖋" },
        { name: "OCR Confidence Drop", key: "OCR Confidence Drop", icon: "📝" },
    ];
    const tags = d.tags || "";
    const fftHigh = (d.fft ?? 0) > 0.5;

    signalsGrid.innerHTML = allSignals.map(s => {
        const active = s.key === "fft_high" ? fftHigh : tags.includes(s.key);
        return `<div class="signal-item ${active ? 'active' : 'inactive'}">
      <span class="signal-icon">${active ? '✔' : '○'}</span>
      <span>${s.name}</span>
    </div>`;
    }).join("");

    // Forensic bars
    const forensicBars = document.getElementById("forensic-bars");
    const metrics = [
        { label: "ELA Score", val: d.ela ?? 0, color: "var(--amber)" },
        { label: "FFT Score", val: d.fft ?? 0, color: "var(--accent)" },
        { label: "ViT Score", val: d.vit ?? 0, color: "var(--red)" },
        { label: "Expiry Risk", val: d.expiry_score ?? 0, color: "var(--red)" },
        { label: "OCR Confidence", val: d.ocr_confidence ?? 0, color: "var(--green)" },
    ];
    forensicBars.innerHTML = metrics.map(m => `
    <div class="forensic-row">
      <span class="forensic-label">${m.label}</span>
      <div class="forensic-track">
        <div class="forensic-fill" style="width:${Math.round(m.val * 100)}%; background:${m.color};"></div>
      </div>
      <span class="forensic-val">${(m.val * 100).toFixed(1)}%</span>
    </div>
  `).join("");

    // Heatmap toggle
    if (d.highlight_path) {
        heatmapWrap.classList.remove("hidden");
        heatmapImg.src = d.highlight_path.startsWith("http") ? d.highlight_path : d.highlight_path;
    } else {
        heatmapWrap.classList.add("hidden");
    }

    // Decision cards reset
    document.querySelectorAll("#decision-grid .decision-card").forEach(c => c.classList.remove("chosen"));
    document.getElementById("decision-toast").classList.add("hidden");
}

// Heatmap toggle
heatmapToggle.addEventListener("change", () => {
    heatmapView.classList.toggle("hidden", !heatmapToggle.checked);
});

// Decision buttons
document.querySelectorAll("#decision-grid .decision-card").forEach(btn => {
    btn.addEventListener("click", async () => {
        if (!lastResult) return;
        const action = btn.dataset.action;
        document.querySelectorAll("#decision-grid .decision-card").forEach(c => c.classList.remove("chosen"));
        btn.classList.add("chosen");
        try {
            await fetch(`${API}/api/decide`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: lastResult.filename, action }),
            });
            const toast = document.getElementById("decision-toast");
            toast.classList.remove("hidden");
            setTimeout(() => toast.classList.add("hidden"), 3000);
        } catch (e) { console.error(e); }
    });
});

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE 2: ANALYTICS
   ═══════════════════════════════════════════════════════════════════════════ */
async function loadAnalytics() {
    try {
        const res = await fetch(`${API}/api/analytics`);
        const d = await res.json();

        document.getElementById("kpi-total").textContent = d.total;
        document.getElementById("kpi-approve").textContent = d.approve;
        document.getElementById("kpi-review").textContent = d.review;
        document.getElementById("kpi-reject").textContent = d.reject;

        // Revenue
        document.getElementById("revenue-value").textContent = `₹ ${d.revenue_protected.toLocaleString("en-IN")}`;

        // Score distribution chart
        const chartEl = document.getElementById("score-chart");
        const maxCount = Math.max(...d.score_distribution.map(b => b.count), 1);
        chartEl.innerHTML = d.score_distribution.map(b => {
            const pct = (b.count / maxCount) * 100;
            const color = parseInt(b.range) >= 60 ? "var(--red)" : parseInt(b.range) >= 30 ? "var(--amber)" : "var(--green)";
            return `<div class="bar-col">
        <span class="bar-count">${b.count}</span>
        <div class="bar-fill" style="height:${pct}%; background:${color};"></div>
        <span class="bar-label">${b.range}</span>
      </div>`;
        }).join("");

        // Categories
        const catEl = document.getElementById("category-list");
        const cats = d.categories || {};
        const maxCat = Math.max(...Object.values(cats), 1);
        const catColors = { "Synthetic / AI": "var(--red)", "Compression": "var(--amber)", "Texture Issues": "var(--accent)", "OCR Problems": "var(--text-dim)" };
        catEl.innerHTML = Object.entries(cats).map(([name, count]) => `
      <div class="category-item">
        <span class="cat-name">${name}</span>
        <div class="cat-bar-track">
          <div class="cat-bar-fill" style="width:${(count / maxCat) * 100}%; background:${catColors[name] || 'var(--accent)'};"></div>
        </div>
        <span class="cat-count">${count}</span>
      </div>
    `).join("");

        // Spike alert
        const spikeEl = document.getElementById("spike-alert");
        if (d.recent_rejects.length) {
            spikeEl.innerHTML = d.recent_rejects.map(r => `
        <div class="spike-item">
          <span>${r.filename}</span>
          <span class="spike-score ${r.score >= 60 ? 'high' : 'medium'}">${r.score}</span>
        </div>
      `).join("");
        } else {
            spikeEl.innerHTML = '<p style="color:var(--text-dim); font-size:.82rem;">No high-risk detections</p>';
        }

        // Average score ring
        const avgScore = d.avg_score || 0;
        const avgFill = document.getElementById("avg-ring-fill");
        avgFill.style.strokeDashoffset = CIRC_AVG - (avgScore / 100) * CIRC_AVG;
        avgFill.style.stroke = avgScore >= 60 ? "var(--red)" : avgScore >= 30 ? "var(--amber)" : "var(--green)";
        animateCounter(document.getElementById("avg-score-val"), Math.round(avgScore));

    } catch (err) {
        console.error("Analytics error:", err);
    }
}

/* ═══════════════════════════════════════════════════════════════════════════
   PAGE 3: QUEUE
   ═══════════════════════════════════════════════════════════════════════════ */
let queuePage = 1;
let queueData = { items: [], total: 0, pages: 1 };

const queueFilter = document.getElementById("queue-filter");
const queueSort = document.getElementById("queue-sort");
const queueTbody = document.getElementById("queue-tbody");
const prevPageBtn = document.getElementById("prev-page");
const nextPageBtn = document.getElementById("next-page");
const pageInfo = document.getElementById("page-info");
const bulkBtn = document.getElementById("bulk-approve-btn");
const selectAllBtn = document.getElementById("select-all-btn");
const checkAll = document.getElementById("check-all");

queueFilter.addEventListener("change", () => { queuePage = 1; loadQueue(); });
queueSort.addEventListener("change", () => { queuePage = 1; loadQueue(); });
prevPageBtn.addEventListener("click", () => { if (queuePage > 1) { queuePage--; loadQueue(); } });
nextPageBtn.addEventListener("click", () => { if (queuePage < queueData.pages) { queuePage++; loadQueue(); } });

checkAll.addEventListener("change", () => {
    document.querySelectorAll(".row-check").forEach(c => { c.checked = checkAll.checked; });
    updateBulkBtn();
});

selectAllBtn.addEventListener("click", () => {
    checkAll.checked = true;
    document.querySelectorAll(".row-check").forEach(c => { c.checked = true; });
    updateBulkBtn();
});

function updateBulkBtn() {
    const checked = document.querySelectorAll(".row-check:checked");
    bulkBtn.disabled = checked.length === 0;
    bulkBtn.textContent = checked.length ? `Bulk Approve (${checked.length})` : "Bulk Approve Selected";
}

bulkBtn.addEventListener("click", async () => {
    const fnames = [...document.querySelectorAll(".row-check:checked")].map(c => c.dataset.filename);
    if (!fnames.length) return;
    try {
        await fetch(`${API}/api/bulk-decide`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filenames: fnames, action: "approve" }),
        });
        loadQueue();
    } catch (e) { console.error(e); }
});

async function loadQueue() {
    try {
        const risk = queueFilter.value;
        const sort = queueSort.value;
        const url = `${API}/api/queue?page=${queuePage}&per_page=20&sort_by=${sort}${risk ? `&risk=${risk}` : ''}`;
        const res = await fetch(url);
        queueData = await res.json();
        renderQueue();
    } catch (err) {
        console.error("Queue error:", err);
    }
}

function renderQueue() {
    const items = queueData.items || [];
    prevPageBtn.disabled = queuePage <= 1;
    nextPageBtn.disabled = queuePage >= queueData.pages;
    pageInfo.textContent = `Page ${queueData.page} of ${queueData.pages} (${queueData.total} items)`;
    checkAll.checked = false;
    updateBulkBtn();

    if (!items.length) {
        queueTbody.innerHTML = '<tr><td colspan="7" style="text-align:center; color:var(--text-dim); padding:28px;">No results found</td></tr>';
        return;
    }

    queueTbody.innerHTML = items.map(item => {
        const scoreClass = item.score >= 60 ? "high" : item.score >= 30 ? "medium" : "low";
        const decClass = item.decision.startsWith("Manual") ? "Manual" : item.decision;
        const tagsHtml = (item.tags || "").split(";").filter(Boolean).map(t => {
            const isWarn = ["Synthetic Pattern", "Compression Artifact"].includes(t.trim());
            return `<span class="tag-mini ${isWarn ? 'warn' : ''}">${t.trim()}</span>`;
        }).join("");
        const statusHtml = item.user_decision
            ? `<span class="status-label decided">✓ ${item.user_decision}</span>`
            : `<span class="status-label pending">Pending</span>`;

        return `<tr>
      <td class="col-check"><input type="checkbox" class="row-check" data-filename="${item.filename}" onchange="updateBulkBtn()" /></td>
      <td>${item.filename}</td>
      <td><span class="score-pill ${scoreClass}">${item.score}</span></td>
      <td><span class="decision-pill ${decClass}">${item.decision}</span></td>
      <td>${tagsHtml || '—'}</td>
      <td>${statusHtml}</td>
      <td><div class="action-btns">
        <button class="action-btn preview-btn" onclick="openPreview('${item.filename}')">Preview</button>
      </div></td>
    </tr>`;
    }).join("");
}

// Make updateBulkBtn global for inline onchange
window.updateBulkBtn = updateBulkBtn;

/* ── Preview Modal ─────────────────────────────────────────────────────── */
const modalOverlay = document.getElementById("preview-modal");
const modalClose = document.getElementById("modal-close");
const modalImg = document.getElementById("modal-img");
const modalTitle = document.getElementById("modal-title");
const modalDetails = document.getElementById("modal-details");
const modalToast = document.getElementById("modal-toast");
let modalFilename = null;

modalClose.addEventListener("click", () => modalOverlay.classList.add("hidden"));
modalOverlay.addEventListener("click", e => { if (e.target === modalOverlay) modalOverlay.classList.add("hidden"); });

window.openPreview = function (filename) {
    const item = (queueData.items || []).find(i => i.filename === filename);
    if (!item) return;
    modalFilename = filename;
    modalTitle.textContent = filename;
    modalImg.src = `${API}/images/${filename}`;
    modalToast.classList.add("hidden");

    modalDetails.innerHTML = `
    <div class="modal-detail"><span class="modal-detail-label">Score</span><span>${item.score}/100</span></div>
    <div class="modal-detail"><span class="modal-detail-label">Decision</span><span>${item.decision}</span></div>
    <div class="modal-detail"><span class="modal-detail-label">ELA</span><span>${(item.ela * 100).toFixed(1)}%</span></div>
    <div class="modal-detail"><span class="modal-detail-label">FFT</span><span>${(item.fft * 100).toFixed(1)}%</span></div>
    <div class="modal-detail"><span class="modal-detail-label">ViT</span><span>${(item.vit * 100).toFixed(1)}%</span></div>
    <div class="modal-detail"><span class="modal-detail-label">OCR Conf</span><span>${(item.ocr_confidence * 100).toFixed(1)}%</span></div>
    <div class="modal-detail"><span class="modal-detail-label">Timeline</span><span>${item.timeline || '—'}</span></div>
    <div class="modal-detail"><span class="modal-detail-label">Expiry</span><span>${item.expiry_text || '—'}</span></div>
  `;

    modalOverlay.classList.remove("hidden");
};

// Modal decision buttons
document.querySelectorAll("[data-modal-action]").forEach(btn => {
    btn.addEventListener("click", async () => {
        if (!modalFilename) return;
        try {
            await fetch(`${API}/api/decide`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ filename: modalFilename, action: btn.dataset.modalAction }),
            });
            modalToast.classList.remove("hidden");
            setTimeout(() => {
                modalOverlay.classList.add("hidden");
                loadQueue();
            }, 1200);
        } catch (e) { console.error(e); }
    });
});

/* ── Helpers ────────────────────────────────────────────────────────────── */
function animateCounter(el, target) {
    const dur = 1000;
    const start = performance.now();
    function step(now) {
        const p = Math.min((now - start) / dur, 1);
        const ease = 1 - Math.pow(1 - p, 3);
        el.textContent = Math.round(target * ease);
        if (p < 1) requestAnimationFrame(step);
    }
    requestAnimationFrame(step);
}

function formatDate(str) {
    if (!str) return "—";
    try {
        const d = new Date(str);
        return d.toLocaleDateString("en-GB", { day: "2-digit", month: "short", year: "numeric" }).toUpperCase();
    } catch { return str; }
}
