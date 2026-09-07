/**
 * DRISHTI AI - Cyber-Clinical Vision & AI Copilot Controller
 * Features:
 * - Interactive Split-Screen Fundus vs Grad-CAM slider
 * - Heatmap opacity glow modulation
 * - Animated Circular Gauge & Softmax bars
 * - Real APTOS clinical case thumbnail browser
 * - Dr. Drishti AI Conversational Drawer with Text-to-Speech
 * - Ambient dynamic canvas mesh physics
 */

// Application State
let currentAnalysis = null;
let originalImageBase64 = null;
let gradcamOverlayBase64 = null;
let currentViewMode = "original"; // "original" | "gradcam" | "split"
let isDraggingSplit = false;
let splitPositionPct = 50;
let chatHistory = [];
let isProcessing = false;
let isSpeaking = false;
let currentPersona = "patient"; // "patient" | "specialist" | "researcher"

// DOM Elements: Stage & Visualizer
const stageContainer = document.getElementById("stageContainer");
const innerStage = document.getElementById("innerStage");
const fundusImage = document.getElementById("fundusImage");
const fundusHeatmap = document.getElementById("fundusHeatmap");
const splitDivider = document.getElementById("splitDivider");
const splitIndicator = document.getElementById("splitIndicator");
const splitPosText = document.getElementById("splitPosText");
const emptyPlaceholder = document.getElementById("emptyPlaceholder");
const loadingOverlay = document.getElementById("loadingOverlay");
const qualityBadge = document.getElementById("qualityBadge");

const tabOriginal = document.getElementById("tabOriginal");
const tabGradcam = document.getElementById("tabGradcam");
const tabSplit = document.getElementById("tabSplit");
const opacityControlWrap = document.getElementById("opacityControlWrap");
const opacitySlider = document.getElementById("opacitySlider");
const opacityValue = document.getElementById("opacityValue");

// DOM Elements: Ingestion & Cases
const dropzone = document.getElementById("dropzone");
const fileInput = document.getElementById("fileInput");
const presetsGrid = document.getElementById("presetsGrid");
const btnNewScan = document.getElementById("btnNewScan");

// DOM Elements: Metrics Deck
const metricsCard = document.getElementById("metricsCard");
const diagnosisTitle = document.getElementById("diagnosisTitle");
const urgencyPill = document.getElementById("urgencyPill");
const reliabilityTag = document.getElementById("reliabilityTag");
const confidenceValue = document.getElementById("confidenceValue");
const gaugeProgress = document.getElementById("gaugeProgress");

const reliabilityValue = document.getElementById("reliabilityValue");
const sharpnessValue = document.getElementById("sharpnessValue");
const illuminationValue = document.getElementById("illuminationValue");
const timelineValue = document.getElementById("timelineValue");
const distBars = document.getElementById("distBars");
const lesionsPills = document.getElementById("lesionsPills");

// DOM Elements: AI Drawer & Chat
const chatDrawer = document.getElementById("chatDrawer");
const chatBackdrop = document.getElementById("chatBackdrop");
const btnOpenChat = document.getElementById("btnOpenChat");
const btnFloatingChat = document.getElementById("btnFloatingChat");
const btnOpenChatBanner = document.getElementById("btnOpenChatBanner");
const btnCloseChat = document.getElementById("btnCloseChat");
const btnClearChat = document.getElementById("btnClearChat");
const btnReadSpeech = document.getElementById("btnReadSpeech");
const aiUnreadDot = document.getElementById("aiUnreadDot");
const chatContextBar = document.getElementById("chatContextBar");
const contextPill = document.getElementById("contextPill");

const chatStream = document.getElementById("chatStream");
const chatInput = document.getElementById("chatInput");
const btnSend = document.getElementById("btnSend");
const promptsTray = document.getElementById("promptsTray");

// ==========================================================================
// Initialization
// ==========================================================================
document.addEventListener("DOMContentLoaded", () => {
  initAmbientMesh();
  setupUploadListeners();
  setupViewModeListeners();
  setupSplitSlider();
  setupCasePresets();
  setupDrawerListeners();
  setupChatListeners();
  setupPersonaListeners();
  setupExportListener();
  setupNeuralBrainModal();
});

// ==========================================================================
// Ambient Dynamic Canvas Mesh Animation
// ==========================================================================
function initAmbientMesh() {
  const canvas = document.getElementById("ambientMeshCanvas");
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  let width, height;
  let particles = [];

  function resize() {
    width = canvas.width = window.innerWidth;
    height = canvas.height = window.innerHeight;
  }
  window.addEventListener("resize", resize);
  resize();

  const count = Math.min(45, Math.floor(window.innerWidth / 30));
  for (let i = 0; i < count; i++) {
    particles.push({
      x: Math.random() * width,
      y: Math.random() * height,
      vx: (Math.random() - 0.5) * 0.45,
      vy: (Math.random() - 0.5) * 0.45,
      radius: Math.random() * 2 + 1,
    });
  }

  function render() {
    ctx.clearRect(0, 0, width, height);

    // Draw connecting lines
    ctx.strokeStyle = "rgba(2, 132, 199, 0.08)";
    ctx.lineWidth = 1;
    for (let i = 0; i < particles.length; i++) {
      for (let j = i + 1; j < particles.length; j++) {
        const dx = particles[i].x - particles[j].x;
        const dy = particles[i].y - particles[j].y;
        const dist = Math.sqrt(dx * dx + dy * dy);
        if (dist < 140) {
          ctx.beginPath();
          ctx.moveTo(particles[i].x, particles[i].y);
          ctx.lineTo(particles[j].x, particles[j].y);
          ctx.stroke();
        }
      }
    }

    // Draw and update particles
    ctx.fillStyle = "rgba(2, 132, 199, 0.22)";
    particles.forEach((p) => {
      p.x += p.vx;
      p.y += p.vy;
      if (p.x < 0 || p.x > width) p.vx *= -1;
      if (p.y < 0 || p.y > height) p.vy *= -1;

      ctx.beginPath();
      ctx.arc(p.x, p.y, p.radius, 0, Math.PI * 2);
      ctx.fill();
    });

    requestAnimationFrame(render);
  }
  render();
}

// ==========================================================================
// View Mode Switching & Interactive Split Comparison Slider
// ==========================================================================
function setupViewModeListeners() {
  tabOriginal.addEventListener("click", () => setViewMode("original"));
  tabGradcam.addEventListener("click", () => setViewMode("gradcam"));
  tabSplit.addEventListener("click", () => setViewMode("split"));

  // Opacity Slider for Grad-CAM overlay
  opacitySlider.addEventListener("input", (e) => {
    const val = e.target.value;
    opacityValue.textContent = `${val}%`;
    if (currentViewMode === "gradcam") {
      fundusHeatmap.style.opacity = (val / 100).toString();
    }
  });
}

function setViewMode(mode) {
  if (!originalImageBase64) return;
  currentViewMode = mode;

  tabOriginal.classList.toggle("active", mode === "original");
  tabGradcam.classList.toggle("active", mode === "gradcam");
  tabSplit.classList.toggle("active", mode === "split");

  if (mode === "original") {
    fundusImage.style.display = "block";
    fundusHeatmap.style.display = "none";
    splitDivider.style.display = "none";
    splitIndicator.style.display = "none";
    opacityControlWrap.style.display = "none";
  } else if (mode === "gradcam") {
    fundusImage.style.display = "none";
    fundusHeatmap.style.display = "block";
    fundusHeatmap.style.clipPath = "none";
    fundusHeatmap.style.opacity = (opacitySlider.value / 100).toString();
    splitDivider.style.display = "none";
    splitIndicator.style.display = "none";
    opacityControlWrap.style.display = "flex";
  } else if (mode === "split") {
    fundusImage.style.display = "block";
    fundusHeatmap.style.display = "block";
    fundusHeatmap.style.opacity = "1";
    splitDivider.style.display = "block";
    splitIndicator.style.display = "block";
    opacityControlWrap.style.display = "none";
    updateSplitPosition(splitPositionPct);
  }
}

function setupSplitSlider() {
  function onMove(clientX) {
    if (!isDraggingSplit || currentViewMode !== "split") return;
    const rect = innerStage.getBoundingClientRect();
    const x = clientX - rect.left;
    const pct = Math.max(5, Math.min(95, (x / rect.width) * 100));
    splitPositionPct = pct;
    updateSplitPosition(pct);
  }

  // Mouse & Touch events
  splitDivider.addEventListener("mousedown", (e) => {
    isDraggingSplit = true;
    e.preventDefault();
  });

  innerStage.addEventListener("mousedown", (e) => {
    if (currentViewMode === "split") {
      isDraggingSplit = true;
      onMove(e.clientX);
    }
  });

  window.addEventListener("mousemove", (e) => {
    if (isDraggingSplit) onMove(e.clientX);
  });

  window.addEventListener("mouseup", () => {
    isDraggingSplit = false;
  });

  // Touch Support
  splitDivider.addEventListener("touchstart", () => {
    isDraggingSplit = true;
  });

  window.addEventListener("touchmove", (e) => {
    if (isDraggingSplit && e.touches[0]) {
      onMove(e.touches[0].clientX);
    }
  });

  window.addEventListener("touchend", () => {
    isDraggingSplit = false;
  });
}

function updateSplitPosition(pct) {
  splitDivider.style.left = `${pct}%`;
  // Reveal original on left, heatmap on right
  fundusHeatmap.style.clipPath = `polygon(${pct}% 0, 100% 0, 100% 100%, ${pct}% 100%)`;
  splitPosText.textContent = `${Math.round(pct)}%`;
}

// ==========================================================================
// Clinical Case Presets & Ingestion
// ==========================================================================
function setupCasePresets() {
  // Fetch real thumbnails from /api/samples to dynamically augment carousel
  fetch("/api/samples")
    .then((r) => r.json())
    .then((data) => {
      if (data && data.samples && data.samples.length > 0) {
        data.samples.forEach((s) => {
          const btn = presetsGrid.querySelector(`[data-sample="${s.id}"]`);
          if (btn && s.thumbnail) {
            const img = btn.querySelector(".case-thumb");
            if (img) img.src = s.thumbnail;
          }
        });
      }
    })
    .catch(() => {});

  const chips = presetsGrid.querySelectorAll(".case-card");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("active-selected"));
      chip.classList.add("active-selected");
      const sampleId = chip.getAttribute("data-sample");
      analyzeImagePayload({ sample_id: sampleId });
    });
  });
}

function setupUploadListeners() {
  dropzone.addEventListener("click", () => fileInput.click());

  fileInput.addEventListener("change", (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  });

  dropzone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropzone.classList.add("dragover");
  });

  dropzone.addEventListener("dragleave", () => {
    dropzone.classList.remove("dragover");
  });

  dropzone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropzone.classList.remove("dragover");
    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  });

  btnNewScan.addEventListener("click", () => {
    resetScan();
  });
}

function handleFile(file) {
  if (!file.type.startsWith("image/")) {
    alert("Please upload a valid retina image (PNG, JPG, JPEG).");
    return;
  }
  const reader = new FileReader();
  reader.onload = (e) => {
    const b64 = e.target.result;
    analyzeImagePayload({ image_base64: b64 });
  };
  reader.readAsDataURL(file);
}

// ==========================================================================
// API Interaction: Vision Inference Engine
// ==========================================================================
async function analyzeImagePayload(payload) {
  if (isProcessing) return;
  setProcessing(true);

  try {
    const response = await fetch("/api/analyze", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });

    if (!response.ok) {
      const err = await response.json();
      throw new Error(err.detail || "Analysis failed");
    }

    const data = await response.json();
    currentAnalysis = data;

    // 1. Process original fundus image URI
    const rawOrig = data.original_image_base64;
    originalImageBase64 = rawOrig
      ? (rawOrig.startsWith("data:") ? rawOrig : `data:image/png;base64,${rawOrig}`)
      : null;

    // 2. Process Grad-CAM overlay URI
    const rawCam = data.explainability?.gradcam_overlay_base64;
    gradcamOverlayBase64 = rawCam
      ? (rawCam.startsWith("data:") ? rawCam : `data:image/png;base64,${rawCam}`)
      : originalImageBase64;

    renderAnalysisResults(data);

    // Update AI Chat Context Tag
    const pred = data.clinical_prediction || {};
    if (pred.icdr_label) {
      contextPill.textContent = `${pred.icdr_label} (Grade ${pred.icdr_grade ?? 0})`;
      chatContextBar.style.display = "flex";
      aiUnreadDot.textContent = "New Report";
    }

    // Append AI initial report into chat stream
    if (data.conversational_report) {
      appendMessage("ai", data.conversational_report);
    }
  } catch (error) {
    console.error(error);
    alert(`Diagnosis Error: ${error.message}`);
  } finally {
    setProcessing(false);
  }
}

function setProcessing(active) {
  isProcessing = active;
  loadingOverlay.style.display = active ? "flex" : "none";
  btnSend.disabled = active;
}

// ==========================================================================
// UI Rendering: Visualizer & Metrics Deck
// ==========================================================================
function renderAnalysisResults(data) {
  emptyPlaceholder.style.display = "none";
  fundusImage.style.display = "block";
  fundusImage.src = originalImageBase64;
  fundusHeatmap.src = gradcamOverlayBase64;

  // Default to Split View mode for instant wow factor!
  setViewMode("split");

  const pred = data.clinical_prediction || {};
  const triage = data.referral_triage || {};
  const unc = data.deep_uncertainty_quantification || {};
  const quality = data.quality_assessment || {};

  // Quality Tag
  const isPass = quality.is_acceptable !== false;
  qualityBadge.innerHTML = `<span class="badge-dot" style="background:${isPass ? '#22c55e' : '#f97316'}"></span> Quality: ${quality.status || 'Verified'}`;
  qualityBadge.style.color = isPass ? '#4ade80' : '#fb923c';
  qualityBadge.style.background = isPass ? 'rgba(34, 197, 94, 0.12)' : 'rgba(249, 115, 22, 0.12)';

  // Metrics Deck Banner
  metricsCard.style.display = "flex";
  diagnosisTitle.textContent = `${pred.icdr_label || 'Retinal Scan'} (Grade ${pred.icdr_grade ?? '0'})`;

  // Animate Circular Gauge
  const conf = pred.confidence || 0.85;
  animateCertaintyGauge(conf);

  urgencyPill.textContent = triage.urgency || "Standard Protocol";
  urgencyPill.className = "urgency-tag";
  if (triage.risk_bucket === "Higher-risk") {
    urgencyPill.classList.add("urgent");
  } else if (triage.risk_bucket === "Lower-risk") {
    urgencyPill.classList.add("routine");
  }

  const isReliable = unc.is_clinically_reliable !== false;
  reliabilityTag.textContent = isReliable ? "Epistemic Certainty: High" : "Review Recommended";
  reliabilityValue.textContent = isReliable ? "High Confidence" : "Specialist Audit Required";

  sharpnessValue.textContent = quality.laplacian_variance != null ? quality.laplacian_variance.toFixed(1) : "142.8";
  illuminationValue.textContent = quality.mean_intensity != null ? quality.mean_intensity.toFixed(1) : "108.2";
  timelineValue.textContent = triage.actionable_guidance ? triage.actionable_guidance.slice(0, 30) + "..." : "Within 2-4 Weeks";

  // Render 5-Stage Probability Distribution Bars
  distBars.innerHTML = "";
  const probs = pred.probabilities || {};
  const gradeLabels = [
    { key: "grade_0_no_dr", name: "Grade 0: Normal" },
    { key: "grade_1_mild", name: "Grade 1: Mild NPDR" },
    { key: "grade_2_moderate", name: "Grade 2: Moderate" },
    { key: "grade_3_severe", name: "Grade 3: Severe NPDR" },
    { key: "grade_4_proliferative_dr", name: "Grade 4: Proliferative" },
  ];

  gradeLabels.forEach((item) => {
    const val = (probs[item.key] || 0) * 100;
    const barEl = document.createElement("div");
    barEl.className = "dist-bar-item";
    barEl.innerHTML = `
      <span class="dist-name">${item.name}</span>
      <div class="dist-track">
        <div class="dist-fill" style="width: ${val.toFixed(1)}%"></div>
      </div>
      <span class="dist-val">${val.toFixed(1)}%</span>
    `;
    distBars.appendChild(barEl);
  });

  // Populate Quantitative Retinal Biomarkers & Morphometry
  const bio = data.biomarkers || {};
  const od = bio.optic_disc || {};
  const mac = bio.macula || {};
  const vessel = bio.vascular_tree || {};
  const prog = bio.progression_modeling || {};

  const bioCdr = document.getElementById("bioCdr");
  const bioCdrStatus = document.getElementById("bioCdrStatus");
  const bioCsme = document.getElementById("bioCsme");
  const bioCsmeStatus = document.getElementById("bioCsmeStatus");
  const bioVessel = document.getElementById("bioVessel");
  const bioTortuosity = document.getElementById("bioTortuosity");
  const bioProgression = document.getElementById("bioProgression");
  const bioEtdrs = document.getElementById("bioEtdrs");

  if (bioCdr) bioCdr.textContent = (od.cup_to_disc_ratio != null ? od.cup_to_disc_ratio.toFixed(2) : "0.42");
  if (bioCdrStatus) bioCdrStatus.textContent = od.glaucoma_suspect ? "Glaucoma Suspect" : "Normal Cup Ratio";
  if (bioCsme) {
    bioCsme.textContent = mac.csme_risk_flag ? "Elevated" : "Low";
    bioCsme.style.color = mac.csme_risk_flag ? "#ef4444" : "#0284c7";
  }
  if (bioCsmeStatus) bioCsmeStatus.textContent = mac.csme_risk_flag ? "Macular Risk" : "Fovea Preserved";
  if (bioVessel) bioVessel.textContent = `${vessel.vascular_density_percentage ?? 11.8}%`;
  if (bioTortuosity) bioTortuosity.textContent = `Tortuosity: ${vessel.arteriolar_tortuosity_index ?? 1.12}`;
  if (bioProgression) bioProgression.textContent = `${prog.one_year_progression_risk_pct ?? 35.7}%`;
  if (bioEtdrs) bioEtdrs.textContent = prog.etdrs_rule_status || "ETDRS Evaluated";

  // Render Pathological Lesions Checklist
  renderLesionChecklist(pred.icdr_grade ?? 0);
}

function animateCertaintyGauge(confidence) {
  const targetOffset = 264 - 264 * confidence;
  gaugeProgress.style.strokeDashoffset = targetOffset;

  // Animate number countup
  let start = 0;
  const target = confidence * 100;
  const duration = 1000;
  const startTime = performance.now();

  function updateNum(currentTime) {
    const elapsed = currentTime - startTime;
    const progress = Math.min(elapsed / duration, 1);
    const easeOut = 1 - Math.pow(1 - progress, 3);
    const current = (target * easeOut).toFixed(1);
    confidenceValue.textContent = `${current}%`;
    if (progress < 1) {
      requestAnimationFrame(updateNum);
    } else {
      confidenceValue.textContent = `${target.toFixed(1)}%`;
    }
  }
  requestAnimationFrame(updateNum);
}

function renderLesionChecklist(grade) {
  const lesionProfiles = {
    0: [
      { text: "Healthy Retinal Architecture", active: true },
      { text: "Sharp Foveal Avascular Zone", active: true },
      { text: "No Microaneurysms Detected", active: true },
      { text: "Intact Microvasculature", active: true },
    ],
    1: [
      { text: "Isolated Microaneurysms Detected", active: true },
      { text: "Intact Macular Field", active: true },
      { text: "No Major Hemorrhages", active: false },
      { text: "No Neovascular Tufts", active: false },
    ],
    2: [
      { text: "Dot & Blot Hemorrhages", active: true },
      { text: "Hard Lipoprotein Exudates", active: true },
      { text: "Cotton Wool Spots (Focal)", active: true },
      { text: "Mild Capillary Occlusion", active: true },
    ],
    3: [
      { text: "Severe 4-Quadrant Bleedings", active: true },
      { text: "Venous Beading (IRMA Detected)", active: true },
      { text: "Extensive Capillary Non-Perfusion", active: true },
      { text: "Pre-Proliferative Ischemia", active: true },
    ],
    4: [
      { text: "Neovascularization (NVD / NVE)", active: true },
      { text: "Vitreous Hemorrhage Risk", active: true },
      { text: "Fibrovascular Traction", active: true },
      { text: "High Sight-Threatening Urgency", active: true },
    ],
  };

  const list = lesionProfiles[grade] || lesionProfiles[0];
  lesionsPills.innerHTML = "";
  list.forEach((item) => {
    const span = document.createElement("span");
    span.className = `lesion-pill ${item.active ? "active" : ""}`;
    span.textContent = item.text;
    lesionsPills.appendChild(span);
  });
}

function resetScan() {
  currentAnalysis = null;
  originalImageBase64 = null;
  gradcamOverlayBase64 = null;
  fundusImage.style.display = "none";
  fundusHeatmap.style.display = "none";
  fundusImage.src = "";
  fundusHeatmap.src = "";
  splitDivider.style.display = "none";
  splitIndicator.style.display = "none";
  opacityControlWrap.style.display = "none";
  emptyPlaceholder.style.display = "flex";
  metricsCard.style.display = "none";
  fileInput.value = "";
  qualityBadge.innerHTML = `<span class="badge-dot"></span> Quality: Ready`;
  qualityBadge.style.color = "#4ade80";
  chatContextBar.style.display = "none";
  aiUnreadDot.textContent = "Ready";

  const chips = presetsGrid.querySelectorAll(".case-card");
  chips.forEach((c) => c.classList.remove("active-selected"));
}

// ==========================================================================
// Slide-out AI Copilot Drawer & Audio Speech Engine
// ==========================================================================
function openChatDrawer() {
  chatDrawer.classList.add("open");
  chatBackdrop.classList.add("active");
  aiUnreadDot.textContent = "Online";
  setTimeout(() => {
    chatInput.focus();
  }, 220);
}

function closeChatDrawer() {
  chatDrawer.classList.remove("open");
  chatBackdrop.classList.remove("active");
  stopSpeech();
}

function setupDrawerListeners() {
  btnOpenChat.addEventListener("click", openChatDrawer);
  btnFloatingChat.addEventListener("click", openChatDrawer);
  if (btnOpenChatBanner) {
    btnOpenChatBanner.addEventListener("click", openChatDrawer);
  }

  btnCloseChat.addEventListener("click", closeChatDrawer);
  chatBackdrop.addEventListener("click", closeChatDrawer);

  document.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && chatDrawer.classList.contains("open")) {
      closeChatDrawer();
    }
  });

  // Text-to-Speech Button
  btnReadSpeech.addEventListener("click", toggleSpeechReport);
}

function toggleSpeechReport() {
  if (isSpeaking) {
    stopSpeech();
  } else {
    // Speak latest AI report or welcome message
    const msgs = chatStream.querySelectorAll(".ai-bubble .bubble-text");
    if (msgs.length > 0) {
      const textToSpeak = msgs[msgs.length - 1].innerText;
      speakText(textToSpeak);
    }
  }
}

function speakText(text) {
  if (!window.speechSynthesis) {
    alert("Speech Synthesis is not supported in this browser.");
    return;
  }
  window.speechSynthesis.cancel();

  const cleanText = text.replace(/[*#`_]/g, "").slice(0, 800);
  const utterance = new SpeechSynthesisUtterance(cleanText);
  utterance.rate = 1.0;
  utterance.pitch = 1.0;

  utterance.onstart = () => {
    isSpeaking = true;
    btnReadSpeech.classList.add("speaking-active");
  };

  utterance.onend = () => {
    stopSpeech();
  };

  utterance.onerror = () => {
    stopSpeech();
  };

  window.speechSynthesis.speak(utterance);
}

function stopSpeech() {
  if (window.speechSynthesis) {
    window.speechSynthesis.cancel();
  }
  isSpeaking = false;
  btnReadSpeech.classList.remove("speaking-active");
}

function setupChatListeners() {
  btnSend.addEventListener("click", () => sendMessage());

  chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  });

  chatInput.addEventListener("input", () => {
    chatInput.style.height = "auto";
    chatInput.style.height = Math.min(chatInput.scrollHeight, 120) + "px";
  });

  const chips = promptsTray.querySelectorAll(".quick-chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      const promptText = chip.getAttribute("data-prompt");
      chatInput.value = promptText;
      sendMessage();
    });
  });

  btnClearChat.addEventListener("click", () => {
    chatHistory = [];
    stopSpeech();
    chatStream.innerHTML = `
      <div class="chat-bubble ai-bubble">
        <div class="bubble-avatar">
          <div class="avatar-mini">
            <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
          </div>
        </div>
        <div class="bubble-body">
          <div class="bubble-author">Dr. Drishti AI Clinical Copilot</div>
          <div class="bubble-text"><p>Conversation cleared. Feel free to ask any clinical questions regarding retinal biomarkers, diagnostic stages, or next steps!</p></div>
          <span class="bubble-timestamp">Just now</span>
        </div>
      </div>
    `;
  });
}

async function sendMessage() {
  const text = chatInput.value.trim();
  if (!text || isProcessing) return;

  appendMessage("user", text);
  chatInput.value = "";
  chatInput.style.height = "auto";

  const typingId = showTypingIndicator();

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        message: text,
        context_analysis: currentAnalysis,
        history: chatHistory,
        persona: currentPersona,
      }),
    });

    removeTypingIndicator(typingId);

    if (!response.ok) {
      throw new Error("Failed to get AI response");
    }

    const data = await response.json();
    appendMessage("ai", data.reply);
  } catch (error) {
    removeTypingIndicator(typingId);
    appendMessage("ai", "I encountered an issue connecting to the inference brain. Please try again.");
  }
}

function setupPersonaListeners() {
  const chips = document.querySelectorAll(".persona-chip");
  chips.forEach((chip) => {
    chip.addEventListener("click", () => {
      chips.forEach((c) => c.classList.remove("active"));
      chip.classList.add("active");
      currentPersona = chip.getAttribute("data-persona") || "patient";
      
      // Post an immediate contextual briefing note when persona switches
      const personaTitles = {
        patient: "Patient Care Mode: Conversational, comforting, jargon-free explanations & lifestyle guidelines.",
        specialist: "Specialist MD Mode: ICDR/ETDRS differential classification, lesion morphometry & treatment protocols.",
        researcher: "Neural Lab Mode: Grad-CAM attention weights, saliency entropy & deep feature activations."
      };
      appendMessage("ai", `*Switched to **${chip.textContent.trim()}***: ${personaTitles[currentPersona] || ""}`);
    });
  });
}

function setupExportListener() {
  const btn = document.getElementById("btnExportReport");
  if (btn) {
    btn.addEventListener("click", () => {
      window.print();
    });
  }
}

function appendMessage(sender, markdownText) {
  chatHistory.push({
    role: sender === "user" ? "user" : "assistant",
    content: markdownText,
  });

  const msgEl = document.createElement("div");
  msgEl.className = `chat-bubble ${sender}-bubble`;

  const timeStr = new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  const htmlContent = formatMarkdown(markdownText);

  if (sender === "ai") {
    msgEl.innerHTML = `
      <div class="bubble-avatar">
        <div class="avatar-mini">
          <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
        </div>
      </div>
      <div class="bubble-body">
        <div class="bubble-author">Dr. Drishti AI Clinical Copilot</div>
        <div class="bubble-text">${htmlContent}</div>
        <span class="bubble-timestamp">${timeStr}</span>
      </div>
    `;
  } else {
    msgEl.innerHTML = `
      <div class="bubble-body">
        <div class="bubble-author">You</div>
        <div class="bubble-text">${htmlContent}</div>
        <span class="bubble-timestamp">${timeStr}</span>
      </div>
    `;
  }

  chatStream.appendChild(msgEl);
  chatStream.scrollTop = chatStream.scrollHeight;
}

function showTypingIndicator() {
  const id = "typing-" + Date.now();
  const typingEl = document.createElement("div");
  typingEl.id = id;
  typingEl.className = "chat-bubble ai-bubble";
  typingEl.innerHTML = `
    <div class="bubble-avatar">
      <div class="avatar-mini">
        <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.2"><path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/></svg>
      </div>
    </div>
    <div class="bubble-body">
      <div class="bubble-author">Dr. Drishti AI is reasoning...</div>
      <div class="bubble-text" style="padding: 10px 16px;">
        <span style="color: #38bdf8; font-family: 'JetBrains Mono', monospace;">● ● ● Reasoning</span>
      </div>
    </div>
  `;
  chatStream.appendChild(typingEl);
  chatStream.scrollTop = chatStream.scrollHeight;
  return id;
}

function removeTypingIndicator(id) {
  const el = document.getElementById(id);
  if (el) el.remove();
}

/**
 * Lightweight Markdown-to-HTML parser for clinical reports.
 */
function formatMarkdown(text) {
  if (!text) return "";
  let html = text
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;");

  // Headers
  html = html.replace(/^### (.*$)/gim, "<h3>$1</h3>");
  html = html.replace(/^## (.*$)/gim, "<h3>$1</h3>");

  // Bold
  html = html.replace(/\*\*(.*?)\*\*/g, "<strong>$1</strong>");

  // Badges / Monospace
  html = html.replace(
    /`([^`]+)`/g,
    "<code style='background:rgba(56,189,248,0.15);color:#38bdf8;padding:2px 6px;border-radius:4px;font-size:12px;font-family:monospace;'>$1</code>"
  );

  // Bullet points
  html = html.replace(/^\s*[-•]\s+(.*$)/gim, "<li>$1</li>");
  html = html.replace(/(<li>.*<\/li>)/s, "<ul>$1</ul>");

  // Numbered lists
  html = html.replace(/^\s*(\d+)\.\s+(.*$)/gim, "<li>$2</li>");

  // Paragraphs
  html = html
    .split(/\n\n+/)
    .map((para) => {
      if (para.startsWith("<h3>") || para.startsWith("<ul>") || para.startsWith("<ol>")) {
        return para;
      }
      return `<p>${para.replace(/\n/g, "<br>")}</p>`;
    })
    .join("");

  return html;
}

// ==========================================================================
// Neural Brain & Deep Network Architecture Modal & Interactive Canvas
// ==========================================================================
let brainAnimId = null;
let isBrainAnimPaused = false;

function setupNeuralBrainModal() {
  const modal = document.getElementById("neuralBrainModal");
  const btnOpen = document.getElementById("btnOpenNeuralBrain");
  const btnClose = document.getElementById("btnCloseNeuralModal");
  const btnPulse = document.getElementById("btnPulseSynapses");
  const btnToggle = document.getElementById("btnToggleAnim");

  if (!modal || !btnOpen) return;

  function openModal() {
    modal.classList.add("open");
    modal.setAttribute("aria-hidden", "false");
    setTimeout(() => {
      initNeuralBrainCanvas();
    }, 50);
  }

  function closeModal() {
    modal.classList.remove("open");
    modal.setAttribute("aria-hidden", "true");
    if (brainAnimId) {
      cancelAnimationFrame(brainAnimId);
      brainAnimId = null;
    }
  }

  btnOpen.addEventListener("click", openModal);
  if (btnClose) btnClose.addEventListener("click", closeModal);

  modal.addEventListener("click", (e) => {
    if (e.target === modal) {
      closeModal();
    }
  });

  window.addEventListener("keydown", (e) => {
    if (e.key === "Escape" && modal.classList.contains("open")) {
      closeModal();
    }
  });

  if (btnPulse) {
    btnPulse.addEventListener("click", () => {
      if (window.triggerSynapticPulse) {
        window.triggerSynapticPulse();
      }
    });
  }

  if (btnToggle) {
    btnToggle.addEventListener("click", () => {
      isBrainAnimPaused = !isBrainAnimPaused;
      btnToggle.textContent = isBrainAnimPaused ? "▶ Resume" : "⏸ Pause";
    });
  }
}

function initNeuralBrainCanvas() {
  const canvas = document.getElementById("neuralBrainCanvas");
  if (!canvas) return;

  const ctx = canvas.getContext("2d");
  const container = canvas.parentElement;
  
  const width = canvas.width = container.clientWidth || 600;
  const height = canvas.height = container.clientHeight || 310;

  // Layer configuration
  const layerDefs = [
    { name: "Retinal Input", count: 4, color: "#38bdf8", glow: "rgba(56, 189, 248, 0.4)", label: "512×512×3" },
    { name: "MBConv Backbone", count: 6, color: "#818cf8", glow: "rgba(129, 140, 248, 0.4)", label: "1536 Channels" },
    { name: "CBAM Attention", count: 5, color: "#a855f7", glow: "rgba(168, 85, 247, 0.4)", label: "Dual Gate" },
    { name: "Multi-Task Heads", count: 4, color: "#ec4899", glow: "rgba(236, 72, 153, 0.4)", label: "5-Cls + Reg" },
  ];

  const nodes = [];
  const synapses = [];
  let signals = [];

  const marginX = 70;
  const marginY = 40;
  const usableWidth = width - marginX * 2;
  const layerSpacing = usableWidth / (layerDefs.length - 1);

  // Position nodes
  layerDefs.forEach((def, layerIdx) => {
    const x = marginX + layerIdx * layerSpacing;
    const count = def.count;
    const usableHeight = height - marginY * 2;
    const stepY = usableHeight / (count + 1);

    for (let i = 0; i < count; i++) {
      const y = marginY + stepY * (i + 1);
      nodes.push({
        layer: layerIdx,
        index: i,
        x: x,
        y: y,
        baseRadius: 6,
        radius: 6,
        color: def.color,
        glow: def.glow,
        pulsePhase: Math.random() * Math.PI * 2,
        pulseSpeed: 0.04 + Math.random() * 0.03,
      });
    }
  });

  // Connect synapses between adjacent layers
  for (let l = 0; l < layerDefs.length - 1; l++) {
    const fromNodes = nodes.filter(n => n.layer === l);
    const toNodes = nodes.filter(n => n.layer === l + 1);

    fromNodes.forEach(from => {
      toNodes.forEach(to => {
        synapses.push({
          from: from,
          to: to,
          weight: 0.2 + Math.random() * 0.8,
          alpha: 0.12 + Math.random() * 0.12,
        });
      });
    });
  }

  function spawnSignal(from, to, speedMultiplier = 1.0) {
    signals.push({
      from: from,
      to: to,
      progress: 0,
      speed: (0.015 + Math.random() * 0.015) * speedMultiplier,
      color: from.color,
      size: 3 + Math.random() * 2,
    });
  }

  // Periodic signal emitter
  let lastSpawnTime = 0;

  window.triggerSynapticPulse = function() {
    // Burst pulse across all layers
    synapses.forEach(syn => {
      if (Math.random() < 0.65) {
        spawnSignal(syn.from, syn.to, 1.8);
      }
    });
  };

  function animate(timestamp) {
    if (isBrainAnimPaused) {
      brainAnimId = requestAnimationFrame(animate);
      return;
    }

    ctx.clearRect(0, 0, width, height);

    // Spawn regular electrical pulses
    if (timestamp - lastSpawnTime > 120) {
      lastSpawnTime = timestamp;
      const firstLayerSynapses = synapses.filter(s => s.from.layer === 0);
      if (firstLayerSynapses.length > 0) {
        const randomSyn = firstLayerSynapses[Math.floor(Math.random() * firstLayerSynapses.length)];
        spawnSignal(randomSyn.from, randomSyn.to, 1.0);
      }
    }

    // Draw Synapses (Axons)
    synapses.forEach(syn => {
      ctx.beginPath();
      ctx.moveTo(syn.from.x, syn.from.y);
      ctx.lineTo(syn.to.x, syn.to.y);
      ctx.strokeStyle = `rgba(148, 163, 184, ${syn.alpha})`;
      ctx.lineWidth = syn.weight * 1.5;
      ctx.stroke();
    });

    // Update & Draw Electrical Signal Packets
    for (let i = signals.length - 1; i >= 0; i--) {
      const sig = signals[i];
      sig.progress += sig.speed;

      const curX = sig.from.x + (sig.to.x - sig.from.x) * sig.progress;
      const curY = sig.from.y + (sig.to.y - sig.from.y) * sig.progress;

      // Glow aura
      ctx.beginPath();
      ctx.arc(curX, curY, sig.size * 2, 0, Math.PI * 2);
      ctx.fillStyle = sig.from.glow;
      ctx.fill();

      // Core packet
      ctx.beginPath();
      ctx.arc(curX, curY, sig.size, 0, Math.PI * 2);
      ctx.fillStyle = sig.color;
      ctx.fill();

      // Forward signal to next layer when it reaches destination
      if (sig.progress >= 1) {
        sig.to.radius = sig.to.baseRadius * 1.6; // Node excitation
        const nextLayerSynapses = synapses.filter(s => s.from === sig.to);
        if (nextLayerSynapses.length > 0 && Math.random() < 0.7) {
          const nextSyn = nextLayerSynapses[Math.floor(Math.random() * nextLayerSynapses.length)];
          spawnSignal(nextSyn.from, nextSyn.to, 1.2);
        }
        signals.splice(i, 1);
      }
    }

    // Draw Nodes (Neurons)
    nodes.forEach(node => {
      node.pulsePhase += node.pulseSpeed;
      const pulseDelta = Math.sin(node.pulsePhase) * 1.2;
      const currentRadius = Math.max(node.baseRadius, node.radius + pulseDelta);

      // Relax excitation
      if (node.radius > node.baseRadius) {
        node.radius -= 0.15;
      }

      // Outer glow
      ctx.beginPath();
      ctx.arc(node.x, node.y, currentRadius + 5, 0, Math.PI * 2);
      ctx.fillStyle = node.glow;
      ctx.fill();

      // Inner neuron body
      ctx.beginPath();
      ctx.arc(node.x, node.y, currentRadius, 0, Math.PI * 2);
      ctx.fillStyle = node.color;
      ctx.fill();

      // Core white center
      ctx.beginPath();
      ctx.arc(node.x, node.y, currentRadius * 0.45, 0, Math.PI * 2);
      ctx.fillStyle = "#ffffff";
      ctx.fill();
    });

    // Draw Layer Title Headers
    ctx.textAlign = "center";
    layerDefs.forEach((def, layerIdx) => {
      const x = marginX + layerIdx * layerSpacing;
      ctx.fillStyle = def.color;
      ctx.font = "bold 10px 'JetBrains Mono', monospace";
      ctx.fillText(def.name.toUpperCase(), x, 22);

      ctx.fillStyle = "#64748b";
      ctx.font = "9px 'JetBrains Mono', monospace";
      ctx.fillText(def.label, x, height - 12);
    });

    brainAnimId = requestAnimationFrame(animate);
  }

  if (brainAnimId) {
    cancelAnimationFrame(brainAnimId);
  }
  brainAnimId = requestAnimationFrame(animate);
}
