let transcriptData = [];
let uniqueSpeakers = new Set();
let speakerColors = {};
let currentSummary = "";
let lastAudioFile = "";
let currentSessionId = "";
let currentProcessedAudio = "";
let currentProcessedVideo = "";
let currentBeforeAudio = "";
let currentAfterAudio = "";
let currentDocumentFilename = "";
let currentDocumentType = "";
let currentDocumentText = "";
let currentDocumentId = "";
let currentDocumentChat = [];
let currentTranscriptChat = [];
let sourceTranscriptData = [];
let sourceSummary = "";
let sourceDocumentText = "";
let translationRequestToken = 0;
let speakerNameMap = {}; 
let speakerOrderMap = {}; 
let isSummaryLoading = false;
let groupedTranscriptCache = [];
let historyEntries = [];
let documentEntries = [];
let currentPolicy = null;
let currentUser = null;
let currentImpersonator = null;
let mediaRecorder = null;
let recordingStream = null;
let isRecording = false;
let isRecordingPaused = false;
let recordChunkIndex = 0;
let liveChunkQueue = Promise.resolve();
let liveTranscriptLinesEl = null;
let liveTranscriptStatusEl = null;
let recordedChunks = [];
let recordingMimeType = "audio/webm";
let liveListeningCardEl = null;
let segmentGroupMap = [];
let transcriptRowEls = [];
let lastAgentPlan = [];
let pendingSelectedFile = null;

const TRANSLATION_LANGUAGE_GROUPS = {
    indian: [
        { label: "Assamese", code: "asm_Beng" },
        { label: "Bengali", code: "ben_Beng" },
        { label: "Bhojpuri", code: "bho_Deva" },
        { label: "Gujarati", code: "guj_Gujr" },
        { label: "Hindi", code: "hin_Deva" },
        { label: "Kannada", code: "kan_Knda" },
        { label: "Kashmiri (Arabic)", code: "kas_Arab" },
        { label: "Kashmiri (Devanagari)", code: "kas_Deva" },
        { label: "Maithili", code: "mai_Deva" },
        { label: "Malayalam", code: "mal_Mlym" },
        { label: "Marathi", code: "mar_Deva" },
        { label: "Meitei (Manipuri)", code: "mni_Beng" },
        { label: "Nepali", code: "npi_Deva" },
        { label: "Odia", code: "ory_Orya" },
        { label: "Punjabi", code: "pan_Guru" },
        { label: "Sanskrit", code: "san_Deva" },
        { label: "Sindhi", code: "snd_Arab" },
        { label: "Tamil", code: "tam_Taml" },
        { label: "Telugu", code: "tel_Telu" },
        { label: "Urdu", code: "urd_Arab" }
    ],
    global: [
        { label: "Afrikaans", code: "afr_Latn" },
        { label: "Arabic", code: "ara_Arab" },
        { label: "Bulgarian", code: "bul_Cyrl" },
        { label: "Chinese (Simplified)", code: "zho_Hans" },
        { label: "Chinese (Traditional)", code: "zho_Hant" },
        { label: "Croatian", code: "hrv_Latn" },
        { label: "Czech", code: "ces_Latn" },
        { label: "Danish", code: "dan_Latn" },
        { label: "Dutch", code: "nld_Latn" },
        { label: "English", code: "eng_Latn" },
        { label: "Filipino", code: "tgl_Latn" },
        { label: "Finnish", code: "fin_Latn" },
        { label: "French", code: "fra_Latn" },
        { label: "German", code: "deu_Latn" },
        { label: "Greek", code: "ell_Grek" },
        { label: "Hebrew", code: "heb_Hebr" },
        { label: "Hungarian", code: "hun_Latn" },
        { label: "Indonesian", code: "ind_Latn" },
        { label: "Italian", code: "ita_Latn" },
        { label: "Japanese", code: "jpn_Jpan" },
        { label: "Korean", code: "kor_Hang" },
        { label: "Malay", code: "zsm_Latn" },
        { label: "Norwegian Bokmal", code: "nob_Latn" },
        { label: "Persian", code: "pes_Arab" },
        { label: "Polish", code: "pol_Latn" },
        { label: "Portuguese", code: "por_Latn" },
        { label: "Romanian", code: "ron_Latn" },
        { label: "Russian", code: "rus_Cyrl" },
        { label: "Serbian", code: "srp_Cyrl" },
        { label: "Spanish", code: "spa_Latn" },
        { label: "Swahili", code: "swh_Latn" },
        { label: "Swedish", code: "swe_Latn" },
        { label: "Thai", code: "tha_Thai" },
        { label: "Turkish", code: "tur_Latn" },
        { label: "Ukrainian", code: "ukr_Cyrl" },
        { label: "Vietnamese", code: "vie_Latn" }
    ]
};

function toggleSidebar() {
    document.getElementById("sidebar").classList.toggle("expanded");
    updateTranscriptDependentUI();
}

const sidebar = document.getElementById("sidebar");
const resizer = document.getElementById("resizer");
resizer.addEventListener("mousedown", (e) => {
    document.addEventListener("mousemove", resizeSidebar);
    document.addEventListener("mouseup", stopResize);
});
function resizeSidebar(e) {
    if (sidebar.classList.contains("expanded")) {
        let newWidth = e.clientX;
        if (newWidth > 150 && newWidth < 500) sidebar.style.width = newWidth + "px";
    }
}
function stopResize() { document.removeEventListener("mousemove", resizeSidebar); }

const pdfGutter = document.getElementById("pdfGutter");
const pdfPanel = document.getElementById("pdfPanel");
if (pdfGutter && pdfPanel) {
    pdfGutter.addEventListener("mousedown", (e) => {
        e.preventDefault();
        document.addEventListener("mousemove", resizePdfPanel);
        document.addEventListener("mouseup", stopPdfResize);
    });
}
function resizePdfPanel(e) {
    if (!pdfPanel || !pdfPanel.classList.contains("open")) return;
    const maxWidth = Math.min(window.innerWidth * 0.7, 900);
    const minWidth = 320;
    const newWidth = Math.min(maxWidth, Math.max(minWidth, window.innerWidth - e.clientX));
    pdfPanel.style.width = `${newWidth}px`;
}
function stopPdfResize() {
    document.removeEventListener("mousemove", resizePdfPanel);
}

const audioFileInput = document.getElementById("audioFile");
const recordBtn = document.getElementById("recordBtn");
const pauseBtn = document.getElementById("pauseBtn");
const stopBtn = document.getElementById("stopBtn");
const finalizeBtn = document.getElementById("finalizeBtn");
const exportTranscriptBtn = document.getElementById("exportTranscriptBtn");
const exportSummaryBtn = document.getElementById("exportSummaryBtn");
const translationTargetSelect = document.getElementById("translationTarget");
const agentQueryInput = document.getElementById("agentQueryInput");

function getAgentBarEl() {
    return document.querySelector(".agent-bar");
}

function setAgentBarVisible(visible) {
    const agentBar = getAgentBarEl();
    if (!agentBar) return;
    agentBar.classList.toggle("hidden", !visible);
}

audioFileInput.addEventListener("change", () => {
    if (audioFileInput.files && audioFileInput.files.length > 0 && !isRecording) {
        processAudio();
    }
});

function openFilePicker() {
    audioFileInput.click();
}

function getAgentQueryText(clearAfterRead = false) {
    if (!agentQueryInput) return "";
    const value = String(agentQueryInput.value || "").trim();
    if (clearAfterRead) {
        agentQueryInput.value = "";
    }
    return value;
}

function setAgentQueryText(value) {
    if (!agentQueryInput) return;
    agentQueryInput.value = String(value || "");
}

function updatePendingUploadUI() {
    const uploadButtons = document.querySelectorAll(".agent-upload-btn");
    uploadButtons.forEach((btn) => {
        btn.classList.toggle("has-file", Boolean(pendingSelectedFile));
        const label = pendingSelectedFile ? `Attached: ${pendingSelectedFile.name}` : "Attach File";
        btn.title = label;
        btn.setAttribute("aria-label", label);
    });
}

function stageSelectedFile(file, silentMode = false) {
    if (!file) return;
    if (!isAudioFile(file.name) && !isVideoFile(file.name) && !isDocumentFile(file.name)) {
        if (!silentMode) {
            alert("Unsupported file. Use audio/video or documents (.pdf/.docx/.txt).");
        }
        return;
    }
    pendingSelectedFile = file;
    lastAudioFile = file.name;
    updatePendingUploadUI();
    if (!silentMode) {
        appendAgentResponseCard(
            "File Attached",
            `${file.name}\nNow type your query and press the send arrow to start processing.`,
            "#8a5b00"
        );
    }
}

function primeAgentQuery(value) {
    setAgentQueryText(value || "");
    if (agentQueryInput) {
        agentQueryInput.focus();
        agentQueryInput.setSelectionRange(agentQueryInput.value.length, agentQueryInput.value.length);
    }
}

function updateAgentWorkspaceUI() {
    const contextPill = document.getElementById("agentContextPill");
    const hintText = document.getElementById("agentHintText");
    if (!contextPill || !hintText) return;

    if (currentDocumentId && currentDocumentFilename) {
        contextPill.textContent = `Document: ${truncateText(currentDocumentFilename, 34)}`;
        hintText.textContent = "Ask the current document for conclusions, sections, numbers, or convert its summary to speech.";
        return;
    }

    if (currentSessionId && Array.isArray(transcriptData) && transcriptData.length > 0) {
        contextPill.textContent = `Transcript: ${transcriptData.length} segments ready`;
        hintText.textContent = "Ask follow-up questions, search topics, summarize the meeting, translate, or generate spoken output.";
        return;
    }

    if ((currentSummary || "").trim()) {
        contextPill.textContent = "Summary ready";
        hintText.textContent = "You can translate the summary, convert it to speech, or ask for a shorter action-item version.";
        return;
    }

    contextPill.textContent = "No active context";
    hintText.textContent = "Tip: type a goal first, then upload a file or query the current transcript/document.";
}

async function parseJsonSafe(response) {
    const text = await response.text();
    if (!text) return { json: null, text: "" };
    try {
        return { json: JSON.parse(text), text };
    } catch (_e) {
        return { json: null, text };
    }
}

function responseErrorMessage(result, text) {
    if (result && result.error) return result.error;
    const trimmed = (text || "").trim();
    if (trimmed.toLowerCase().startsWith("<!doctype")) {
        return "Unauthorized or session expired.";
    }
    return trimmed || "Request failed.";
}

function initTranslationLanguageDropdown() {
    if (!translationTargetSelect) return;

    translationTargetSelect.innerHTML = "";

    const defaultOption = document.createElement("option");
    defaultOption.value = "";
    defaultOption.textContent = "Select Translation Language";
    translationTargetSelect.appendChild(defaultOption);

    const appendGroup = (label, items) => {
        const group = document.createElement("optgroup");
        group.label = label;
        items.forEach((item) => {
            const option = document.createElement("option");
            option.value = item.code;
            option.textContent = item.label;
            group.appendChild(option);
        });
        translationTargetSelect.appendChild(group);
    };

    appendGroup("Indian Languages", TRANSLATION_LANGUAGE_GROUPS.indian);
    appendGroup("Global Languages", TRANSLATION_LANGUAGE_GROUPS.global);
    translationTargetSelect.addEventListener("change", () => {
        applySelectedLanguageToAllTexts();
    });
}

function cloneTranscriptSegments(segments) {
    if (!Array.isArray(segments)) return [];
    return segments.map((seg) => ({
        ...seg,
        text: String((seg && seg.text) || "")
    }));
}

function setSourceTranscript(segments) {
    sourceTranscriptData = cloneTranscriptSegments(segments);
}

function setSourceSummary(summaryText) {
    sourceSummary = String(summaryText || "");
}

function setSourceDocumentText(docText) {
    sourceDocumentText = String(docText || "");
}

async function renderCurrentContent() {
    if (Array.isArray(transcriptData) && transcriptData.length > 0) {
        rebuildSpeakerState();
        await renderChatDelayed();
        setupRenameSidebar();
        document.getElementById("sumBtn").style.display = "flex";
    } else if (currentDocumentFilename) {
        renderDocumentResult();
        document.getElementById("sumBtn").style.display = "none";
    } else {
        await renderChatDelayed();
        document.getElementById("sumBtn").style.display = "none";
    }
    updateAgentWorkspaceUI();
    updateTranscriptDependentUI();
}

function setUploadHeroVisible(visible) {
    const hero = document.querySelector("#chat .upload-hero");
    if (!hero) return;
    hero.classList.toggle("hidden", !visible);
}

function clearChatRows() {
    const chat = document.getElementById("chat");
    if (!chat) return;
    const old = chat.querySelectorAll(".transcription");
    old.forEach(r => r.remove());
}


function setSummaryButtonState() {
    const sumBtn = document.getElementById("sumBtn");
    if (!sumBtn) return;
    const hasSummary = Boolean((currentSummary || "").trim());
    sumBtn.disabled = hasSummary;
    if (hasSummary) {
        sumBtn.classList.add("disabled");
        sumBtn.title = "Summary already generated for this file";
    } else {
        sumBtn.classList.remove("disabled");
        sumBtn.title = "Generate summary";
    }
}

function updateTranscriptDependentUI() {
    const hasTranscript = Array.isArray(transcriptData) && transcriptData.length > 0;
    const isExpanded = sidebar.classList.contains("expanded");
    const sideHeading = document.getElementById("sideHeading");
    const renameBox = document.getElementById("renameBox");
    const keywordBtn = document.getElementById("keywordBtn");

    [finalizeBtn, exportTranscriptBtn, exportSummaryBtn].forEach((btn) => {
        if (!btn) return;
        btn.classList.toggle("hidden", !hasTranscript);
    });
    if (keywordBtn) keywordBtn.classList.toggle("hidden", !hasTranscript);

    if (sideHeading) {
        sideHeading.style.display = hasTranscript && isExpanded ? "block" : "none";
    }
    if (!hasTranscript && renameBox) {
        renameBox.innerHTML = "";
    }
    setSummaryButtonState();
}

function showKeywordSearch() {
    const card = document.getElementById("semanticSearchCard");
    if (!card) renderKeywordSearchBar();
    const target = document.getElementById("semanticSearchCard");
    if (!target) return;
    target.scrollIntoView({ behavior: "smooth", block: "start" });
    const input = document.getElementById("semanticSearchInput");
    if (input) setTimeout(() => input.focus(), 200);
}

function initDropZone() {
    const dropZone = document.getElementById("dropZoneCard");
    if (!dropZone) return;

    ["dragenter", "dragover"].forEach((evtName) => {
        dropZone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add("drag-over");
        });
    });

    ["dragleave", "drop"].forEach((evtName) => {
        dropZone.addEventListener(evtName, (e) => {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove("drag-over");
        });
    });

    dropZone.addEventListener("drop", (e) => {
        const files = e.dataTransfer && e.dataTransfer.files ? e.dataTransfer.files : null;
        if (!files || files.length === 0) return;

        const dt = new DataTransfer();
        dt.items.add(files[0]);
        audioFileInput.files = dt.files;
        processAudio();
    });
}

function setRecordingButtons(active, paused = false) {
    if (!recordBtn || !pauseBtn || !stopBtn) return;
    recordBtn.classList.toggle("hidden", active);
    pauseBtn.classList.toggle("hidden", !active);
    stopBtn.classList.toggle("hidden", !active);
    pauseBtn.textContent = paused ? "Resume" : "Pause";
}

function ensureLiveTranscriptCard() {
    if (
        liveTranscriptLinesEl &&
        liveTranscriptStatusEl &&
        liveTranscriptLinesEl.isConnected &&
        liveTranscriptStatusEl.isConnected
    ) {
        return;
    }

    const chat = document.getElementById("chat");
    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:#ef4444">REC</div>
        <div class="content live-transcript-card">
            <span class="live-transcript-badge">Live Transcription</span>
            <div id="liveTranscriptStatus" style="font-size:12px;color:#fecaca;margin-bottom:6px;">Recording...</div>
            <div id="liveTranscriptLines" class="live-transcript-lines"></div>
        </div>
    `;
    chat.appendChild(row);
    liveTranscriptStatusEl = row.querySelector("#liveTranscriptStatus");
    liveTranscriptLinesEl = row.querySelector("#liveTranscriptLines");
    if (currentSummary) {
        renderSummaryCard(currentSummary);
        setSummaryButtonState();
    }
    chat.scrollTop = chat.scrollHeight;
}

function appendLiveTranscript(text) {
    const value = (text || "").trim();
    if (!value) return;
    ensureLiveTranscriptCard();
    if (liveTranscriptLinesEl.textContent.trim()) {
        liveTranscriptLinesEl.textContent += `\n${value}`;
    } else {
        liveTranscriptLinesEl.textContent = value;
    }
    const chat = document.getElementById("chat");
    chat.scrollTop = chat.scrollHeight;
}

function renderListeningPromptCard() {
    if (liveListeningCardEl && liveListeningCardEl.isConnected) {
        return liveListeningCardEl;
    }
    const chat = document.getElementById("chat");
    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:#ef4444">REC</div>
        <div class="content summary-card" style="position:relative;">
            <div class="summary-loading">
                <span class="summary-spinner" aria-hidden="true"></span>
                <span class="summary-loading-text">Listening...</span>
            </div>
        </div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    liveListeningCardEl = row;
    return row;
}

function clearListeningPromptCard() {
    if (liveListeningCardEl && liveListeningCardEl.parentNode) {
        liveListeningCardEl.parentNode.removeChild(liveListeningCardEl);
    }
    liveListeningCardEl = null;
}

function preferredRecorderMimeType() {
    const types = [
        "audio/webm;codecs=opus",
        "audio/webm",
        "audio/ogg;codecs=opus",
        "audio/mp4"
    ];
    for (const t of types) {
        if (window.MediaRecorder && MediaRecorder.isTypeSupported && MediaRecorder.isTypeSupported(t)) {
            return t;
        }
    }
    return "";
}

function extensionForMimeType(mimeType) {
    const value = String(mimeType || "").toLowerCase();
    if (value.includes("ogg")) return "ogg";
    if (value.includes("mp4")) return "mp4";
    if (value.includes("wav")) return "wav";
    return "webm";
}

async function sendLiveChunk(blob, chunkIndex) {
    if (!blob || blob.size === 0) return;
    const formData = new FormData();
    const ext = extensionForMimeType(blob.type);
    const filename = `chunk_${chunkIndex}.${ext}`;
    formData.append("audio_chunk", blob, filename);
    formData.append("chunk_index", String(chunkIndex));

    try {
        const response = await fetch("/transcribe_chunk", {
            method: "POST",
            body: formData
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Live transcription failed");
        }
        appendLiveTranscript(result.text || "");
        if (liveTranscriptStatusEl && !isRecordingPaused) {
            liveTranscriptStatusEl.textContent = "Recording...";
        }
    } catch (e) {
        if (liveTranscriptStatusEl) {
            liveTranscriptStatusEl.textContent = `Live transcription error: ${e.message || "unknown error"}`;
        }
    }
}

async function startRecording() {
    if (isRecording) return;
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        alert("Microphone recording is not supported in this browser.");
        return;
    }

    try {
        recordingStream = await navigator.mediaDevices.getUserMedia({ audio: true });
        const options = {};
        const mimeType = preferredRecorderMimeType();
        if (mimeType) options.mimeType = mimeType;

        mediaRecorder = new MediaRecorder(recordingStream, options);
        recordingMimeType = mimeType || mediaRecorder.mimeType || "audio/webm";
        recordedChunks = [];
        isRecording = true;
        isRecordingPaused = false;
        recordChunkIndex = 0;
        liveChunkQueue = Promise.resolve();
        renderListeningPromptCard();
        setRecordingButtons(true, false);

        mediaRecorder.ondataavailable = (event) => {
            if (!event.data || event.data.size === 0) return;
            recordedChunks.push(event.data);
            recordChunkIndex += 1;
        };

        mediaRecorder.onerror = () => {
            // Keep UI silent during live recording flow.
        };

        mediaRecorder.onstop = async () => {
            clearListeningPromptCard();
            if (recordingStream) {
                recordingStream.getTracks().forEach((t) => t.stop());
            }
            recordingStream = null;
            mediaRecorder = null;
            isRecording = false;
            isRecordingPaused = false;
            setRecordingButtons(false, false);

            const finalBlob = new Blob(recordedChunks, { type: recordingMimeType || "audio/webm" });
            recordedChunks = [];
            if (!finalBlob || finalBlob.size === 0) {
                return;
            }

            const ext = extensionForMimeType(recordingMimeType);
            const filename = `live_meeting_${new Date().toISOString().replace(/[:.]/g, "-")}.${ext}`;
            const liveFile = new File([finalBlob], filename, {
                type: recordingMimeType || "audio/webm",
                lastModified: Date.now(),
            });

            stageSelectedFile(liveFile, true);
        };

        mediaRecorder.start(1000);
    } catch (e) {
        alert(e.message || "Microphone permission denied.");
        isRecording = false;
        isRecordingPaused = false;
        recordedChunks = [];
        clearListeningPromptCard();
        setRecordingButtons(false, false);
    }
}

function togglePauseRecording() {
    if (!mediaRecorder || !isRecording) return;
    if (mediaRecorder.state === "recording") {
        mediaRecorder.pause();
        isRecordingPaused = true;
        setRecordingButtons(true, true);
    } else if (mediaRecorder.state === "paused") {
        mediaRecorder.resume();
        isRecordingPaused = false;
        setRecordingButtons(true, false);
    }
}

function stopRecording() {
    if (!mediaRecorder || !isRecording) return;
    if (mediaRecorder.state !== "inactive") {
        try {
            mediaRecorder.requestData();
        } catch (_e) {
            // Ignore and proceed to stop.
        }
        mediaRecorder.stop();
    }
}

function formatHistoryLabel(iso) {
    if (!iso) return "";
    const d = new Date(iso);
    if (isNaN(d.getTime())) return iso;
    return d.toLocaleString();
}

function truncateText(value, maxLen = 28) {
    const txt = String(value || "").trim();
    if (txt.length <= maxLen) return txt;
    return `${txt.slice(0, maxLen - 1)}…`;
}

function renderHistoryList() {
    const list = document.getElementById("historyList");
    if (!list) return;

    if (!historyEntries || historyEntries.length === 0) {
        list.innerHTML = `<div class="history-empty">No history yet</div>`;
        return;
    }

    const canRename = currentPolicy ? (currentPolicy.permissions || []).includes("history:rename") : true;
    const canDelete = currentPolicy ? (currentPolicy.permissions || []).includes("history:delete") : true;

    list.innerHTML = historyEntries.map((entry) => {
        const activeClass = entry.session_id === currentSessionId ? "active" : "";
        const title = truncateText(entry.title || entry.processed_file || entry.session_id);
        const meta = `${entry.segments || 0} segments • ${formatHistoryLabel(entry.updated_at)}`;
        return `
            <div class="history-item ${activeClass}" title="${escapeHTMLText(entry.title || entry.session_id)}">
                <button class="history-open-btn" onclick="openHistorySession('${entry.session_id}')">
                    <span class="history-title">${escapeHTMLText(title)}</span>
                    <span class="history-meta">${escapeHTMLText(meta)}</span>
                </button>
                <div class="history-actions">
                    ${canRename ? `<button class="history-action-btn" onclick="renameHistorySession('${entry.session_id}')" title="Rename">✎</button>` : ""}
                    ${canDelete ? `<button class="history-action-btn delete" onclick="deleteHistorySession('${entry.session_id}')" title="Delete">🗑</button>` : ""}
                </div>
            </div>
        `;
    }).join("");
}

function renderDocumentHistory() {
    const list = document.getElementById("documentHistoryList");
    if (!list) return;

    if (!documentEntries || documentEntries.length === 0) {
        list.innerHTML = `<div class="history-empty">No documents yet</div>`;
        return;
    }

    const canRename = currentPolicy ? (currentPolicy.permissions || []).includes("history:rename") : true;
    const canDelete = currentPolicy ? (currentPolicy.permissions || []).includes("history:delete") : true;

    list.innerHTML = documentEntries.map((entry) => {
        const activeClass = entry.document_id === currentDocumentId ? "active" : "";
        const title = truncateText(entry.filename || entry.document_id);
        const meta = `${entry.chunk_count || 0} chunks • ${formatHistoryLabel(entry.updated_at)}`;
        return `
            <div class="history-item ${activeClass}" title="${escapeHTMLText(entry.filename || entry.document_id)}">
                <button class="history-open-btn" onclick="openDocumentEntry('${entry.document_id}')">
                    <span class="history-title">${escapeHTMLText(title)}</span>
                    <span class="history-meta">${escapeHTMLText(meta)}</span>
                </button>
                <div class="history-actions">
                    ${canRename ? `<button class="history-action-btn" onclick="renameDocumentEntry('${entry.document_id}')" title="Rename">✎</button>` : ""}
                    ${canDelete ? `<button class="history-action-btn delete" onclick="deleteDocumentEntry('${entry.document_id}')" title="Delete">🗑</button>` : ""}
                </div>
            </div>
        `;
    }).join("");
}

async function refreshHistory() {
    try {
        const response = await fetch("/history");
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to load history");
        }
        historyEntries = result.history || [];
        renderHistoryList();
    } catch (e) {
        const list = document.getElementById("historyList");
        if (list) {
            list.innerHTML = `<div class="history-empty">History unavailable</div>`;
        }
    }
}

async function refreshDocumentHistory() {
    try {
        const response = await fetch("/api/documents");
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to load documents");
        }
        documentEntries = result.documents || [];
        renderDocumentHistory();
    } catch (e) {
        const list = document.getElementById("documentHistoryList");
        if (list) {
            list.innerHTML = `<div class="history-empty">Documents unavailable</div>`;
        }
    }
}

async function loadPolicy() {
    try {
        const response = await fetch("/me");
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to load policy");
        }
        currentPolicy = result.policy || null;
        currentUser = result.user || null;
        currentImpersonator = result.impersonator || null;
    } catch (_e) {
        currentPolicy = null;
        currentUser = null;
        currentImpersonator = null;
    }

    const perms = currentPolicy ? currentPolicy.permissions || [] : [];
    const has = (p) => perms.includes(p);
    const clearBtn = document.getElementById("clearHistoryBtn");
    if (clearBtn) {
        clearBtn.style.display = has("history:delete") ? "inline-flex" : "none";
    }
    const sumBtn = document.getElementById("sumBtn");
    if (sumBtn) {
        sumBtn.style.display = has("summary:generate") ? "flex" : "none";
    }
    const exportTranscriptBtn = document.getElementById("exportTranscriptBtn");
    if (exportTranscriptBtn) {
        exportTranscriptBtn.style.display = has("export:transcript") ? "flex" : "none";
    }
    const exportSummaryBtn = document.getElementById("exportSummaryBtn");
    if (exportSummaryBtn) {
        exportSummaryBtn.style.display = has("export:summary") ? "flex" : "none";
    }
    const adminBtn = document.getElementById("adminBtn");
    if (adminBtn) {
        const isSuper = currentUser && currentUser.role_name === "super_admin";
        const isAdmin = currentUser && currentUser.role_name === "admin";
        if (isSuper) {
            adminBtn.textContent = "Create Admin/User";
            adminBtn.style.display = "flex";
        } else if (isAdmin) {
            adminBtn.textContent = "Create Users";
            adminBtn.style.display = "flex";
        } else {
            adminBtn.style.display = "none";
        }
    }
    const impersonateBtn = document.getElementById("impersonateBtn");
    if (impersonateBtn) {
        const isSuper = currentUser && currentUser.role_name === "super_admin";
        const isAdmin = currentUser && currentUser.role_name === "admin";
        if (currentImpersonator) {
            impersonateBtn.textContent = "Return to Admin";
            impersonateBtn.classList.remove("hidden");
        } else if (isSuper || isAdmin) {
            impersonateBtn.textContent = "Switch User";
            impersonateBtn.classList.remove("hidden");
        } else {
            impersonateBtn.classList.add("hidden");
        }
    }
    if (currentUser && currentUser.name) {
        renderGreetingCard(currentUser.name);
    }
}

async function handleImpersonate() {
    if (currentImpersonator) {
        try {
            const response = await fetch("/api/impersonate/stop", { method: "POST" });
            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || "Failed to return");
            }
            location.reload();
        } catch (e) {
            window.alert(e.message || "Failed to return to admin.");
        }
        return;
    }
    const email = window.prompt("Enter the user email to switch into:");
    if (!email) return;
    try {
        const response = await fetch("/api/impersonate", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ email: email.trim() }),
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to switch user");
        }
        location.reload();
    } catch (e) {
        window.alert(e.message || "Failed to switch user.");
    }
}

function rebuildSpeakerState() {
    uniqueSpeakers.clear();
    speakerNameMap = {};
    speakerOrderMap = {};

    let speakerIndex = 0;
    let tempSpeakerOrder = [];
    transcriptData.forEach(seg => {
        if (!uniqueSpeakers.has(seg.speaker)) {
            uniqueSpeakers.add(seg.speaker);
            speakerOrderMap[seg.speaker] = speakerIndex;
            tempSpeakerOrder.push(seg.speaker);
            speakerIndex++;
        }
        if (!speakerNameMap[seg.speaker]) {
            speakerNameMap[seg.speaker] = seg.speaker;
        }
    });

    tempSpeakerOrder.forEach((speaker, idx) => {
        speakerColors[speaker] = getSpeakerColor(idx);
    });
}

async function openHistorySession(sessionId) {
    if (!sessionId) return;
    document.getElementById("loadingOverlay").style.display = "flex";
    try {
        const response = await fetch(`/history/${encodeURIComponent(sessionId)}`);
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to open history");
        }

        currentSessionId = result.session_id || sessionId;
        transcriptData = result.transcript || [];
        currentSummary = result.summary || "";
        currentBeforeAudio = result.before_audio_file || result.processed_file || "";
        currentAfterAudio = result.after_audio_file || result.processed_file || "";
        currentProcessedAudio = currentAfterAudio || "";
        currentProcessedVideo = result.source_video || "";
        currentDocumentFilename = "";
        currentDocumentType = "";
        currentDocumentText = "";
        currentDocumentId = "";
        currentDocumentChat = [];
        currentTranscriptChat = result.qa_history || [];
        lastAudioFile = currentAfterAudio || result.processed_file || result.title || sessionId;
        setSourceTranscript(transcriptData);
        setSourceSummary(currentSummary);
        setSourceDocumentText("");

        await applySelectedLanguageToAllTexts(false);
        renderHistoryList();
        renderDocumentHistory();
        updateSidebarMiniPreview();
        setSummaryButtonState();
    } catch (e) {
        alert(e.message || "Failed to open history");
    } finally {
        document.getElementById("loadingOverlay").style.display = "none";
    }
}

async function openDocumentEntry(docId) {
    if (!docId) return;
    document.getElementById("loadingOverlay").style.display = "flex";
    try {
        const response = await fetch(`/api/documents/${encodeURIComponent(docId)}`);
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to open document");
        }

        currentDocumentId = result.document_id || docId;
        currentDocumentFilename = result.filename || "";
        currentDocumentType = (result.document_type || "").toLowerCase();
        currentDocumentText = result.text_preview || "";
        currentDocumentChat = result.chat_history || [];
        currentTranscriptChat = [];
        currentSummary = result.summary || "";
        currentSessionId = "";
        transcriptData = [];
        currentProcessedAudio = "";
        currentProcessedVideo = "";
        currentBeforeAudio = "";
        currentAfterAudio = "";
        setSourceTranscript([]);
        setSourceSummary(currentSummary);
        setSourceDocumentText(currentDocumentText);

        await applySelectedLanguageToAllTexts(false);
        renderDocumentHistory();
        renderHistoryList();
        updateSidebarMiniPreview();
        setSummaryButtonState();
    } catch (e) {
        alert(e.message || "Failed to open document");
    } finally {
        document.getElementById("loadingOverlay").style.display = "none";
    }
}

async function renameDocumentEntry(docId) {
    const entry = (documentEntries || []).find((x) => x.document_id === docId);
    const currentName = entry ? (entry.filename || entry.document_id) : docId;
    const nextName = window.prompt("Rename document:", currentName);
    if (nextName === null) return;
    const clean = nextName.trim();
    if (!clean) return;

    try {
        const response = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ filename: clean })
        });
        const { json, text } = await parseJsonSafe(response);
        const result = json || {};
        if (!response.ok) {
            throw new Error(responseErrorMessage(result, text));
        }
        await refreshDocumentHistory();
        setSummaryButtonState();
    } catch (e) {
        alert(e.message || "Failed to rename document");
    }
}

async function deleteDocumentEntry(docId) {
    const ok = window.confirm("Delete this document?");
    if (!ok) return;
    try {
        const response = await fetch(`/api/documents/${encodeURIComponent(docId)}`, {
            method: "DELETE"
        });
        const { json, text } = await parseJsonSafe(response);
        const result = json || {};
        if (!response.ok) {
            throw new Error(responseErrorMessage(result, text));
        }
        if (currentDocumentId === docId) {
            currentDocumentId = "";
            currentDocumentFilename = "";
            currentDocumentType = "";
            currentDocumentText = "";
            currentDocumentChat = [];
            currentTranscriptChat = [];
            currentSummary = "";
            setSourceSummary("");
            setSourceDocumentText("");
            await renderCurrentContent();
            updateSidebarMiniPreview();
        setSummaryButtonState();
        }
        await refreshDocumentHistory();
        setSummaryButtonState();
    } catch (e) {
        alert(e.message || "Failed to delete document");
    }
}

async function clearAllDocuments() {
    if (!documentEntries || documentEntries.length === 0) return;
    const ok = window.confirm("Clear all documents?");
    if (!ok) return;
    try {
        const response = await fetch("/api/documents", { method: "DELETE" });
        const { json, text } = await parseJsonSafe(response);
        const result = json || {};
        if (!response.ok) {
            throw new Error(responseErrorMessage(result, text));
        }
    } catch (e) {
        alert(e.message || "Failed to clear documents");
        return;
    }
    currentDocumentId = "";
    currentDocumentFilename = "";
    currentDocumentType = "";
    currentDocumentText = "";
    currentDocumentChat = [];
    currentTranscriptChat = [];
    currentSummary = "";
    setSourceSummary("");
    setSourceDocumentText("");
    await renderCurrentContent();
    updateSidebarMiniPreview();
        setSummaryButtonState();
    await refreshDocumentHistory();
        setSummaryButtonState();
}

async function renameHistorySession(sessionId) {
    const entry = (historyEntries || []).find((x) => x.session_id === sessionId);
    const currentTitle = (entry && entry.title) ? entry.title : sessionId;
    const nextTitle = window.prompt("Rename chat title:", currentTitle);
    if (nextTitle === null) return;
    const clean = nextTitle.trim();
    if (!clean) return;

    try {
        const response = await fetch(`/history/${encodeURIComponent(sessionId)}`, {
            method: "PATCH",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ title: clean })
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to rename history");
        }
        await refreshHistory();
    } catch (e) {
        alert(e.message || "Failed to rename history");
    }
}

async function deleteHistorySession(sessionId) {
    const ok = window.confirm("Delete this history item?");
    if (!ok) return;

    try {
        const response = await fetch(`/history/${encodeURIComponent(sessionId)}`, {
            method: "DELETE"
        });
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to delete history");
        }

        if (currentSessionId === sessionId) {
            currentSessionId = "";
            transcriptData = [];
            currentSummary = "";
            currentProcessedAudio = "";
            currentProcessedVideo = "";
            currentBeforeAudio = "";
            currentAfterAudio = "";
            currentDocumentFilename = "";
            currentDocumentType = "";
            currentDocumentText = "";
            currentDocumentId = "";
            currentDocumentChat = [];
            currentTranscriptChat = [];
            updateSidebarMiniPreview();
        setSummaryButtonState();
            setSourceTranscript([]);
            setSourceSummary("");
            setSourceDocumentText("");
            groupedTranscriptCache = [];
            uniqueSpeakers.clear();
            speakerNameMap = {};
            speakerOrderMap = {};
            await renderChatDelayed();
            setupRenameSidebar();
            document.getElementById("sumBtn").style.display = "none";
            updateTranscriptDependentUI();
        }

        await refreshHistory();
    } catch (e) {
        alert(e.message || "Failed to delete history");
    }
}

async function clearAllHistory() {
    if (!historyEntries || historyEntries.length === 0) return;
    const ok = window.confirm("Clear all history entries?");
    if (!ok) return;

    const ids = historyEntries.map((x) => x.session_id).filter(Boolean);
    for (const sessionId of ids) {
        try {
            await fetch(`/history/${encodeURIComponent(sessionId)}`, { method: "DELETE" });
        } catch (_e) {
            // Continue deleting remaining items.
        }
    }

    currentSessionId = "";
    transcriptData = [];
    currentSummary = "";
    currentProcessedAudio = "";
    currentProcessedVideo = "";
    currentBeforeAudio = "";
    currentAfterAudio = "";
    currentDocumentFilename = "";
    currentDocumentType = "";
    currentDocumentText = "";
    currentDocumentId = "";
    currentDocumentChat = [];
    setSourceTranscript([]);
    setSourceSummary("");
    setSourceDocumentText("");
    groupedTranscriptCache = [];
    uniqueSpeakers.clear();
    speakerNameMap = {};
    speakerOrderMap = {};
    await renderChatDelayed();
    setupRenameSidebar();
    document.getElementById("sumBtn").style.display = "none";
    updateTranscriptDependentUI();
    await refreshHistory();
    await refreshDocumentHistory();
        setSummaryButtonState();
    updateSidebarMiniPreview();
        setSummaryButtonState();
}

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function getSpeakerColor(idx) {
    // Golden-angle hue stepping gives visually distinct, non-repeating speaker colors.
    const hue = (idx * 137.508) % 360;
    const saturation = 68 + ((idx % 3) * 6); // 68, 74, 80
    const lightness = 42 + (((idx + 1) % 3) * 5); // 47, 52, 42
    return {
        main: `hsl(${hue.toFixed(1)} ${saturation}% ${lightness}%)`,
        glow: `hsl(${hue.toFixed(1)} ${Math.min(88, saturation + 8)}% ${Math.min(62, lightness + 12)}% / 0.14)`
    };
}

function isAudioFile(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    return ["wav", "mp3", "aac", "aiff", "wma", "amr", "opus", "webm", "ogg", "m4a"].includes(ext);
}

function isVideoFile(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    return ["mp4", "mkv", "avi", "mov", "wmv", "mpeg", "3gp"].includes(ext);
}

function isDocumentFile(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    return ["pdf", "docx", "txt", "png", "jpg", "jpeg", "tif", "tiff", "bmp"].includes(ext);
}

async function processSelectedFile(selectedFile, silentMode = false, explicitQuery = "") {
    if (!selectedFile) return;
    lastAudioFile = selectedFile.name;
    
    // Show Loader
    document.getElementById("loadingOverlay").style.display = "flex";

    try {
        if (selectedFile && !isAudioFile(selectedFile.name) && !isVideoFile(selectedFile.name) && !isDocumentFile(selectedFile.name)) {
            throw new Error("Unsupported file. Use audio/video or documents (.pdf/.docx/.txt).");
        }

        const formData = new FormData();
        formData.append("file", selectedFile);
        formData.append(
            "query",
            explicitQuery || (
                isDocumentFile(selectedFile.name)
                    ? "Ingest this document, summarize it, and prepare it for question answering."
                    : "Transcribe this media, diarize speakers, and prepare it for follow-up questions."
            )
        );
        const response = await fetch("/api/agent/chat", {
            method: "POST",
            body: formData
        });

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Agent processing failed");
        }

        await handleAgentResponse(result, { source: "upload" });
        setAgentBarVisible(false);
        pendingSelectedFile = null;
        updatePendingUploadUI();
        if (audioFileInput) {
            audioFileInput.value = "";
        }
    } catch (e) { 
        document.getElementById("loadingOverlay").style.display = "none";
        if (!silentMode) {
            alert(e.message || "Connection failed.");
        }
    }
}

async function processAudio(silentMode = false) {
    if (isRecording) {
        alert("Stop recording before processing another file.");
        return;
    }
    const selectedFile = audioFileInput.files && audioFileInput.files.length > 0 ? audioFileInput.files[0] : null;
    stageSelectedFile(selectedFile, silentMode);
}

async function renderChatDelayed() {
    // Hide Loader once rendering starts
    document.getElementById("loadingOverlay").style.display = "none";
    
    const chat = document.getElementById("chat");
    const old = chat.querySelectorAll(".transcription");
    old.forEach(r => r.remove());

    if (transcriptData.length === 0) {
        setUploadHeroVisible(true);
        return;
    }

    setUploadHeroVisible(false);

    if (currentProcessedVideo) {
        const videoRow = document.createElement("div");
        videoRow.className = "message-row transcription";
        videoRow.innerHTML = `
            <div class="avatar" style="background:#f97316">▶</div>
            <div class="content video-preview-content" style="border-left: 4px solid #f97316; background: rgba(249, 115, 22, 0.12);">
                <span style="font-size:10px; font-weight:900; color:#f97316; text-transform:uppercase;">VIDEO PREVIEW</span><br>
                <video class="video-preview-player" controls preload="metadata" style="width:600px; height:400px; max-width:100%; margin-top:8px; border-radius:12px; background:#000;">
                    <source src="/videos/${encodeURIComponent(currentProcessedVideo)}">
                    Your browser does not support video playback.
                </video>
                <span style="display:block; font-size:10px; color:var(--muted); margin-top:5px; font-weight:600;">${currentProcessedVideo}</span>
            </div>
        `;
        chat.appendChild(videoRow);

        const videoEl = videoRow.querySelector(".video-preview-player");
        if (videoEl) {
            videoEl.addEventListener("loadedmetadata", () => {
                const isLandscape = videoEl.videoWidth >= videoEl.videoHeight;
                videoEl.style.width = isLandscape ? "600px" : "400px";
                videoEl.style.height = isLandscape ? "400px" : "600px";
            });
        }
    }

    const previewBeforeAudio = currentBeforeAudio || currentProcessedAudio;
    const previewAfterAudio = currentAfterAudio || currentProcessedAudio;

    if (previewBeforeAudio || previewAfterAudio) {
        const beforeName = previewBeforeAudio || "Not Available";
        const afterName = previewAfterAudio || "Not Available";
        const beforeSrc = previewBeforeAudio ? `/audio/${encodeURIComponent(previewBeforeAudio)}` : "";
        const afterSrc = previewAfterAudio ? `/audio/${encodeURIComponent(previewAfterAudio)}` : "";

        const audioRow = document.createElement("div");
        audioRow.className = "message-row transcription";
        audioRow.innerHTML = `
            <div class="avatar" style="background:#ef4444">♫</div>
            <div class="content audio-preview-content" style="border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.12);">
                <span style="font-size:10px; font-weight:900; color:#ef4444; text-transform:uppercase;">AUDIO PREVIEW</span><br>
                <div class="audio-compare-grid">
                    <div class="audio-compare-item">
                        <span class="audio-compare-label">Before</span>
                        ${beforeSrc ? `
                            <audio controls preload="metadata" style="width:100%; margin-top:8px;">
                                <source src="${beforeSrc}">
                                Your browser does not support audio playback.
                            </audio>
                        ` : `<div class="audio-compare-empty">Not available</div>`}
                        <span class="audio-compare-name">${escapeHTMLText(beforeName)}</span>
                    </div>
                    <div class="audio-compare-item">
                        <span class="audio-compare-label">After (Demucs)</span>
                        ${afterSrc ? `
                            <audio controls preload="metadata" style="width:100%; margin-top:8px;">
                                <source src="${afterSrc}">
                                Your browser does not support audio playback.
                            </audio>
                        ` : `<div class="audio-compare-empty">Not available</div>`}
                        <span class="audio-compare-name">${escapeHTMLText(afterName)}</span>
                    </div>
                </div>
            </div>
        `;
        chat.appendChild(audioRow);
    }

    const groupedTranscript = [];
    let currentGroup = null;

    transcriptData.forEach((seg, segIndex) => {
        if (currentGroup && currentGroup.speaker === seg.speaker) {
            currentGroup.texts.push(seg.text);
            currentGroup.end = seg.end;
            currentGroup.segmentIndices.push(segIndex);
        } else {
            if (currentGroup) groupedTranscript.push(currentGroup);
            currentGroup = {
                speaker: seg.speaker,
                texts: [seg.text],
                start: seg.start,
                end: seg.end,
                segmentIndices: [segIndex]
            };
        }
    });
    if (currentGroup) groupedTranscript.push(currentGroup);
    groupedTranscriptCache = groupedTranscript;
    segmentGroupMap = new Array(transcriptData.length);
    transcriptRowEls = [];
    groupedTranscriptCache.forEach((group, idx) => {
        (group.segmentIndices || []).forEach((segIdx) => {
            segmentGroupMap[segIdx] = idx;
        });
    });

    for (let i = 0; i < groupedTranscript.length; i++) {
        const group = groupedTranscript[i];
        const row = document.createElement("div");
        row.className = "message-row transcription";
        row.dataset.groupIndex = String(i);
        const colorSet = speakerColors[group.speaker];
        const ts = `[${formatTime(group.start || 0)} - ${formatTime(group.end || 0)}]`;
        
        const combinedText = group.texts.length > 1 
            ? group.texts.map(t => `• ${t}`).join('<br>') 
            : group.texts[0];

        row.innerHTML = `<div class="avatar" style="background: ${colorSet.main}">${group.speaker[0]}</div><div class="content" style="border-left: 4px solid ${colorSet.main}; background: ${colorSet.glow}"><div class="translate-transcript-icon" onclick="translateTranscriptByIndex(${i})" title="Translate Transcript"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M5 8h14"></path><path d="M5 12h8"></path><path d="M13 19l4-8 4 8"></path><path d="M14.5 16h5"></path></svg></div><div class="copy-transcript-icon" onclick="copyTranscriptByIndex(${i})" title="Copy Transcript"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg></div><div class="delete-transcript-icon" onclick="deleteTranscriptByIndex(${i})" title="Delete Transcript Block"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M3 6h18"></path><path d="M8 6V4h8v2"></path><path d="M19 6l-1 14H6L5 6"></path><path d="M10 11v6"></path><path d="M14 11v6"></path></svg></div><span style="font-size:10px; font-weight:900; color:${colorSet.main}; text-transform:uppercase;">${group.speaker}</span><br>${combinedText}<span style="display:block; font-size:10px; color:var(--muted); margin-top:5px; font-weight:600;">${ts}</span></div>`;
        chat.appendChild(row);
        transcriptRowEls[i] = row;
    }
    renderTranscriptQAPanel();
    chat.scrollTop = chat.scrollHeight;
}


function renderTranscriptQAPanel() {
    const chat = document.getElementById("chat");
    if (!chat) return;

    const existing = document.getElementById("transcriptQaPanel");
    if (existing) existing.remove();

    if (!Array.isArray(transcriptData) || transcriptData.length === 0) {
        return;
    }

    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.id = "transcriptQaPanel";
    row.innerHTML = `
        <div class="avatar" style="background:#f59e0b">AG</div>
        <div class="content doc-qa-content" style="border-left: 4px solid #d97706;">
            <span class="doc-qa-title">Agent Q&A</span>
            <div class="doc-qa-subtitle">Ask follow-up questions about this transcript in the same agent style.</div>
            <div class="doc-qa-messages" id="transcriptQaMessages"></div>
            <div class="doc-qa-input-row">
                <textarea id="transcriptQaInput" placeholder="Ask a question about this transcript..." rows="2"></textarea>
                <button class="doc-qa-send" id="transcriptQaSend" type="button">Ask</button>
            </div>
            <div class="doc-qa-hint">Answers use semantic transcript search + context.</div>
        </div>
    `;
    chat.appendChild(row);

    const input = document.getElementById("transcriptQaInput");
    const sendBtn = document.getElementById("transcriptQaSend");
    if (sendBtn) {
        sendBtn.addEventListener("click", () => askTranscriptQuestion());
    }
    if (input) {
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                askTranscriptQuestion();
            }
        });
    }
    renderTranscriptChatHistory();
}

function renderTranscriptChatHistory() {
    const list = document.getElementById("transcriptQaMessages");
    if (!list) return;
    const items = Array.isArray(currentTranscriptChat) ? currentTranscriptChat : [];
    if (!items.length) {
        list.innerHTML = "<div class='doc-qa-empty'>No questions asked yet.</div>";
        return;
    }

    list.innerHTML = items.map((item, idx) => {
        const role = item.role === "assistant" ? "AI" : "You";
        const body = escapeHTMLText(item.content || "");
        const sources = Array.isArray(item.sources) && item.sources.length
            ? `<div class='doc-qa-sources'>` + item.sources.map((s) => {
                const segIdx = Number(s.segment_index);
                const label = Number.isFinite(segIdx)
                    ? `Seg ${segIdx + 1} · ${formatTime(s.start || 0)}-${formatTime(s.end || 0)}`
                    : "Segment";
                return `<button type='button' class='doc-qa-source-btn' data-seg='${segIdx}'>${label}</button>`;
              }).join("") + `</div>`
            : "";
        const actions = item.role === "assistant"
            ? `
                <div class="doc-qa-actions">
                    <button type="button" class="doc-qa-action-btn" data-action="copy" data-index="${idx}" title="Copy answer">Copy</button>
                    <button type="button" class="doc-qa-action-btn" data-action="translate" data-index="${idx}" title="Translate answer">Translate</button>
                    <button type="button" class="doc-qa-action-btn" data-action="speak" data-index="${idx}" title="Convert answer to speech">Speak</button>
                </div>
            `
            : "";
        return `
            <div class="doc-qa-message ${item.role === "assistant" ? "assistant" : "user"}">
                <div class="doc-qa-role">${role}</div>
                <div class="doc-qa-text">${body}</div>
                ${actions}
                ${sources}
            </div>
        `;
    }).join("");

    const sourceButtons = list.querySelectorAll('.doc-qa-source-btn');
    sourceButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const idx = Number(btn.dataset.seg);
            if (Number.isFinite(idx)) {
                focusTranscriptSegment(idx);
            }
        });
    });

    const actionButtons = list.querySelectorAll('.doc-qa-action-btn');
    actionButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const idx = Number(btn.dataset.index);
            if (!Number.isFinite(idx)) return;
            const action = btn.dataset.action || "";
            if (action === "copy") {
                copyTranscriptAnswerByIndex(idx);
            } else if (action === "translate") {
                translateTranscriptAnswerByIndex(idx);
            } else if (action === "speak") {
                speakTranscriptAnswerByIndex(idx);
            }
        });
    });

    list.scrollTop = list.scrollHeight;
}

function focusTranscriptSegment(segmentIndex) {
    if (!Array.isArray(transcriptRowEls) || !segmentGroupMap) return;
    const groupIndex = segmentGroupMap[segmentIndex];
    if (typeof groupIndex !== "number") return;
    const row = transcriptRowEls[groupIndex];
    if (!row) return;

    transcriptRowEls.forEach((el) => el && el.classList.remove("search-hit"));
    row.classList.add("search-hit");
    row.scrollIntoView({ behavior: "smooth", block: "center" });

    window.setTimeout(() => {
        row.classList.remove("search-hit");
    }, 2000);
}

function copyTranscriptAnswerByIndex(index) {
    const item = Array.isArray(currentTranscriptChat) ? currentTranscriptChat[index] : null;
    if (!item || item.role !== "assistant") return;
    const text = String(item.content || "").trim();
    if (!text) return;
    navigator.clipboard.writeText(text);
    alert("Answer copied to clipboard!");
}

async function translateTranscriptAnswerByIndex(index) {
    const item = Array.isArray(currentTranscriptChat) ? currentTranscriptChat[index] : null;
    if (!item || item.role !== "assistant") return;
    const text = String(item.content || "").trim();
    if (!text) return;

    const targetLang = requestTargetLanguage();
    if (!targetLang) return;

    try {
        const result = await translateViaAPI({
            text,
            target_lang: targetLang
        });
        const translated = String((result && result.text) || "").trim();
        if (!translated) {
            throw new Error("Translation returned empty text");
        }
        currentTranscriptChat[index].content = translated;
        renderTranscriptChatHistory();
    } catch (e) {
        alert(e.message || "Answer translation failed.");
    }
}

async function speakTranscriptAnswerByIndex(index) {
    const item = Array.isArray(currentTranscriptChat) ? currentTranscriptChat[index] : null;
    if (!item || item.role !== "assistant") return;
    const text = String(item.content || "").trim();
    if (!text) return;

    const targetLang = translationTargetSelect ? (translationTargetSelect.value || "").trim() : "";
    try {
        const result = await agentQueryJSON({
            tool: "text_to_speech",
            query: "convert this answer to speech",
            text,
            tts_lang: targetLang || "en"
        });
        await handleAgentResponse(result, { source: "tts" });
    } catch (e) {
        alert(e.message || "Text-to-speech failed.");
    }
}

async function askTranscriptQuestion() {
    const input = document.getElementById("transcriptQaInput");
    const sendBtn = document.getElementById("transcriptQaSend");
    if (!input || !sendBtn) return;
    if (!currentSessionId) {
        alert("Transcript context not ready yet.");
        return;
    }

    const question = (input.value || "").trim();
    if (!question) return;
    input.value = "";

    currentTranscriptChat = Array.isArray(currentTranscriptChat) ? currentTranscriptChat : [];
    currentTranscriptChat.push({ role: "user", content: question });
    renderTranscriptChatHistory();

    sendBtn.disabled = true;
    sendBtn.textContent = "Thinking...";
    try {
        const result = await agentQueryJSON({
            query: question,
            session_id: currentSessionId,
            question,
            top_k: 5
        });
        await handleAgentResponse(result, { source: "transcript_qa" });
    } catch (e) {
        currentTranscriptChat.push({ role: "assistant", content: e.message || "Failed to answer." });
        renderTranscriptChatHistory();
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = "Ask";
    }
}

function renderKeywordSearchBar() {
    const chat = document.getElementById("chat");
    if (!chat) return;
    const existing = document.getElementById("semanticSearchCard");
    if (existing) existing.remove();
    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.id = "semanticSearchCard";
    row.innerHTML = `
        <div class="avatar" style="background:#0f172a">🔎</div>
        <div class="content search-card">
            <div class="search-title">Keyword spotting</div>
            <div class="search-subtitle">Exact match first. If none, semantic meaning search kicks in.</div>
            <div class="search-controls">
                <input class="search-input" id="semanticSearchInput" placeholder="Search exact word or meaning...">
                <button class="search-btn" onclick="runSemanticSearch()">Search</button>
                <button class="search-btn ghost" onclick="clearSemanticSearch()">Clear</button>
            </div>
            <div class="search-count" id="semanticSearchCount"></div>
            <div class="search-results" id="semanticSearchResults"></div>
        </div>
    `;
    chat.appendChild(row);
}

function moveKeywordSearchToEnd() {
    const chat = document.getElementById("chat");
    const card = document.getElementById("semanticSearchCard");
    if (!chat || !card) return;
    chat.appendChild(card);
}

async function runSemanticSearch() {
    const input = document.getElementById("semanticSearchInput");
    const resultsEl = document.getElementById("semanticSearchResults");
    const countEl = document.getElementById("semanticSearchCount");
    if (!input || !resultsEl) return;
    const query = (input.value || "").trim();
    if (!query) return;
    if (!currentSessionId) {
        resultsEl.innerHTML = "<div class='search-empty'>No transcript session loaded.</div>";
        if (countEl) countEl.textContent = "";
        return;
    }
    resultsEl.innerHTML = "Searching...";
    if (countEl) countEl.textContent = "";
    transcriptRowEls.forEach((row) => row && row.classList.remove("search-hit"));
    transcriptRowEls.forEach((row) => row && row.classList.remove("search-dim"));
    const exactMatches = [];
    if (Array.isArray(transcriptData) && transcriptData.length) {
        const needle = query.toLowerCase();
        transcriptData.forEach((seg, idx) => {
            const text = String(seg.text || "");
            if (text.toLowerCase().includes(needle)) {
                exactMatches.push({
                    segment_index: idx,
                    text: text,
                    speaker: seg.speaker || "",
                    start: seg.start || 0,
                    end: seg.end || 0,
                    score: 1.0,
                });
            }
        });
    }
    if (exactMatches.length) {
        const matchedGroups = new Set();
        if (countEl) countEl.textContent = `${exactMatches.length} exact matches`;
        resultsEl.innerHTML = exactMatches.map((hit) => {
            const groupIndex = segmentGroupMap[hit.segment_index];
            if (typeof groupIndex === "number" && transcriptRowEls[groupIndex]) {
                transcriptRowEls[groupIndex].classList.add("search-hit");
                matchedGroups.add(groupIndex);
            }
            const ts = `[${formatTime(hit.start || 0)} - ${formatTime(hit.end || 0)}]`;
            return `
                <div class="search-row">
                    <div class="search-meta">Exact • ${escapeHTMLText(hit.speaker || "Speaker")} • ${ts}</div>
                    <div class="search-text">${escapeHTMLText(hit.text || "")}</div>
                </div>
            `;
        }).join("");
        transcriptRowEls.forEach((row, idx) => {
            if (!matchedGroups.has(idx)) {
                row.classList.add("search-dim");
            }
        });
        return;
    }
    try {
        const result = await agentQueryJSON({
            tool: "search_history",
            query,
            session_id: currentSessionId,
            top_k: 6
        });
        const results = (((result || {}).result) || {}).results || [];
        if (!results.length) {
            resultsEl.innerHTML = "<div class='search-empty'>No matches found.</div>";
            if (countEl) countEl.textContent = "0 matches";
            return;
        }
        if (countEl) countEl.textContent = `${results.length} semantic matches`;
        resultsEl.innerHTML = results.map((hit) => {
            const groupIndex = segmentGroupMap[hit.segment_index];
            if (typeof groupIndex === "number" && transcriptRowEls[groupIndex]) {
                transcriptRowEls[groupIndex].classList.add("search-hit");
            }
            const ts = `[${formatTime(hit.start || 0)} - ${formatTime(hit.end || 0)}]`;
            return `
                <div class="search-row">
                    <div class="search-meta">Semantic • ${escapeHTMLText(hit.speaker || "Speaker")} • ${ts}</div>
                    <div class="search-text">${escapeHTMLText(hit.text || "")}</div>
                </div>
            `;
        }).join("");
    } catch (e) {
        resultsEl.innerHTML = `<div class='search-empty'>${escapeHTMLText(e.message || "Search failed.")}</div>`;
        if (countEl) countEl.textContent = "";
    }
}

function clearSemanticSearch() {
    const input = document.getElementById("semanticSearchInput");
    const resultsEl = document.getElementById("semanticSearchResults");
    const countEl = document.getElementById("semanticSearchCount");
    if (input) input.value = "";
    if (resultsEl) resultsEl.innerHTML = "";
    if (countEl) countEl.textContent = "";
    transcriptRowEls.forEach((row) => row && row.classList.remove("search-hit"));
    transcriptRowEls.forEach((row) => row && row.classList.remove("search-dim"));
}

function renderDocumentResult() {
    document.getElementById("loadingOverlay").style.display = "none";
    setUploadHeroVisible(true);

    const chat = document.getElementById("chat");
    const old = chat.querySelectorAll(".transcription");
    old.forEach((r) => r.remove());

    const row = document.createElement("div");
    row.className = "message-row transcription";

    row.innerHTML = `
        <div class="avatar" style="background:#ef4444">DOC</div>
        <div class="content doc-preview-content" style="border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.10);">
            <div class="doc-layout single">
                <div class="doc-layout-left" id="docLayoutLeft">
                    <div class="doc-info-card">
                        <div class="doc-info-title">Document</div>
                        <div class="doc-info-name">${escapeHTMLText(currentDocumentFilename)}</div>
                    </div>
                    <div class="content doc-summary-card" id="docSummaryCard" style="position:relative;"></div>
                </div>
            </div>
        </div>
    `;

    chat.appendChild(row);
    updateSidebarMiniPreview();
        setSummaryButtonState();
    renderDocumentQAPanel();
    if (currentSummary) {
        const target = document.getElementById("docSummaryCard");
        if (target) {
            renderSummaryCard(currentSummary, target);
        }
    }
    chat.scrollTop = chat.scrollHeight;
}

function renderDocumentQAPanel() {
    const chat = document.getElementById("chat");
    const existing = document.getElementById("documentQaPanel");
    if (existing) existing.remove();

    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.id = "documentQaPanel";
    row.innerHTML = `
        <div class="avatar" style="background:#f59e0b">AG</div>
        <div class="content doc-qa-content" style="border-left: 4px solid #d97706;">
            <span class="doc-qa-title">Agent Q&A</span>
            <div class="doc-qa-subtitle">Ask questions about this document with the same agent workspace theme.</div>
            <div class="doc-qa-messages" id="docQaMessages"></div>
            <div class="doc-qa-chunk-preview hidden" id="docChunkPreview">
                <div class="doc-qa-chunk-header">
                    <div class="doc-qa-chunk-title" id="docChunkTitle">Chunk</div>
                    <button type="button" class="doc-qa-chunk-close" onclick="closeChunkPreview()">Close</button>
                </div>
                <div class="doc-qa-chunk-body" id="docChunkBody"></div>
            </div>
            <div class="doc-qa-input-row">
                <textarea id="docQaInput" placeholder="Ask a question about this document..." rows="2"></textarea>
                <button class="doc-qa-send" id="docQaSend" type="button">Ask</button>
            </div>
            <div class="doc-qa-hint">History-aware answers are enabled for this document.</div>
        </div>
    `;
    const target = document.getElementById("docLayoutLeft");
    if (target) {
        target.appendChild(row);
    } else {
        chat.appendChild(row);
    }

    const input = document.getElementById("docQaInput");
    const sendBtn = document.getElementById("docQaSend");
    if (sendBtn) {
        sendBtn.addEventListener("click", () => askDocumentQuestion());
    }
    if (input) {
        input.addEventListener("keydown", (e) => {
            if (e.key === "Enter" && !e.shiftKey) {
                e.preventDefault();
                askDocumentQuestion();
            }
        });
    }
    renderDocumentChatHistory();
}

function updateSidebarMiniPreview() {
    const panel = document.getElementById("docMiniPanel");
    const frame = document.getElementById("docMiniFrame");
    const name = document.getElementById("docMiniName");
    if (!panel || !frame || !name) return;
    if (currentDocumentType === "pdf" && currentDocumentFilename) {
        frame.src = `/documents/${encodeURIComponent(currentDocumentFilename)}`;
        name.textContent = currentDocumentFilename;
        panel.classList.remove("hidden");
    } else {
        frame.src = "";
        name.textContent = "";
        panel.classList.add("hidden");
    }
}

function openDocumentPreview(mode) {
    if (currentDocumentType !== "pdf" || !currentDocumentFilename) return;
    openPdfPanel(mode === "zoom" ? 1.25 : 1.0);
}

function closeDocumentPreview() {
    const panel = document.getElementById("pdfPanel");
    const frame = document.getElementById("pdfFrameMain");
    const gutter = document.getElementById("pdfGutter");
    if (frame) frame.src = "";
    if (panel) panel.classList.remove("open");
    if (gutter) gutter.classList.remove("active");
    updatePdfZoomIndicator(1.0);
}

let pdfZoom = 1.0;

function openPdfPanel(zoomLevel = 1.0) {
    const panel = document.getElementById("pdfPanel");
    const frame = document.getElementById("pdfFrameMain");
    const gutter = document.getElementById("pdfGutter");
    if (!panel || !frame) return;
    frame.src = `/documents/${encodeURIComponent(currentDocumentFilename)}`;
    panel.style.width = "";
    panel.classList.add("open");
    if (gutter) gutter.classList.add("active");
    pdfZoom = zoomLevel;
    applyPdfZoom();
}

function applyPdfZoom() {
    const frame = document.getElementById("pdfFrameMain");
    if (!frame) return;
    frame.style.transform = `scale(${pdfZoom})`;
    updatePdfZoomIndicator(pdfZoom);
}

function updatePdfZoomIndicator(value) {
    const indicator = document.getElementById("pdfZoomIndicator");
    if (!indicator) return;
    indicator.textContent = `${Math.round(value * 100)}%`;
}

function setPdfZoom(delta) {
    pdfZoom = Math.min(2.0, Math.max(0.6, pdfZoom + delta));
    applyPdfZoom();
}

function resetPdfZoom() {
    pdfZoom = 1.0;
    applyPdfZoom();
}

function renderDocumentChatHistory() {
    const list = document.getElementById("docQaMessages");
    if (!list) return;
    const items = Array.isArray(currentDocumentChat) ? currentDocumentChat : [];
    if (!items.length) {
        list.innerHTML = "<div class='doc-qa-empty'>No questions asked yet.</div>";
        return;
    }
    list.innerHTML = items.map((item, idx) => {
        const role = item.role === "assistant" ? "AI" : "You";
        const body = escapeHTMLText(item.content || "");
        const sources = Array.isArray(item.sources) && item.sources.length
            ? `<div class='doc-qa-sources'>` + item.sources.map((s) => {
                const idx = Number(s.index);
                const label = Number.isFinite(idx) ? `Chunk ${idx + 1}` : "Chunk";
                return `<button type='button' class='doc-qa-source-btn' data-index='${idx}'>${label}</button>`;
              }).join("") + `</div>`
            : "";
        const actions = item.role === "assistant"
            ? `
                <div class="doc-qa-actions">
                    <button type="button" class="doc-qa-action-btn" data-action="copy" data-index="${idx}" title="Copy answer">Copy</button>
                    <button type="button" class="doc-qa-action-btn" data-action="translate" data-index="${idx}" title="Translate answer">Translate</button>
                    <button type="button" class="doc-qa-action-btn" data-action="speak" data-index="${idx}" title="Convert answer to speech">Speak</button>
                </div>
            `
            : "";
        return `
            <div class="doc-qa-message ${item.role === "assistant" ? "assistant" : "user"}">
                <div class="doc-qa-role">${role}</div>
                <div class="doc-qa-text">${body}</div>
                ${actions}
                ${sources}
            </div>
        `;
    }).join("");

    const sourceButtons = list.querySelectorAll('.doc-qa-source-btn');
    sourceButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const idx = Number(btn.dataset.index);
            if (Number.isFinite(idx)) {
                openChunkPreview(idx);
            }
        });
    });

    const actionButtons = list.querySelectorAll('.doc-qa-action-btn');
    actionButtons.forEach((btn) => {
        btn.addEventListener('click', () => {
            const idx = Number(btn.dataset.index);
            if (!Number.isFinite(idx)) return;
            const action = btn.dataset.action || "";
            if (action === "copy") {
                copyDocumentAnswerByIndex(idx);
            } else if (action === "translate") {
                translateDocumentAnswerByIndex(idx);
            } else if (action === "speak") {
                speakDocumentAnswerByIndex(idx);
            }
        });
    });

    list.scrollTop = list.scrollHeight;
}


function closeChunkPreview() {
    const panel = document.getElementById("docChunkPreview");
    const body = document.getElementById("docChunkBody");
    const title = document.getElementById("docChunkTitle");
    if (body) body.textContent = "";
    if (title) title.textContent = "";
    if (panel) panel.classList.add("hidden");
}

async function openChunkPreview(chunkIndex) {
    if (!currentDocumentId) return;
    const panel = document.getElementById("docChunkPreview");
    const body = document.getElementById("docChunkBody");
    const title = document.getElementById("docChunkTitle");
    if (!panel || !body || !title) return;

    title.textContent = `Chunk ${Number(chunkIndex) + 1}`;
    body.textContent = "Loading chunk...";
    panel.classList.remove("hidden");

    try {
        const response = await fetch(`/api/documents/${encodeURIComponent(currentDocumentId)}/chunks/${chunkIndex}`);
        const result = await response.json();
        if (!response.ok) {
            throw new Error(result.error || "Failed to load chunk");
        }
        body.textContent = result.text || "";
    } catch (e) {
        body.textContent = e.message || "Failed to load chunk";
    }
}


function copyDocumentAnswerByIndex(index) {
    const item = Array.isArray(currentDocumentChat) ? currentDocumentChat[index] : null;
    if (!item || item.role !== "assistant") return;
    const text = String(item.content || "").trim();
    if (!text) return;
    navigator.clipboard.writeText(text);
    alert("Answer copied to clipboard!");
}

async function translateDocumentAnswerByIndex(index) {
    const item = Array.isArray(currentDocumentChat) ? currentDocumentChat[index] : null;
    if (!item || item.role !== "assistant") return;
    const text = String(item.content || "").trim();
    if (!text) return;

    const targetLang = requestTargetLanguage();
    if (!targetLang) return;

    try {
        const result = await translateViaAPI({
            text,
            target_lang: targetLang
        });
        const translated = String((result && result.text) || "").trim();
        if (!translated) {
            throw new Error("Translation returned empty text");
        }
        currentDocumentChat[index].content = translated;
        renderDocumentChatHistory();
    } catch (e) {
        alert(e.message || "Answer translation failed.");
    }
}

async function speakDocumentAnswerByIndex(index) {
    const item = Array.isArray(currentDocumentChat) ? currentDocumentChat[index] : null;
    if (!item || item.role !== "assistant") return;
    const text = String(item.content || "").trim();
    if (!text) return;

    const targetLang = translationTargetSelect ? (translationTargetSelect.value || "").trim() : "";
    try {
        const result = await agentQueryJSON({
            tool: "text_to_speech",
            query: "convert this answer to speech",
            text,
            tts_lang: targetLang || "en"
        });
        await handleAgentResponse(result, { source: "tts" });
    } catch (e) {
        alert(e.message || "Text-to-speech failed.");
    }
}

async function askDocumentQuestion() {
    const input = document.getElementById("docQaInput");
    const sendBtn = document.getElementById("docQaSend");
    if (!input || !sendBtn) return;
    if (!currentDocumentId) {
        alert("Document context not ready yet.");
        return;
    }
    const question = (input.value || "").trim();
    if (!question) return;
    input.value = "";

    currentDocumentChat = Array.isArray(currentDocumentChat) ? currentDocumentChat : [];
    currentDocumentChat.push({ role: "user", content: question });
    renderDocumentChatHistory();

    sendBtn.disabled = true;
    sendBtn.textContent = "Thinking...";
    try {
        const result = await agentQueryJSON({
            query: question,
            document_id: currentDocumentId,
            question,
            top_k: 5
        });
        await handleAgentResponse(result, { source: "document_qa" });
    } catch (e) {
        currentDocumentChat.push({ role: "assistant", content: e.message || "Failed to answer." });
        renderDocumentChatHistory();
    } finally {
        sendBtn.disabled = false;
        sendBtn.textContent = "Ask";
    }
}

function setupRenameSidebar() {
    const box = document.getElementById("renameBox");
    box.innerHTML = "";
    
    const sortedSpeakers = Array.from(uniqueSpeakers).sort((a, b) => {
        return speakerOrderMap[a] - speakerOrderMap[b];
    });
    
    sortedSpeakers.forEach(s => {
        const div = document.createElement("div");
        div.className = "rename-item";
        const speakerNo = (speakerOrderMap[s] ?? 0) + 1;
        div.innerHTML = `<input type="text" id="name_${s}" placeholder="Enter speaker ${speakerNo} Name">`;
        box.appendChild(div);
    });
    updateTranscriptDependentUI();
}

async function applyNames() {
    let nameMap = {};
    let newSpeakerColors = {};
    uniqueSpeakers.forEach(s => {
        const inputEl = document.getElementById("name_" + s);
        const val = inputEl ? inputEl.value.trim() : "";
        const final = val || s;
        nameMap[s] = final;
        newSpeakerColors[final] = speakerColors[s];
        speakerNameMap[s] = final;
    });
    transcriptData = transcriptData.map(seg => ({ ...seg, speaker: nameMap[seg.speaker] }));
    sourceTranscriptData = sourceTranscriptData.map(seg => ({ ...seg, speaker: nameMap[seg.speaker] }));
    speakerColors = newSpeakerColors;
    uniqueSpeakers = new Set(Object.values(nameMap));

    currentSummary = "";
    setSourceSummary("");
    await applySelectedLanguageToAllTexts(false);
    await persistSessionTranscript();
}

async function persistSessionTranscript() {
    if (!currentSessionId) return;
    try {
        await fetch(`/history/${encodeURIComponent(currentSessionId)}/transcript`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                transcript: transcriptData,
                summary: currentSummary || ""
            })
        });
        await refreshHistory();
    } catch (e) {
        // Non-blocking persistence; UI should continue even if save fails.
    }
}

function buildExportTranscriptText() {
    if (!transcriptData || transcriptData.length === 0) return "";

    const uniqueSpeakerList = [...new Set(transcriptData.map(seg => seg.speaker.toUpperCase()))];
    let content = `ATTENDEES:\n`;
    uniqueSpeakerList.forEach(speaker => {
        content += `- ${speaker}\n`;
    });
    content += `\nTRANSCRIPTS:\n`;

    const groupedTranscript = [];
    let currentGroup = null;
    transcriptData.forEach(seg => {
        if (currentGroup && currentGroup.speaker === seg.speaker) {
            currentGroup.texts.push(seg.text);
            currentGroup.end = seg.end;
        } else {
            if (currentGroup) groupedTranscript.push(currentGroup);
            currentGroup = { speaker: seg.speaker, texts: [seg.text], start: seg.start, end: seg.end };
        }
    });
    if (currentGroup) groupedTranscript.push(currentGroup);

    groupedTranscript.forEach(group => {
        content += `\n${group.speaker.toUpperCase()} [${formatTime(group.start || 0)} - ${formatTime(group.end || 0)}]:\n`;
        group.texts.forEach(text => {
            content += `- ${text}\n`;
        });
    });

    return content;
}


function deleteTranscriptByIndex(index) {
    const group = groupedTranscriptCache[index];
    if (!group || !Array.isArray(group.segmentIndices)) return;
    const ok = window.confirm("Delete this transcript block?");
    if (!ok) return;

    const toRemove = new Set(group.segmentIndices);
    transcriptData = transcriptData.filter((_, idx) => !toRemove.has(idx));
    sourceTranscriptData = sourceTranscriptData.filter((_, idx) => !toRemove.has(idx));

    renderChatDelayed();
    persistSessionTranscript();
}

function copyTranscriptByIndex(index) {
    const group = groupedTranscriptCache[index];
    if (!group) return;

    let text = `${group.speaker.toUpperCase()} [${formatTime(group.start || 0)} - ${formatTime(group.end || 0)}]:\n`;
    group.texts.forEach(line => {
        text += `- ${line}\n`;
    });

    navigator.clipboard.writeText(text.trim());
    alert("Transcript copied to clipboard!");
}

function requestTargetLanguage() {
    if (!translationTargetSelect) return null;
    const selected = (translationTargetSelect.value || "").trim();
    if (!selected) {
        alert("Select a translation language from the dropdown in the top bar.");
        return null;
    }
    return selected;
}

async function translateViaAPI(payload) {
    const result = await agentQueryJSON({
        tool: "translate_text",
        query: "translate this content",
        ...payload
    });
    return (result && result.result) || {};
}

async function applySelectedLanguageToAllTexts(showErrors = true) {
    const token = ++translationRequestToken;
    const targetLang = translationTargetSelect ? (translationTargetSelect.value || "").trim() : "";

    if (!targetLang) {
        transcriptData = cloneTranscriptSegments(sourceTranscriptData);
        currentSummary = sourceSummary || "";
        currentDocumentText = sourceDocumentText || "";
        await renderCurrentContent();
        return;
    }

    const hasTranscript = Array.isArray(sourceTranscriptData) && sourceTranscriptData.length > 0;
    const hasSummary = Boolean((sourceSummary || "").trim());
    const hasDocument = Boolean((sourceDocumentText || "").trim());
    if (!hasTranscript && !hasSummary && !hasDocument) {
        return;
    }

    document.getElementById("loadingOverlay").style.display = "flex";
    try {
        const jobs = [];
        if (hasTranscript) {
            jobs.push(
                translateViaAPI({
                    texts: sourceTranscriptData.map((seg) => seg.text || ""),
                    target_lang: targetLang
                }).then((result) => ({ type: "transcript", result }))
            );
        }
        if (hasSummary) {
            jobs.push(
                translateViaAPI({
                    text: sourceSummary,
                    target_lang: targetLang
                }).then((result) => ({ type: "summary", result }))
            );
        }
        if (hasDocument) {
            jobs.push(
                translateViaAPI({
                    text: sourceDocumentText,
                    target_lang: targetLang
                }).then((result) => ({ type: "document", result }))
            );
        }

        const translated = await Promise.all(jobs);
        if (token !== translationRequestToken) return;

        transcriptData = cloneTranscriptSegments(sourceTranscriptData);
        currentSummary = sourceSummary || "";
        currentDocumentText = sourceDocumentText || "";

        translated.forEach((item) => {
            if (item.type === "transcript") {
                const texts = Array.isArray(item.result && item.result.texts) ? item.result.texts : [];
                transcriptData = transcriptData.map((seg, idx) => ({
                    ...seg,
                    text: String(texts[idx] || seg.text || "")
                }));
            } else if (item.type === "summary") {
                currentSummary = String((item.result && item.result.text) || currentSummary || "");
            } else if (item.type === "document") {
                currentDocumentText = String((item.result && item.result.text) || currentDocumentText || "");
            }
        });

        await renderCurrentContent();
    } catch (e) {
        if (token !== translationRequestToken) return;
        if (showErrors) {
            alert(e.message || "Automatic translation failed.");
        }
    } finally {
        if (token === translationRequestToken) {
            document.getElementById("loadingOverlay").style.display = "none";
        }
    }
}

async function translateTranscriptByIndex(index) {
    const group = groupedTranscriptCache[index];
    if (!group || !Array.isArray(group.texts) || group.texts.length === 0) return;

    const targetLang = requestTargetLanguage();
    if (!targetLang) return;

    try {
        const result = await translateViaAPI({
            texts: group.texts,
            target_lang: targetLang
        });
        const translated = Array.isArray(result.texts) ? result.texts : [];
        if (translated.length !== group.texts.length) {
            throw new Error("Incomplete translation response");
        }

        group.segmentIndices.forEach((segIdx, i) => {
            if (typeof segIdx === "number" && transcriptData[segIdx]) {
                transcriptData[segIdx].text = translated[i] || transcriptData[segIdx].text;
            }
            if (typeof segIdx === "number" && sourceTranscriptData[segIdx]) {
                sourceTranscriptData[segIdx].text = translated[i] || sourceTranscriptData[segIdx].text;
            }
        });

        await renderChatDelayed();
        await persistSessionTranscript();
    } catch (e) {
        alert(e.message || "Transcript translation failed.");
    }
}

function getResolvedSummaryText(summaryText) {
    let updatedSummary = summaryText || "";
    const sortedEntries = Object.entries(speakerNameMap).sort((a, b) => b[0].length - a[0].length);
    for (let [originalSpeaker, finalizedName] of sortedEntries) {
        const regex = new RegExp('\\b' + originalSpeaker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'gi');
        updatedSummary = updatedSummary.replace(regex, finalizedName);
    }
    return updatedSummary;
}

function escapeHTMLText(text) {
    return (text || "")
        .replace(/&/g, "&amp;")
        .replace(/</g, "&lt;")
        .replace(/>/g, "&gt;");
}

function isSummaryHeadingLine(line) {
    const value = (line || "").trim();
    const headingPatterns = [
        /^MINUTES OF A MEETING$/i,
        /^TITLE\s*:.*/i,
        /^DATE\s*:.*/i,
        /^PLACE\s*:.*/i,
        /^INTRODUCTION$/i,
        /^ATTENDEES$/i,
        /^SUMMARY OF THE MEETING$/i,
        /^KEY ASPECTS DISCUSSED\s*:?$/i,
        /^ACTION ITEMS AND ASSIGNED TO\s*:?$/i,
        /^DEADLINES FOR THE TASKS\s*:?$/i,
        /^THANK YOU$/i
    ];
    return headingPatterns.some((pattern) => pattern.test(value));
}

function renderSummaryCard(summaryText, targetCard = null) {
    const updatedSummary = getResolvedSummaryText(summaryText);

    const formattedLines = updatedSummary
        .split("\n")
        .map((line) => {
            const normalized = line.replace(/^\* /, "• ").replace(/^- /, "• ");
            const safe = escapeHTMLText(normalized);
            return isSummaryHeadingLine(normalized) ? `<b>${safe}</b>` : safe;
        });
    const formatted = formattedLines.join("<br>");

    const summaryMarkup = `
        <div class="translate-sum-icon" onclick="speakSummary()" title="Generate Speech" style="right:118px;">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5"></polygon>
                <path d="M15.5 8.5a5 5 0 0 1 0 7"></path>
                <path d="M19 5a10 10 0 0 1 0 14"></path>
            </svg>
        </div>
        <div class="translate-sum-icon" onclick="translateSummary()" title="Translate Summary">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M5 8h14"></path>
                <path d="M5 12h8"></path>
                <path d="M13 19l4-8 4 8"></path>
                <path d="M14.5 16h5"></path>
            </svg>
        </div>
        <div class="copy-sum-icon" onclick="copySummary()" title="Copy Summary"
            style="position:absolute;top:10px;right:10px;cursor:pointer;opacity:0.7">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        </div>
        <div class="delete-sum-icon" onclick="deleteSummary()" title="Delete Summary"
            style="position:absolute;top:10px;right:64px;cursor:pointer;opacity:0.7">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <path d="M3 6h18"></path>
                <path d="M8 6V4h8v2"></path>
                <path d="M19 6l-1 14H6L5 6"></path>
                <path d="M10 11v6"></path>
                <path d="M14 11v6"></path>
            </svg>
        </div>
        ${formatted}
    `;

    const chat = document.getElementById("chat");
    if (targetCard) {
        targetCard.id = "sumCard";
        targetCard.innerHTML = summaryMarkup;
        moveKeywordSearchToEnd();
        chat.scrollTop = chat.scrollHeight;
        return;
    }

    if (currentDocumentFilename) {
        const docTarget = document.getElementById("docSummaryCard");
        if (docTarget) {
            docTarget.id = "sumCard";
            docTarget.innerHTML = summaryMarkup;
        }
        return;
    }

    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:#10b981">Σ</div>
        <div class="content summary-card" id="sumCard" style="position:relative;">
            ${summaryMarkup}
        </div>
    `;
    chat.appendChild(row);
    moveKeywordSearchToEnd();
    chat.scrollTop = chat.scrollHeight;
}

function renderGreetingCard(name) {
    const chat = document.getElementById("chat");
    if (!chat) return;
    if (chat.querySelector(".agent-greeting-row")) return;
    const row = document.createElement("div");
    row.className = "message-row transcription agent-greeting-row";
    row.innerHTML = `
        <div class="avatar" style="background: #f59e0b">AI</div>
        <div class="content">Hello ${escapeHTMLText(name || "there")}, welcome to the AI Knowledge Studio 🧠✨.</div>
    `;
    chat.insertBefore(row, chat.firstChild);
}

async function fetchAdminUsers() {
    const response = await fetch("/api/users");
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to load users");
    }
    return result.users || [];
}

async function fetchDepartments() {
    const response = await fetch("/api/departments");
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to load departments");
    }
    return result.departments || [];
}

async function createDepartment(name) {
    const response = await fetch("/api/departments", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to create department");
    }
    return result;
}

async function deleteDepartment(departmentId) {
    const response = await fetch(`/api/departments/${encodeURIComponent(departmentId)}`, {
        method: "DELETE",
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to delete department");
    }
}

async function updateDepartment(departmentId, name) {
    const response = await fetch(`/api/departments/${encodeURIComponent(departmentId)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ name }),
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to update department");
    }
}
async function fetchAuditEvents() {
    const response = await fetch("/api/audit");
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to load audit log");
    }
    return result.events || [];
}

async function deleteAuditEvent(eventId) {
    const response = await fetch(`/api/audit/${encodeURIComponent(eventId)}`, {
        method: "DELETE",
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to delete audit");
    }
}

async function clearAllAudits() {
    const response = await fetch("/api/audit", {
        method: "DELETE",
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to clear audits");
    }
}

async function createAdminUser(payload) {
    const response = await fetch("/api/users", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to create user");
    }
    return result;
}

async function updateAdminUser(username, payload) {
    const response = await fetch(`/api/users/${encodeURIComponent(username)}`, {
        method: "PATCH",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to update user");
    }
    return result;
}

async function deleteAdminUser(username) {
    const response = await fetch(`/api/users/${encodeURIComponent(username)}`, {
        method: "DELETE",
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Failed to delete user");
    }
}

function formatAuditEvent(ev) {
    const ts = new Date(ev.created_at || ev.ts || "").toLocaleString("en-GB");
    const actor = ev.email || ev.username || "";
    const action = ev.action || "event";
    return { ts, actor, action };
}

async function showAdminPanel() {
    const chat = document.getElementById("chat");
    if (!chat) return;
    if (!currentUser) {
        await loadPolicy();
    }
    clearChatRows();
    setUploadHeroVisible(false);
    const row = document.createElement("div");
    row.className = "message-row transcription";
    const roleOptions = [];
    if (currentUser && currentUser.role_name === "super_admin") {
        roleOptions.push("<option value='admin'>admin</option>");
    }
    roleOptions.push("<option value='user'>user</option>");
    const deptDefault = currentUser && currentUser.department ? currentUser.department : "general";
    row.innerHTML = `
        <div class="avatar" style="background:#111827">ADM</div>
        <div class="content admin-panel" id="adminPanel">
            <div class="admin-card">
                <h3>User Management</h3>
                <div class="admin-grid">
                    <input class="admin-input" id="adminCreateName" placeholder="Full Name">
                    <input class="admin-input" id="adminCreateEmail" placeholder="Email">
                    <input class="admin-input" id="adminCreatePassword" placeholder="Password">
                    <select class="admin-input" id="adminCreateRole">${roleOptions.join("")}</select>
                    <select class="admin-input" id="adminCreateDept">
                        <option value="${deptDefault}">${deptDefault}</option>
                    </select>
                </div>
                <div class="admin-actions">
                    <button class="admin-btn" onclick="handleCreateUser()">Create User</button>
                </div>
                <div class="admin-note">Admins can create users in their department. Superusers can create admins and users for any department.</div>
            </div>
            <div class="admin-card" id="departmentPanel" style="display:none;">
                <h3>Departments</h3>
                <div class="admin-grid">
                    <input class="admin-input" id="adminCreateDepartment" placeholder="New Department">
                </div>
                <div class="admin-actions">
                    <button class="admin-btn" onclick="handleCreateDepartment()">Add Department</button>
                </div>
                <div class="admin-table" id="adminDepartmentsTable"></div>
            </div>
            <div class="admin-card">
                <h3>Users</h3>
                <div class="admin-table" id="adminUsersTable"></div>
            </div>
            <div class="admin-card">
                <h3>Audit Log</h3>
                <div class="admin-actions" id="auditActions" style="margin-bottom:8px;"></div>
                <div class="admin-grid" style="margin-bottom:10px;">
                    <input class="admin-input" id="auditFilterDate" type="date">
                    <select class="admin-input" id="auditFilterDept">
                        <option value="">All Departments</option>
                    </select>
                    <input class="admin-input" id="auditFilterEmail" placeholder="Filter by Email">
                    <button class="admin-btn" onclick="refreshAdminPanel()">Apply Filters</button>
                    <button class="admin-btn" onclick="clearAuditFilters()">Clear Filters</button>
                </div>
                <div class="admin-table" id="adminAuditTable"></div>
            </div>
        </div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    await refreshAdminPanel();
}

async function refreshAdminPanel() {
    const usersTable = document.getElementById("adminUsersTable");
    const auditTable = document.getElementById("adminAuditTable");
    const departmentsTable = document.getElementById("adminDepartmentsTable");
    if (!usersTable || !auditTable) return;

    usersTable.innerHTML = "Loading...";
    auditTable.innerHTML = "Loading...";
    let departments = [];
    try {
        let events = await fetchAuditEvents();
        const [users, fetchedDepartments] = await Promise.all([fetchAdminUsers(), fetchDepartments()]);
        departments = fetchedDepartments || [];
        usersTable.innerHTML = users.length ? "" : "<div class='admin-note'>No users found.</div>";
        const deptSelect = document.getElementById("adminCreateDept");
        if (deptSelect) {
            const depOptions = (departments && departments.length ? departments : [{ name: "general" }])
                .map((d) => `<option value="${escapeHTMLText(d.name)}">${escapeHTMLText(d.name)}</option>`)
                .join("");
            deptSelect.innerHTML = depOptions;
            if (currentUser && currentUser.role_name !== "super_admin") {
                deptSelect.value = currentUser.department || "general";
                deptSelect.disabled = true;
            } else {
                deptSelect.disabled = false;
            }
        }
        const auditDeptSelect = document.getElementById("auditFilterDept");
        const isSuper = currentUser && currentUser.role_name === "super_admin";
        if (auditDeptSelect) {
            const depOptions = [`<option value="">All Departments</option>`]
                .concat((departments && departments.length ? departments : [{ name: "general" }])
                    .map((d) => `<option value="${escapeHTMLText(d.name)}">${escapeHTMLText(d.name)}</option>`))
                .join("");
            auditDeptSelect.innerHTML = depOptions;
            if (!isSuper) {
                auditDeptSelect.value = currentUser && currentUser.department ? currentUser.department : "";
                auditDeptSelect.disabled = true;
                auditDeptSelect.style.display = "none";
            } else {
                auditDeptSelect.disabled = false;
                auditDeptSelect.style.display = "";
            }
        }
        const departmentPanel = document.getElementById("departmentPanel");
        if (departmentPanel) {
            departmentPanel.style.display = "block";
            const deptInput = document.getElementById("adminCreateDepartment");
            const deptBtn = departmentPanel.querySelector(".admin-actions .admin-btn");
            if (!isSuper) {
                departmentPanel.style.display = "none";
                if (deptInput) deptInput.disabled = true;
                if (deptBtn) deptBtn.disabled = true;
            } else if (departmentsTable) {
                departmentsTable.innerHTML = departments.length ? "" : "<div class='admin-note'>No departments yet.</div>";
                departments.forEach((d) => {
                    const row = document.createElement("div");
                    row.className = "admin-row";
                    row.innerHTML = `
                        <div class="title">${escapeHTMLText(d.name)}</div>
                        <div class="admin-actions">
                            <button class="admin-btn" onclick="handleRenameDepartment('${d.id}', '${escapeHTMLText(d.name)}')">Rename</button>
                            <button class="admin-btn" onclick="handleDeleteDepartment('${d.id}', '${escapeHTMLText(d.name)}')">Delete</button>
                        </div>
                    `;
                    departmentsTable.appendChild(row);
                });
            }
        }
        users.forEach((u) => {
            const row = document.createElement("div");
            row.className = "admin-row";
            const roleSelect = currentUser && currentUser.role_name === "super_admin"
                ? `<select class="admin-input" onchange="handleChangeRole('${u.id}', this.value)">
                        <option value="admin" ${u.role_name === "admin" ? "selected" : ""}>admin</option>
                        <option value="user" ${u.role_name === "user" ? "selected" : ""}>user</option>
                   </select>`
                : `<span class="meta">${escapeHTMLText(u.role_name)}</span>`;
            const deptSelect = currentUser && currentUser.role_name === "super_admin"
                ? `<select class="admin-input" onchange="handleChangeDept('${u.id}', this.value)">
                        ${departments.map((d) => `<option value="${escapeHTMLText(d.name)}" ${u.department === d.name ? "selected" : ""}>${escapeHTMLText(d.name)}</option>`).join("")}
                   </select>`
                : `<span class="meta">${escapeHTMLText(u.department || "general")}</span>`;
            row.innerHTML = `
                <div>
                    <div class="title">${escapeHTMLText(u.name || "")} • ${escapeHTMLText(u.role_name)}</div>
                    <div class="meta">${escapeHTMLText(u.email)} • Dept: ${escapeHTMLText(u.department || "general")}</div>
                </div>
                <div class="admin-actions">
                    <button class="admin-btn" onclick="handleResetPassword('${u.id}')">Reset Password</button>
                    ${roleSelect}
                    ${deptSelect}
                    <button class="admin-btn" onclick="handleDeleteUser('${u.id}')">Delete</button>
                </div>
            `;
            usersTable.appendChild(row);
        });

        const auditActions = document.getElementById("auditActions");
        if (auditActions) {
            auditActions.innerHTML = "";
            if (currentUser && currentUser.role_name === "super_admin") {
                const btn = document.createElement("button");
                btn.className = "admin-btn";
                btn.textContent = "Clear All Audits";
                btn.onclick = async () => {
                    const ok = window.confirm("Clear all audits?");
                    if (!ok) return;
                    await clearAllAudits();
                    await refreshAdminPanel();
                };
                auditActions.appendChild(btn);
            }
        }

        const dateFilter = document.getElementById("auditFilterDate");
        const deptFilter = document.getElementById("auditFilterDept");
        const emailFilter = document.getElementById("auditFilterEmail");
        if (dateFilter && dateFilter.value) {
            events = events.filter((ev) => (ev.created_at || "").startsWith(dateFilter.value));
        }
        if (deptFilter && deptFilter.value && isSuper) {
            events = events.filter((ev) => (ev.department || "").toLowerCase() === deptFilter.value.toLowerCase());
        }
        if (emailFilter && emailFilter.value) {
            const needle = emailFilter.value.toLowerCase();
            events = events.filter((ev) => (ev.email || "").toLowerCase().includes(needle));
        }

        auditTable.innerHTML = events.length ? "" : "<div class='admin-note'>No events yet.</div>";
        events.forEach((ev) => {
            const row = document.createElement("div");
            row.className = "admin-row";
            const info = formatAuditEvent(ev);
            const loginTime = info.action === "auth:login" ? info.ts : "";
            const logoutTime = info.action === "auth:logout" ? info.ts : "";
            const deptText = ev.department || "-";
            const historyText = `Department: ${deptText} • Login: ${loginTime || "-"} • Logout: ${logoutTime || "-"} • History: ${info.action}`;
            row.innerHTML = `
                <div>
                    <div class="title">${escapeHTMLText(info.actor || "unknown")}</div>
                    <div class="meta">${escapeHTMLText(historyText)}</div>
                </div>
                <div class="admin-actions">
                    <button class="admin-btn" onclick="handleDeleteAudit('${ev.id}')">Delete</button>
                </div>
            `;
            auditTable.appendChild(row);
        });
    } catch (e) {
        usersTable.innerHTML = `<div class='admin-note'>${escapeHTMLText(e.message || "Failed to load users.")}</div>`;
        auditTable.innerHTML = `<div class='admin-note'>${escapeHTMLText(e.message || "Failed to load audit.")}</div>`;
    }
}

async function handleCreateUser() {
    const name = (document.getElementById("adminCreateName").value || "").trim();
    const email = (document.getElementById("adminCreateEmail").value || "").trim();
    const password = (document.getElementById("adminCreatePassword").value || "").trim();
    const role = (document.getElementById("adminCreateRole").value || "").trim() || "user";
    const department = (document.getElementById("adminCreateDept").value || "").trim() || "general";
    if (!name || !email || !password) return;
    await createAdminUser({ name, email, password, role, department });
    await refreshAdminPanel();
}

async function handleResetPassword(userId) {
    const password = window.prompt("Enter new password:");
    if (!password) return;
    await updateAdminUser(userId, { password });
    await refreshAdminPanel();
}

async function handleChangeRole(userId, role) {
    const value = (role || "").trim();
    if (!value) return;
    await updateAdminUser(userId, { role: value });
    await refreshAdminPanel();
}

async function handleChangeDept(userId, department) {
    const value = (department || "").trim();
    if (!value) return;
    await updateAdminUser(userId, { department: value });
    await refreshAdminPanel();
}

async function handleDeleteUser(userId) {
    const ok = window.confirm(`Delete user?`);
    if (!ok) return;
    await deleteAdminUser(userId);
    await refreshAdminPanel();
}

async function handleDeleteAudit(eventId) {
    const ok = window.confirm("Delete this audit entry?");
    if (!ok) return;
    await deleteAuditEvent(eventId);
    await refreshAdminPanel();
}

function clearAuditFilters() {
    const dateFilter = document.getElementById("auditFilterDate");
    const deptFilter = document.getElementById("auditFilterDept");
    const emailFilter = document.getElementById("auditFilterEmail");
    if (dateFilter) dateFilter.value = "";
    if (deptFilter) deptFilter.value = "";
    if (emailFilter) emailFilter.value = "";
    refreshAdminPanel();
}

async function handleCreateDepartment() {
    const name = (document.getElementById("adminCreateDepartment").value || "").trim();
    if (!name) return;
    await createDepartment(name);
    document.getElementById("adminCreateDepartment").value = "";
    await refreshAdminPanel();
}

async function handleDeleteDepartment(departmentId, name) {
    const ok = window.confirm(`Delete department '${name}'? This will delete all users in it.`);
    if (!ok) return;
    await deleteDepartment(departmentId);
    await refreshAdminPanel();
}

async function handleRenameDepartment(departmentId, name) {
    const next = window.prompt("Rename department:", name || "");
    if (!next) return;
    await updateDepartment(departmentId, next);
    await refreshAdminPanel();
}

function renderSummaryLoadingCard() {
    const chat = document.getElementById("chat");
    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:#10b981">Σ</div>
        <div class="content summary-card" id="summaryPendingCard" style="position:relative;">
            <div class="summary-loading">
                <span class="summary-spinner" aria-hidden="true"></span>
                <span class="summary-loading-text">Generating Summary...</span>
            </div>
        </div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
    return document.getElementById("summaryPendingCard");
}

function requestMeetingDetails() {
    const meetingTitle = window.prompt("Enter Meeting Title:");
    if (meetingTitle === null || !meetingTitle.trim()) return null;

    const meetingDate = window.prompt("Enter Meeting Date:");
    if (meetingDate === null || !meetingDate.trim()) return null;

    const meetingPlace = window.prompt("Enter Meeting Place:");
    if (meetingPlace === null || !meetingPlace.trim()) return null;

    return {
        meeting_title: meetingTitle.trim(),
        meeting_date: meetingDate.trim(),
        meeting_place: meetingPlace.trim()
    };
}

async function showSummary() {
    if (!transcriptData || transcriptData.length === 0) return;
    if (isSummaryLoading) return;
    if (Boolean((currentSummary || "").trim())) return;

    if (currentSummary) {
        renderSummaryCard(currentSummary);
        setSummaryButtonState();
        return;
    }

    const meetingDetails = requestMeetingDetails();
    if (!meetingDetails) return;

    isSummaryLoading = true;
    const pendingCard = renderSummaryLoadingCard();

    try {
        const content = buildExportTranscriptText();
        const selectedTargetLang = translationTargetSelect ? (translationTargetSelect.value || "").trim() : "";
        const result = await agentQueryJSON({
            query: selectedTargetLang ? "summarize this meeting and translate it" : "summarize this meeting",
            content: content,
            session_id: currentSessionId,
            meeting_title: meetingDetails.meeting_title,
            meeting_date: meetingDetails.meeting_date,
            meeting_place: meetingDetails.meeting_place,
            target_lang: selectedTargetLang || undefined
        });

        const freshSummary = ((((result || {}).result) || {}).summary) || "";
        setSourceSummary(freshSummary);
        currentSummary = ((((result || {}).result) || {}).translated_summary) || freshSummary;
        renderSummaryCard(currentSummary, pendingCard);
        setSummaryButtonState();
        await refreshHistory();
    } catch (e) {
        if (pendingCard) {
            pendingCard.textContent = "Summary generation failed.";
            pendingCard.removeAttribute("id");
        }
    } finally {
        isSummaryLoading = false;
    }
}

loadPolicy().then(() => {
    refreshHistory();
    refreshDocumentHistory();
    updateAgentWorkspaceUI();
});
initTranslationLanguageDropdown();
initDropZone();
updateTranscriptDependentUI();
setUploadHeroVisible(true);
updateAgentWorkspaceUI();
if (agentQueryInput) {
    agentQueryInput.addEventListener("keydown", (e) => {
        if (e.key === "Enter" && !e.shiftKey) {
            e.preventDefault();
            submitAgentQuery();
        }
    });
}


function deleteSummary() {
    const ok = window.confirm("Delete the summary?");
    if (!ok) return;
    currentSummary = "";
    setSourceSummary("");
    renderCurrentContent();
    setSummaryButtonState();
    persistSessionTranscript();
}

function copySummary() {
    const text = document.getElementById("sumCard").innerText;
    navigator.clipboard.writeText(text);
    alert("Summary copied to clipboard!");
}

async function translateSummary() {
    const baseSummary = sourceSummary || currentSummary;
    if (!baseSummary || !baseSummary.trim()) return;

    const targetLang = requestTargetLanguage();
    if (!targetLang) return;

    try {
        const result = await translateViaAPI({
            text: baseSummary,
            target_lang: targetLang
        });
        currentSummary = result.text || baseSummary;
        const sumCard = document.getElementById("sumCard");
        if (sumCard) {
            renderSummaryCard(currentSummary, sumCard);
        } else {
            renderSummaryCard(currentSummary);
        setSummaryButtonState();
        }
        await persistSessionTranscript();
    } catch (e) {
        alert(e.message || "Summary translation failed.");
    }
}

// function exportData(format) {
//     let updatedSummary = currentSummary;
//     const sortedEntries = Object.entries(speakerNameMap).sort((a, b) => b[0].length - a[0].length);
//     for (let [originalSpeaker, finalizedName] of sortedEntries) {
//         const regex = new RegExp('\\b' + originalSpeaker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'gi');
//         updatedSummary = updatedSummary.replace(regex, finalizedName);
//     }
    
//     let content = `CONVERSATION SUMMARY:\n${updatedSummary}\n\nTRANSCRIPT:\n`;
//     transcriptData.forEach(seg => { 
//         content += `[${formatTime(seg.start || 0)} - ${formatTime(seg.end || 0)}] [${seg.speaker}]: ${seg.text}\n\n`; 
//     });
//     const blob = new Blob([content], { type: "text/plain" });
//     const link = document.createElement("a");
//     link.href = URL.createObjectURL(blob);
//     link.download = `${lastAudioFile.split('.')[0] || 'ASR_Export'}.${format === 'word' ? 'docx' : 'pdf'}`;
//     link.click();
// }

function exportTranscript() {
    if (!transcriptData || transcriptData.length === 0) return;

    // Group consecutive lines by speaker
    const groupedTranscript = [];
    let currentGroup = null;

    transcriptData.forEach(seg => {
        if (currentGroup && currentGroup.speaker === seg.speaker) {
            currentGroup.texts.push(seg.text);
            currentGroup.end = seg.end;
        } else {
            if (currentGroup) groupedTranscript.push(currentGroup);
            currentGroup = { speaker: seg.speaker, texts: [seg.text], start: seg.start, end: seg.end };
        }
    });
    if (currentGroup) groupedTranscript.push(currentGroup);

    // Unique speakers for ATTENDEES
    const uniqueSpeakers = [...new Set(transcriptData.map(seg => seg.speaker.toUpperCase()))];

    // Build content
    let content = `<p><b>ATTENDEES:</b></p><ul>`;
    uniqueSpeakers.forEach(speaker => {
        content += `<li>${speaker}</li>`;
    });
    content += `</ul><p><b>TRANSCRIPTS</b></p>`;

    // Build transcript blocks
    groupedTranscript.forEach(group => {
        content += `<p><b>${group.speaker.toUpperCase()}</b></p><ul>`;
        group.texts.forEach(text => {
            content += `<li>${text}</li>`;
        });
        content += `</ul><p><span style="color:red">[${formatTime(group.start || 0)} - ${formatTime(group.end || 0)}]</span></p>`;
    });

    // Create Word-compatible Blob
    const preamble = `
        <html xmlns:o='urn:schemas-microsoft-com:office:office' 
              xmlns:w='urn:schemas-microsoft-com:office:word' 
              xmlns='http://www.w3.org/TR/REC-html40'>
        <head><meta charset='utf-8'><title>ASR Export</title></head>
        <body>${content}</body></html>
    `;

    const blob = new Blob([preamble], { type: "application/msword" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${lastAudioFile.split('.')[0] || 'ASR_Export'}.doc`;
    link.click();
}

function exportSummary() {
    if (!currentSummary || !currentSummary.trim()) {
        alert("Summary is not available. Click Summarize first.");
        return;
    }

    const updatedSummary = getResolvedSummaryText(currentSummary);
    const lines = updatedSummary.split("\n");

    let content = "";
    lines.forEach((line) => {
        const normalized = line.replace(/^\* /, "• ").replace(/^- /, "• ").trim();
        if (!normalized) {
            content += "<p>&nbsp;</p>";
            return;
        }
        const safe = escapeHTMLText(normalized);
        content += isSummaryHeadingLine(normalized) ? `<p><b>${safe}</b></p>` : `<p>${safe}</p>`;
    });

    const preamble = `
        <html xmlns:o='urn:schemas-microsoft-com:office:office'
              xmlns:w='urn:schemas-microsoft-com:office:word'
              xmlns='http://www.w3.org/TR/REC-html40'>
        <head><meta charset='utf-8'><title>Summary Export</title></head>
        <body>${content}</body></html>
    `;

    const blob = new Blob([preamble], { type: "application/msword" });
    const link = document.createElement("a");
    link.href = URL.createObjectURL(blob);
    link.download = `${lastAudioFile.split('.')[0] || 'ASR_Export'}_summary.doc`;
    link.click();
}

async function agentQueryJSON(payload) {
    const response = await fetch("/api/agent/query", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload || {})
    });
    const result = await response.json();
    if (!response.ok) {
        throw new Error(result.error || "Agent request failed");
    }
    return result;
}

function appendChatBubble(role, text) {
    const chat = document.getElementById("chat");
    if (!chat) return;
    const safeRole = role === "user" ? "user" : "assistant";
    const badge = safeRole === "user" ? "YOU" : "AI";
    const row = document.createElement("div");
    row.className = `message-row transcription chat-bubble-row ${safeRole}`;
    row.innerHTML = `
        <div class="avatar chat-bubble-avatar">${badge}</div>
        <div class="content chat-bubble-content">${escapeHTMLText(text || "")}</div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
}

function appendAgentResponseCard(title, text, accent = "#d97706") {
    const chat = document.getElementById("chat");
    if (!chat) return;
    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:${accent}">AG</div>
        <div class="content agent-response-card" style="border-left: 4px solid ${accent}; background: rgba(245, 158, 11, 0.10);">
            <div class="agent-response-title">${escapeHTMLText(title || "Agent Response")}</div>
            <div class="agent-response-text">${escapeHTMLText(text || "")}</div>
        </div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
}

async function handleAgentResponse(agentResult, options = {}) {
    const result = (agentResult && agentResult.result) || {};
    lastAgentPlan = Array.isArray(agentResult && agentResult.plan) ? agentResult.plan : [];

    if (result.answer && (agentResult && agentResult.selected_tool) === "chat_response") {
        appendChatBubble("assistant", result.answer);
        return;
    }

    if (result.answer && options.source === "transcript_qa") {
        currentTranscriptChat = result.history || currentTranscriptChat;
        const sources = result.sources || [];
        if (currentTranscriptChat.length) {
            const lastIdx = currentTranscriptChat.length - 1;
            if (currentTranscriptChat[lastIdx].role === "assistant") {
                currentTranscriptChat[lastIdx].sources = sources;
            }
        }
        renderTranscriptChatHistory();
        return;
    }

    if (result.answer && options.source === "document_qa") {
        currentDocumentChat = result.history || currentDocumentChat;
        const sources = result.sources || [];
        if (currentDocumentChat.length) {
            const lastIdx = currentDocumentChat.length - 1;
            if (currentDocumentChat[lastIdx].role === "assistant") {
                currentDocumentChat[lastIdx].sources = sources;
            }
        }
        renderDocumentChatHistory();
        return;
    }

    if (result.document_id || result.document_filename) {
        transcriptData = [];
        groupedTranscriptCache = [];
        currentSummary = result.summary || "";
        currentProcessedAudio = "";
        currentProcessedVideo = "";
        currentBeforeAudio = "";
        currentAfterAudio = "";
        currentDocumentFilename = result.document_filename || "";
        currentDocumentType = (result.document_type || "").toLowerCase();
        currentDocumentId = result.document_id || "";
        currentDocumentChat = result.chat_history || [];
        currentTranscriptChat = [];
        currentDocumentText = result.document_text || result.text_preview || "";
        setSourceTranscript([]);
        setSourceSummary(currentSummary);
        setSourceDocumentText(currentDocumentText);
        await applySelectedLanguageToAllTexts(false);
        await refreshDocumentHistory();
        setSummaryButtonState();
        if (result.answer && options.source !== "document_qa") {
            appendAgentResponseCard("Document Answer", result.answer, "#0ea5e9");
        }
        return;
    }

    if (Array.isArray(result.transcript) || result.session_id) {
        transcriptData = result.transcript || [];
        currentSummary = result.summary || "";
        currentSessionId = result.session_id || currentSessionId;
        currentDocumentFilename = "";
        currentDocumentType = "";
        currentDocumentText = "";
        currentDocumentId = "";
        currentDocumentChat = [];
        currentTranscriptChat = result.history || result.qa_history || [];
        currentProcessedVideo = result.source_video || "";
        currentBeforeAudio = result.before_audio_file || result.processed_file || "";
        currentAfterAudio = result.after_audio_file || result.processed_file || "";
        if (currentAfterAudio) {
            lastAudioFile = currentAfterAudio;
            currentProcessedAudio = currentAfterAudio;
        }
        setSourceTranscript(transcriptData);
        setSourceSummary(currentSummary);
        setSourceDocumentText("");
        await applySelectedLanguageToAllTexts(false);
        await refreshHistory();
        await refreshDocumentHistory();
        setSummaryButtonState();
        if (result.answer && options.source === "transcript_qa") {
            currentTranscriptChat = result.history || currentTranscriptChat;
            const sources = result.sources || [];
            if (currentTranscriptChat.length) {
                const lastIdx = currentTranscriptChat.length - 1;
                if (currentTranscriptChat[lastIdx].role === "assistant") {
                    currentTranscriptChat[lastIdx].sources = sources;
                }
            }
            renderTranscriptChatHistory();
        } else if (result.answer) {
            appendAgentResponseCard("Transcript Answer", result.answer, "#6366f1");
        }
        return;
    }

    if (result.summary || result.translated_summary) {
        const sourceSummaryText = result.summary || "";
        setSourceSummary(sourceSummaryText);
        currentSummary = result.translated_summary || result.summary || "";
        renderSummaryCard(currentSummary);
        setSummaryButtonState();
        return;
    }

    if (result.text || (Array.isArray(result.texts) && result.texts.length)) {
        appendAgentResponseCard("Translation", result.text || result.texts.join("\n"), "#0f766e");
        return;
    }

    if (result.audio_file || result.audio_url) {
        appendGeneratedAudioCard("Generated Speech", result.audio_file || result.audio_url, result.audio_url || "");
        return;
    }

    if (Array.isArray(result.results) && result.results.length) {
        appendAgentResponseCard("Search Results", result.results.map((item) => item.text || "").join("\n\n"), "#334155");
    }
}

function buildAgentContextPayload(queryText) {
    const payload = { query: queryText };
    if (/\b(text to speech|tts|speak|voice|read aloud)\b/i.test(queryText || "")) {
        if ((currentSummary || "").trim()) {
            payload.text = currentSummary;
        } else if ((currentDocumentText || "").trim()) {
            payload.text = currentDocumentText;
        } else if (Array.isArray(transcriptData) && transcriptData.length > 0) {
            payload.text = buildExportTranscriptText();
        }
    }
    if (Array.isArray(transcriptData) && transcriptData.length > 0) {
        payload.content = buildExportTranscriptText();
    }
    if (currentDocumentId) {
        payload.document_id = currentDocumentId;
        payload.question = queryText;
    } else if (currentSessionId) {
        payload.session_id = currentSessionId;
        payload.question = queryText;
    }

    const targetLang = translationTargetSelect ? (translationTargetSelect.value || "").trim() : "";
    if (targetLang && /\btranslate\b/i.test(queryText || "")) {
        payload.target_lang = targetLang;
    }
    return payload;
}

async function submitAgentQuery() {
    const queryText = getAgentQueryText(false);
    if (pendingSelectedFile) {
        if (!queryText) {
            alert("Add your query first, then press the send arrow.");
            return;
        }
        getAgentQueryText(true);
        await processSelectedFile(pendingSelectedFile, false, queryText);
        return;
    }

    if (!queryText) return;
    getAgentQueryText(true);
    const payload = buildAgentContextPayload(queryText);
    const isPlainChatMode = Boolean(
        !pendingSelectedFile &&
        !payload.session_id &&
        !payload.document_id &&
        !payload.content &&
        !payload.text &&
        !payload.target_lang
    );

    if (isPlainChatMode) {
        appendChatBubble("user", queryText);
    }

    if (!isPlainChatMode) {
        document.getElementById("loadingOverlay").style.display = "flex";
    }
    try {
        const result = await agentQueryJSON(payload);
        await handleAgentResponse(result, { source: "agent_bar" });
        const resultPayload = (result && result.result) || {};
        if (!resultPayload.answer && !resultPayload.summary && !resultPayload.text && !(Array.isArray(resultPayload.transcript) && resultPayload.transcript.length)) {
            appendAgentResponseCard(
                "Agent Plan",
                (lastAgentPlan || []).map((step) => `${step.tool}: ${step.reason}`).join("\n") || "Task completed."
            );
        }
        if ((result && result.selected_tool) !== "chat_response") {
            setAgentBarVisible(false);
        }
    } catch (e) {
        if (isPlainChatMode) {
            appendChatBubble("assistant", e.message || "Agent request failed");
        } else {
            appendAgentResponseCard("Agent Error", e.message || "Agent request failed", "#b91c1c");
        }
    } finally {
        if (!isPlainChatMode) {
            document.getElementById("loadingOverlay").style.display = "none";
        }
    }
}

function appendGeneratedAudioCard(title, audioFile, audioUrl = "") {
    const chat = document.getElementById("chat");
    if (!chat) return;
    const src = audioUrl || `/audio/${encodeURIComponent(audioFile || "")}`;
    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:#0f766e">TTS</div>
        <div class="content agent-response-card" style="border-left: 4px solid #0f766e; background: rgba(15, 118, 110, 0.10);">
            <div class="agent-response-title">${escapeHTMLText(title || "Generated Speech")}</div>
            <audio controls preload="metadata" style="width:100%; margin-top:8px;">
                <source src="${src}" type="audio/mpeg">
                Your browser does not support audio playback.
            </audio>
            <div class="agent-response-text" style="margin-top:8px;">${escapeHTMLText(audioFile || "")}</div>
        </div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
}

async function speakSummary() {
    const text = String(currentSummary || "").trim();
    if (!text) {
        alert("Summary is not available yet.");
        return;
    }

    const targetLang = translationTargetSelect ? (translationTargetSelect.value || "").trim() : "";
    try {
        const result = await agentQueryJSON({
            tool: "text_to_speech",
            query: "convert this summary to speech",
            text,
            tts_lang: targetLang || "en"
        });
        await handleAgentResponse(result, { source: "tts" });
    } catch (e) {
        alert(e.message || "Text-to-speech failed.");
    }
}
