let transcriptData = [];
let uniqueSpeakers = new Set();
let speakerColors = {};
let currentSummary = "";
let lastAudioFile = "";
let currentProcessedAudio = "";
let currentProcessedVideo = "";
let currentDocumentFilename = "";
let currentDocumentType = "";
let currentDocumentText = "";
let speakerNameMap = {}; 
let speakerOrderMap = {}; 
let isSummaryLoading = false;
let groupedTranscriptCache = [];

function toggleSidebar() { document.getElementById("sidebar").classList.toggle("expanded"); }

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

const audioFileInput = document.getElementById("audioFile");
const filePathInput = document.getElementById("filePath");
const sourceIndicator = document.getElementById("sourceIndicator");

function updateSourceIndicator() {
    const hasFile = audioFileInput.files && audioFileInput.files.length > 0;
    const hasPath = filePathInput.value.trim().length > 0;

    if (hasFile) {
        sourceIndicator.textContent = "Source: Uploaded file";
    } else if (hasPath) {
        sourceIndicator.textContent = "Source: Path file";
    } else {
        sourceIndicator.textContent = "Source: Not selected";
    }
}

audioFileInput.addEventListener("change", () => {
    if (audioFileInput.files && audioFileInput.files.length > 0) {
        filePathInput.value = audioFileInput.files[0].name;
    }
    updateSourceIndicator();
});

filePathInput.addEventListener("input", () => {
    // If user types manually, treat it as a path-source flow.
    if (audioFileInput.files && audioFileInput.files.length > 0) {
        audioFileInput.value = "";
    }
    updateSourceIndicator();
});

function openFilePicker() {
    audioFileInput.click();
}

updateSourceIndicator();

function formatTime(seconds) {
    const mins = Math.floor(seconds / 60);
    const secs = Math.floor(seconds % 60);
    return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
}

function getSpeakerColor(idx) {
    const palette = [
        { main: '#7c5cff', glow: 'rgba(124, 92, 255, 0.1)' },   
        { main: '#10b981', glow: 'rgba(16, 185, 129, 0.1)' },   
        { main: '#f59e0b', glow: 'rgba(245, 158, 11, 0.1)' },   
        { main: '#3b82f6', glow: 'rgba(59, 130, 246, 0.1)' },   
        { main: '#ec4899', glow: 'rgba(236, 72, 153, 0.1)' }    
    ];
    return palette[idx % palette.length];
}

function isAudioFile(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    return ["wav", "mp3", "aac", "aiff", "wma", "amr", "opus"].includes(ext);
}

function isVideoFile(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    return ["mp4", "mkv", "avi", "mov", "wmv", "mpeg", "3gp"].includes(ext);
}

function isDocumentFile(name) {
    const ext = (name.split(".").pop() || "").toLowerCase();
    return ["pdf", "docx", "txt"].includes(ext);
}

async function processAudio() {
    const inputValue = filePathInput.value.trim();
    const selectedFile = audioFileInput.files && audioFileInput.files.length > 0 ? audioFileInput.files[0] : null;

    if (!selectedFile && !inputValue) return;
    lastAudioFile = selectedFile ? selectedFile.name : inputValue;
    
    // Show Loader and hide input
    document.getElementById("loadingOverlay").style.display = "flex";
    document.getElementById("inputGroup").classList.add("hidden");

    try {
        if (selectedFile && isDocumentFile(selectedFile.name)) {
            const meetingDetails = requestMeetingDetails();
            if (!meetingDetails) {
                document.getElementById("loadingOverlay").style.display = "none";
                document.getElementById("inputGroup").classList.remove("hidden");
                return;
            }

            const formData = new FormData();
            formData.append("document_file", selectedFile);
            formData.append("meeting_title", meetingDetails.meeting_title);
            formData.append("meeting_date", meetingDetails.meeting_date);
            formData.append("meeting_place", meetingDetails.meeting_place);

            const response = await fetch("/process_document", {
                method: "POST",
                body: formData
            });

            const result = await response.json();
            if (!response.ok) {
                throw new Error(result.error || "Document processing failed");
            }

            transcriptData = [];
            groupedTranscriptCache = [];
            currentSummary = result.summary || "";
            currentProcessedAudio = "";
            currentProcessedVideo = "";
            currentDocumentFilename = result.document_filename || "";
            currentDocumentType = (result.document_type || "").toLowerCase();
            currentDocumentText = result.document_text || "";
            document.getElementById("sumBtn").style.display = "none";

            renderDocumentResult();
            return;
        }

        if (selectedFile && !isAudioFile(selectedFile.name) && !isVideoFile(selectedFile.name)) {
            throw new Error("Unsupported file. Use audio/video or documents (.pdf/.docx/.txt).");
        }

        let response;

        if (selectedFile) {
            const formData = new FormData();
            formData.append("audio_file", selectedFile);
            response = await fetch("/process", {
                method: "POST",
                body: formData
            });
        } else {
            response = await fetch("/process", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ file_path: inputValue })
            });
        }

        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Audio processing failed");
        }
        
        transcriptData = result.transcript;
        currentSummary = result.summary;
        currentDocumentFilename = "";
        currentDocumentType = "";
        currentDocumentText = "";
        currentProcessedVideo = result.source_video || "";
        if (result.processed_file) {
            lastAudioFile = result.processed_file;
            currentProcessedAudio = result.processed_file;
        }
        
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
        
        renderChatDelayed();
        setupRenameSidebar();
        document.getElementById("sumBtn").style.display = "flex";
    } catch (e) { 
        document.getElementById("loadingOverlay").style.display = "none";
        document.getElementById("inputGroup").classList.remove("hidden");
        alert(e.message || "Connection failed."); 
    }
}

async function renderChatDelayed() {
    // Hide Loader once rendering starts
    document.getElementById("loadingOverlay").style.display = "none";
    
    const chat = document.getElementById("chat");
    const old = chat.querySelectorAll(".transcription");
    old.forEach(r => r.remove());

    if (transcriptData.length === 0) return;

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

    if (currentProcessedAudio) {
        const audioRow = document.createElement("div");
        audioRow.className = "message-row transcription";
        audioRow.innerHTML = `
            <div class="avatar" style="background:#ef4444">♫</div>
            <div class="content audio-preview-content" style="border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.12);">
                <span style="font-size:10px; font-weight:900; color:#ef4444; text-transform:uppercase;">AUDIO PREVIEW</span><br>
                <audio controls preload="metadata" style="width:100%; margin-top:8px;">
                    <source src="/audio/${encodeURIComponent(currentProcessedAudio)}" type="audio/wav">
                    Your browser does not support audio playback.
                </audio>
                <span style="display:block; font-size:10px; color:var(--muted); margin-top:5px; font-weight:600;">${currentProcessedAudio}</span>
            </div>
        `;
        chat.appendChild(audioRow);
    }

    const groupedTranscript = [];
    let currentGroup = null;

    transcriptData.forEach(seg => {
        if (currentGroup && currentGroup.speaker === seg.speaker) {
            currentGroup.texts.push(seg.text);
            currentGroup.end = seg.end;
        } else {
            if (currentGroup) groupedTranscript.push(currentGroup);
            currentGroup = {
                speaker: seg.speaker,
                texts: [seg.text],
                start: seg.start,
                end: seg.end
            };
        }
    });
    if (currentGroup) groupedTranscript.push(currentGroup);
    groupedTranscriptCache = groupedTranscript;

    for (let i = 0; i < groupedTranscript.length; i++) {
        const group = groupedTranscript[i];
        const row = document.createElement("div");
        row.className = "message-row transcription";
        const colorSet = speakerColors[group.speaker];
        const ts = `[${formatTime(group.start || 0)} - ${formatTime(group.end || 0)}]`;
        
        const combinedText = group.texts.length > 1 
            ? group.texts.map(t => `• ${t}`).join('<br>') 
            : group.texts[0];

        row.innerHTML = `<div class="avatar" style="background: ${colorSet.main}">${group.speaker[0]}</div><div class="content" style="border-left: 4px solid ${colorSet.main}; background: ${colorSet.glow}"><div class="copy-transcript-icon" onclick="copyTranscriptByIndex(${i})" title="Copy Transcript"><svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><rect x="9" y="9" width="13" height="13" rx="2"></rect><path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path></svg></div><span style="font-size:10px; font-weight:900; color:${colorSet.main}; text-transform:uppercase;">${group.speaker}</span><br>${combinedText}<span style="display:block; font-size:10px; color:var(--muted); margin-top:5px; font-weight:600;">${ts}</span></div>`;
        chat.appendChild(row);
    }
    chat.scrollTop = chat.scrollHeight;
}

function renderDocumentResult() {
    document.getElementById("loadingOverlay").style.display = "none";

    const chat = document.getElementById("chat");
    const old = chat.querySelectorAll(".transcription");
    old.forEach((r) => r.remove());

    const row = document.createElement("div");
    row.className = "message-row transcription";

    let previewMarkup = "";
    if (currentDocumentType === "pdf") {
        previewMarkup = `
            <div class="doc-preview-panel">
                <iframe class="doc-preview-frame" src="/documents/${encodeURIComponent(currentDocumentFilename)}"></iframe>
            </div>
        `;
    } else {
        previewMarkup = `
            <div class="doc-preview-panel">
                <div class="doc-preview-text">${escapeHTMLText(currentDocumentText || "No preview available.")}</div>
            </div>
        `;
    }

    row.innerHTML = `
        <div class="avatar" style="background:#ef4444">DOC</div>
        <div class="content doc-preview-content" style="border-left: 4px solid #ef4444; background: rgba(239, 68, 68, 0.10);">
            <span style="font-size:10px; font-weight:900; color:#ef4444; text-transform:uppercase;">DOCUMENT PREVIEW</span><br>
            <span style="display:block; font-size:12px; color:var(--muted); margin-top:5px;">${escapeHTMLText(currentDocumentFilename)}</span>
            ${previewMarkup}
        </div>
    `;

    chat.appendChild(row);
    if (currentSummary) {
        renderSummaryCard(currentSummary);
    }
    chat.scrollTop = chat.scrollHeight;
}

function setupRenameSidebar() {
    const box = document.getElementById("renameBox");
    document.getElementById("sideHeading").style.display = "block";
    box.innerHTML = "";
    
    const sortedSpeakers = Array.from(uniqueSpeakers).sort((a, b) => {
        return speakerOrderMap[a] - speakerOrderMap[b];
    });
    
    sortedSpeakers.forEach(s => {
        const div = document.createElement("div");
        div.className = "rename-item";
        div.innerHTML = `<label>RENAME ${s}</label><input type="text" id="name_${s}" placeholder="Enter name...">`;
        box.appendChild(div);
    });
}

function applyNames() {
    let nameMap = {};
    let newSpeakerColors = {};
    uniqueSpeakers.forEach(s => {
        const val = document.getElementById("name_" + s).value.trim();
        const final = val || s;
        nameMap[s] = final;
        newSpeakerColors[final] = speakerColors[s];
        speakerNameMap[s] = final;
    });
    transcriptData = transcriptData.map(seg => ({ ...seg, speaker: nameMap[seg.speaker] }));
    speakerColors = newSpeakerColors;
    uniqueSpeakers = new Set(Object.values(nameMap));
    
    renderChatDelayed();
    setupRenameSidebar();
    currentSummary = "";
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
        <div class="copy-sum-icon" onclick="copySummary()" title="Copy Summary"
            style="position:absolute;top:10px;right:10px;cursor:pointer;opacity:0.7">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
            </svg>
        </div>
        ${formatted}
    `;

    const chat = document.getElementById("chat");
    if (targetCard) {
        targetCard.id = "sumCard";
        targetCard.innerHTML = summaryMarkup;
        chat.scrollTop = chat.scrollHeight;
        return;
    }

    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:#10b981">Σ</div>
        <div class="content summary-card" id="sumCard" style="
            padding:16px; border-radius:12px; background:#111827; color:white; line-height:1.6; position:relative;">
            ${summaryMarkup}
        </div>
    `;
    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
}

function renderSummaryLoadingCard() {
    const chat = document.getElementById("chat");
    const row = document.createElement("div");
    row.className = "message-row transcription";
    row.innerHTML = `
        <div class="avatar" style="background:#10b981">Σ</div>
        <div class="content summary-card" id="summaryPendingCard" style="
            padding:16px; border-radius:12px; background:#111827; color:white; line-height:1.6; position:relative;">
            <div class="summary-loading">
                <span class="summary-loading-text">Generating Summary</span>
                <span class="dot"></span>
                <span class="dot"></span>
                <span class="dot"></span>
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

    if (currentSummary) {
        renderSummaryCard(currentSummary);
        return;
    }

    const meetingDetails = requestMeetingDetails();
    if (!meetingDetails) return;

    isSummaryLoading = true;
    const pendingCard = renderSummaryLoadingCard();

    try {
        const content = buildExportTranscriptText();
        const response = await fetch("/summarize_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({
                content: content,
                meeting_title: meetingDetails.meeting_title,
                meeting_date: meetingDetails.meeting_date,
                meeting_place: meetingDetails.meeting_place
            })
        });
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Summary generation failed");
        }

        currentSummary = result.summary || "";
        renderSummaryCard(currentSummary, pendingCard);
    } catch (e) {
        if (pendingCard) {
            pendingCard.textContent = "Summary generation failed.";
            pendingCard.removeAttribute("id");
        }
    } finally {
        isSummaryLoading = false;
    }
}

function copySummary() {
    const text = document.getElementById("sumCard").innerText;
    navigator.clipboard.writeText(text);
    alert("Summary copied to clipboard!");
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
