let transcriptData = [];
let uniqueSpeakers = new Set();
let speakerColors = {};
let currentSummary = "";
let lastAudioFile = "";
let speakerNameMap = {}; 
let speakerOrderMap = {}; 

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

async function processAudio() {
    const input = document.getElementById("filename");
    const filename = input.value.trim();
    if (!filename) return;
    lastAudioFile = filename;
    
    // Show Loader and hide input
    document.getElementById("loadingOverlay").style.display = "flex";
    document.getElementById("inputGroup").classList.add("hidden");

    try {
        const response = await fetch("/process", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ filename: filename }) });
        const result = await response.json();
        
        transcriptData = result.transcript;
        currentSummary = result.summary;
        
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
        alert("Connection failed."); 
    }
}

async function renderChatDelayed() {
    // Hide Loader once rendering starts
    document.getElementById("loadingOverlay").style.display = "none";
    
    const chat = document.getElementById("chat");
    const old = chat.querySelectorAll(".transcription");
    old.forEach(r => r.remove());

    if (transcriptData.length === 0) return;

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

    for (const group of groupedTranscript) {
        const row = document.createElement("div");
        row.className = "message-row transcription";
        const colorSet = speakerColors[group.speaker];
        const ts = `[${formatTime(group.start || 0)} - ${formatTime(group.end || 0)}]`;
        
        const combinedText = group.texts.length > 1 
            ? group.texts.map(t => `• ${t}`).join('<br>') 
            : group.texts[0];

        row.innerHTML = `<div class="avatar" style="background: ${colorSet.main}">${group.speaker[0]}</div><div class="content" style="border-left: 4px solid ${colorSet.main}; background: ${colorSet.glow}"><span style="font-size:10px; font-weight:900; color:${colorSet.main}; text-transform:uppercase;">${group.speaker}</span><br>${combinedText}<span style="display:block; font-size:10px; color:var(--muted); margin-top:5px; font-weight:600;">${ts}</span></div>`;
        chat.appendChild(row);
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

function renderSummaryCard(summaryText) {
    const chat = document.getElementById("chat");
    const row = document.createElement("div");
    row.className = "message-row transcription";

    function escapeHTML(text) {
        return text.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
    }

    let updatedSummary = summaryText;
    const sortedEntries = Object.entries(speakerNameMap).sort((a, b) => b[0].length - a[0].length);
    for (let [originalSpeaker, finalizedName] of sortedEntries) {
        const regex = new RegExp('\\b' + originalSpeaker.replace(/[.*+?^${}()|[\]\\]/g, '\\$&') + '\\b', 'gi');
        updatedSummary = updatedSummary.replace(regex, finalizedName);
    }

    let formatted = escapeHTML(updatedSummary)
        .replace(/\*\*(.*?)\*\*/g, "$1")
        .replace(/^One meaningfull paragraph Summary:/gim, "<h3>📌 Minutes of Meeting Summaried </h3>")
        .replace(/^\* /gm, "• ")
        .replace(/^- /gm, "• ")
        .replace(/\n/g, "<br>");

    row.innerHTML = `
        <div class="avatar" style="background:#10b981">Σ</div>
        <div class="content summary-card" id="sumCard" style="
            padding:16px; border-radius:12px; background:#111827; color:white; line-height:1.6; position:relative;">
            <div class="copy-sum-icon" onclick="copySummary()" title="Copy Summary"
                style="position:absolute;top:10px;right:10px;cursor:pointer;opacity:0.7">
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                    <rect x="9" y="9" width="13" height="13" rx="2"></rect>
                    <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1"></path>
                </svg>
            </div>
            ${formatted}
        </div>
    `;

    chat.appendChild(row);
    chat.scrollTop = chat.scrollHeight;
}

async function showSummary() {
    if (!transcriptData || transcriptData.length === 0) return;

    if (currentSummary) {
        renderSummaryCard(currentSummary);
        return;
    }

    document.getElementById("loadingOverlay").style.display = "flex";
    document.querySelector(".loading-text").innerText = "GENERATING SUMMARY...";

    try {
        const content = buildExportTranscriptText();
        const response = await fetch("/summarize_text", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ content: content })
        });
        const result = await response.json();

        if (!response.ok) {
            throw new Error(result.error || "Summary generation failed");
        }

        currentSummary = result.summary || "";
        renderSummaryCard(currentSummary);
    } catch (e) {
        alert("Summary generation failed.");
    } finally {
        document.getElementById("loadingOverlay").style.display = "none";
        document.querySelector(".loading-text").innerText = "GENERATING TRANSCRIPTION...";
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

function exportData(format) {
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


