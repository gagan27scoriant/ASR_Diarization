(function () {
  const statusEl = document.getElementById("status");
  const pagesEl = document.getElementById("pages");

  function setStatus(msg, isError = false) {
    if (!statusEl) return;
    statusEl.textContent = msg || "";
    statusEl.style.color = isError ? "#b91c1c" : "#64748b";
  }

  function parseParams() {
    const params = new URLSearchParams(window.location.search || "");
    return {
      file: params.get("file") || "",
      mode: (params.get("mode") || "full").toLowerCase(),
    };
  }

  async function renderPage(pdf, pageNum, scale) {
    const page = await pdf.getPage(pageNum);
    const viewport = page.getViewport({ scale });
    const canvas = document.createElement("canvas");
    const context = canvas.getContext("2d", { alpha: false });
    canvas.width = Math.floor(viewport.width);
    canvas.height = Math.floor(viewport.height);

    const wrapper = document.createElement("div");
    wrapper.className = "page-frame";
    wrapper.appendChild(canvas);

    await page.render({
      canvasContext: context,
      viewport,
    }).promise;

    return wrapper;
  }

  async function boot() {
    const { file, mode } = parseParams();
    if (!file) {
      setStatus("Missing PDF file path.", true);
      return;
    }
    if (!window.pdfjsLib) {
      setStatus("PDF.js failed to load.", true);
      return;
    }

    try {
      const loadingTask = window.pdfjsLib.getDocument({
        url: file,
        withCredentials: true,
        disableWorker: true,
      });
      const pdf = await loadingTask.promise;

      const maxPages = mode === "mini" ? 1 : 80;
      const renderCount = Math.min(pdf.numPages, maxPages);
      const scale = mode === "mini" ? 0.7 : 1.25;

      setStatus(`Rendering ${renderCount} page${renderCount === 1 ? "" : "s"}...`);

      const fragment = document.createDocumentFragment();
      for (let i = 1; i <= renderCount; i++) {
        const pageNode = await renderPage(pdf, i, scale);
        fragment.appendChild(pageNode);
      }
      pagesEl.innerHTML = "";
      pagesEl.appendChild(fragment);

      if (pdf.numPages > renderCount) {
        setStatus(`Showing first ${renderCount} of ${pdf.numPages} pages.`);
      } else {
        setStatus(`Loaded ${pdf.numPages} page${pdf.numPages === 1 ? "" : "s"}.`);
      }
    } catch (err) {
      setStatus(`Failed to load PDF: ${err && err.message ? err.message : "Unknown error"}`, true);
    }
  }

  boot();
})();
