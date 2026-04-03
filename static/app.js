(() => {
  const cfg = window.__APP_CONFIG__ || { minChars: 10 };
  const minChars = Number(cfg.minChars) || 10;

  const form = document.getElementById("newsForm");
  const textarea = document.getElementById("news");
  const alertEl = document.getElementById("alert");
  const wordCountEl = document.getElementById("wordCount");

  const resultCard = document.getElementById("resultCard");
  const badge = document.getElementById("predictionBadge");
  const confidenceText = document.getElementById("confidenceText");
  const segReal = document.getElementById("segReal");
  const segFake = document.getElementById("segFake");
  const realProbEl = document.getElementById("realProb");
  const fakeProbEl = document.getElementById("fakeProb");
  const resultWordCount = document.getElementById("resultWordCount");

  const btn = document.getElementById("predictBtn");
  const btnText = document.getElementById("predictBtnText");
  const spinner = document.getElementById("loadingSpinner");
  const clearBtn = document.getElementById("clearBtn");

  const themeToggle = document.getElementById("themeToggle");
  const themeToggleIcon = document.getElementById("themeToggleIcon");

  const categorySelect = document.getElementById("category");
  const fetchNewsBtn = document.getElementById("fetchNewsBtn");
  const liveSourceEl = document.getElementById("liveSource");
  const headlineSelect = document.getElementById("headlineSelect");
  const useHeadlineBtn = document.getElementById("useHeadlineBtn");
  const randomHeadlineBtn = document.getElementById("randomHeadlineBtn");

  let fetchedHeadlines = [];

  // Performance + live metals header widgets.
  const metricAccuracyEl = document.getElementById("metricAccuracy");
  const metricPrecisionEl = document.getElementById("metricPrecision");
  const metricRecallEl = document.getElementById("metricRecall");
  const metricF1El = document.getElementById("metricF1");

  // Viral headlines widget.
  const refreshViralBtn = document.getElementById("refreshViralBtn");
  const viralCategory = document.getElementById("viralCategory");
  const viralSelect = document.getElementById("viralSelect");
  const useViralBtn = document.getElementById("useViralBtn");
  const viralSourceEl = document.getElementById("viralSource");
  let viralItems = [];

  async function loadMetrics() {
    try {
      const res = await fetch("/metrics");
      const data = await res.json();
      if (!res.ok) throw new Error(data && data.error ? data.error : "Failed");

      if (metricAccuracyEl) metricAccuracyEl.textContent = data.accuracy != null ? `${data.accuracy}%` : "—";
      if (metricPrecisionEl) metricPrecisionEl.textContent = data.precision != null ? `${data.precision}%` : "—";
      if (metricRecallEl) metricRecallEl.textContent = data.recall != null ? `${data.recall}%` : "—";
      if (metricF1El) metricF1El.textContent = data.f1 != null ? `${data.f1}%` : "—";
    } catch (e) {
      // Keep placeholders if unavailable.
    }
  }

  function setViralSource(text) {
    if (!viralSourceEl) return;
    viralSourceEl.textContent = text || "";
  }

  async function loadViral() {
    if (!viralSelect || !useViralBtn) return;
    const cat = (viralCategory && viralCategory.value) || "top";
    viralSelect.disabled = true;
    useViralBtn.disabled = true;
    setViralSource("");
    viralSelect.innerHTML = `<option value="">Loading viral headlines…</option>`;

    try {
      const res = await fetch(`/viral?category=${encodeURIComponent(cat)}&limit=25`);
      const data = await res.json();
      if (!res.ok) throw new Error(data && data.error ? data.error : "Failed");

      viralItems = Array.isArray(data.items) ? data.items : [];
      if (!viralItems.length) throw new Error("No viral headlines found.");

      viralSelect.innerHTML = "";
      viralItems.forEach((it, idx) => {
        const opt = document.createElement("option");
        opt.value = String(idx);
        opt.textContent = it.title || `Viral headline ${idx + 1}`;
        viralSelect.appendChild(opt);
      });
      viralSelect.disabled = false;
      useViralBtn.disabled = false;
      setViralSource("Aggregated from multiple global sources (RSS).");
    } catch (e) {
      viralItems = [];
      viralSelect.innerHTML = `<option value="">Unable to load viral headlines</option>`;
      setViralSource("Try again in a moment.");
    }
  }

  function setAlert(message, isError = true) {
    alertEl.textContent = message;
    alertEl.hidden = !message;
    alertEl.style.borderColor = isError ? "rgba(231, 76, 60, 0.35)" : "rgba(46, 204, 113, 0.35)";
    alertEl.style.background = isError ? "rgba(231, 76, 60, 0.12)" : "rgba(46, 204, 113, 0.12)";
  }

  function setLoading(loading) {
    btn.disabled = loading;
    spinner.hidden = !loading;
    btnText.textContent = loading ? "Analyzing..." : "Analyze";
  }

  function setTheme(theme) {
    document.documentElement.setAttribute("data-theme", theme);
    const icon = theme === "light" ? "☀️" : "🌙";
    if (themeToggleIcon) themeToggleIcon.textContent = icon;
  }

  function initTheme() {
    const saved = window.localStorage.getItem("theme");
    if (saved === "light" || saved === "dark") {
      setTheme(saved);
    } else {
      // Default based on OS preference (falls back to dark if unavailable).
      const preferLight = window.matchMedia && window.matchMedia("(prefers-color-scheme: light)").matches;
      setTheme(preferLight ? "light" : "dark");
    }
  }

  function updateWordCount() {
    const text = (textarea.value || "").trim();
    const words = text ? text.split(/\s+/).filter(Boolean).length : 0;
    wordCountEl.textContent = `${words} words`;
  }

  function resetUI() {
    textarea.value = "";
    setAlert("");
    resultCard.hidden = true;
    badge.classList.remove("real", "fake");
    badge.textContent = "—";
    confidenceText.textContent = "—";
    realProbEl.textContent = "—";
    fakeProbEl.textContent = "—";
    resultWordCount.textContent = "—";
    segReal.style.width = "0%";
    segFake.style.width = "0%";
    fetchedHeadlines = [];
    if (headlineSelect) {
      headlineSelect.innerHTML = `<option value="">Headlines will appear here…</option>`;
      headlineSelect.disabled = true;
    }
    if (useHeadlineBtn) useHeadlineBtn.disabled = true;
    if (randomHeadlineBtn) randomHeadlineBtn.disabled = true;
    setLiveSource("");
    updateWordCount();
  }

  function setLiveSource(info) {
    if (!liveSourceEl) return;
    if (!info) {
      liveSourceEl.textContent = "";
      return;
    }
    liveSourceEl.textContent = `Loaded from: ${info}`;
  }

  async function predictNow(newsText) {
    setAlert("");
    setLoading(true);

    try {
      const res = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ news: newsText }),
      });

      const data = await res.json();
      if (!res.ok) {
        const msg = data && data.error ? data.error : "Prediction failed.";
        setAlert(msg);
        return;
      }

      const prediction = data.prediction;
      const confidence = data.confidence;
      const realPct = data.real_prob;
      const fakePct = data.fake_prob;

      badge.classList.remove("real", "fake");
      badge.classList.add(prediction === "Real" ? "real" : "fake");
      badge.textContent = prediction;

      confidenceText.textContent = `${confidence.toFixed(2)}%`;
      realProbEl.textContent = `${realPct.toFixed(2)}%`;
      fakeProbEl.textContent = `${fakePct.toFixed(2)}%`;

      // Segmented confidence bar: real and fake probabilities.
      segReal.style.width = `${realPct}%`;
      segFake.style.width = `${fakePct}%`;

      resultWordCount.textContent = `${data.word_count} words`;

      // Animation on update.
      resultCard.hidden = false;
      resultCard.classList.remove("animate-in");
      // Force reflow so the animation always triggers.
      void resultCard.offsetWidth;
      resultCard.classList.add("animate-in");
    } catch (e) {
      setAlert("Something went wrong. Please try again.");
    } finally {
      setLoading(false);
    }
  }

  function validate(newsText) {
    const text = (newsText || "").trim();
    if (!text) return { ok: false, message: "Please enter some news text." };
    if (text.length < minChars) return { ok: false, message: `Please enter at least ${minChars} characters.` };
    return { ok: true };
  }

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    const newsText = textarea.value;
    const v = validate(newsText);
    if (!v.ok) {
      setAlert(v.message);
      return;
    }
    await predictNow(newsText.trim());
  });

  clearBtn.addEventListener("click", () => resetUI());

  textarea.addEventListener("input", () => {
    updateWordCount();
    if (alertEl && !alertEl.hidden) {
      // Clear validation warning as the user types.
      setAlert("");
    }
  });

  // Live news fetching: pull a headline + snippet for the selected category.
  fetchNewsBtn.addEventListener("click", async () => {
    const category = categorySelect.value || "general";
    setAlert("");
    setLiveSource("");
    resultCard.hidden = true;
    setLoading(true);

    try {
      const res = await fetch(`/news?category=${encodeURIComponent(category)}&limit=20`);
      const data = await res.json();
      if (!res.ok) {
        const msg = data && data.error ? data.error : "Unable to fetch live news.";
        setAlert(msg);
        return;
      }

      fetchedHeadlines = Array.isArray(data.items) ? data.items : [];
      if (!fetchedHeadlines.length) {
        setAlert("No headlines found. Try another category.");
        return;
      }

      // Populate dropdown.
      headlineSelect.innerHTML = "";
      fetchedHeadlines.forEach((it, idx) => {
        const opt = document.createElement("option");
        opt.value = String(idx);
        opt.textContent = it.title || `Headline ${idx + 1}`;
        headlineSelect.appendChild(opt);
      });
      headlineSelect.disabled = false;
      useHeadlineBtn.disabled = false;
      randomHeadlineBtn.disabled = false;
      setLiveSource("Multiple open RSS sources (varies by category)");
    } catch (e) {
      setAlert("Unable to reach live news service.");
    } finally {
      setLoading(false);
    }
  });

  async function useHeadlineAt(index) {
    const it = fetchedHeadlines[index];
    if (!it) return;
    textarea.value = it.text || it.title || "";
    updateWordCount();
    if (it.source) setLiveSource(it.source);

    const v = validate(textarea.value);
    if (v.ok) await predictNow(textarea.value.trim());
    else setAlert(v.message);
  }

  useHeadlineBtn.addEventListener("click", async () => {
    const idx = Number(headlineSelect.value);
    if (!Number.isFinite(idx)) return;
    await useHeadlineAt(idx);
  });

  randomHeadlineBtn.addEventListener("click", async () => {
    if (!fetchedHeadlines.length) return;
    const idx = Math.floor(Math.random() * fetchedHeadlines.length);
    headlineSelect.value = String(idx);
    await useHeadlineAt(idx);
  });

  // Sample buttons: fill textarea, then analyze automatically.
  async function runDemoSample(kind) {
    setAlert("");
    setLiveSource("");
    resultCard.hidden = true;
    setLoading(true);

    try {
      const minConf = kind === "real" ? 85 : 80;
      const res = await fetch(`/demo-sample?label=${encodeURIComponent(kind)}&min_conf=${minConf}&max_conf=90`);
      const data = await res.json();
      if (!res.ok) {
        const msg = data && data.error ? data.error : "Unable to load demo sample.";
        setAlert(msg);
        return;
      }

      const item = data.item || {};
      const model = data.model || {};

      textarea.value = item.text || "";
      updateWordCount();
      setLiveSource(item.source ? `${item.source} (dataset sample)` : "dataset sample");

      // Directly render the prediction without doing a second /predict call.
      const prediction = model.prediction;
      const confidence = Number(model.confidence || 0);
      const realPct = Number(model.real_prob || 0);
      const fakePct = Number(model.fake_prob || 0);

      badge.classList.remove("real", "fake");
      badge.classList.add(prediction === "Real" ? "real" : "fake");
      badge.textContent = prediction;

      confidenceText.textContent = `${confidence.toFixed(2)}%`;
      realProbEl.textContent = `${realPct.toFixed(2)}%`;
      fakeProbEl.textContent = `${fakePct.toFixed(2)}%`;

      segReal.style.width = `${realPct}%`;
      segFake.style.width = `${fakePct}%`;

      resultWordCount.textContent = `${model.word_count || 0} words`;

      resultCard.hidden = false;
      resultCard.classList.remove("animate-in");
      void resultCard.offsetWidth;
      resultCard.classList.add("animate-in");
    } catch (e) {
      setAlert("Unable to load demo sample.");
    } finally {
      setLoading(false);
    }
  }

  document.querySelectorAll(".sample-btn").forEach((btn) => {
    btn.addEventListener("click", async () => {
      const kind = btn.getAttribute("data-sample");
      await runDemoSample(kind);
    });
  });

  themeToggle.addEventListener("click", () => {
    const current = document.documentElement.getAttribute("data-theme") || "dark";
    const next = current === "dark" ? "light" : "dark";
    window.localStorage.setItem("theme", next);
    setTheme(next);
  });

  initTheme();
  updateWordCount();
  loadMetrics();
  loadViral();

  if (refreshViralBtn) refreshViralBtn.addEventListener("click", () => loadViral());
  if (viralCategory) viralCategory.addEventListener("change", () => loadViral());
  if (useViralBtn) {
    useViralBtn.addEventListener("click", async () => {
      const idx = Number(viralSelect.value);
      const it = viralItems[idx];
      if (!it) return;
      textarea.value = it.text || it.title || "";
      updateWordCount();
      setLiveSource(it.source || "");
      const v = validate(textarea.value);
      if (v.ok) await predictNow(textarea.value.trim());
      else setAlert(v.message);
    });
  }
})();

