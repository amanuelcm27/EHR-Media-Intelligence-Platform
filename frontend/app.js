"use strict";

const TYPE_META = {
  DocumentReference: {
    label: "Note",
    badge: "bg-sky-50 text-sky-700 ring-sky-600/20",
    dot: "bg-sky-500",
  },
  Observation: {
    label: "Lab",
    badge: "bg-amber-50 text-amber-800 ring-amber-600/20",
    dot: "bg-amber-500",
  },
  DiagnosticReport: {
    label: "Report",
    badge: "bg-violet-50 text-violet-700 ring-violet-600/20",
    dot: "bg-violet-500",
  },
  Encounter: {
    label: "Encounter",
    badge: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
    dot: "bg-emerald-500",
  },
  Summary: {
    label: "AI summary",
    badge: "bg-teal-50 text-teal-700 ring-teal-600/20",
    dot: "bg-teal-500",
  },
  Patient: {
    label: "Patient",
    badge: "bg-slate-100 text-slate-700 ring-slate-600/20",
    dot: "bg-slate-500",
  },
};

const CONFIDENCE_META = {
  high: "bg-emerald-50 text-emerald-700 ring-emerald-600/20",
  moderate: "bg-amber-50 text-amber-800 ring-amber-600/20",
  low: "bg-rose-50 text-rose-700 ring-rose-600/20",
};

const SCORE_META = {
  high: "bg-teal-50 text-teal-700 ring-teal-600/20",
  mid: "bg-slate-100 text-slate-600 ring-slate-500/20",
  low: "bg-slate-50 text-slate-500 ring-slate-400/20",
};

const SUGGESTIONS = [
  "heart failure with shortness of breath",
  "sepsis antibiotics lactate",
  "critical potassium hyperkalemia",
  "appendicitis surgery",
  "diabetes follow up",
];

const state = {
  query: "",
  types: new Set(),
  start: "",
  end: "",
  results: [],
  loading: false,
  controller: null,
  requestId: 0,
  timer: null,
  lastTrigger: null,
  detail: null,
  detailController: null,
  activeTab: "summary",
  panels: {},
  health: null,
};

const $ = (id) => document.getElementById(id);
const els = {};

function el(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined && text !== null) node.textContent = String(text);
  return node;
}

function svgIcon(path, className = "h-4 w-4") {
  const template = document.createElement("template");
  template.innerHTML = `<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.8" class="${className}" aria-hidden="true"><path d="${path}" stroke-linecap="round" stroke-linejoin="round"/></svg>`;
  return template.content.firstElementChild;
}

function formatDate(value) {
  if (!value) return "Undated";
  const parsed = new Date(`${String(value).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(parsed.getTime())) return String(value);
  return new Intl.DateTimeFormat("en-US", {
    month: "short",
    day: "numeric",
    year: "numeric",
  }).format(parsed);
}

function formatMrn(mrn) {
  if (!mrn) return "MRN unknown";
  const value = String(mrn);
  if (value.toUpperCase().startsWith("MRN-")) return value.toUpperCase();
  return /^\d+$/.test(value) ? `MRN-${value}` : `MRN-${value.toUpperCase()}`;
}

function ageFromBirthDate(value) {
  if (!value) return null;
  const birth = new Date(`${String(value).slice(0, 10)}T00:00:00`);
  if (Number.isNaN(birth.getTime())) return null;
  const today = new Date();
  let age = today.getFullYear() - birth.getFullYear();
  const month = today.getMonth() - birth.getMonth();
  if (month < 0 || (month === 0 && today.getDate() < birth.getDate())) age -= 1;
  return age >= 0 ? age : null;
}

function typeMeta(type) {
  return (
    TYPE_META[type] || {
      label: type || "Resource",
      badge: "bg-slate-100 text-slate-700 ring-slate-600/20",
      dot: "bg-slate-500",
    }
  );
}

function confidenceMeta(label) {
  return CONFIDENCE_META[label] || CONFIDENCE_META.moderate;
}

function scoreMeta(score) {
  if (score >= 0.7) return SCORE_META.high;
  if (score >= 0.5) return SCORE_META.mid;
  return SCORE_META.low;
}

function escapeRegExp(value) {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function highlighted(text, query) {
  const fragment = document.createDocumentFragment();
  const source = String(text || "");
  const terms = Array.from(
    new Set(
      String(query || "")
        .toLowerCase()
        .split(/[^a-z0-9]+/)
        .filter((term) => term.length > 2)
    )
  );
  if (!terms.length) {
    fragment.append(document.createTextNode(source));
    return fragment;
  }
  const pattern = new RegExp(`(${terms.map(escapeRegExp).join("|")})`, "gi");
  const parts = source.split(pattern);
  parts.forEach((part, index) => {
    if (index % 2 === 1) {
      fragment.append(el("mark", "rounded bg-teal-100 px-0.5 text-teal-900", part));
    } else if (part) {
      fragment.append(document.createTextNode(part));
    }
  });
  return fragment;
}

function badge(type) {
  const meta = typeMeta(type);
  const node = el(
    "span",
    `type-badge ${meta.badge}`
  );
  node.append(el("span", `h-1.5 w-1.5 rounded-full ${meta.dot}`));
  node.append(document.createTextNode(meta.label));
  return node;
}

function updateClearButton() {
  const hasValue = Boolean(els.searchInput.value);
  els.clearSearch.classList.toggle("hidden", !hasValue);
  els.clearSearch.classList.toggle("flex", hasValue);
}

function renderFilterState() {
  document.querySelectorAll("#type-filters [data-type]").forEach((button) => {
    const type = button.dataset.type;
    const active = type === "all" ? state.types.size === 0 : state.types.has(type);
    button.setAttribute("aria-pressed", String(active));
  });
}

function updateUrl() {
  const params = new URLSearchParams();
  if (state.query) params.set("q", state.query);
  if (state.types.size) params.set("types", Array.from(state.types).join(","));
  if (state.start) params.set("start", state.start);
  if (state.end) params.set("end", state.end);
  const query = params.toString();
  const next = query ? `${window.location.pathname}?${query}` : window.location.pathname;
  window.history.replaceState(null, "", next);
}

function hydrateFromUrl() {
  const params = new URLSearchParams(window.location.search);
  const query = params.get("q") || "";
  const types = (params.get("types") || "")
    .split(",")
    .map((item) => item.trim())
    .filter((item) => TYPE_META[item]);
  types.forEach((type) => state.types.add(type));
  state.start = params.get("start") || "";
  state.end = params.get("end") || "";
  els.startDate.value = state.start;
  els.endDate.value = state.end;
  els.searchInput.value = query;
  renderFilterState();
  updateClearButton();
  if (query.trim()) {
    doSearch(query.trim());
  } else {
    renderWelcome();
  }
}

async function loadHealth() {
  try {
    const response = await fetch("/health");
    if (!response.ok) throw new Error(`Health check failed (${response.status})`);
    const data = await response.json();
    state.health = data;
    const stats = data.stats || {};
    els.indexStatus.replaceChildren();
    const dot = el("span", "relative flex h-2 w-2");
    dot.append(el("span", "absolute inline-flex h-full w-full animate-ping rounded-full bg-teal-400 opacity-60"));
    dot.append(el("span", "relative inline-flex h-2 w-2 rounded-full bg-teal-500"));
    els.indexStatus.append(dot);
    els.indexStatus.append(
      document.createTextNode(
        `${stats.patients || 0} patients · ${stats.chunks || 0} chunks indexed`
      )
    );
    els.indexStatus.className =
      "inline-flex items-center gap-2 rounded-full border border-teal-200 bg-teal-50 px-3 py-1.5 text-xs font-semibold text-teal-800 shadow-sm";
    if (state.query === "" && !els.resultGrid.querySelector("[data-result-card]")) {
      renderWelcome();
    }
  } catch (error) {
    els.indexStatus.replaceChildren();
    els.indexStatus.append(el("span", "h-2 w-2 rounded-full bg-amber-500"));
    els.indexStatus.append(document.createTextNode("Index unavailable"));
    els.indexStatus.className =
      "inline-flex items-center gap-2 rounded-full border border-amber-200 bg-amber-50 px-3 py-1.5 text-xs font-semibold text-amber-800 shadow-sm";
    els.indexStatus.title = error.message;
  }
}

function setLoading(loading, query) {
  state.loading = loading;
  els.resultGrid.setAttribute("aria-busy", String(loading));
  if (loading) {
    els.resultMeta.textContent = "Searching…";
    els.loadingAnnouncer.textContent = `Searching for ${query}`;
    renderSkeletons();
  } else {
    els.loadingAnnouncer.textContent = "";
  }
}

function renderSkeletons() {
  els.resultGrid.replaceChildren();
  for (let index = 0; index < 3; index += 1) {
    const card = el(
      "div",
      "animate-pulse rounded-2xl border border-slate-200 bg-white p-5 shadow-soft"
    );
    card.setAttribute("aria-hidden", "true");
    card.append(el("div", "h-5 w-24 rounded-full bg-slate-100"));
    card.append(el("div", "mt-4 h-4 w-2/3 rounded bg-slate-100"));
    card.append(el("div", "mt-2 h-3 w-1/2 rounded bg-slate-100"));
    card.append(el("div", "mt-4 h-3 w-full rounded bg-slate-100"));
    card.append(el("div", "mt-2 h-3 w-5/6 rounded bg-slate-100"));
    card.append(el("div", "mt-5 h-12 w-full rounded-xl bg-slate-100"));
    els.resultGrid.append(card);
  }
}

function scheduleSearch(immediate) {
  window.clearTimeout(state.timer);
  const query = els.searchInput.value.trim();
  updateClearButton();
  if (!query) {
    state.query = "";
    state.results = [];
    updateUrl();
    renderWelcome();
    return;
  }
  if (immediate) {
    doSearch(query);
  } else {
    state.timer = window.setTimeout(() => doSearch(query), 250);
  }
}

async function doSearch(query) {
  if (state.controller) state.controller.abort();
  const requestId = state.requestId + 1;
  state.requestId = requestId;
  const controller = new AbortController();
  state.controller = controller;
  state.query = query;
  updateUrl();
  setLoading(true, query);

  const payload = { query, top_k: 5 };
  if (state.types.size) payload.resource_types = Array.from(state.types);
  if (state.start) payload.start_date = state.start;
  if (state.end) payload.end_date = state.end;

  try {
    const response = await fetch("/search", {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    let data = {};
    try {
      data = await response.json();
    } catch {
      data = {};
    }
    if (!response.ok) {
      throw new Error(data.detail || `Search failed (${response.status})`);
    }
    if (requestId !== state.requestId) return;
    state.results = Array.isArray(data.results) ? data.results : [];
    renderResults(data);
  } catch (error) {
    if (error.name === "AbortError" || requestId !== state.requestId) return;
    renderError(error.message || "Search failed");
  } finally {
    if (requestId === state.requestId) {
      setLoading(false);
      els.resultGrid.setAttribute("aria-busy", "false");
    }
  }
}

function resultMetaText(data) {
  const count = Number(data.count || 0);
  const took = Math.max(0, Math.round(Number(data.took_ms || 0)));
  const noun = count === 1 ? "result" : "results";
  const provider = data.index && data.index.provider ? ` · ${data.index.provider}` : "";
  return `${count} ${noun} · ${took} ms${provider}`;
}

function renderResults(data) {
  els.resultGrid.replaceChildren();
  els.resultMeta.textContent = resultMetaText(data);
  els.loadingAnnouncer.textContent = `${data.count || 0} results found`;

  if (!state.results.length) {
    renderEmpty(data.query || state.query);
    return;
  }

  state.results.forEach((result, index) => {
    els.resultGrid.append(createResultCard(result, index));
  });
  els.resultGrid.setAttribute("aria-busy", "false");
}

function createResultCard(result, index) {
  const card = el("article", "group result-card animate-fade-up");
  card.dataset.resultCard = "true";
  card.dataset.patientId = result.patient_id;
  card.dataset.index = String(index);
  card.tabIndex = 0;
  card.setAttribute("role", "button");
  card.setAttribute("aria-haspopup", "dialog");
  card.setAttribute(
    "aria-label",
    `Open ${result.patient_name}, ${typeMeta(result.resource_type).label}, ${formatDate(
      result.effective_date
    )}`
  );
  card.style.animationDelay = `${Math.min(index * 35, 210)}ms`;

  const top = el("div", "flex items-start justify-between gap-3");
  top.append(badge(result.resource_type));
  const score = Math.round(Math.max(0, Number(result.score || 0)) * 100);
  top.append(
    el(
      "span",
      `inline-flex items-center rounded-full px-2.5 py-1 text-[11px] font-bold ring-1 ring-inset ${scoreMeta(
        Number(result.score || 0)
      )}`,
      `${score}% match`
    )
  );
  card.append(top);

  const heading = el("div", "mt-3");
  heading.append(el("h3", "text-base font-bold tracking-tight text-slate-900", result.patient_name));
  heading.append(
    el(
      "p",
      "mt-0.5 text-xs font-medium text-slate-500",
      `${formatMrn(result.mrn)} · ${formatDate(result.effective_date)}`
    )
  );
  card.append(heading);

  if (result.title) {
    const title = el("p", "mt-3 text-sm font-semibold text-slate-700");
    title.append(highlighted(result.title, state.query));
    card.append(title);
  }

  const snippet = el("p", "mt-2 line-clamp-3 text-sm leading-6 text-slate-600");
  snippet.append(highlighted(result.snippet || result.text || "", state.query));
  card.append(snippet);

  const ai = el(
    "div",
    "mt-auto border-t border-slate-100 pt-3"
  );
  const aiLabel = el("p", "flex items-center gap-1.5 text-[10px] font-bold uppercase tracking-[0.16em] text-teal-700");
  aiLabel.append(svgIcon("M12 3v2m0 14v2M5.2 5.2l1.4 1.4m10.8 10.8 1.4 1.4M3 12h2m14 0h2M5.2 18.8l1.4-1.4M17.4 6.6l1.4-1.4", "h-3.5 w-3.5"));
  aiLabel.append(document.createTextNode("AI summary"));
  if (result.summary_confidence !== null && result.summary_confidence !== undefined) {
    aiLabel.append(
      el(
        "span",
        `ml-auto rounded-full px-1.5 py-0.5 text-[9px] font-bold ring-1 ring-inset ${confidenceMeta(
          Number(result.summary_confidence) >= 0.75
            ? "high"
            : Number(result.summary_confidence) >= 0.5
              ? "moderate"
              : "low"
        )}`,
        `${Math.round(Number(result.summary_confidence) * 100)}% confidence`
      )
    );
  }
  ai.append(aiLabel);
  ai.append(
    el(
      "p",
      "mt-1 line-clamp-2 text-xs leading-5 text-slate-500",
      result.ai_snippet || "No AI summary available for this record."
    )
  );
  card.append(ai);

  const footer = el(
    "p",
    "mt-3 flex items-center gap-1 text-[11px] font-semibold text-teal-700 opacity-0 transition group-hover:opacity-100 group-focus-visible:opacity-100"
  );
  footer.append(document.createTextNode("Open patient record"));
  footer.append(svgIcon("M5 12h14m-5-5 5 5-5 5", "h-3.5 w-3.5"));
  card.append(footer);

  const open = () => openDrawer(result.patient_id, card);
  card.addEventListener("click", open);
  card.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      open();
    }
  });
  return card;
}

function renderWelcome() {
  els.resultGrid.replaceChildren();
  els.resultMeta.textContent = "Ready";
  els.loadingAnnouncer.textContent = "";
  els.resultGrid.setAttribute("aria-busy", "false");

  const wrapper = el(
    "div",
    "col-span-full overflow-hidden rounded-3xl border border-slate-200 bg-white shadow-soft"
  );
  const body = el("div", "p-6 sm:p-8");
  const icon = el(
    "div",
    "flex h-12 w-12 items-center justify-center rounded-2xl bg-teal-50 text-teal-700 ring-1 ring-inset ring-teal-600/10"
  );
  icon.append(svgIcon("m21 21-4.35-4.35M17 11a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z", "h-6 w-6"));
  body.append(icon);
  body.append(
    el(
      "h3",
      "mt-4 text-xl font-bold tracking-tight text-slate-900",
      "Search across notes, labs, reports and AI summaries"
    )
  );
  body.append(
    el(
      "p",
      "mt-2 max-w-2xl text-sm leading-6 text-slate-600",
      "Results are ranked with sentence-transformer embeddings, then filtered by resource type and clinical date. Select any card to open the full patient record."
    )
  );

  const features = el("div", "mt-5 grid gap-3 sm:grid-cols-3");
  [
    ["Semantic retrieval", "Finds concepts, not just exact keywords."],
    ["FHIR R4 validated", "Every resource is checked against the R4 schema."],
    ["Explainable", "Scores, source snippets and AI summaries on every card."],
  ].forEach(([title, text]) => {
    const item = el("div", "rounded-2xl border border-slate-100 bg-slate-50/70 p-4");
    item.append(el("p", "text-xs font-bold uppercase tracking-wide text-teal-700", title));
    item.append(el("p", "mt-1 text-xs leading-5 text-slate-600", text));
    features.append(item);
  });
  body.append(features);

  const stats = state.health && state.health.stats;
  if (stats) {
    const resources = stats.resources || {};
    body.append(
      el(
        "p",
        "mt-5 text-xs font-medium text-slate-500",
        `${stats.patients} synthetic patients · ${resources.DocumentReference ?? "—"} documents · ${resources.Observation ?? "—"} lab results · ${stats.chunks} indexed chunks`
      )
    );
  }

  const suggestions = el("div", "mt-5 flex flex-wrap items-center gap-2");
  suggestions.append(el("span", "text-xs font-semibold text-slate-500", "Start with:"));
  SUGGESTIONS.slice(0, 4).forEach((query) => {
    const button = el(
      "button",
      "rounded-full border border-slate-200 bg-white px-3 py-1 text-xs font-medium text-slate-600 shadow-sm transition hover:border-teal-300 hover:text-teal-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-500",
      query
    );
    button.type = "button";
    button.addEventListener("click", () => applyQuery(query));
    suggestions.append(button);
  });
  body.append(suggestions);
  wrapper.append(body);
  els.resultGrid.append(wrapper);
}

function renderEmpty(query) {
  const wrapper = el(
    "div",
    "col-span-full rounded-3xl border border-dashed border-slate-300 bg-white p-8 text-center shadow-sm"
  );
  const icon = el(
    "div",
    "mx-auto flex h-12 w-12 items-center justify-center rounded-2xl bg-slate-100 text-slate-500"
  );
  icon.append(svgIcon("m21 21-4.35-4.35M17 11a6 6 0 1 1-12 0 6 6 0 0 1 12 0Z", "h-6 w-6"));
  wrapper.append(icon);
  wrapper.append(el("h3", "mt-4 text-lg font-bold text-slate-900", `No matches for “${query}”`));
  wrapper.append(
    el(
      "p",
      "mx-auto mt-2 max-w-lg text-sm leading-6 text-slate-600",
      state.types.size || state.start || state.end
        ? "Try removing a filter, widening the date range, or using broader clinical language."
        : "Try broader clinical language, a diagnosis, a medication, or a lab name."
    )
  );
  const actions = el("div", "mt-5 flex flex-wrap items-center justify-center gap-2");
  const reset = el(
    "button",
    "rounded-full bg-teal-600 px-4 py-2 text-xs font-semibold text-white shadow-sm transition hover:bg-teal-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2",
    "Reset filters"
  );
  reset.type = "button";
  reset.addEventListener("click", () => {
    clearFilters();
    scheduleSearch(true);
  });
  actions.append(reset);
  SUGGESTIONS.slice(1, 4).forEach((suggestion) => {
    const button = el(
      "button",
      "rounded-full border border-slate-200 bg-white px-4 py-2 text-xs font-semibold text-slate-600 transition hover:border-teal-300 hover:text-teal-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-500",
      suggestion
    );
    button.type = "button";
    button.addEventListener("click", () => applyQuery(suggestion));
    actions.append(button);
  });
  wrapper.append(actions);
  els.resultGrid.append(wrapper);
}

function renderError(message) {
  els.resultGrid.replaceChildren();
  els.resultMeta.textContent = "Search failed";
  const wrapper = el(
    "div",
    "col-span-full rounded-3xl border border-rose-200 bg-rose-50 p-6 text-left shadow-sm"
  );
  wrapper.setAttribute("role", "alert");
  wrapper.append(el("h3", "text-base font-bold text-rose-900", "Search could not be completed"));
  wrapper.append(el("p", "mt-1 text-sm leading-6 text-rose-800", message));
  const retry = el(
    "button",
    "mt-4 rounded-full bg-rose-700 px-4 py-2 text-xs font-semibold text-white transition hover:bg-rose-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-2",
    "Try again"
  );
  retry.type = "button";
  retry.addEventListener("click", () => doSearch(state.query));
  wrapper.append(retry);
  els.resultGrid.append(wrapper);
  els.loadingAnnouncer.textContent = "Search failed";
}

function clearFilters() {
  state.types.clear();
  state.start = "";
  state.end = "";
  els.startDate.value = "";
  els.endDate.value = "";
  renderFilterState();
}

function applyQuery(query) {
  els.searchInput.value = query;
  updateClearButton();
  els.searchInput.focus();
  scheduleSearch(true);
}

function handleGridKeydown(event) {
  const cards = Array.from(els.resultGrid.querySelectorAll("[data-result-card]"));
  if (!cards.length) return;
  const current = document.activeElement.closest("[data-result-card]");
  if (!current) return;
  const index = cards.indexOf(current);
  if (event.key === "ArrowDown" || event.key === "ArrowRight") {
    event.preventDefault();
    cards[Math.min(index + 1, cards.length - 1)].focus();
  } else if (event.key === "ArrowUp" || event.key === "ArrowLeft") {
    event.preventDefault();
    cards[Math.max(index - 1, 0)].focus();
  } else if (event.key === "Home") {
    event.preventDefault();
    cards[0].focus();
  } else if (event.key === "End") {
    event.preventDefault();
    cards[cards.length - 1].focus();
  }
}

function setDrawerVisibility(open) {
  if (open) {
    els.drawerRoot.classList.remove("hidden");
    els.drawerRoot.setAttribute("aria-hidden", "false");
    document.body.classList.add("overflow-hidden");
    window.requestAnimationFrame(() => {
      els.drawerOverlay.classList.remove("opacity-0");
      els.drawerOverlay.classList.add("opacity-100");
      els.drawer.classList.remove("translate-x-full");
      els.drawer.classList.add("translate-x-0");
    });
  } else {
    els.drawerOverlay.classList.remove("opacity-100");
    els.drawerOverlay.classList.add("opacity-0");
    els.drawer.classList.remove("translate-x-0");
    els.drawer.classList.add("translate-x-full");
    els.drawerRoot.setAttribute("aria-hidden", "true");
    document.body.classList.remove("overflow-hidden");
    window.setTimeout(() => {
      if (els.drawerRoot.getAttribute("aria-hidden") === "true") {
        els.drawerRoot.classList.add("hidden");
      }
    }, 220);
  }
}

async function openDrawer(patientId, trigger) {
  state.lastTrigger = trigger || document.activeElement;
  state.activeTab = "summary";
  state.detail = null;
  state.panels = {};
  els.drawerTitle.textContent = "Loading patient record…";
  els.drawerMeta.textContent = "";
  els.drawerTabsWrap.classList.add("hidden");
  renderDrawerSkeleton();
  setDrawerVisibility(true);
  window.setTimeout(() => els.drawerClose.focus(), 60);

  if (state.detailController) state.detailController.abort();
  const controller = new AbortController();
  state.detailController = controller;
  try {
    const response = await fetch(`/patients/${encodeURIComponent(patientId)}`, {
      signal: controller.signal,
    });
    const data = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(data.detail || `Request failed (${response.status})`);
    state.detail = data;
    renderDetail(data);
  } catch (error) {
    if (error.name === "AbortError") return;
    renderDrawerError(error.message || "Could not load the patient record");
  }
}

function closeDrawer() {
  if (state.detailController) state.detailController.abort();
  setDrawerVisibility(false);
  const trigger = state.lastTrigger;
  state.lastTrigger = null;
  if (trigger && typeof trigger.focus === "function") {
    window.setTimeout(() => trigger.focus(), 30);
  }
}

function renderDrawerSkeleton() {
  const wrapper = el("div", "space-y-4");
  const card = el("div", "animate-pulse rounded-2xl border border-slate-200 bg-white p-5");
  card.append(el("div", "h-4 w-32 rounded bg-slate-100"));
  card.append(el("div", "mt-3 h-3 w-full rounded bg-slate-100"));
  card.append(el("div", "mt-2 h-3 w-11/12 rounded bg-slate-100"));
  card.append(el("div", "mt-2 h-3 w-4/5 rounded bg-slate-100"));
  wrapper.append(card);
  els.drawerContent.replaceChildren(wrapper);
}

function renderDrawerError(message) {
  els.drawerTitle.textContent = "Record unavailable";
  els.drawerMeta.textContent = "";
  const wrapper = el("div", "rounded-2xl border border-rose-200 bg-rose-50 p-5");
  wrapper.setAttribute("role", "alert");
  wrapper.append(el("h3", "text-base font-bold text-rose-900", "Could not load this patient record"));
  wrapper.append(el("p", "mt-1 text-sm leading-6 text-rose-800", message));
  const retry = el(
    "button",
    "mt-4 rounded-full bg-rose-700 px-4 py-2 text-xs font-semibold text-white transition hover:bg-rose-800 focus:outline-none focus-visible:ring-2 focus-visible:ring-rose-500 focus-visible:ring-offset-2",
    "Retry"
  );
  retry.type = "button";
  retry.addEventListener("click", () => openDrawer(state.lastTrigger?.dataset?.patientId, state.lastTrigger));
  wrapper.append(retry);
  els.drawerContent.replaceChildren(wrapper);
}

function renderDetail(data) {
  const patient = data.patient || {};
  const name = patient.display_name || [patient.given_name, patient.family_name].filter(Boolean).join(" ") || "Unknown patient";
  const age = ageFromBirthDate(patient.birth_date);
  const gender = patient.gender && patient.gender !== "unknown" ? patient.gender : null;
  const metaParts = [formatMrn(patient.mrn)];
  if (age !== null) metaParts.push(`${age} years`);
  if (gender) metaParts.push(gender);
  if (data.documents && data.documents.length) {
    metaParts.push(`${data.documents.length} document${data.documents.length === 1 ? "" : "s"}`);
  }
  els.drawerTitle.textContent = name;
  els.drawerMeta.textContent = metaParts.join(" · ");
  els.drawerTabsWrap.classList.remove("hidden");
  $("document-count").textContent = String((data.documents || []).length);
  $("resource-count").textContent = String((data.resources || []).length);

  state.panels = {
    summary: buildSummaryPanel(data),
    documents: buildDocumentsPanel(data),
    resources: buildResourcesPanel(data),
  };
  els.drawerContent.replaceChildren(
    state.panels.summary,
    state.panels.documents,
    state.panels.resources
  );
  activateTab(state.activeTab || "summary", false);
}

function validationBanner(data) {
  const validation = data.validation;
  if (!validation) {
    const banner = el(
      "div",
      "rounded-2xl border border-slate-200 bg-slate-50 p-4 text-sm text-slate-600"
    );
    banner.append(el("p", "font-semibold text-slate-800", "FHIR validation not available"));
    banner.append(el("p", "mt-1 text-xs", "The Bundle has not been validated for this record."));
    return banner;
  }
  const valid = Boolean(validation.valid);
  const banner = el(
    "div",
    valid
      ? "rounded-2xl border border-emerald-200 bg-emerald-50 p-4"
      : "rounded-2xl border border-amber-200 bg-amber-50 p-4"
  );
  const heading = el(
    "p",
    valid
      ? "flex items-center gap-2 text-sm font-bold text-emerald-900"
      : "flex items-center gap-2 text-sm font-bold text-amber-900"
  );
  heading.append(
    svgIcon(
      valid ? "m5 13 4 4L19 7" : "M12 9v4m0 4h.01M10.3 3.9 2.4 17.5A2 2 0 0 0 4.1 20h15.8a2 2 0 0 0 1.7-2.5L13.7 3.9a2 2 0 0 0-3.4 0Z",
      "h-4 w-4"
    )
  );
  const issues = Array.isArray(validation.issues) ? validation.issues : [];
  const errors =
    validation.error_count !== undefined
      ? validation.error_count
      : issues.filter((issue) => issue.severity === "error").length;
  const warnings =
    validation.warning_count !== undefined
      ? validation.warning_count
      : issues.filter((issue) => issue.severity === "warning").length;
  heading.append(
    document.createTextNode(
      valid ? "FHIR R4 validated" : `FHIR R4 validation found ${errors} error(s)`
    )
  );
  banner.append(heading);
  banner.append(
    el(
      "p",
      valid ? "mt-1 text-xs text-emerald-800" : "mt-1 text-xs text-amber-800",
      `${validation.bundle_id || "Bundle"} · ${errors} errors · ${warnings} warnings`
    )
  );
  return banner;
}

function buildSummaryPanel(data) {
  const panel = el("div", "space-y-4");
  panel.id = "panel-summary";
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", "tab-summary");
  panel.tabIndex = 0;
  panel.append(validationBanner(data));

  const summary = data.summary;
  if (!summary) {
    const empty = el("div", "rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-center");
    empty.append(el("p", "text-sm font-semibold text-slate-800", "No AI summary available"));
    empty.append(el("p", "mt-1 text-xs text-slate-500", "The summary cache may still be warming up."));
    panel.append(empty);
    return panel;
  }

  const card = el("section", "rounded-2xl border border-slate-200 bg-white p-5 shadow-sm");
  const header = el("div", "flex flex-wrap items-start justify-between gap-3");
  const title = el("div");
  title.append(el("p", "text-[11px] font-bold uppercase tracking-[0.16em] text-teal-700", "AI clinical summary"));
  title.append(el("h3", "mt-1 text-lg font-bold tracking-tight text-slate-900", "Chief concern and clinical picture"));
  header.append(title);
  const confidence = el(
    "span",
    `rounded-full px-2.5 py-1 text-[11px] font-bold uppercase tracking-wide ring-1 ring-inset ${confidenceMeta(summary.confidence_label)}`,
    `${summary.confidence_label || "moderate"} · ${Math.round(Number(summary.confidence || 0) * 100)}%`
  );
  header.append(confidence);
  card.append(header);
  card.append(el("p", "mt-4 text-sm leading-7 text-slate-700", summary.summary || ""));
  card.append(
    el(
      "p",
      "mt-3 text-[11px] text-slate-400",
      `${summary.provider || "unknown"} · ${summary.model || "model"} · ${summary.word_count || 0} words${summary.cached ? " · cached" : ""}`
    )
  );

  const sections = summary.sections || {};
  const grid = el("div", "mt-5 grid gap-3 sm:grid-cols-2");
  grid.append(
    sectionBlock("Chief concern", sections.chief_concern || "Not documented", "text-sm leading-6 text-slate-700")
  );
  grid.append(sectionBlock("Diagnoses", listChips(sections.diagnoses || [])));
  grid.append(sectionBlock("Recent media", listItems(sections.recent_media || [], "text-xs leading-5 text-slate-600")));
  grid.append(sectionBlock("Flagged anomalies", listItems(sections.flagged_anomalies || [], "text-xs leading-5 text-amber-800", true)));
  card.append(grid);

  const disclaimer = el("div", "mt-4 rounded-xl border border-amber-200 bg-amber-50 p-3");
  disclaimer.append(el("p", "text-[11px] font-bold uppercase tracking-wide text-amber-800", "Decision support only"));
  disclaimer.append(el("p", "mt-1 text-xs leading-5 text-amber-800", summary.disclaimer || ""));
  card.append(disclaimer);
  panel.append(card);

  const flags = data.patient && data.patient.data_quality_flags;
  if (Array.isArray(flags) && flags.length) {
    const flagCard = el("section", "rounded-2xl border border-slate-200 bg-white p-5 shadow-sm");
    flagCard.append(el("h3", "text-sm font-bold text-slate-900", "Data quality flags"));
    const list = el("ul", "mt-3 space-y-2");
    flags.slice(0, 8).forEach((flag) => {
      const item = el("li", "flex gap-2 text-xs leading-5 text-slate-600");
      const dot =
        flag.severity === "critical"
          ? "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-rose-500"
          : flag.severity === "warning"
            ? "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500"
            : "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-400";
      item.append(el("span", dot));
      item.append(el("span", "", flag.message));
      list.append(item);
    });
    flagCard.append(list);
    panel.append(flagCard);
  }
  return panel;
}

function sectionBlock(title, content, contentClass) {
  const block = el("div", "rounded-xl border border-slate-100 bg-slate-50/70 p-4");
  block.append(el("p", "text-[10px] font-bold uppercase tracking-[0.16em] text-slate-500", title));
  if (typeof content === "string") {
    block.append(el("p", `mt-1.5 ${contentClass || "text-xs leading-5 text-slate-600"}`, content));
  } else if (content) {
    block.append(content);
  }
  return block;
}

function listChips(values) {
  if (!values.length) {
    return el("p", "mt-1.5 text-xs leading-5 text-slate-500", "None documented");
  }
  const wrapper = el("div", "mt-2 flex flex-wrap gap-1.5");
  values.forEach((value) => {
    wrapper.append(
      el("span", "rounded-full bg-white px-2 py-1 text-[11px] font-medium text-slate-700 ring-1 ring-inset ring-slate-200", value)
    );
  });
  return wrapper;
}

function listItems(values, className, warning = false) {
  if (!values.length) {
    return el("p", "mt-1.5 text-xs leading-5 text-slate-500", "None");
  }
  const list = el("ul", "mt-1.5 space-y-1");
  values.slice(0, 8).forEach((value) => {
    const item = el("li", `flex gap-2 ${className || "text-xs leading-5 text-slate-600"}`);
    item.append(
      el(
        "span",
        warning
          ? "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-amber-500"
          : "mt-1.5 h-1.5 w-1.5 shrink-0 rounded-full bg-slate-400"
      )
    );
    item.append(el("span", "", value));
    list.append(item);
  });
  return list;
}

function buildDocumentsPanel(data) {
  const panel = el("div", "space-y-3");
  panel.id = "panel-documents";
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", "tab-documents");
  panel.tabIndex = 0;
  panel.hidden = true;

  const documents = data.documents || [];
  if (!documents.length) {
    panel.append(el("p", "rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500", "No clinical documents for this patient."));
    return panel;
  }

  documents.forEach((document, index) => {
    const details = el("details", "group rounded-2xl border border-slate-200 bg-white shadow-sm open:shadow-md");
    details.open = index === 0;
    const summary = el("summary", "flex cursor-pointer list-none items-start justify-between gap-3 p-4 focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-500 focus-visible:ring-offset-2");
    const left = el("div", "min-w-0");
    left.append(el("p", "text-sm font-bold text-slate-900", document.title || "Clinical document"));
    left.append(
      el(
        "p",
        "mt-1 text-xs text-slate-500",
        `${typeMeta("DocumentReference").label} · ${formatDate(document.authored_on)}${document.author ? ` · ${document.author}` : ""} · ${document.word_count || 0} words`
      )
    );
    summary.append(left);
    const chevron = svgIcon("m6 9 6 6 6-6", "h-4 w-4 shrink-0 text-slate-400 transition group-open:rotate-180");
    summary.append(chevron);
    details.append(summary);

    const body = el("div", "border-t border-slate-100 px-4 pb-4 pt-3");
    body.append(el("p", "whitespace-pre-line text-sm leading-6 text-slate-700", document.text || ""));
    if (Array.isArray(document.anomalies) && document.anomalies.length) {
      const flags = el("div", "mt-3 flex flex-wrap gap-1.5");
      document.anomalies.forEach((anomaly) => {
        flags.append(
          el(
            "span",
            "rounded-full bg-amber-50 px-2 py-1 text-[10px] font-semibold text-amber-800 ring-1 ring-inset ring-amber-600/20",
            anomaly.description || anomaly.code
          )
        );
      });
      body.append(flags);
    }
    details.append(body);
    panel.append(details);
  });
  return panel;
}

function resourceTitle(resource) {
  if (!resource) return "FHIR resource";
  if (resource.resourceType === "DocumentReference") {
    return resource.description || resource.type?.text || "Clinical document";
  }
  if (resource.resourceType === "Observation") {
    return resource.code?.text || resource.code?.coding?.[0]?.display || "Observation";
  }
  if (resource.resourceType === "DiagnosticReport") {
    return resource.code?.text || "Diagnostic report";
  }
  if (resource.resourceType === "Encounter") {
    return resource.type?.[0]?.text || resource.class?.display || "Encounter";
  }
  if (resource.resourceType === "Patient") {
    return resource.name?.[0]?.text || [resource.name?.[0]?.given?.join(" "), resource.name?.[0]?.family].filter(Boolean).join(" ") || "Patient";
  }
  return resource.resourceType || "FHIR resource";
}

function resourceDate(resource) {
  return (
    resource?.date ||
    resource?.effectiveDateTime ||
    resource?.issued ||
    resource?.period?.start ||
    null
  );
}

function sanitizeForDisplay(resource) {
  const clone = JSON.parse(JSON.stringify(resource));
  if (clone.resourceType === "DocumentReference" && Array.isArray(clone.content)) {
    clone.content.forEach((item) => {
      if (item.attachment && typeof item.attachment.data === "string") {
        item.attachment.data = `[base64 omitted: ${item.attachment.data.length} chars]`;
      }
    });
  }
  return clone;
}

function buildResourcesPanel(data) {
  const panel = el("div", "space-y-4");
  panel.id = "panel-resources";
  panel.setAttribute("role", "tabpanel");
  panel.setAttribute("aria-labelledby", "tab-resources");
  panel.tabIndex = 0;
  panel.hidden = true;

  const resources = data.resources || [];
  if (!resources.length) {
    panel.append(el("p", "rounded-2xl border border-dashed border-slate-300 bg-white p-6 text-center text-sm text-slate-500", "No linked FHIR resources."));
    return panel;
  }

  const groups = new Map();
  resources.forEach((resource) => {
    const type = resource.resourceType || "Resource";
    if (!groups.has(type)) groups.set(type, []);
    groups.get(type).push(resource);
  });

  groups.forEach((items, type) => {
    const section = el("section", "rounded-2xl border border-slate-200 bg-white p-4 shadow-sm");
    const header = el("div", "flex items-center justify-between gap-2");
    header.append(badge(type));
    header.append(el("span", "text-xs font-semibold text-slate-500", `${items.length} resource${items.length === 1 ? "" : "s"}`));
    section.append(header);
    const list = el("div", "mt-3 space-y-2");
    items.forEach((resource) => {
      const card = el("div", "rounded-xl border border-slate-100 bg-slate-50/70 p-3");
      const top = el("div", "flex flex-wrap items-start justify-between gap-2");
      const left = el("div", "min-w-0");
      left.append(el("p", "truncate text-xs font-bold text-slate-800", resourceTitle(resource)));
      left.append(
        el(
          "p",
          "mt-0.5 font-mono text-[10px] text-slate-400",
          `${resource.resourceType}/${resource.id || "—"}${resource.status ? ` · ${resource.status}` : ""}${resourceDate(resource) ? ` · ${formatDate(resourceDate(resource))}` : ""}`
        )
      );
      top.append(left);
      const toggle = el(
        "button",
        "rounded-lg border border-slate-200 bg-white px-2 py-1 text-[10px] font-semibold text-slate-600 transition hover:border-teal-300 hover:text-teal-700 focus:outline-none focus-visible:ring-2 focus-visible:ring-teal-500",
        "JSON"
      );
      toggle.type = "button";
      toggle.setAttribute("aria-expanded", "false");
      top.append(toggle);
      card.append(top);
      const pre = el("pre", "fhir-json hidden");
      toggle.addEventListener("click", () => {
        const expanded = toggle.getAttribute("aria-expanded") === "true";
        toggle.setAttribute("aria-expanded", String(!expanded));
        if (!expanded && !pre.textContent) {
          pre.textContent = JSON.stringify(sanitizeForDisplay(resource), null, 2);
        }
        pre.classList.toggle("hidden", expanded);
        toggle.textContent = expanded ? "JSON" : "Hide";
      });
      card.append(pre);
      list.append(card);
    });
    section.append(list);
    panel.append(section);
  });
  return panel;
}

function activateTab(tab, focus) {
  const tabs = Array.from(document.querySelectorAll("#drawer-tabs [role='tab']"));
  tabs.forEach((button) => {
    const selected = button.dataset.tab === tab;
    button.setAttribute("aria-selected", String(selected));
    button.tabIndex = selected ? 0 : -1;
    if (selected && focus) button.focus();
  });
  Object.entries(state.panels).forEach(([name, panel]) => {
    if (panel) panel.hidden = name !== tab;
  });
  state.activeTab = tab;
}

function handleTabKeydown(event) {
  const tabs = Array.from(document.querySelectorAll("#drawer-tabs [role='tab']"));
  const current = tabs.indexOf(document.activeElement);
  if (current < 0) return;
  let next = null;
  if (event.key === "ArrowRight") next = (current + 1) % tabs.length;
  if (event.key === "ArrowLeft") next = (current - 1 + tabs.length) % tabs.length;
  if (event.key === "Home") next = 0;
  if (event.key === "End") next = tabs.length - 1;
  if (next === null) return;
  event.preventDefault();
  activateTab(tabs[next].dataset.tab, true);
}

function trapFocus(event) {
  if (event.key !== "Tab") return;
  const focusable = Array.from(
    els.drawer.querySelectorAll(
      "a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, [tabindex]:not([tabindex='-1'])"
    )
  ).filter((node) => node.offsetParent !== null);
  if (!focusable.length) return;
  const first = focusable[0];
  const last = focusable[focusable.length - 1];
  if (event.shiftKey && document.activeElement === first) {
    event.preventDefault();
    last.focus();
  } else if (!event.shiftKey && document.activeElement === last) {
    event.preventDefault();
    first.focus();
  }
}

function bindEvents() {
  els.searchForm.addEventListener("submit", (event) => {
    event.preventDefault();
    scheduleSearch(true);
  });
  els.searchInput.addEventListener("input", () => {
    updateClearButton();
    scheduleSearch(false);
  });
  els.clearSearch.addEventListener("click", () => {
    els.searchInput.value = "";
    updateClearButton();
    state.query = "";
    state.results = [];
    updateUrl();
    renderWelcome();
    els.searchInput.focus();
  });

  document.querySelectorAll("#query-hints [data-query]").forEach((button) => {
    button.addEventListener("click", () => applyQuery(button.dataset.query || ""));
  });

  document.querySelectorAll("#type-filters [data-type]").forEach((button) => {
    button.addEventListener("click", () => {
      const type = button.dataset.type;
      if (type === "all") {
        state.types.clear();
      } else if (state.types.has(type)) {
        state.types.delete(type);
      } else {
        state.types.add(type);
      }
      renderFilterState();
      if (state.query) scheduleSearch(true);
    });
  });

  els.startDate.addEventListener("change", () => {
    state.start = els.startDate.value;
    if (state.query) scheduleSearch(true);
    else updateUrl();
  });
  els.endDate.addEventListener("change", () => {
    state.end = els.endDate.value;
    if (state.query) scheduleSearch(true);
    else updateUrl();
  });
  els.resetFilters.addEventListener("click", () => {
    clearFilters();
    if (state.query) scheduleSearch(true);
    else updateUrl();
  });

  els.resultGrid.addEventListener("keydown", handleGridKeydown);

  els.drawerClose.addEventListener("click", closeDrawer);
  els.drawerOverlay.addEventListener("click", closeDrawer);
  els.drawer.addEventListener("keydown", trapFocus);
  document.querySelectorAll("#drawer-tabs [role='tab']").forEach((button) => {
    button.addEventListener("click", () => activateTab(button.dataset.tab, false));
  });
  els.drawerTabs.addEventListener("keydown", handleTabKeydown);

  document.addEventListener("keydown", (event) => {
    const target = document.activeElement;
    const tag = target ? target.tagName : "";
    const typing =
      ["INPUT", "TEXTAREA", "SELECT"].includes(tag) || (target && target.isContentEditable);
    if (event.key === "/" && !typing && els.drawerRoot.getAttribute("aria-hidden") === "true") {
      event.preventDefault();
      els.searchInput.focus();
      els.searchInput.select();
    }
    if (event.key === "Escape" && els.drawerRoot.getAttribute("aria-hidden") === "false") {
      event.preventDefault();
      closeDrawer();
    }
  });

  window.addEventListener("popstate", () => {
    state.types.clear();
    state.start = "";
    state.end = "";
    hydrateFromUrl();
  });
}

function init() {
  Object.assign(els, {
    searchForm: $("search-form"),
    searchInput: $("search-input"),
    clearSearch: $("clear-search"),
    startDate: $("start-date"),
    endDate: $("end-date"),
    resetFilters: $("reset-filters"),
    resultGrid: $("result-grid"),
    resultMeta: $("result-meta"),
    loadingAnnouncer: $("loading-announcer"),
    indexStatus: $("index-status"),
    drawerRoot: $("drawer-root"),
    drawerOverlay: $("drawer-overlay"),
    drawer: $("drawer"),
    drawerClose: $("drawer-close"),
    drawerTitle: $("drawer-title"),
    drawerMeta: $("drawer-meta"),
    drawerTabsWrap: $("drawer-tabs-wrap"),
    drawerTabs: $("drawer-tabs"),
    drawerContent: $("drawer-content"),
  });
  renderFilterState();
  bindEvents();
  loadHealth();
  hydrateFromUrl();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init);
} else {
  init();
}
