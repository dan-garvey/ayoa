"use strict";

const $ = (id) => document.getElementById(id);
const reader = $("reader");
const input = $("turn-text");
const storage = {
  get(key) { try { return localStorage.getItem(`ayoa:${key}`); } catch { return null; } },
  set(key, value) { try { localStorage.setItem(`ayoa:${key}`, value); } catch { /* The open composer still retains the text. */ } },
  remove(key) { try { localStorage.removeItem(`ayoa:${key}`); } catch { /* Storage may be disabled. */ } },
};
let token = "";
let selected = null;
let current = null;
let rendered = "";
let sessionSignature = "";
let storyChoices = [];
let handoffPacket = null;
let renameTarget = null;
let refreshing = false;
let refreshAgain = false;
let toastTimer;
const sending = new Set();
const mobile = matchMedia("(max-width: 700px)");
const compareParam = new URLSearchParams(location.search).get("compare");
let comparing = compareParam === null ? storage.get("compare") === "true" : compareParam === "1";

async function api(path, body) {
  if (comparing) {
    const url = new URL(path, location.origin);
    url.searchParams.set("compare", "1");
    path = url.pathname + url.search;
  }
  const response = await fetch(path, {
    method: body === undefined ? "GET" : "POST",
    headers: body === undefined ? {} : { "Content-Type": "application/json", "X-Chat-Token": token },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: "no-store",
  });
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || "Could not finish this request.");
  return result;
}

function query(path, id = selected) { return `${path}?id=${encodeURIComponent(id)}`; }
function node(tag, className, text) {
  const el = document.createElement(tag);
  if (className) el.className = className;
  if (text !== undefined) el.textContent = text;
  return el;
}
function notice(message) {
  $("notice-text").textContent = message;
  $("notice").hidden = !message;
}
function toast(message) {
  clearTimeout(toastTimer);
  $("toast").textContent = message;
  $("toast").hidden = false;
  toastTimer = setTimeout(() => { $("toast").hidden = true; }, 2600);
}
async function copy(text) {
  try { await navigator.clipboard.writeText(text); toast("Copied to clipboard"); }
  catch { notice("Clipboard access is unavailable. Select the text to copy it, or download the story."); }
}
function download(text, filename) {
  const url = URL.createObjectURL(new Blob([text], { type: "text/plain;charset=utf-8" }));
  const link = node("a");
  link.href = url;
  link.download = filename;
  link.click();
  setTimeout(() => URL.revokeObjectURL(url), 1000);
}
function resizeInput() {
  input.style.height = "auto";
  input.style.height = `${Math.min(input.scrollHeight, mobile.matches ? 160 : 220)}px`;
}
function nearBottom() { return reader.scrollHeight - reader.scrollTop - reader.clientHeight < 160; }
function latestMessage() {
  return [...$("conversation").querySelectorAll(".message.story")].at(-1);
}
function goLatest() {
  const target = latestMessage();
  if (target) reader.scrollTop += target.getBoundingClientRect().top - reader.getBoundingClientRect().top - 28;
  else reader.scrollTop = reader.scrollHeight;
  $("latest").hidden = true;
}
function updateLatest() {
  const target = latestMessage();
  $("latest").hidden = !current || !target || target.getBoundingClientRect().top < reader.getBoundingClientRect().top + 160;
}
function toggleSidebar(open) {
  const visible = mobile.matches && open;
  $("sidebar").classList.toggle("open", visible);
  $("sidebar").inert = mobile.matches && !visible;
  document.querySelector("main").inert = visible;
  $("scrim").hidden = !visible;
  $("menu").setAttribute("aria-expanded", String(visible));
  if (visible) $("new-story").focus();
}

function renderSessions(sessions) {
  const signature = JSON.stringify([selected, sessions]);
  if (sessionSignature === signature) return;
  sessionSignature = signature;
  const list = $("session-list");
  list.replaceChildren();
  if (!sessions.length) list.append(node("p", "session-empty", "Your stories will find a home here."));
  for (const session of sessions) {
    const button = node("button", "session");
    button.setAttribute("aria-current", session.id === selected ? "page" : "false");
    const symbol = node("span", "session-symbol", "◫");
    symbol.setAttribute("aria-hidden", "true");
    const label = node("span");
    label.append(node("span", "session-name", session.title));
    const date = session.updated_at ? new Date(session.updated_at).toLocaleString(undefined, { month: "short", day: "numeric", hour: "numeric", minute: "2-digit" }) : "";
    label.append(node("span", "session-meta", session.unavailable ? "Unable to read saved files" : `${session.player} · ${session.turn_count} ${session.turn_count === 1 ? "passage" : "passages"}\n${date}`));
    button.append(symbol, label);
    button.addEventListener("click", () => selectSession(session.id));
    list.append(button);
  }
}

function proseContent(html, className = "prose") {
  const prose = node("div", className);
  // Only the server's HTML-disabled Markdown renderer supplies these fragments.
  prose.innerHTML = html;
  for (const link of prose.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return prose;
}

function message(role, html, label) {
  const article = node("article", `message ${role}`);
  article.setAttribute("aria-label", role === "player" ? "Your turn" : "Story passage");
  const meta = node("div", "message-meta");
  if (role === "story") {
    const mark = node("span", "message-mark", "✧");
    mark.setAttribute("aria-hidden", "true");
    meta.append(mark);
  }
  meta.append(node("span", "", label));
  article.append(meta, proseContent(html));
  return article;
}

function passage(text, html, number, role = "story", summaryHtml = null) {
  const label = role === "previous" ? "PREVIOUS VERSION" : role === "replacement" ? "CURRENT VERSION" : "THE STORY";
  const article = message(role, html, label);
  const kind = role === "story" ? "passage" : role === "previous" ? "previous version" : "current version";
  if (role !== "story") article.setAttribute("aria-label", `${label.toLocaleLowerCase()} ${number + 1}`);
  const tools = node("div", "message-tools");
  const copyButton = node("button", "copy-passage", `Copy ${kind}`);
  copyButton.setAttribute("aria-label", `Copy ${kind} ${number + 1}`);
  copyButton.addEventListener("click", () => copy(text));
  tools.append(copyButton, node("span", "passage-number", `Passage ${number + 1}`));
  article.append(tools);
  if (comparing) {
    if (summaryHtml) {
      const details = node("details", "reasoning-summary");
      details.append(node("summary", "", "Reasoning summary"), proseContent(summaryHtml, "prose reasoning-prose"));
      article.append(details);
    } else {
      article.append(node("p", "summary-unavailable", "No reasoning summary was captured for this response."));
    }
  }
  return article;
}

function comparison(turn, number, pending = null) {
  const pair = node("section", `message comparison story${pending ? " pending-comparison" : ""}`);
  pair.setAttribute("aria-label", `Passage ${number + 1} comparison`);
  const previous = pending ? passage(turn.output, turn.output_html, number, "previous", turn.summary_html) : passage(turn.previous, turn.previous_html, number, "previous", turn.previous_summary_html);
  let replacement;
  if (!pending) replacement = passage(turn.output, turn.output_html, number, "replacement", turn.summary_html);
  else {
    replacement = message("replacement", "", "NEW VERSION");
    replacement.querySelector(".prose").append(node("p", "comparison-waiting", pending.failed ? "Regeneration paused. The existing passage is saved." : "Regenerating the passage…"));
  }
  pair.append(previous, replacement);
  return pair;
}

function checkupNotes(checkup) {
  const details = node("details", "backstage-checkup");
  details.append(node("summary", "", `Backstage checkup · before passage ${checkup.before_turn + 1}`));
  details.append(proseContent(checkup.output_html, "prose checkup-prose"));
  if (checkup.summary_html) {
    const reasoning = node("details", "reasoning-summary");
    reasoning.append(node("summary", "", "Checkup reasoning summary"), proseContent(checkup.summary_html, "prose reasoning-prose"));
    details.append(reasoning);
  }
  return details;
}

function setComparison(enabled) {
  comparing = enabled;
  storage.set("compare", String(enabled));
  $("compare").setAttribute("aria-pressed", String(enabled));
  $("comparison-note").hidden = !enabled;
  $("conversation").classList.toggle("comparing", enabled);
  const url = new URL(location.href);
  if (enabled) url.searchParams.set("compare", "1");
  else url.searchParams.delete("compare");
  history.replaceState(null, "", url);
  if (current) {
    renderView({ ...current, compare: enabled });
    refresh();
  }
}

function confirmSubmission(view) {
  const saved = storage.get(`submission:${view.id}`);
  if (!saved) return;
  let submission;
  try { submission = JSON.parse(saved); } catch { storage.remove(`submission:${view.id}`); return; }
  const accepted = submission.submission_id && (view.pending?.submission_id === submission.submission_id
    || view.turns.some((turn) => turn.submission_id === submission.submission_id));
  if (accepted) {
    if (storage.get(`draft:${view.id}`) === submission.text) storage.remove(`draft:${view.id}`);
    storage.remove(`submission:${view.id}`);
    if (selected === view.id && input.value === submission.text) { input.value = ""; resizeInput(); }
  }
}

function renderView(view) {
  if (view.id !== selected) return;
  if (Boolean(view.compare) !== comparing) return;
  if (current && view.turn_count < current.turn_count) return;
  const previousCount = current?.turn_count || 0;
  const replaced = current && view.turn_count === current.turn_count && view.turns.at(-1)?.submission_id !== current.turns.at(-1)?.submission_id;
  current = view;
  confirmSubmission(view);
  $("welcome").hidden = true;
  $("conversation").hidden = false;
  $("composer-area").hidden = false;
  $("story-title").textContent = view.title;
  $("story-kicker").textContent = `PLAYING AS ${view.player.toLocaleUpperCase()}`;
  document.title = `${view.title} · Ayoa`;
  const signature = JSON.stringify([view.id, comparing, view.turns, view.pending, view.checkups, view.checkup_every]);
  if (signature !== rendered) {
    const initial = !rendered;
    const follow = nearBottom() || initial;
    const oldTop = reader.scrollTop;
    rendered = signature;
    const conversation = $("conversation");
    conversation.replaceChildren();
    if (comparing && view.checkup_every !== undefined) {
      conversation.append(node("p", "checkup-cadence", view.checkup_every ? `Backstage checkups: every ${view.checkup_every} player messages. The opening counts; regenerations do not. Notes guide subsequent responses until the next checkup.` : "Backstage checkups are disabled for this story."));
    }
    const checkups = new Map((comparing ? view.checkups || [] : []).map(item => [item.before_turn, item]));
    if (!view.turns.length && !view.pending) {
      const empty = node("div", "start-session");
      empty.append(node("span", "eyebrow", "A FRESH PAGE"), node("h2", "", "Your story is ready."), node("p", "", "Begin with the opening scene, or write your own first turn below."));
      const begin = node("button", "button primary", "Begin story ↗");
      begin.addEventListener("click", () => sendTurn("Begin the story."));
      empty.append(begin);
      conversation.append(empty);
    }
    for (const turn of view.turns) {
      const player = message("player", turn.input_html, "YOU");
      const regenerating = view.pending?.kind === "regenerate" && turn.number === view.turn_count - 1 ? view.pending : null;
      const story = comparing && (typeof turn.previous === "string" || regenerating) ? comparison(turn, turn.number, regenerating) : passage(turn.output, turn.output_html, turn.number, "story", turn.summary_html);
      conversation.append(player);
      if (checkups.has(turn.number)) conversation.append(checkupNotes(checkups.get(turn.number)));
      conversation.append(story, node("div", "turn-divider"));
    }
    if (view.pending) {
      conversation.append(message(view.pending.kind === "regenerate" ? "regeneration" : "player", view.pending.input_html, view.pending.kind === "regenerate" ? "REGENERATION INSTRUCTIONS" : "YOU"));
      if (checkups.has(view.turn_count)) conversation.append(checkupNotes(checkups.get(view.turn_count)));
    }
    if (view.pending && sending.has(view.id)) reader.scrollTop = reader.scrollHeight;
    else if (follow && view.turn_count && (initial || replaced || view.turn_count > previousCount)) goLatest();
    else if (follow) reader.scrollTop = reader.scrollHeight;
    else reader.scrollTop = oldTop;
    updateLatest();
  }
  updateControls();
}

function updateControls() {
  if (!current) return;
  const busy = current.busy || sending.has(selected);
  const pending = current.pending;
  const waiting = Boolean(pending || busy);
  input.disabled = waiting;
  $("send").disabled = waiting || !input.value.trim();
  $("regenerate").disabled = waiting || !current.turn_count || !input.value.trim();
  $("export").disabled = !current.turn_count || waiting;
  $("rename-player").disabled = waiting;
  $("response-status").hidden = !waiting;
  $("status-chip").hidden = false;
  $("activity-dot").classList.toggle("active", busy);
  $("resume").hidden = !pending || busy || pending.handoff_ready;
  $("handoff").hidden = !pending?.handoff_ready || busy;
  let status = "Your turn";
  if (busy) {
    status = pending?.kind === "regenerate" ? "Regenerating the passage…" : "Writing the next passage…";
    if (comparing && pending?.stage === "checkup") status = "Reviewing adherence and story development…";
  }
  else if (pending?.failed) status = "The response paused. Your turn is saved.";
  else if (pending?.handoff_ready) status = "Manual response needed. Open the handoff to add it.";
  else if (pending) status = "A response is unfinished. Your turn is saved.";
  $("response-status-text").textContent = status;
  $("status-chip").textContent = busy ? "Writing…" : pending ? "Waiting" : "Your turn";
  $("composer-prompt").textContent = waiting ? "Your place is saved." : "Your words. Your choices.";
  input.placeholder = waiting ? "Waiting for the next passage…" : "What do you say or do?";
}

async function refresh() {
  if (refreshing) { refreshAgain = true; return; }
  refreshing = true;
  const id = selected;
  const previousView = current;
  try {
    const list = await api("/api/sessions");
    renderSessions(list.sessions);
    if (id) {
      const view = await api(query("/api/session", id));
      if (selected === id && current === previousView) renderView(view);
      else if (selected === id) refreshAgain = true;
    }
  } catch (error) {
    if (selected === id) notice(`${error.message} Your unsent text stays in the composer.`);
  } finally {
    refreshing = false;
    if (refreshAgain) { refreshAgain = false; refresh(); }
  }
}

async function selectSession(id) {
  if (selected && current && !input.disabled) storage.set(`draft:${selected}`, input.value);
  selected = id;
  current = null;
  rendered = "";
  $("conversation").replaceChildren(node("p", "session-empty", "Opening story…"));
  $("conversation").hidden = false;
  $("welcome").hidden = true;
  $("story-title").textContent = "Opening story…";
  $("story-kicker").textContent = "YOUR STORIES";
  $("status-chip").hidden = true;
  $("response-status").hidden = true;
  $("latest").hidden = true;
  $("export").disabled = true;
  $("rename-player").disabled = true;
  input.value = storage.get(`draft:${id}`) || "";
  input.disabled = true;
  $("send").disabled = true;
  $("regenerate").disabled = true;
  storage.set("selected", id);
  history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  notice("");
  toggleSidebar(false);
  await refresh();
  resizeInput();
}

async function sendTurn(text = input.value, kind = "turn") {
  if (!current || current.pending || current.busy || sending.has(selected) || !text.trim()) return;
  if (kind === "regenerate" && !current.turn_count) return;
  const id = selected;
  // UUID v4 also works on plain HTTP over a home network.
  const bytes = crypto.getRandomValues(new Uint8Array(16));
  bytes[6] = (bytes[6] & 0x0f) | 0x40;
  bytes[8] = (bytes[8] & 0x3f) | 0x80;
  const submissionId = Array.from(bytes, byte => byte.toString(16).padStart(2, "0")).join("");
  const submission = { id, text, submission_id: submissionId, expected_version: current.version };
  input.value = text;
  storage.set(`draft:${id}`, text);
  storage.set(`submission:${id}`, JSON.stringify(submission));
  sending.add(id);
  notice("");
  updateControls();
  const request = api(`/api/${kind}`, submission);
  refresh();
  try {
    const view = await request;
    confirmSubmission(view);
    if (selected === id) renderView(view);
  } catch (error) {
    if (selected === id) notice(error.message);
  } finally {
    sending.delete(id);
    await refresh();
    if (selected === id) { updateControls(); if (!input.disabled && !mobile.matches) input.focus(); }
  }
}

async function resume() {
  if (!current || sending.has(selected)) return;
  const id = selected;
  sending.add(id);
  notice("");
  updateControls();
  try { renderView(await api("/api/resume", { id })); }
  catch (error) { if (selected === id) notice(error.message); }
  finally { sending.delete(id); await refresh(); }
}

function newStory() {
  toggleSidebar(false);
  $("new-error").hidden = true;
  $("new-dialog").showModal();
  $("story-choice").focus();
}
function chooseStory() {
  $("new-player").value = storyChoices.find((story) => story.id === $("story-choice").value)?.player || "";
}
async function createStory(event) {
  event.preventDefault();
  $("create-story").disabled = true;
  $("new-error").hidden = true;
  try {
    const view = await api("/api/sessions", { story: $("story-choice").value, player_name: $("new-player").value });
    $("new-dialog").close();
    await selectSession(view.id);
    // Render the confirmed new session even if an older poll was in flight.
    renderView(view);
    sendTurn("Begin the story.");
  } catch (error) {
    $("new-error").textContent = error.message;
    $("new-error").hidden = false;
  } finally { $("create-story").disabled = false; }
}

function openRename() {
  if (!current || current.pending || current.busy || sending.has(selected)) return;
  renameTarget = { id: selected, expected_version: current.version };
  $("rename-name").value = current.player;
  $("rename-error").hidden = true;
  $("rename-dialog").showModal();
  $("rename-name").focus();
  $("rename-name").select();
}
async function renamePlayer(event) {
  event.preventDefault();
  if (!renameTarget) return;
  const target = renameTarget;
  $("save-name").disabled = true;
  $("rename-error").hidden = true;
  try {
    renderView(await api("/api/rename", { ...target, player_name: $("rename-name").value }));
    $("rename-dialog").close();
    toast("Protagonist name saved");
  } catch (error) {
    $("rename-error").textContent = `${error.message} Close and reopen this dialog to try again.`;
    $("rename-error").hidden = false;
  } finally {
    $("save-name").disabled = false;
    await refresh();
  }
}

async function loadHandoff(id) {
  const packet = await api(query("/api/handoff", id));
  handoffPacket = { ...packet, id };
  $("request-text").value = packet.text;
  $("response-text").value = "";
  $("response-file").value = "";
  $("handoff-stage").textContent = packet.stage === "checkup" ? "BACKSTAGE CHECKUP HANDOFF" : packet.kind === "regenerate" ? "REGENERATION HANDOFF" : "RESPONSE HANDOFF";
  $("handoff-note").textContent = packet.stage === "checkup" ? "Accepting these notes prepares the story response. The notes stay out of the story." : "Accepting the response publishes this passage.";
}
async function openHandoff() {
  try {
    await loadHandoff(selected);
    $("handoff-error").hidden = true;
    $("handoff-dialog").showModal();
    $("copy-request").focus();
  } catch (error) { notice(error.message); }
}
async function acceptResponse(event) {
  event.preventDefault();
  if (!handoffPacket || !$("response-text").value.trim()) return;
  const packet = handoffPacket;
  $("accept-response").disabled = true;
  $("handoff-error").hidden = true;
  try {
    const view = await api("/api/accept", { id: packet.id, request_id: packet.request_id, text: $("response-text").value });
    renderView(view);
    $("handoff-dialog").close();
    toast(packet.stage === "checkup" ? "Checkup saved. Open the next handoff for the passage." : "Passage saved");
    await refresh();
  } catch (error) {
    $("handoff-error").textContent = `${error.message} Close and reopen the handoff to load its current request.`;
    $("handoff-error").hidden = false;
  } finally { $("accept-response").disabled = false; }
}

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  storage.set("theme", theme);
  $("theme").textContent = theme === "dark" ? "Use light appearance" : "Use dark appearance";
}
function setSize(size) {
  size = Math.min(26, Math.max(16, size || 20));
  document.documentElement.style.setProperty("--story-size", `${size}px`);
  $("text-size").textContent = String(size);
  storage.set("size", String(size));
  $("smaller").disabled = size <= 16;
  $("larger").disabled = size >= 26;
}

$("menu").addEventListener("click", () => toggleSidebar(!$("sidebar").classList.contains("open")));
$("scrim").addEventListener("click", () => { toggleSidebar(false); $("menu").focus(); });
mobile.addEventListener("change", () => toggleSidebar(false));
document.addEventListener("keydown", (event) => { if (event.key === "Escape") toggleSidebar(false); });
$("new-story").addEventListener("click", newStory);
$("welcome-start").addEventListener("click", newStory);
$("new-form").addEventListener("submit", createStory);
$("story-choice").addEventListener("change", chooseStory);
$("rename-player").addEventListener("click", openRename);
$("rename-form").addEventListener("submit", renamePlayer);
$("composer-form").addEventListener("submit", (event) => { event.preventDefault(); sendTurn(); });
$("regenerate").addEventListener("click", () => sendTurn(input.value, "regenerate"));
input.addEventListener("input", () => { if (selected) storage.set(`draft:${selected}`, input.value); resizeInput(); updateControls(); });
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) { event.preventDefault(); sendTurn(input.value, event.shiftKey ? "regenerate" : "turn"); }
});
$("resume").addEventListener("click", resume);
$("handoff").addEventListener("click", openHandoff);
$("handoff-form").addEventListener("submit", acceptResponse);
$("copy-request").addEventListener("click", () => { if (handoffPacket) copy(handoffPacket.text); });
$("download-request").addEventListener("click", () => { if (handoffPacket) download(handoffPacket.text, `request-${handoffPacket.request_id}.txt`); });
$("response-file").addEventListener("change", async () => {
  const file = $("response-file").files[0];
  if (file) {
    if (file.size > 1_900_000) { $("handoff-error").textContent = "This response file is too large."; $("handoff-error").hidden = false; return; }
    $("response-text").value = await file.text();
  }
});
$("export").addEventListener("click", () => { if (selected) { const link = node("a"); link.href = query("/api/transcript"); link.download = "story.md"; link.click(); } });
$("latest").addEventListener("click", () => goLatest());
reader.addEventListener("scroll", updateLatest);
$("dismiss-notice").addEventListener("click", () => notice(""));
$("theme").addEventListener("click", () => setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark"));
$("compare").addEventListener("click", () => setComparison(!comparing));
$("smaller").addEventListener("click", () => setSize(Number($("text-size").textContent) - 2));
$("larger").addEventListener("click", () => setSize(Number($("text-size").textContent) + 2));
for (const button of document.querySelectorAll("[data-close]")) button.addEventListener("click", () => $(button.dataset.close).close());
for (const dialog of document.querySelectorAll("dialog")) dialog.addEventListener("click", (event) => {
  if (event.target === dialog) {
    const rect = dialog.getBoundingClientRect();
    if (event.clientX < rect.left || event.clientX > rect.right || event.clientY < rect.top || event.clientY > rect.bottom) dialog.close();
  }
});
setTheme(storage.get("theme") || "light");
setSize(Number(storage.get("size")) || 20);
setComparison(comparing);
toggleSidebar(false);
if (/Mac|iPhone|iPad/.test(navigator.platform)) $("send-shortcut").textContent = "⌘";

async function start() {
  try {
    const boot = await api("/api/bootstrap");
    token = boot.token;
    storyChoices = boot.stories;
    $("story-choice").replaceChildren(...storyChoices.map((story) => {
      const option = node("option", "", story.title); option.value = story.id; return option;
    }));
    chooseStory();
    $("create-story").disabled = !storyChoices.length;
    $("mode-note").textContent = boot.automatic ? "The story will respond automatically after each turn." : "Manual mode: you must provide each response through the handoff controls.";
    const list = await api("/api/sessions");
    renderSessions(list.sessions);
    let hash = "";
    try { hash = decodeURIComponent(location.hash.slice(1)); } catch { /* Ignore a malformed bookmark. */ }
    const preferred = hash || storage.get("selected");
    const target = list.sessions.find((session) => session.id === preferred) || list.sessions.find((session) => !session.unavailable);
    if (target) await selectSession(target.id);
    setInterval(() => { if (!document.hidden) refresh(); }, 1600);
    document.addEventListener("visibilitychange", () => { if (!document.hidden) refresh(); });
  } catch (error) { notice(`${error.message} Reload this page when the local chat is available.`); }
}
start();
