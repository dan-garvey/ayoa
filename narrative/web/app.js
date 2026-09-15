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

async function api(path, body) {
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
  const prose = node("div", "prose");
  // Only the server's HTML-disabled Markdown renderer supplies these fragments.
  prose.innerHTML = html;
  for (const link of prose.querySelectorAll("a")) {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  article.append(meta, prose);
  return article;
}

function confirmSubmission(view) {
  const saved = storage.get(`submission:${view.id}`);
  if (!saved) return;
  let submission;
  try { submission = JSON.parse(saved); } catch { storage.remove(`submission:${view.id}`); return; }
  const accepted = (view.pending && view.turn_count === submission.expected_turns && view.pending.input === submission.text)
    || view.turns[submission.expected_turns]?.input === submission.text;
  if (accepted) {
    if (storage.get(`draft:${view.id}`) === submission.text) storage.remove(`draft:${view.id}`);
    storage.remove(`submission:${view.id}`);
    if (selected === view.id && input.value === submission.text) { input.value = ""; resizeInput(); }
  }
}

function renderView(view) {
  if (view.id !== selected) return;
  if (current && view.turn_count < current.turn_count) return;
  const previousCount = current?.turn_count || 0;
  current = view;
  confirmSubmission(view);
  $("welcome").hidden = true;
  $("conversation").hidden = false;
  $("composer-area").hidden = false;
  $("story-title").textContent = view.title;
  $("story-kicker").textContent = `PLAYING AS ${view.player.toLocaleUpperCase()}`;
  document.title = `${view.title} · Ayoa`;
  const signature = JSON.stringify([view.id, view.turns, view.pending?.input]);
  if (signature !== rendered) {
    const initial = !rendered;
    const follow = nearBottom() || initial;
    const oldTop = reader.scrollTop;
    rendered = signature;
    const conversation = $("conversation");
    conversation.replaceChildren();
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
      const story = message("story", turn.output_html, "THE STORY");
      const tools = node("div", "message-tools");
      const copyButton = node("button", "copy-passage", "Copy passage");
      copyButton.setAttribute("aria-label", `Copy passage ${turn.number + 1}`);
      copyButton.addEventListener("click", () => copy(turn.output));
      tools.append(copyButton, node("span", "passage-number", `Passage ${turn.number + 1}`));
      story.append(tools);
      conversation.append(player, story, node("div", "turn-divider"));
    }
    if (view.pending) conversation.append(message("player", view.pending.input_html, "YOU"));
    if (view.pending && sending.has(view.id)) reader.scrollTop = reader.scrollHeight;
    else if (follow && view.turn_count && (initial || view.turn_count > previousCount)) goLatest();
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
  $("export").disabled = !current.turn_count || waiting;
  $("rename-player").disabled = waiting;
  $("response-status").hidden = !waiting;
  $("status-chip").hidden = false;
  $("activity-dot").classList.toggle("active", busy);
  $("resume").hidden = !pending || busy || pending.handoff_ready;
  $("handoff").hidden = !pending?.handoff_ready || busy;
  let status = "Your turn";
  if (busy) status = pending?.stage === "revision" ? "Refining the passage…" : "Writing the next passage…";
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
  storage.set("selected", id);
  history.replaceState(null, "", `#${encodeURIComponent(id)}`);
  notice("");
  toggleSidebar(false);
  await refresh();
  resizeInput();
}

async function sendTurn(text = input.value) {
  if (!current || current.pending || current.busy || sending.has(selected) || !text.trim()) return;
  const id = selected;
  const submission = { id, text, expected_turns: current.turn_count, expected_version: current.version };
  input.value = text;
  storage.set(`draft:${id}`, text);
  storage.set(`submission:${id}`, JSON.stringify(submission));
  sending.add(id);
  notice("");
  updateControls();
  const request = api("/api/turn", { id, text, expected_version: submission.expected_version });
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
  $("handoff-stage").textContent = packet.stage === "revision" ? "FINAL RESPONSE HANDOFF" : "DRAFT RESPONSE HANDOFF";
}
async function openHandoff() {
  try {
    await loadHandoff(selected);
    $("handoff-error").hidden = true;
    $("handoff-note").textContent = "Only the final response becomes part of the published story.";
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
    if (view.pending?.handoff_ready) {
      await loadHandoff(packet.id);
      $("handoff-note").textContent = "Draft received. Complete this final response to publish the passage.";
      $("copy-request").focus();
    } else {
      $("handoff-dialog").close();
      if (view.pending) toast("Response saved. Continue the unfinished turn.");
      else toast("Passage saved");
    }
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
input.addEventListener("input", () => { if (selected) storage.set(`draft:${selected}`, input.value); resizeInput(); updateControls(); });
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && (event.ctrlKey || event.metaKey) && !event.isComposing) { event.preventDefault(); sendTurn(); }
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
