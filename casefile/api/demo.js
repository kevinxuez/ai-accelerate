"use strict";

const ACCESS_ROLE = "student";
const DEFAULT_PLACEHOLDER = "Ask the Supervisor for evidence, an argument, coaching, or progress...";
const byId = (id) => document.getElementById(id);
const form = byId("request-form");
const messageInput = byId("message");
const sendButton = byId("send");
const resultArea = document.querySelector(".result-area");
const attachmentInput = byId("attachment");
const removeAttachmentButton = byId("remove-attachment");
let sessionId = newSessionId();

const ingestionPrompt = (side) => `Import the attached DOCX as ${side} evidence. Show every proposed card, flag, exclusion reason, marked span, and provenance before asking for confirmation.`;

function newSessionId() {
  return globalThis.crypto?.randomUUID?.() || `session-${Date.now()}-${Math.random().toString(16).slice(2)}`;
}

function node(tag, className = "", text = null) {
  const element = document.createElement(tag);
  if (className) element.className = className;
  if (text !== null && text !== undefined) element.textContent = String(text);
  return element;
}

function titleCase(value) {
  return String(value || "unknown").replaceAll("_", " ").replace(/\b\w/g, (letter) => letter.toUpperCase());
}

function shortId(value) {
  return value ? String(value).slice(0, 12) : "—";
}

function formatJson(value) {
  return JSON.stringify(value, null, 2);
}

function list(values, className = "clean-list") {
  const target = node("ul", className);
  for (const value of values || []) target.append(node("li", "", value));
  return target;
}

function labeledCard(label, heading) {
  const card = node("article", "artifact-card");
  card.append(node("div", "evidence-label", label), node("h3", "", heading));
  return card;
}

function metadataPills(values) {
  const row = node("div", "pill-list");
  for (const value of values.filter(Boolean)) row.append(node("span", "pill", value));
  return row;
}

function renderMarkedText(text, readSpans = [], emphasisSpans = []) {
  const body = node("p", "marked-body");
  const safeText = String(text || "");
  const boundaries = new Set([0, safeText.length]);
  for (const span of [...readSpans, ...emphasisSpans]) {
    boundaries.add(Math.max(0, Math.min(safeText.length, span.start)));
    boundaries.add(Math.max(0, Math.min(safeText.length, span.end)));
  }
  const points = [...boundaries].sort((left, right) => left - right);
  for (let index = 0; index < points.length - 1; index += 1) {
    const start = points[index];
    const end = points[index + 1];
    if (end <= start) continue;
    const classes = ["source-mark"];
    if (readSpans.some((span) => span.start <= start && span.end >= end)) classes.push("read");
    if (emphasisSpans.some((span) => span.start <= start && span.end >= end)) classes.push("emphasis");
    body.append(node("span", classes.join(" "), safeText.slice(start, end)));
  }
  return body;
}

function renderEvidenceCard(card, index, label = "Confirmed source card") {
  const target = node("article", "evidence-card");
  target.dataset.cardId = card.card_id;
  target.append(node("div", "evidence-label", `${label} ${index}`));
  target.append(node("h3", "", card.header || "Untitled evidence card"));
  if (card.tag) target.append(node("p", "tagline", card.tag));
  if (card.citation) target.append(node("p", "citation", card.citation));
  target.append(metadataPills([
    card.side && `Side: ${titleCase(card.side)}`,
    card.resolution && `Resolution: ${card.resolution}`,
    card.source_filename && `File: ${card.source_filename}`,
    Number.isFinite(card.retrieval_score) && `Score: ${card.retrieval_score.toFixed(3)}`,
  ]));
  if ((card.read_spans || []).length || (card.emphasis_spans || []).length) {
    const legend = node("div", "marking-legend");
    legend.append(node("span", "marking-key read", "Read span"), node("span", "marking-key emphasis", "Emphasis span"));
    target.append(legend);
  }
  target.append(renderMarkedText(card.body, card.read_spans, card.emphasis_spans));
  target.append(node("p", "source-provenance", `Card ID: ${card.card_id}`));
  return target;
}

function renderEvidencePacket(packet) {
  const stack = node("div", "evidence-stack");
  const summary = labeledCard("EvidencePacket", packet.request_summary);
  summary.append(metadataPills([
    packet.side && `Side: ${titleCase(packet.side)}`,
    `Resolution: ${packet.resolution}`,
    packet.empty_result ? "No confirmed matches" : `${packet.cards.length} confirmed card(s)`,
  ]));
  summary.append(node("h4", "", "Queries executed"), list(packet.queries_executed));
  summary.append(node("h4", "", "Confirmed files considered"), list(packet.confirmed_source_files_considered));
  const provenance = packet.provenance;
  summary.append(node("p", "source-provenance", `Ledger v${provenance.ledger_schema_version} · ${provenance.retrieval_backend} · ${provenance.embedding_model} · confirmed only`));
  stack.append(summary);
  packet.cards.forEach((card, index) => stack.append(renderEvidenceCard(card, index + 1)));
  return stack;
}

function renderRulePacket(packet) {
  const stack = node("div", "evidence-stack");
  stack.append(labeledCard("RulePacket", `${packet.event} · ${packet.request_summary}`));
  for (const chunk of packet.chunks) {
    const card = labeledCard(`Rule section ${chunk.section_number}`, chunk.section_title);
    card.append(node("p", "citation", `${chunk.document} · score ${chunk.score.toFixed(3)}`));
    card.append(node("p", "evidence-body", chunk.text));
    stack.append(card);
  }
  return stack;
}

function renderTopicPacket(packet) {
  const card = labeledCard("TopicPacket", packet.event);
  card.append(node("p", "argument-part", packet.resolution));
  card.append(metadataPills([
    packet.provider,
    `Backend: ${packet.backend}`,
    packet.synthetic ? "Synthetic fixture" : "Provider result",
    packet.effective_from && `From ${packet.effective_from}`,
    packet.effective_to && `To ${packet.effective_to}`,
  ]));
  card.append(node("p", "source-provenance", packet.source_ref));
  return card;
}

function renderPreviewCard(card, index) {
  const target = node("article", "evidence-card");
  const disposition = card.indexable ? ((card.flags || []).length ? "flagged" : "indexable") : "excluded";
  target.append(node("div", "evidence-label", `Preview card ${index} · ${disposition}`));
  target.append(node("h3", "", card.header || "Untitled evidence card"));
  if (card.tag) target.append(node("p", "tagline", card.tag));
  if (card.citation) target.append(node("p", "citation", card.citation));
  target.append(metadataPills([
    `Side: ${titleCase(card.side)}`,
    `Evidence: ${titleCase(card.evidence_type)}`,
    ...(card.topic_tags || []).map((tag) => `Topic: ${tag}`),
  ]));
  if ((card.flags || []).length) {
    const warning = node("div", "alert danger");
    warning.append(node("strong", "", "Flags / exclusion reasons"), list(card.flags));
    target.append(warning);
  }
  if (card.explanation) target.append(node("p", "argument-part", card.explanation));
  target.append(renderMarkedText(card.body, card.read_spans, card.emphasis_spans));
  target.append(node("p", "source-provenance", `Proposed card ID: ${card.card_id}`));
  return target;
}

function renderIngestionPreview(preview) {
  const stack = node("div", "ingest-preview");
  const counts = {
    indexable: preview.cards.filter((card) => card.indexable).length,
    flagged: preview.cards.filter((card) => (card.flags || []).length > 0).length,
    excluded: preview.cards.filter((card) => !card.indexable).length,
  };
  const summary = labeledCard("IngestionPreview · confirmation required", preview.source_filename);
  summary.append(metadataPills([
    `Indexable: ${counts.indexable}`,
    `Flagged: ${counts.flagged}`,
    `Excluded: ${counts.excluded}`,
    `Side: ${titleCase(preview.side)}`,
  ]));
  summary.append(node("p", "source-provenance", `SHA-256: ${preview.source_sha256} · job ${preview.job_id}`));
  const provenance = preview.provenance;
  summary.append(node("p", "source-provenance", `Provenance: ${provenance.extraction_backend} extraction · ${provenance.boundary_method} boundaries (${provenance.boundary_prompt}) · ${provenance.labeling_method} labels (${provenance.labeling_prompt}) · ${provenance.model} · unconfirmed`));
  if (preview.warnings.length) {
    const warnings = node("div", "alert notice");
    warnings.append(node("strong", "", "Preview warnings"), list(preview.warnings));
    summary.append(warnings);
  }
  stack.append(summary);
  preview.cards.forEach((card, index) => stack.append(renderPreviewCard(card, index + 1)));
  const confirm = node("button", "confirm-button", "Confirm evidence import");
  confirm.type = "button";
  confirm.dataset.testid = "confirm-ingest-button";
  confirm.addEventListener("click", () => confirmIngestion(preview.confirmation_token, confirm));
  stack.append(confirm);
  return stack;
}

function renderIngestionCommit(result) {
  const card = labeledCard("IngestionCommitResult", result.source_filename);
  card.append(metadataPills([
    `Written: ${result.written_cards}`,
    `Searchable: ${result.searchable_cards}`,
    `Ledger v${result.ledger_schema_version}`,
    result.index_rebuilt && "Retrieval ready",
  ]));
  card.append(node("p", "source-provenance", `Job: ${result.job_id}`));
  return card;
}

function renderArgumentDraft(draft, allArtifacts) {
  const stack = node("div", "evidence-stack");
  const argument = node("article", "argument-card");
  argument.dataset.testid = "argument-draft";
  argument.append(node("div", "evidence-label", `${titleCase(draft.side)} structured ArgumentDraft`));
  argument.append(node("h3", "", draft.title), node("p", "citation", draft.resolution));
  for (const [label, key] of [
    ["Claim", "claim"], ["Warrant", "warrant"], ["Evidence", "evidence"],
    ["Impact", "impact"], ["Resolution link", "resolution_link"], ["Likely response", "likely_response"],
  ]) {
    const sectionData = draft[key];
    const section = node("section", "argument-section");
    const heading = node("div", "argument-section-head");
    heading.append(node("h4", "", label), node("span", `support-badge ${sectionData.support.replaceAll("_", "-")}`, titleCase(sectionData.support)));
    section.append(heading, node("p", "argument-part", sectionData.text));
    if (sectionData.card_ids.length) section.append(node("p", "card-id-list", `Cards: ${sectionData.card_ids.join(", ")}`));
    argument.append(section);
  }
  stack.append(argument);
  if (draft.unsupported_facts.length) {
    const unsupported = node("div", "alert danger");
    unsupported.append(node("strong", "", "Unsupported facts"), list(draft.unsupported_facts));
    stack.append(unsupported);
  }
  const evidenceCards = new Map();
  for (const packet of allArtifacts.filter((artifact) => artifact.artifact_type === "evidence_packet")) {
    for (const card of packet.cards) evidenceCards.set(card.card_id, card);
  }
  if (draft.source_card_ids.length) stack.append(node("div", "evidence-label", "Full confirmed source cards"));
  draft.source_card_ids.forEach((cardId, index) => {
    const card = evidenceCards.get(cardId);
    if (card) stack.append(renderEvidenceCard(card, index + 1));
    else stack.append(node("div", "alert danger", `Cited card ${cardId} is missing from the returned EvidencePacket.`));
  });
  return stack;
}

function renderDrillPlan(plan) {
  const card = labeledCard("DrillPlan", plan.title);
  card.append(metadataPills([
    `${plan.duration_minutes} minutes`, titleCase(plan.side), plan.speech_position,
    `Student: ${plan.student_id}`,
  ]));
  card.append(node("p", "argument-part", plan.personalization_summary));
  card.append(node("h4", "", "Focus"), list(plan.focus));
  card.append(node("h4", "", "Instructions"));
  const ordered = node("ol", "clean-list");
  plan.instructions.forEach((instruction) => ordered.append(node("li", "", instruction)));
  card.append(ordered);
  if (plan.evidence_card_ids.length) card.append(node("p", "card-id-list", `Evidence cards: ${plan.evidence_card_ids.join(", ")}`));
  return card;
}

function renderCoachTurn(turn) {
  const card = node("article", "coach-card");
  card.append(node("div", "evidence-label", "Simulated coach · not a human coach"));
  card.append(node("h3", "", `${turn.speech_position} · ${turn.focus}`));
  card.append(node("p", "argument-part", turn.feedback));
  card.append(node("h4", "", "Your next question"), node("p", "argument-part", turn.question));
  if (turn.evidence_card_ids.length) card.append(node("p", "card-id-list", `Evidence cards: ${turn.evidence_card_ids.join(", ")}`));
  return card;
}

function renderProgressSummary(summary) {
  const stack = node("div", "record-stack");
  const overview = labeledCard("ProgressSummary", `Student ${summary.student_id}`);
  overview.append(node("p", "argument-part", summary.summary));
  stack.append(overview);
  for (const record of summary.records) {
    const card = node("article", "record-card");
    card.append(node("div", "evidence-label", "Progress record"));
    card.append(node("h3", "", `${record.date} · ${record.speech_position}`));
    card.append(node("p", "citation", record.resolution || "No resolution recorded"));
    card.append(node("p", "evidence-body", record.assessment_text));
    card.append(metadataPills((record.weakness_tags || []).map((tag) => `Focus: ${tag}`)));
    card.append(node("p", "source-provenance", `Author: ${record.author_id}`));
    stack.append(card);
  }
  return stack;
}

function renderAssessmentProposal(proposal) {
  const card = labeledCard("AssessmentProposal · confirmation required", `${proposal.student_id} · ${proposal.speech_position}`);
  card.append(node("p", "citation", proposal.resolution || "No resolution recorded"));
  card.append(node("p", "argument-part", proposal.assessment_text));
  card.append(metadataPills(proposal.weakness_tags.map((tag) => `Weakness: ${tag}`)));
  return card;
}

function renderCalendarEvent(event) {
  const card = labeledCard("CalendarEvent", `Human coaching for ${event.student_id}`);
  card.append(node("p", "argument-part", `${new Date(event.start).toLocaleString()} – ${new Date(event.end).toLocaleString()}`));
  card.append(metadataPills([
    event.timezone, `Backend: ${event.backend}`, event.synthetic ? "Synthetic fixture" : "Provider event",
    event.attendee_email && `Attendee: ${event.attendee_email}`,
  ]));
  card.append(node("p", "source-provenance", `Event ID: ${event.event_id}`));
  return card;
}

function renderArtifact(artifact, allArtifacts) {
  switch (artifact.artifact_type) {
    case "evidence_packet": return renderEvidencePacket(artifact);
    case "rule_packet": return renderRulePacket(artifact);
    case "topic_packet": return renderTopicPacket(artifact);
    case "ingestion_preview": return renderIngestionPreview(artifact);
    case "ingestion_commit_result": return renderIngestionCommit(artifact);
    case "argument_draft": return renderArgumentDraft(artifact, allArtifacts);
    case "drill_plan": return renderDrillPlan(artifact);
    case "coach_turn": return renderCoachTurn(artifact);
    case "progress_summary": return renderProgressSummary(artifact);
    case "assessment_proposal": return renderAssessmentProposal(artifact);
    case "calendar_event": return renderCalendarEvent(artifact);
    default: return node("div", "alert danger", `Unsupported artifact type: ${artifact.artifact_type || "missing discriminator"}`);
  }
}

function renderTypedError(data) {
  const error = data.error || {};
  const card = node("div", "alert danger");
  card.dataset.testid = "typed-error";
  card.append(node("strong", "", `${error.code || "UNKNOWN_ERROR"}: ${error.message || "The request failed."}`));
  const grid = node("div", "error-grid");
  for (const [label, value] of [
    ["Stage", error.stage], ["Agent", error.agent || "—"], ["Tool", error.tool || "—"],
    ["Retryable", error.retryable ? "Yes" : "No"], ["Request ID", data.request_id || "—"], ["Session ID", data.session_id || "—"],
  ]) {
    const item = node("div");
    item.append(node("span", "", label), node("strong", "", value || "—"));
    grid.append(item);
  }
  card.append(grid);
  if (error.details && Object.keys(error.details).length) card.append(node("pre", "response-copy", formatJson(error.details)));
  return card;
}

function renderResult(data) {
  const target = byId("result");
  target.replaceChildren();
  if (data.status === "failed") {
    target.append(renderTypedError(data));
    return;
  }
  if (data.status === "needs_input") target.append(node("div", "alert notice state-banner", "Needs input · reply in this session to continue."));
  if (data.status === "needs_confirmation") target.append(node("div", "alert notice state-banner", "Needs confirmation · review the staged action before confirming."));
  const artifacts = Array.isArray(data.artifacts) ? data.artifacts : [];
  const artifactStack = node("div", "evidence-stack");
  for (const artifact of artifacts) artifactStack.append(renderArtifact(artifact, artifacts));
  if (artifacts.length) target.append(artifactStack);
  if (data.response) target.append(node("pre", "response-copy", data.response));
}

function emptyTrace(targetId, text) {
  const target = byId(targetId);
  target.replaceChildren();
  const item = node("div", "task-item");
  item.append(node("span", "", text), node("b", "", "—"));
  target.append(item);
}

function renderTrace(data, elapsed) {
  byId("trace-agent").textContent = titleCase(data.active_agent || data.error?.agent || "supervisor");
  byId("trace-status").textContent = titleCase(data.status);
  byId("trace-latency").textContent = `${elapsed} ms`;
  byId("trace-request").textContent = shortId(data.request_id);
  byId("trace-session").textContent = shortId(data.session_id || sessionId);

  const handoffs = [...(data.agent_trace || [])].sort((left, right) => left.sequence - right.sequence);
  const handoffTarget = byId("trace-handoffs");
  handoffTarget.replaceChildren();
  if (!handoffs.length) emptyTrace("trace-handoffs", "No agent events returned");
  for (const entry of handoffs) {
    const item = node("div", "task-item");
    const route = entry.from_agent || entry.to_agent ? ` · ${entry.from_agent || "start"} → ${entry.to_agent || entry.agent}` : "";
    item.title = entry.summary;
    item.append(node("span", "", `${entry.agent} · ${entry.event}${route}`), node("b", "", `#${entry.sequence}`));
    handoffTarget.append(item);
  }

  const toolTarget = byId("trace-tools");
  toolTarget.replaceChildren();
  const toolCalls = [...(data.tool_trace || [])].sort((left, right) => left.sequence - right.sequence);
  if (!toolCalls.length) emptyTrace("trace-tools", "No tool calls returned");
  const grouped = new Map();
  for (const call of toolCalls) {
    if (!grouped.has(call.agent)) grouped.set(call.agent, []);
    grouped.get(call.agent).push(call);
  }
  for (const [agent, calls] of grouped) {
    const group = node("div", "trace-agent-group");
    group.append(node("strong", "", titleCase(agent)));
    for (const call of calls) {
      const item = node("div", "task-item");
      item.title = call.result_summary || call.error_code || formatJson(call.arguments || {});
      item.append(node("span", "", `#${call.sequence} ${call.tool} · ${call.stage}`), node("b", "", call.status));
      group.append(item);
    }
    toolTarget.append(group);
  }

  const models = [...(data.model_trace || [])].sort((left, right) => left.sequence - right.sequence);
  const modelTarget = byId("trace-models");
  modelTarget.replaceChildren();
  if (!models.length) emptyTrace("trace-models", "No model calls returned");
  for (const call of models) {
    const item = node("div", "task-item");
    item.title = `${call.prompt_template} · prompt ${shortId(call.prompt_sha256)} · response ${shortId(call.response_sha256)}`;
    item.append(node("span", "", `#${call.sequence} ${call.agent} · ${call.schema_name}`), node("b", "", `${call.status}${call.latency_ms === null ? "" : ` · ${call.latency_ms} ms`}`));
    modelTarget.append(item);
  }
  renderDeveloperModelCalls(models);
}

function renderDeveloperModelCalls(models) {
  const exposed = models.filter((call) => call.rendered_system_prompt || call.rendered_user_payload || call.model_response);
  const details = byId("model-details");
  const target = byId("model-call-details");
  details.hidden = exposed.length === 0;
  target.replaceChildren();
  for (const call of exposed) {
    const card = node("div", "model-call");
    card.append(node("strong", "", `${titleCase(call.agent)} · ${call.model} · ${call.prompt_template}`));
    if (call.rendered_system_prompt) card.append(node("span", "evidence-label", "System prompt"), node("pre", "", call.rendered_system_prompt));
    if (call.rendered_user_payload) card.append(node("span", "evidence-label", "User payload"), node("pre", "", call.rendered_user_payload));
    if (call.model_response) card.append(node("span", "evidence-label", "Model response"), node("pre", "", call.model_response));
    target.append(card);
  }
}

function setBusy(busy) {
  sendButton.disabled = busy;
  sendButton.querySelector("span").textContent = busy ? "Running…" : "Run request";
  resultArea.setAttribute("aria-busy", String(busy));
  byId("attachment-label").classList.toggle("disabled", busy);
}

function setSessionState(text) {
  byId("session-state").textContent = text;
}

function updateAttachmentControl() {
  const file = attachmentInput.files[0];
  byId("attachment-meta").textContent = file ? `${file.name} · ${(file.size / 1024).toFixed(1)} KB` : "No file attached · DOCX only, 10 MB maximum";
  removeAttachmentButton.hidden = !file;
}

function clearAttachment() {
  attachmentInput.value = "";
  updateAttachmentControl();
}

function resetSession({ clearPrompt = false, clearOutput = false } = {}) {
  sessionId = newSessionId();
  byId("trace-session").textContent = shortId(sessionId);
  messageInput.placeholder = DEFAULT_PLACEHOLDER;
  setSessionState("Session ready");
  clearAttachment();
  if (clearPrompt) messageInput.value = "";
  if (clearOutput) {
    byId("submitted-prompt").hidden = true;
    byId("prompt-entered").textContent = "";
    byId("result").replaceChildren(node("div", "empty-state", "Ready for a new multi-agent request."));
    byId("raw-details").hidden = true;
    byId("trace-agent").textContent = "—";
    byId("trace-status").textContent = "—";
    byId("trace-latency").textContent = "—";
    byId("trace-request").textContent = "—";
    emptyTrace("trace-handoffs", "No handoffs yet");
    emptyTrace("trace-tools", "No tool calls yet");
    emptyTrace("trace-models", "No model calls yet");
    renderDeveloperModelCalls([]);
  }
}

function updateComposer(data) {
  const artifactTypes = new Set((data.artifacts || []).map((artifact) => artifact.artifact_type));
  messageInput.value = "";
  if (data.status === "needs_input") {
    messageInput.placeholder = "Reply with the missing information to resume this session...";
    setSessionState("Waiting for input");
  } else if (data.status === "needs_confirmation") {
    messageInput.placeholder = "Confirm the staged action, or describe a requested change...";
    setSessionState("Waiting for confirmation");
  } else if (artifactTypes.has("coach_turn")) {
    messageInput.placeholder = "Reply to the simulated coach, or ask to end coaching...";
    setSessionState("Simulated coaching active");
  } else if (artifactTypes.has("argument_draft")) {
    messageInput.placeholder = "Ask the Strategist to revise this argument...";
    setSessionState("Argument ready for revision");
  } else {
    messageInput.placeholder = DEFAULT_PLACEHOLDER;
    setSessionState(data.status === "failed" ? "Request failed visibly" : "Session ready");
  }
}

async function readResponse(response) {
  const text = await response.text();
  try {
    return text ? JSON.parse(text) : {};
  } catch (error) {
    return {
      status: "failed",
      request_id: response.headers.get("x-request-id") || "browser-unparseable-response",
      session_id: sessionId,
      error: { code: "INVALID_RESPONSE", message: "The API did not return JSON.", stage: "browser.response", agent: null, tool: null, retryable: false, details: { http_status: response.status } },
      agent_trace: [], tool_trace: [], model_trace: [],
    };
  }
}

async function confirmIngestion(token, button) {
  button.disabled = true;
  setBusy(true);
  const started = performance.now();
  let data;
  try {
    const response = await fetch("/ingestion/confirm", {
      method: "POST",
      headers: { "content-type": "application/json", "accept": "application/json" },
      body: JSON.stringify({
        confirmation_token: token,
        role: ACCESS_ROLE,
        user_id: byId("uid").value.trim(),
        resolution: byId("resolution").value,
        idempotency_key: globalThis.crypto?.randomUUID?.() || `confirm-${Date.now()}`,
      }),
    });
    data = await readResponse(response);
    const elapsed = Math.round(performance.now() - started);
    if (!response.ok || data.status === "failed") {
      renderResult(data);
      renderTrace(data, elapsed);
      updateComposer(data);
    } else {
      byId("result").replaceChildren(renderArtifact(data, [data]));
      byId("result-meta").textContent = `Ingestion confirmed · ${elapsed} ms`;
      clearAttachment();
      setSessionState("Evidence imported");
    }
    byId("raw-json").textContent = formatJson(data);
    byId("raw-details").hidden = false;
  } catch (error) {
    showTransportError(error, Math.round(performance.now() - started));
  } finally {
    setBusy(false);
  }
}

function showTransportError(error, elapsed) {
  const data = {
    status: "failed", request_id: "browser-transport-failure", session_id: sessionId,
    error: { code: "TRANSPORT_ERROR", message: error.message, stage: "browser.fetch", agent: null, tool: null, retryable: true, details: {} },
    agent_trace: [], tool_trace: [], model_trace: [],
  };
  renderResult(data);
  renderTrace(data, elapsed);
  updateComposer(data);
  byId("raw-json").textContent = formatJson(data);
  byId("raw-details").hidden = false;
}

async function submitRequest() {
  const message = messageInput.value.trim();
  if (!message) return messageInput.focus();
  byId("prompt-entered").textContent = message;
  byId("submitted-prompt").hidden = false;
  const attachment = attachmentInput.files[0];
  setBusy(true);
  byId("result-meta").textContent = attachment ? "Supervisor received attachment" : "Supervisor is working";
  byId("result").replaceChildren(node("div", "empty-state", "The Supervisor is coordinating the four-agent graph…"));
  const started = performance.now();
  let data;
  try {
    let response;
    if (attachment) {
      const payload = new FormData();
      payload.append("message", message);
      payload.append("role", ACCESS_ROLE);
      payload.append("user_id", byId("uid").value.trim());
      payload.append("resolution", byId("resolution").value);
      payload.append("session_id", sessionId);
      payload.append("attachment", attachment, attachment.name);
      response = await fetch("/chat/with-attachment", { method: "POST", headers: { "accept": "application/json" }, body: payload });
    } else {
      response = await fetch("/chat", {
        method: "POST",
        headers: { "content-type": "application/json", "accept": "application/json" },
        body: JSON.stringify({ message, role: ACCESS_ROLE, user_id: byId("uid").value.trim(), resolution: byId("resolution").value, session_id: sessionId }),
      });
    }
    data = await readResponse(response);
    if (data.session_id) sessionId = data.session_id;
    const elapsed = Math.round(performance.now() - started);
    renderResult(data);
    renderTrace(data, elapsed);
    updateComposer(data);
    byId("result-meta").textContent = `${titleCase(data.status)} · ${titleCase(data.active_agent || data.error?.agent || "supervisor")} · ${elapsed} ms`;
    byId("raw-json").textContent = formatJson(data);
    byId("raw-details").hidden = false;
  } catch (error) {
    showTransportError(error, Math.round(performance.now() - started));
  } finally {
    setBusy(false);
  }
}

function setHealthRow(id, value, good = true) {
  const row = byId(id);
  row.classList.toggle("good", good);
  row.classList.toggle("warn", !good);
  row.querySelector("strong span").textContent = value;
}

async function loadHealth() {
  try {
    const response = await fetch("/health/ready", { headers: { "accept": "application/json" } });
    const data = await response.json();
    setHealthRow("status-api", data.status, response.ok);
    setHealthRow("status-agent", data.graph, response.ok);
    setHealthRow("status-retrieval", `${data.retrieval} · ${data.embedding_model || "embedding unknown"}`, response.ok);
    setHealthRow("status-model", data.model, response.ok);
    byId("status-calendar").hidden = false;
    setHealthRow("status-calendar", data.calendar, response.ok);
  } catch (error) {
    for (const id of ["status-api", "status-agent", "status-retrieval", "status-model"]) setHealthRow(id, "unavailable", false);
  }
}

byId("uid").addEventListener("change", () => resetSession());
byId("resolution").addEventListener("change", () => resetSession());
byId("new-session").addEventListener("click", () => { resetSession({ clearPrompt: true, clearOutput: true }); messageInput.focus(); });
attachmentInput.addEventListener("change", () => {
  const file = attachmentInput.files[0];
  if (file && !file.name.toLowerCase().endsWith(".docx")) {
    showTransportError(new Error("Only Word .docx attachments are supported."), 0);
    clearAttachment();
    return;
  }
  if (file && file.size > 10 * 1024 * 1024) {
    showTransportError(new Error("The attachment must be 10 MB or smaller."), 0);
    clearAttachment();
    return;
  }
  if (file && !messageInput.value.trim()) messageInput.value = ingestionPrompt(byId("side").value);
  updateAttachmentControl();
});
removeAttachmentButton.addEventListener("click", clearAttachment);
form.addEventListener("submit", (event) => { event.preventDefault(); submitRequest(); });
messageInput.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter") { event.preventDefault(); form.requestSubmit(); }
});

messageInput.placeholder = DEFAULT_PLACEHOLDER;
updateAttachmentControl();
setSessionState("Session ready");
byId("trace-session").textContent = shortId(sessionId);
loadHealth();
