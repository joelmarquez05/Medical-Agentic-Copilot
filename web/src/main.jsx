import React, { useEffect, useRef, useState } from "react";
import { createRoot } from "react-dom/client";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import {
  Activity,
  ArrowDown,
  ArrowRight,
  ArrowUp,
  BookOpen,
  Check,
  ChevronDown,
  ChevronRight,
  CircleHelp,
  Copy,
  ExternalLink,
  FileText,
  FolderOpen,
  Globe2,
  History,
  LoaderCircle,
  Menu,
  MessageSquare,
  PanelRightClose,
  Paperclip,
  Plus,
  Search,
  Settings2,
  ShieldCheck,
  Sparkles,
  UserRound,
  UsersRound,
  X,
} from "lucide-react";
import "./style.css";

const DEFAULTS = {
  name: "Medical Copilot",
  accent: "#B90000",
  font: "editorial",
  font_scale: 1,
  density: "comfortable",
  logo: null,
};
const MODEL_NAMES = { gemma: "Gemma 4", "kimi-k3": "Kimi K3" };

async function api(path, options = {}) {
  const response = await fetch("/api" + path, {
    ...options,
    headers: {
      ...(options.body && !(options.body instanceof FormData)
        ? { "Content-Type": "application/json" }
        : {}),
      ...options.headers,
    },
  });
  const data = await response.json();
  if (!response.ok)
    throw new Error(
      typeof data.detail === "string"
        ? data.detail
        : data.detail?.map((x) => x.msg).join(" ") ||
          "The request could not be completed.",
    );
  return data;
}
const post = (path, data = {}) =>
  api(path, { method: "POST", body: JSON.stringify(data) });
const iconSize = 18;
const shortDate = (value) =>
  new Date(value).toLocaleDateString("en-GB", {
    day: "numeric",
    month: "short",
  });
const fieldName = (value) =>
  ({
    weight_kg: "Weight",
    albumin_g_dl: "Serum albumin",
    snack: "Preferred snack",
  })[value] || value.replaceAll("_", " ").replace(/^./, (c) => c.toUpperCase());

function Modal({ title, children, onClose, wide = false }) {
  const ref = useRef(null);
  useEffect(() => {
    ref.current?.showModal();
    return () => ref.current?.close();
  }, []);
  return (
    <dialog
      ref={ref}
      className={"modal " + (wide ? "wide" : "")}
      onCancel={onClose}
      onClick={(e) => {
        if (e.target === ref.current) onClose();
      }}
    >
      <div className="modal-heading">
        <h2>{title}</h2>
        <button
          className="icon-button"
          aria-label="Close dialog"
          onClick={onClose}
        >
          <X size={20} />
        </button>
      </div>
      {children}
    </dialog>
  );
}

function AnswerText({ text, sources, onSource }) {
  let object;
  try {
    const fenced = text?.match(/^\s*```(?:json)?\s*(\{[\s\S]*\})\s*```\s*$/);
    const value = JSON.parse(fenced ? fenced[1] : text);
    if (value && typeof value === "object" && !Array.isArray(value))
      object = value;
  } catch {}
  if (object)
    return (
      <dl className="answer-fields">
        {Object.entries(object).map(([key, value]) => (
          <div key={key}>
            <dt>{fieldName(key)}</dt>
            <dd>
              {value === null
                ? "Not recorded"
                : typeof value === "object"
                  ? JSON.stringify(value)
                  : String(value)}
            </dd>
          </div>
        ))}
      </dl>
    );
  let display = text || "";
  sources.forEach((source, i) => {
    display = display.replaceAll(source.id, `[${i + 1}](#source/${source.id})`);
  });
  return (
    <Markdown
      remarkPlugins={[remarkGfm]}
      components={{
        a: ({ href, children }) =>
          href?.startsWith("#source/") ? (
            <button
              className="inline-citation"
              onClick={() => onSource(href.slice(8))}
            >
              {children}
            </button>
          ) : (
            <a href={href} target="_blank" rel="noreferrer">
              {children}
            </a>
          ),
      }}
    >
      {display}
    </Markdown>
  );
}

function Message({ message, onSource, onActivity, retry, notify }) {
  const result = message.result;
  const response = result?.response;
  const pending = ["queued", "running"].includes(message.state);
  const sources = result?.sources || [];
  const cited = sources.filter((s) => s.cited);
  const isOperational = response?.type === "operational_message";
  const clarification = response?.type === "clarify";
  return (
    <article className="exchange">
      <div className="question-row">
        <div className="user-mark">
          <UserRound size={16} />
        </div>
        <h2>{message.question}</h2>
      </div>
      <div className="response-row">
        <div className="response-heading">
          <BookOpen size={17} />
          <span>{result?.archive ? "Saved response" : "Medical Copilot"}</span>
          <span className="response-model">
            {result?.model?.name || MODEL_NAMES[message.model_id || "gemma"]}
          </span>
          {result?.archive && (
            <span className="archive-label">
              14 Oct 2026 · No new inference
            </span>
          )}
          {pending && <span className="working-dot" />}
        </div>
        {pending ? (
          <button className="progress-line" onClick={() => onActivity(message)}>
            <LoaderCircle size={16} className="spin" />
            <span>{message.events.at(-1)?.label || "Request queued"}</span>
            <ChevronRight size={15} />
          </button>
        ) : result?.error ? (
          <div className="answer-error">
            <p>{result.error}</p>
            <button
              className="text-button"
              onClick={() => retry(message.question)}
            >
              Try again <ArrowRight size={14} />
            </button>
          </div>
        ) : response ? (
          <>
            {clarification && (
              <p className="clarification-label">
                <CircleHelp size={16} /> One detail before continuing
              </p>
            )}
            <div
              className={"answer-body " + (isOperational ? "answer-error" : "")}
            >
              <AnswerText
                text={clarification ? response.question : response.answer}
                sources={sources}
                onSource={(id) => onSource(message, id)}
              />
            </div>
            {response.justification && !isOperational && (
              <details className="support-note">
                <summary>Answer rationale</summary>
                <div className="answer-body">
                  <AnswerText
                    text={response.justification}
                    sources={sources}
                    onSource={(id) => onSource(message, id)}
                  />
                </div>
              </details>
            )}
            {!!response.limitations?.length && (
              <div className="limitations">
                {response.limitations.map((line, i) => (
                  <p key={i}>{line}</p>
                ))}
              </div>
            )}
            {!!cited.length && (
              <div className="citation-list">
                {cited.map((source, i) => (
                  <button
                    className="citation-chip"
                    key={source.id}
                    onClick={() => onSource(message, source.id)}
                  >
                    <span>{sources.indexOf(source) + 1}</span>
                    <span>{source.title}</span>
                    {source.pdf_page && <small>p. {source.pdf_page}</small>}
                    <ChevronRight size={13} />
                  </button>
                ))}
              </div>
            )}
            {!cited.length && !clarification && !isOperational && (
              <p className="no-citations">
                {message.result.memory_saved
                  ? "Based on available context; no retrieved source cited."
                  : "No retrieved source cited."}
              </p>
            )}
            <div className="answer-actions">
              <button
                className="icon-button"
                aria-label="Copy answer"
                onClick={async () => {
                  try {
                    await navigator.clipboard.writeText(
                      response.answer || response.question,
                    );
                    notify("Answer copied");
                  } catch {
                    notify("Clipboard access is unavailable");
                  }
                }}
              >
                <Copy size={15} />
              </button>
              <button
                className="text-button muted"
                onClick={() => onActivity(message)}
              >
                <Activity size={14} /> View activity
              </button>
              {result.memory_saved && (
                <span className="saved-state">
                  <Check size={12} /> Saved
                </span>
              )}
              {!result.memory_saved && !result.archive && (
                <span className="unsaved-state">Memory not saved</span>
              )}
            </div>
          </>
        ) : null}
      </div>
    </article>
  );
}

function App() {
  const [boot, setBoot] = useState(null),
    [appearance, setAppearance] = useState(DEFAULTS);
  const [active, setActive] = useState(null),
    [context, setContext] = useState(null),
    [messages, setMessages] = useState([]);
  const [draft, setDraft] = useState(""),
    [sending, setSending] = useState(false),
    [uploading, setUploading] = useState(false);
  const [preferredModel, setPreferredModel] = useState(() => {
    const saved = localStorage.getItem("medical-copilot-model");
    return MODEL_NAMES[saved] ? saved : "gemma";
  });
  const [changingModel, setChangingModel] = useState(false);
  const [panel, setPanel] = useState(null),
    [selectedMessage, setSelectedMessage] = useState(null),
    [selectedSource, setSelectedSource] = useState(null);
  const [settings, setSettings] = useState(false),
    [search, setSearch] = useState(""),
    [toast, setToast] = useState("");
  const [sidebar, setSidebar] = useState(false),
    [initialError, setInitialError] = useState(""),
    [workingExample, setWorkingExample] = useState(false);
  const activeRef = useRef(null),
    textRef = useRef(null),
    uploadRef = useRef(null),
    bottomRef = useRef(null),
    toastTimer = useRef(null);
  const busy =
    sending ||
    changingModel ||
    messages.some((m) => ["queued", "running"].includes(m.state));
  const currentPatient = context?.session?.patient_id || null;
  const currentModel = context?.session?.model_id || preferredModel;

  function notify(text) {
    setToast(text);
    clearTimeout(toastTimer.current);
    toastTimer.current = setTimeout(() => setToast(""), 4500);
  }
  async function refresh() {
    const data = await api("/bootstrap");
    setBoot(data);
    return data;
  }
  function activate(id) {
    activeRef.current = id;
    setActive(id);
    setContext(null);
    setMessages([]);
    setPanel(null);
    setSelectedMessage(null);
    setSelectedSource(null);
    setSidebar(false);
    localStorage.setItem("medical-copilot-conversation", id || "");
  }
  async function load(id) {
    const data = await api("/sessions/" + id);
    if (activeRef.current === id) {
      setContext(data);
      setMessages(data.messages);
    }
    return data;
  }
  useEffect(() => {
    refresh()
      .then((data) => {
        setAppearance(data.appearance);
        const saved = localStorage.getItem("medical-copilot-conversation");
        if (data.sessions.some((s) => s.id === saved)) activate(saved);
      })
      .catch((error) => setInitialError(error.message));
    return () => clearTimeout(toastTimer.current);
  }, []);
  useEffect(() => {
    if (active) load(active).catch((error) => notify(error.message));
  }, [active]);
  useEffect(() => {
    document.documentElement.style.setProperty("--accent", appearance.accent);
    document.documentElement.style.setProperty(
      "--text-scale",
      appearance.font_scale,
    );
    document.documentElement.dataset.font = appearance.font;
    document.documentElement.dataset.density = appearance.density;
    document.title = appearance.name;
  }, [appearance]);
  const pendingIds = messages
    .filter((m) => ["queued", "running"].includes(m.state))
    .map((m) => m.id)
    .join(",");
  useEffect(() => {
    const streams = pendingIds
      .split(",")
      .filter(Boolean)
      .map((id) => {
        const stream = new EventSource("/api/runs/" + id + "/events");
        stream.onmessage = (event) => {
          const item = JSON.parse(event.data);
          if (activeRef.current !== active) {
            stream.close();
            return;
          }
          if (item.type === "activity")
            setMessages((rows) =>
              rows.map((row) =>
                row.id === id &&
                !row.events.some((e) => e.sequence === item.event.sequence)
                  ? {
                      ...row,
                      state:
                        item.event.kind === "queued" ? row.state : "running",
                      events: [...row.events, item.event],
                    }
                  : row,
              ),
            );
          if (item.type === "complete") {
            stream.close();
            setMessages((rows) =>
              rows.map((row) => (row.id === id ? item.message : row)),
            );
            load(active).catch(() => {});
            refresh().catch(() => {});
          }
        };
        return stream;
      });
    return () => streams.forEach((stream) => stream.close());
  }, [active, pendingIds]);
  useEffect(() => {
    if (messages.length)
      bottomRef.current?.scrollIntoView({
        behavior: matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "instant"
          : "smooth",
        block: "end",
      });
  }, [messages.length, pendingIds]);
  useEffect(() => {
    if (textRef.current) {
      textRef.current.style.height = "auto";
      textRef.current.style.height =
        Math.min(textRef.current.scrollHeight, 180) + "px";
    }
  }, [draft]);

  async function newConversation(patientId = null) {
    const created = await post("/sessions", {
      patient_id: patientId,
      model_id: currentModel,
    });
    activate(created.id);
    setDraft("");
    await refresh();
    return created.id;
  }
  async function ensureConversation() {
    return activeRef.current || (await newConversation());
  }
  async function changeModel(modelId) {
    if (busy || uploading) return;
    const id = activeRef.current;
    setChangingModel(true);
    try {
      if (id) {
        const session = await api("/sessions/" + id + "/model", {
          method: "PUT",
          body: JSON.stringify({ model_id: modelId }),
        });
        if (activeRef.current === id)
          setContext((value) => (value ? { ...value, session } : value));
      }
      localStorage.setItem("medical-copilot-model", modelId);
      setPreferredModel(modelId);
      await refresh();
      notify(MODEL_NAMES[modelId] + " selected for the next answer");
    } catch (error) {
      notify(error.message);
    } finally {
      setChangingModel(false);
    }
  }
  async function changePatient(value) {
    if ((value || null) === currentPatient && active) return;
    try {
      await newConversation(value || null);
      notify(
        value
          ? "Started a conversation with this patient context"
          : "Started a conversation with no patient context",
      );
    } catch (error) {
      notify(error.message);
    }
  }
  async function submit(event) {
    event?.preventDefault();
    if (!draft.trim() || busy || uploading) return;
    const question = draft.trim();
    setSending(true);
    let id;
    try {
      id = await ensureConversation();
      const message = await post("/sessions/" + id + "/messages", {
        question,
        request_id: crypto.randomUUID(),
      });
      if (activeRef.current === id) {
        setMessages((rows) => [
          ...rows.filter((r) => r.id !== message.id),
          message,
        ]);
        setDraft("");
      }
      await refresh();
    } catch (error) {
      setDraft(question);
      notify(error.message);
      if (id) await load(id).catch(() => {});
    } finally {
      setSending(false);
    }
  }
  async function upload(file) {
    if (!file) return;
    if (file.size > 50 * 1024 * 1024)
      return notify("Choose a PDF smaller than 50 MiB");
    setUploading(true);
    try {
      const id = await ensureConversation();
      const form = new FormData();
      form.append("file", file);
      await api("/sessions/" + id + "/documents", {
        method: "POST",
        body: form,
      });
      await load(id);
      setPanel("files");
      notify("Document attached and ready to search");
    } catch (error) {
      notify(error.message);
    } finally {
      setUploading(false);
      if (uploadRef.current) uploadRef.current.value = "";
    }
  }
  async function example(id) {
    if (busy || uploading || workingExample) return;
    setWorkingExample(true);
    try {
      if (id === "saved") {
        const saved = await post("/examples/saved");
        activate(saved.id);
        await refresh();
        return;
      }
      const data = await post("/examples/" + id + "/prepare", {
        model_id: currentModel,
      });
      activate(data.session_id);
      setDraft(data.question);
      await refresh();
      notify(data.notice);
      setPanel(null);
      setTimeout(() => textRef.current?.focus(), 100);
    } catch (error) {
      notify(error.message);
    } finally {
      setWorkingExample(false);
    }
  }
  function showSource(message, id) {
    setSelectedMessage(message.id);
    setSelectedSource(id);
    setPanel("sources");
  }
  function showActivity(message) {
    setSelectedMessage(message.id);
    setPanel("activity");
  }
  const inspected =
    messages.find((m) => m.id === selectedMessage) || messages.at(-1);
  const source = inspected?.result?.sources?.find(
    (s) => s.id === selectedSource,
  );
  const sessions =
    boot?.sessions.filter(
      (s) =>
        (s.message_count > 0 || s.id === active || s.demo) &&
        (s.title + " " + (s.patient_name || ""))
          .toLowerCase()
          .includes(search.toLowerCase()),
    ) || [];

  return (
    <div className="app-shell">
      {sidebar && (
        <button
          className="sidebar-scrim"
          aria-label="Close navigation"
          onClick={() => setSidebar(false)}
        />
      )}
      <aside className={"sidebar " + (sidebar ? "open" : "")}>
        <a
          className="brand"
          href="#"
          onClick={(e) => {
            e.preventDefault();
            activate(null);
            setDraft("");
          }}
        >
          {appearance.logo ? (
            <img src={appearance.logo} alt="" />
          ) : (
            <span className="brand-symbol">
              <BookOpen size={23} strokeWidth={1.6} />
            </span>
          )}
          <span>{appearance.name}</span>
        </a>
        <button
          className="new-consultation"
          onClick={() => newConversation().catch((e) => notify(e.message))}
        >
          <Plus size={17} /> New consultation<span>↗</span>
        </button>
        <nav className="primary-nav" aria-label="Workspace">
          <button
            className={!panel ? "selected" : ""}
            onClick={() => setPanel(null)}
          >
            <BookOpen size={iconSize} /> Research
          </button>
          <button
            className={panel === "patient" ? "selected" : ""}
            onClick={() => setPanel(panel === "patient" ? null : "patient")}
          >
            <UsersRound size={iconSize} /> Patients
            <span>{boot?.patients.length || ""}</span>
          </button>
          <button
            className={panel === "examples" ? "selected" : ""}
            onClick={() => setPanel(panel === "examples" ? null : "examples")}
          >
            <Sparkles size={iconSize} /> Examples
          </button>
        </nav>
        <div className="history-heading">
          <span>Conversations</span>
          <Search size={14} />
        </div>
        <label className="history-search">
          <Search size={14} />
          <input
            aria-label="Search conversations"
            placeholder="Find a conversation"
            value={search}
            onChange={(e) => setSearch(e.target.value)}
          />
        </label>
        <div className="session-list">
          {sessions.map((session) => (
            <button
              key={session.id}
              className={
                "session-item " + (session.id === active ? "active" : "")
              }
              onClick={() => activate(session.id)}
            >
              <MessageSquare size={15} />
              <span>
                <strong>{session.title}</strong>
                <small>
                  {session.patient_name || "No patient context"}
                  {session.demo ? " · Example" : ""}
                </small>
              </span>
            </button>
          ))}
          {!sessions.length && (
            <p className="empty-history">
              {search
                ? "No conversations found."
                : "Your consultations will appear here."}
            </p>
          )}
        </div>
        <div className="sidebar-bottom">
          <button onClick={() => setSettings(true)}>
            <Settings2 size={17} /> Appearance & settings
          </button>
          <div className="local-label">
            <span /> Local workspace <small>Saved on this computer</small>
          </div>
        </div>
      </aside>
      <div className="workspace">
        <header className="topbar">
          <div className="context-control">
            <button
              className="icon-button mobile-menu"
              aria-label="Open navigation"
              onClick={() => setSidebar(true)}
            >
              <Menu size={20} />
            </button>
            {currentPatient ? <UserRound size={17} /> : <Globe2 size={17} />}
            <select
              aria-label="Patient context"
              value={currentPatient || ""}
              onChange={(e) => changePatient(e.target.value)}
              disabled={!!active && !context}
            >
              <option value="">No patient context</option>
              {boot?.patients.map((patient) => (
                <option value={patient.patient_id} key={patient.patient_id}>
                  {patient.name}
                </option>
              ))}
            </select>
            <ChevronDown size={14} />
            <span className="optional-label">
              {currentPatient
                ? "Patient context active"
                : "Patient context optional"}
            </span>
          </div>
          <div className="top-actions">
            <label
              className="model-indicator"
              title="Choose the model for the next answer"
            >
              <span className="model-dot" />
              <select
                aria-label="Answer model"
                value={currentModel}
                onChange={(event) => changeModel(event.target.value)}
                disabled={busy || uploading || !boot || (!!active && !context)}
              >
                {(boot?.models || []).map((model) => (
                  <option value={model.id} key={model.id}>
                    {model.name}
                  </option>
                ))}
              </select>
              <ChevronDown size={12} />
            </label>
            <span className="topbar-divider" />
            <button
              className={"icon-button " + (panel === "files" ? "active" : "")}
              aria-label="Attached files"
              title="Attached files"
              onClick={() => setPanel(panel === "files" ? null : "files")}
            >
              <FolderOpen size={19} />
              {!!context?.files.length && (
                <span className="file-count">{context.files.length}</span>
              )}
            </button>
            <button
              className={
                "icon-button " + (panel === "activity" ? "active" : "")
              }
              aria-label="Agent activity"
              title="Agent activity"
              onClick={() => setPanel(panel === "activity" ? null : "activity")}
            >
              <Activity size={19} />
            </button>
          </div>
        </header>
        <main className={"main-layout " + (panel ? "has-panel" : "")}>
          <section
            className={"conversation " + (!messages.length ? "welcome" : "")}
            aria-label="Clinical conversation"
          >
            <div className="conversation-scroll">
              {!messages.length ? (
                <div className="welcome-content">
                  <div className="welcome-mark">
                    <BookOpen size={29} strokeWidth={1.3} />
                    <span>Medicine, with the evidence in view.</span>
                  </div>
                  <h1>
                    Evidence for your
                    <br />
                    next clinical question.
                  </h1>
                  <p className="welcome-description">
                    Explore the literature, ask your documents, and bring
                    <br className="desktop-break" /> patient context into the
                    conversation when it matters.
                  </p>
                  {currentPatient && (
                    <div className="context-banner">
                      <UserRound size={16} />
                      <span>
                        Using records for{" "}
                        <strong>{context.session.patient_name}</strong>
                      </span>
                      <button onClick={() => setPanel("patient")}>
                        View context <ChevronRight size={14} />
                      </button>
                    </div>
                  )}
                  {initialError && (
                    <p className="answer-error" role="alert">
                      {initialError}{" "}
                      <button onClick={() => location.reload()}>Reload</button>
                    </p>
                  )}
                  <div className="example-grid">
                    {boot?.examples
                      .filter((item) => item.featured)
                      .map((item, i) => {
                        const Icon = [Search, FileText, History][i];
                        return (
                          <button
                            key={item.id}
                            disabled={workingExample || busy || uploading}
                            className="example-item"
                            onClick={() => example(item.id)}
                          >
                            <Icon size={19} strokeWidth={1.5} />
                            <strong>{item.title}</strong>
                            <span>{item.description}</span>
                            <ArrowRight size={16} className="example-arrow" />
                          </button>
                        );
                      })}
                  </div>
                </div>
              ) : (
                <div className="message-column">
                  {messages.map((message) => (
                    <Message
                      key={message.id}
                      message={message}
                      onSource={showSource}
                      onActivity={showActivity}
                      retry={(question) => {
                        setDraft(question);
                        textRef.current?.focus();
                      }}
                      notify={notify}
                    />
                  ))}
                  <div ref={bottomRef} />
                </div>
              )}
            </div>
            <div className="composer-area">
              <form
                className={"composer " + (busy ? "is-busy" : "")}
                onSubmit={submit}
              >
                <textarea
                  ref={textRef}
                  aria-label="Ask a clinical question"
                  placeholder={
                    currentPatient
                      ? "Ask with this patient’s context…"
                      : "Ask a clinical question…"
                  }
                  value={draft}
                  maxLength={12000}
                  rows={2}
                  onChange={(e) => setDraft(e.target.value)}
                  onKeyDown={(e) => {
                    if (
                      e.key === "Enter" &&
                      !e.shiftKey &&
                      !e.nativeEvent.isComposing
                    ) {
                      e.preventDefault();
                      submit();
                    }
                  }}
                  disabled={sending}
                />
                <div className="composer-bottom">
                  <div className="composer-tools">
                    <button
                      type="button"
                      className="attach-button"
                      disabled={uploading || busy}
                      onClick={() => uploadRef.current?.click()}
                    >
                      {uploading ? (
                        <LoaderCircle size={17} className="spin" />
                      ) : (
                        <Paperclip size={17} />
                      )}
                      <span>
                        {uploading ? "Reading document…" : "Attach PDF"}
                      </span>
                    </button>
                    {!!context?.files.length && (
                      <button
                        type="button"
                        className="attached-pill"
                        onClick={() => setPanel("files")}
                      >
                        <FileText size={13} />
                        {context.files.length} attached
                      </button>
                    )}
                  </div>
                  <button
                    className="send-button"
                    type="submit"
                    aria-label="Send question"
                    disabled={!draft.trim() || busy || uploading || !boot}
                  >
                    {busy ? (
                      <LoaderCircle size={19} className="spin" />
                    ) : (
                      <ArrowUp size={20} />
                    )}
                  </button>
                </div>
              </form>
              <div className="composer-caption">
                <span>
                  <ShieldCheck size={12} /> Sources remain available for
                  inspection.
                </span>
                <span>
                  Enter to send <span className="caption-separator">·</span>{" "}
                  Shift + Enter for a new line
                </span>
              </div>
            </div>
          </section>
          {panel && (
            <aside className="inspector" aria-label="Context panel">
              <div className="inspector-heading">
                <h2>
                  {
                    {
                      sources: "Sources",
                      files: "Documents",
                      activity: "Agent activity",
                      patient: "Patient context",
                      examples: "Try an example",
                    }[panel]
                  }
                </h2>
                <button
                  className="icon-button"
                  aria-label="Close context panel"
                  onClick={() => setPanel(null)}
                >
                  <PanelRightClose size={18} />
                </button>
              </div>
              <div className="inspector-content">
                {panel === "sources" && (
                  <Sources
                    message={inspected}
                    source={source}
                    select={(id) => setSelectedSource(id)}
                  />
                )}
                {panel === "files" && (
                  <>
                    <p className="panel-description">
                      {currentPatient
                        ? "Files registered to this patient are available across their conversations."
                        : "Documents attached to this conversation with no patient context."}
                    </p>
                    {context?.files.map((file) => (
                      <a
                        key={file.document_id}
                        className="document-item"
                        href={`/api/sessions/${active}/documents/${file.document_id}/file`}
                        target="_blank"
                        rel="noreferrer"
                      >
                        <FileText size={21} />
                        <span>
                          <strong>{file.title}</strong>
                          <small>
                            {file.page_count}{" "}
                            {file.page_count === 1 ? "page" : "pages"} ·
                            Selectable text
                          </small>
                        </span>
                        <ExternalLink size={14} />
                      </a>
                    ))}
                    {!context?.files.length && (
                      <div className="panel-empty">
                        <FileText size={31} />
                        <h3>Bring a document into view.</h3>
                        <p>
                          Attach a PDF to ask about its contents and inspect the
                          retrieved passages.
                        </p>
                      </div>
                    )}
                    <button
                      className="secondary-button full"
                      disabled={uploading || busy}
                      onClick={() => uploadRef.current?.click()}
                    >
                      <Plus size={16} /> Attach a PDF
                    </button>
                    <p className="fine-print">
                      Selectable text · Up to 50 MiB and 250 pages
                    </p>
                  </>
                )}
                {panel === "activity" && <ActivityPanel message={inspected} />}
                {panel === "patient" && (
                  <PatientPanel
                    context={context}
                    patients={boot?.patients || []}
                    onSelect={changePatient}
                    onCreated={async (patient) => {
                      await newConversation(patient.patient_id);
                      setPanel("patient");
                      await refresh();
                    }}
                    onUpdate={async () => {
                      if (active) await load(active);
                    }}
                    notify={notify}
                  />
                )}
                {panel === "examples" && (
                  <>
                    <p className="panel-description">
                      Each click prepares the context and an editable question.
                      Press Send to ask the live agent.
                    </p>
                    {[
                      { id: "original", title: "Quick examples" },
                      {
                        id: "continuity",
                        title: "A three-step continuity flow",
                      },
                    ].map((flow) => (
                      <section className="example-flow" key={flow.id}>
                        <h3 className="small-heading">{flow.title}</h3>
                        {flow.id === "continuity" && (
                          <p className="fine-print">
                            Try these in order: find evidence, read the case,
                            then continue in a new conversation.
                          </p>
                        )}
                        {boot?.examples
                          .filter((item) => item.flow === flow.id)
                          .map((item, i) => (
                            <button
                              className="drawer-example"
                              key={item.id}
                              onClick={() => example(item.id)}
                              disabled={workingExample || busy || uploading}
                            >
                              <span className="example-step">{i + 1}</span>
                              <span>
                                <strong>{item.title}</strong>
                                <small>{item.description}</small>
                              </span>
                              <ChevronRight size={16} />
                            </button>
                          ))}
                      </section>
                    ))}
                    <div className="panel-divider" />
                    <h3 className="small-heading">Presentation backup</h3>
                    <p className="panel-description">
                      Open a clearly labelled response saved during the previous
                      evaluation. This makes no model request.
                    </p>
                    <button
                      className="secondary-button full"
                      disabled={workingExample || busy || uploading}
                      onClick={() => example("saved")}
                    >
                      <History size={16} /> Open saved PDF example
                    </button>
                  </>
                )}
              </div>
            </aside>
          )}
        </main>
      </div>
      <input
        ref={uploadRef}
        type="file"
        accept="application/pdf,.pdf"
        className="visually-hidden"
        aria-label="Select a PDF"
        onChange={(e) => upload(e.target.files[0])}
      />
      {settings && (
        <AppearanceModal
          appearance={appearance}
          onClose={() => setSettings(false)}
          canReset={!busy && !uploading && !workingExample}
          onReset={async () => {
            await post("/reset", { confirm: true });
            activate(null);
            setDraft("");
            setSearch("");
            await refresh();
            setSettings(false);
            notify(
              "Conversations cleared. Patient records and documents are kept.",
            );
          }}
          onSave={async (value) => {
            try {
              const saved = await api("/appearance", {
                method: "PUT",
                body: JSON.stringify(value),
              });
              setAppearance(saved);
              setSettings(false);
              notify("Workspace appearance saved");
            } catch (error) {
              notify(error.message);
            }
          }}
        />
      )}
      {toast && (
        <div className="toast" role="status">
          <span>{toast}</span>
          <button
            className="icon-button"
            aria-label="Dismiss notification"
            onClick={() => setToast("")}
          >
            <X size={15} />
          </button>
        </div>
      )}
    </div>
  );
}

function Sources({ message, source, select }) {
  const items = message?.result?.sources || [];
  if (!items.length)
    return (
      <div className="panel-empty">
        <BookOpen size={31} />
        <h3>No retrieved sources yet.</h3>
        <p>
          When the agent retrieves a paper or PDF passage, its delivered text
          will appear here.
        </p>
      </div>
    );
  return (
    <>
      {source ? (
        <>
          <button
            className="text-button muted back-button"
            onClick={() => select(null)}
          >
            ← All sources
          </button>
          <div className="source-type">
            {source.pdf_page ? "Document passage" : "PubMed record"}
            {source.cited && (
              <span>
                <Check size={12} /> Cited
              </span>
            )}
          </div>
          <h3 className="source-title">{source.title}</h3>
          <p className="source-meta">
            {source.pdf_page
              ? `PDF page ${source.pdf_page}`
              : [source.year, source.pmid ? `PMID ${source.pmid}` : ""]
                  .filter(Boolean)
                  .join(" · ")}
          </p>
          <div className="passage-heading">Text delivered to the model</div>
          <blockquote className="source-passage">{source.text}</blockquote>
          {source.truncated && (
            <p className="fine-print">
              This delivered excerpt is truncated; it is not the full article.
            </p>
          )}
          <a
            className="secondary-button full"
            href={source.file_url || source.url}
            target="_blank"
            rel="noreferrer"
          >
            <ExternalLink size={15} />
            {source.pdf_page ? "Open PDF at this page" : "Open PubMed record"}
          </a>
          <p className="fine-print">
            An existing reference does not by itself establish support for every
            statement.
          </p>
        </>
      ) : (
        <>
          <p className="panel-description">
            Actual source views delivered during this answer. Cited references
            are marked separately.
          </p>
          {items.map((item, i) => (
            <button
              className="source-item"
              key={item.id}
              onClick={() => select(item.id)}
            >
              <span className="source-number">{i + 1}</span>
              <span>
                <strong>{item.title}</strong>
                <small>
                  {item.pdf_page
                    ? `PDF page ${item.pdf_page}`
                    : `PubMed${item.year ? " · " + item.year : ""}`}
                  {item.cited ? " · Cited" : " · Retrieved"}
                </small>
              </span>
              <ChevronRight size={14} />
            </button>
          ))}
        </>
      )}
    </>
  );
}

function ActivityPanel({ message }) {
  if (!message)
    return (
      <div className="panel-empty">
        <Activity size={31} />
        <h3>Follow the work.</h3>
        <p>
          Tool calls and saved interactions will appear here as the agent runs.
        </p>
      </div>
    );
  const usage = message.result?.usage;
  return (
    <>
      <p className="panel-description">Recorded actions for this response.</p>
      <ol className="activity-list">
        {message.events.map((event) => (
          <li
            key={event.sequence}
            className={event.kind === "error" ? "failed" : ""}
          >
            <span className="activity-marker">
              {event.kind === "error" ? <X size={11} /> : <Check size={11} />}
            </span>
            <div>
              <strong>{event.label}</strong>
              {event.arguments?.query && <p>“{event.arguments.query}”</p>}
              {event.arguments?.fact_key && (
                <p>{fieldName(event.arguments.fact_key)}</p>
              )}
              {event.delivered_views !== undefined && (
                <small>{event.delivered_views} source views delivered</small>
              )}
              <time>
                {new Date(event.at).toLocaleTimeString("en-GB", {
                  hour: "2-digit",
                  minute: "2-digit",
                  second: "2-digit",
                })}
              </time>
            </div>
          </li>
        ))}
      </ol>
      {usage && (
        <>
          <div className="panel-divider" />
          <h3 className="small-heading">
            Request details{message.result.archive ? " · Historical" : ""}
          </h3>
          <dl className="usage-list">
            <div>
              <dt>Model requests</dt>
              <dd>{usage.model_requests}</dd>
            </div>
            <div>
              <dt>Input tokens</dt>
              <dd>{usage.input_tokens.toLocaleString()}</dd>
            </div>
            <div>
              <dt>Known inference cost</dt>
              <dd>${usage.known_cost_usd.toFixed(6)}</dd>
            </div>
            {!!usage.unknown_charge_records && (
              <div>
                <dt>Unreported charges</dt>
                <dd>{usage.unknown_charge_records}</dd>
              </div>
            )}
          </dl>
        </>
      )}
    </>
  );
}

function PatientPanel({
  context,
  patients,
  onSelect,
  onCreated,
  onUpdate,
  notify,
}) {
  const [name, setName] = useState(""),
    [creating, setCreating] = useState(false),
    [edit, setEdit] = useState(null),
    [form, setForm] = useState(false);
  const patient = context?.session?.patient_id;
  return (
    <>
      <p className="panel-description">
        {patient
          ? "Explicit records remain separate from the assistant’s public conversation history."
          : "Add patient context when the question needs it. You can also ask without patient context."}
      </p>
      {patient && (
        <>
          <div className="patient-card">
            <span className="patient-avatar">
              <UserRound size={24} />
            </span>
            <div>
              <h3>{context.session.patient_name}</h3>
              <span>Selected patient</span>
            </div>
          </div>
          <div className="section-heading">
            <h3 className="small-heading">Current records</h3>
            <button
              className="text-button"
              onClick={() => {
                setEdit(null);
                setForm(true);
              }}
            >
              <Plus size={14} /> Add
            </button>
          </div>
          {Object.values(context.facts).map((fact) => (
            <div className="fact-item" key={fact.fact_id}>
              <div>
                <strong>{fieldName(fact.fact_key)}</strong>
                <p>
                  {String(fact.value)} {fact.unit || ""}
                </p>
                <small>Observed {shortDate(fact.observed_at)}</small>
              </div>
              <button
                className="text-button muted"
                onClick={() => {
                  setEdit(fact);
                  setForm(true);
                }}
              >
                Correct
              </button>
            </div>
          ))}
          {!Object.keys(context.facts).length && (
            <p className="fine-print">
              No explicit patient measurements have been recorded.
            </p>
          )}
          <p className="fine-print">
            New observations and corrections are recorded explicitly. Assistant
            responses do not become verified records.
          </p>
          <details className="history-details">
            <summary>Dated record history</summary>
            {context.fact_history.map((fact) => (
              <div
                className={
                  "version-item " + (fact.superseded ? "superseded" : "")
                }
                key={fact.fact_id}
              >
                <strong>
                  {fieldName(fact.fact_key)}: {String(fact.value)} {fact.unit}
                </strong>
                <span>
                  {shortDate(fact.observed_at)}
                  {fact.superseded ? " · Superseded" : ""}
                </span>
                <small>{fact.source_ref}</small>
              </div>
            ))}
          </details>
          <div className="panel-divider" />
        </>
      )}
      <h3 className="small-heading">Patients</h3>
      {patients.map((person) => (
        <button
          className={
            "patient-switch " +
            (person.patient_id === patient ? "selected" : "")
          }
          onClick={() => onSelect(person.patient_id)}
          key={person.patient_id}
        >
          <UserRound size={17} />
          <span>{person.name}</span>
          {person.patient_id === patient ? (
            <Check size={15} />
          ) : (
            <ChevronRight size={15} />
          )}
        </button>
      ))}
      <form
        className="new-patient-form"
        onSubmit={async (e) => {
          e.preventDefault();
          if (!name.trim()) return;
          setCreating(true);
          try {
            await onCreated(await post("/patients", { name: name.trim() }));
            setName("");
          } catch (error) {
            notify(error.message);
          } finally {
            setCreating(false);
          }
        }}
      >
        <label htmlFor="patient-name">Create a patient</label>
        <div>
          <input
            id="patient-name"
            placeholder="Patient name or identifier"
            value={name}
            maxLength={120}
            onChange={(e) => setName(e.target.value)}
          />
          <button
            className="icon-button"
            disabled={creating || !name.trim()}
            aria-label="Create patient"
          >
            <Plus size={19} />
          </button>
        </div>
      </form>
      {form && (
        <FactModal
          fact={edit}
          patient={patient}
          onClose={() => setForm(false)}
          onSave={async (value) => {
            try {
              await post("/patients/" + patient + "/facts", value);
              await onUpdate();
              setForm(false);
              notify(
                edit
                  ? "Correction saved; the previous version is retained"
                  : "Patient record saved",
              );
            } catch (error) {
              notify(error.message);
            }
          }}
        />
      )}
    </>
  );
}

function FactModal({ fact, onClose, onSave }) {
  const [key, setKey] = useState(fact?.fact_key || "weight_kg"),
    [value, setValue] = useState(
      fact?.value === undefined ? "" : String(fact.value),
    );
  const [unit, setUnit] = useState(fact?.unit || "kg"),
    [date, setDate] = useState(
      (fact?.observed_at || new Date().toISOString()).slice(0, 10),
    ),
    [source, setSource] = useState("Explicitly entered in the local workspace"),
    [saving, setSaving] = useState(false);
  const [customName, setCustomName] = useState("");
  return (
    <Modal
      title={fact ? "Correct a recorded observation" : "Add a patient record"}
      onClose={onClose}
    >
      <form
        className="modal-form"
        onSubmit={async (e) => {
          e.preventDefault();
          setSaving(true);
          try {
            await onSave({
              fact_key:
                key === "custom"
                  ? customName
                      .trim()
                      .toLowerCase()
                      .replace(/[^a-z0-9]+/g, "_")
                      .replace(/^([^a-z])/, "record_$1")
                      .slice(0, 80)
                  : key,
              value:
                value.trim() !== "" && Number.isFinite(Number(value))
                  ? Number(value)
                  : value,
              unit: unit || null,
              observed_at: date + "T12:00:00+00:00",
              source_ref: source,
              origin: "user_provided",
              ...(fact ? { supersedes_fact_id: fact.fact_id } : {}),
            });
          } finally {
            setSaving(false);
          }
        }}
      >
        <label>
          Measurement or note
          {fact ? (
            <input value={fieldName(key)} disabled />
          ) : (
            <select
              value={key}
              onChange={(e) => {
                setKey(e.target.value);
                setUnit(
                  {
                    weight_kg: "kg",
                    albumin_g_dl: "g/dL",
                    snack: "",
                    custom: "",
                  }[e.target.value],
                );
              }}
            >
              <option value="weight_kg">Weight</option>
              <option value="albumin_g_dl">Serum albumin</option>
              <option value="snack">Preferred snack</option>
              <option value="custom">Another measurement or note</option>
            </select>
          )}
        </label>
        {key === "custom" && (
          <label>
            Name
            <input
              required
              maxLength={80}
              placeholder="For example, appetite"
              value={customName}
              onChange={(e) => setCustomName(e.target.value)}
            />
          </label>
        )}
        <div className="form-row">
          <label>
            Value
            <input
              required
              value={value}
              onChange={(e) => setValue(e.target.value)}
            />
          </label>
          <label>
            Unit
            <input
              value={unit}
              onChange={(e) => setUnit(e.target.value)}
              maxLength={40}
            />
          </label>
        </div>
        <label>
          Observation date
          <input
            required
            type="date"
            value={date}
            onChange={(e) => setDate(e.target.value)}
          />
        </label>
        <label>
          Source or note
          <input
            required
            value={source}
            maxLength={300}
            onChange={(e) => setSource(e.target.value)}
          />
        </label>
        <p className="fine-print">
          {fact
            ? "This explicitly replaces the selected observation. Its previous value remains in the dated history."
            : "A new observation retains its observation date and recorded origin."}
        </p>
        <button className="primary-button" disabled={saving}>
          {saving ? "Saving…" : fact ? "Save correction" : "Save record"}
        </button>
      </form>
    </Modal>
  );
}

function AppearanceModal({ appearance, onClose, onSave, onReset, canReset }) {
  const [value, setValue] = useState(appearance),
    [saving, setSaving] = useState(false),
    [error, setError] = useState("");
  const [confirmReset, setConfirmReset] = useState(false),
    [resetting, setResetting] = useState(false),
    [resetError, setResetError] = useState("");
  const update = (key, item) => setValue((old) => ({ ...old, [key]: item }));
  return (
    <Modal title="Make the workspace yours" onClose={onClose}>
      <form
        className="modal-form"
        onSubmit={async (e) => {
          e.preventDefault();
          setSaving(true);
          try {
            await onSave(value);
          } finally {
            setSaving(false);
          }
        }}
      >
        <p className="modal-description">
          A quiet interface, tuned to the way you read.
        </p>
        <label>
          Workspace name
          <input
            required
            value={value.name}
            maxLength={50}
            onChange={(e) => update("name", e.target.value)}
          />
        </label>
        <div className="form-row">
          <label>
            Accent colour
            <div className="colour-control">
              <input
                type="color"
                aria-label="Accent colour"
                value={value.accent}
                onChange={(e) => update("accent", e.target.value)}
              />
              <span>{value.accent.toUpperCase()}</span>
            </div>
          </label>
          <label>
            Typography
            <select
              value={value.font}
              onChange={(e) => update("font", e.target.value)}
            >
              <option value="editorial">Editorial serif</option>
              <option value="modern">Modern sans serif</option>
            </select>
          </label>
        </div>
        <div className="form-row">
          <label>
            Density
            <select
              value={value.density}
              onChange={(e) => update("density", e.target.value)}
            >
              <option value="comfortable">Comfortable</option>
              <option value="compact">Compact</option>
            </select>
          </label>
          <label>
            Reading size · {Math.round(value.font_scale * 100)}%
            <input
              type="range"
              min=".9"
              max="1.25"
              step=".05"
              value={value.font_scale}
              onChange={(e) => update("font_scale", Number(e.target.value))}
            />
          </label>
        </div>
        <label>
          Logo{" "}
          <span className="optional-field">
            Optional · PNG, JPEG or WebP · 200 KiB
          </span>
          <input
            type="file"
            accept="image/png,image/jpeg,image/webp"
            onChange={(e) => {
              const file = e.target.files[0];
              if (!file) return;
              if (file.size > 200 * 1024)
                return setError("Choose a logo below 200 KiB.");
              const reader = new FileReader();
              reader.onload = () => {
                update("logo", reader.result);
                setError("");
              };
              reader.readAsDataURL(file);
            }}
          />
        </label>
        {value.logo && (
          <div className="logo-preview">
            <img src={value.logo} alt="Workspace logo preview" />
            <button
              type="button"
              className="text-button"
              onClick={() => update("logo", null)}
            >
              Remove logo
            </button>
          </div>
        )}
        {error && <p className="answer-error">{error}</p>}
        <div className="panel-divider" />
        <div className="model-setting">
          <BookOpen size={20} />
          <div>
            <strong>Gemma 4 and Kimi K3</strong>
            <span>OpenRouter · Reasoning enabled</span>
            <small>
              Choose the model in the conversation header. Credentials stay on
              the local server.
            </small>
          </div>
        </div>
        <section className="reset-section">
          <h3 className="small-heading">Start with a clear workspace</h3>
          <p>
            Clear every conversation and its remembered public history. Patient
            records, documents and appearance settings stay available.
          </p>
          {!canReset && (
            <p>Wait for the current action to finish before resetting.</p>
          )}
          {confirmReset ? (
            <div className="reset-confirmation">
              <p>Delete all conversations? This cannot be undone.</p>
              <div className="reset-actions">
                <button
                  type="button"
                  className="text-button muted"
                  disabled={resetting}
                  onClick={() => setConfirmReset(false)}
                >
                  Cancel
                </button>
                <button
                  type="button"
                  className="reset-button"
                  disabled={!canReset || resetting || saving}
                  onClick={async () => {
                    setResetting(true);
                    setResetError("");
                    try {
                      await onReset();
                    } catch (error) {
                      setResetError(error.message);
                    } finally {
                      setResetting(false);
                    }
                  }}
                >
                  {resetting ? "Clearing…" : "Delete all conversations"}
                </button>
              </div>
            </div>
          ) : (
            <button
              type="button"
              className="reset-button"
              disabled={!canReset || saving}
              onClick={() => setConfirmReset(true)}
            >
              Reset conversations
            </button>
          )}
          {resetError && (
            <p className="answer-error" role="alert">
              {resetError}
            </p>
          )}
        </section>
        <div className="modal-actions">
          <button
            className="text-button muted"
            type="button"
            onClick={() => setValue(DEFAULTS)}
          >
            Restore defaults
          </button>
          <button className="primary-button" disabled={saving || resetting}>
            {saving ? "Saving…" : "Save appearance"}
          </button>
        </div>
      </form>
    </Modal>
  );
}

createRoot(document.getElementById("root")).render(<App />);
