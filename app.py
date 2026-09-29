"""
Jarvis - a LangChain agent with real weather and stock tools, served as a
Streamlit chat app.

Run locally:   python -m streamlit run app.py
Deploy:        Streamlit Community Cloud (see README.md)

How to see what's happening:
  * In the chat: the live "status" box shows each step (model call, tool call,
    timings, errors).
  * Sidebar -> "Run health check": tests your key, the model, and both tools.
  * Terminal (local) / "Manage app" logs (cloud): one log line per step.
  * Optional: LangSmith tracing (see README) for a full visual trace.
"""

import logging
import os
import time
import uuid

import requests
import streamlit as st
import threading

from langchain.agents import create_agent
from langchain.agents.middleware import (AgentMiddleware, ModelFallbackMiddleware,
                                         ModelRetryMiddleware)
from langchain_core.messages import AIMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter
from langgraph.checkpoint.memory import InMemorySaver

from tools import get_stock_price, get_weather

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-7s  %(message)s", force=True)
log = logging.getLogger("jarvis")



# ---------------------------------------------------------------- settings --
def _setting(name: str, default: str | None = None) -> str | None:
    """Read from Streamlit secrets first, then environment variables."""
    try:
        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:  # no secrets file at all
        pass
    return os.environ.get(name, default)


API_KEY = _setting("OPENROUTER_API_KEY")
DEFAULT_MODELS = ("inclusionai/ling-3.0-flash-sante:free, google/gemma-4-26b-a4b-it:free, "
                  "nvidia/nemotron-3-super-120b-a12b:free, google/gemma-4-31b-it:free, "
                  "qwen/qwen3.8-27b:free, openrouter/free, nvidia/nemotron-3.5-lightning:free")
# MODEL can be one id, or several separated by commas. They are tried IN ORDER:
# if one fails (rate limit, overloaded, timeout...), the next one is used.
MODELS = [m.strip() for m in _setting("MODEL", DEFAULT_MODELS).split(",") if m.strip()]
MODEL = MODELS[0]
# Seconds to wait for ONE model reply before giving up (free models can be slow)
LLM_TIMEOUT = int(_setting("LLM_TIMEOUT", "180"))
# If EVERY model fails, wait and try the whole list again this many times
# (waits 5s, then 10s, 20s... between rounds - busy free providers often recover).
RETRY_ROUNDS = int(_setting("RETRY_ROUNDS", "2"))
# How much the model "thinks" before answering: none | minimal | low | medium | high.
# Lower = much faster. Empty = the model's own default.
REASONING_EFFORT = (_setting("REASONING_EFFORT", "low") or "").strip().lower()

# Optional LangSmith tracing: add LANGSMITH_API_KEY to secrets to switch it on
if _setting("LANGSMITH_API_KEY"):
    os.environ["LANGSMITH_API_KEY"] = _setting("LANGSMITH_API_KEY")
    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_PROJECT"] = _setting("LANGSMITH_PROJECT", "jarvis-agent")

SYSTEM_PROMPT = (
    "You are an AI assistant, your name is Jarvis developed by Harish.\n"
    "You always respond to the user's query politely. DO NOT USE PROFANE WORDS.\n"
    "Use the get_weather tool for weather questions and the get_stock_price tool "
    "for share-price questions. Never make up weather or prices."
)

st.set_page_config(page_title="Jarvis", page_icon="🤖")
st.title("🤖 Jarvis")
st.caption(f"My first LangChain agent · model: `{MODEL}`"
           + (f" (+{len(MODELS) - 1} fallbacks)" if len(MODELS) > 1 else "") + " · "
           "weather by Open-Meteo.com · prices by Yahoo Finance")

if not API_KEY:
    st.error("OPENROUTER_API_KEY is not set. Add it in the app's **Secrets** "
             "(Streamlit Cloud) or in `.streamlit/secrets.toml` (local).")
    st.stop()


# ------------------------------------------------------------------- agent --
def build_llm(model: str, api_key: str, timeout_s: int, effort: str = ""):
    # NOTE: ChatOpenRouter's timeout is in MILLISECONDS, hence * 1000.
    # max_retries=0: fail fast so we can move on to the next model quickly,
    # instead of the SDK silently retrying the same busy model for minutes.
    return ChatOpenRouter(model=model, temperature=0, openrouter_api_key=api_key,
                          timeout=timeout_s * 1000, max_retries=0,
                          app_title="Jarvis agent",
                          reasoning={"effort": effort} if effort else None)


# Every model attempt (success or failure), per conversation thread_id, so the
# status box can show fallbacks. (The agent runs its steps on worker threads,
# so we key by the conversation id from the LangGraph config.)
# st.cache_resource keeps ONE shared dict across Streamlit reruns (the agent is
# cached too, so both must point at the same object).
@st.cache_resource
def _attempt_store():
    return {}, threading.Lock()


ATTEMPTS, _ATTEMPTS_LOCK = _attempt_store()


def _record_attempt(item):
    try:
        from langgraph.config import get_config
        tid = get_config().get("configurable", {}).get("thread_id")
    except Exception:
        tid = None
    if tid:
        with _ATTEMPTS_LOCK:
            ATTEMPTS.setdefault(tid, []).append(item)


def short_error(exc: Exception) -> str:
    """Turn long SDK errors into one readable line."""
    text = str(exc)
    for marker, label in (("provider_overloaded", "provider overloaded"),
                          ("'timeout'", "provider timed out"),
                          ("TooManyRequests", "rate-limited (too many requests)"),
                          ("Forbidden", "not allowed for this app"),
                          ("only available on", "not allowed for this app")):
        if marker in text or marker in type(exc).__name__:
            return label
    if "timed out" in text.lower():
        return "timed out"
    return f"{type(exc).__name__}: {text[:120]}"


class AttemptLogger(AgentMiddleware):
    """Innermost middleware: sees each single model attempt (incl. fallbacks)."""

    def wrap_model_call(self, request, handler):
        name = getattr(request.model, "model_name", "?")
        t0 = time.time()
        try:
            response = handler(request)
        except Exception as exc:
            secs = time.time() - t0
            log.warning("ATTEMPT %s failed after %.1fs: %s", name, secs, short_error(exc))
            _record_attempt((name, False, secs, short_error(exc)))
            raise
        secs = time.time() - t0
        log.info("ATTEMPT %s ok (%.1fs)", name, secs)
        _record_attempt((name, True, secs, ""))
        return response


def _retryable(exc: Exception) -> bool:
    """Don't retry things that will never work (bad key, forbidden, bad model id)."""
    t = f"{type(exc).__name__} {exc}".lower()
    return not any(k in t for k in ("unauthorized", "401", "invalid api key"))


@st.cache_resource
def build_agent(models: tuple, api_key: str, timeout_s: int, effort: str, rounds: int):
    log.info("Building agent: %s | timeout %ss | reasoning %s | retry rounds %s",
             " -> ".join(models), timeout_s, effort or "default", rounds)
    llms = [build_llm(m, api_key, timeout_s, effort) for m in models]
    middleware = []
    if rounds > 0:  # outer: if the WHOLE list fails, wait and try the list again
        middleware.append(ModelRetryMiddleware(max_retries=rounds, retry_on=_retryable,
                                               on_failure="error", initial_delay=5.0,
                                               backoff_factor=2.0, max_delay=60.0))
    if len(llms) > 1:  # middle: try the next model when one fails
        middleware.append(ModelFallbackMiddleware(*llms[1:]))
    middleware.append(AttemptLogger())  # inner: log every single attempt
    return create_agent(
        model=llms[0],
        system_prompt=SYSTEM_PROMPT,
        tools=[get_weather, get_stock_price],
        middleware=middleware,
        checkpointer=InMemorySaver(),
    )


agent = build_agent(tuple(MODELS), API_KEY, LLM_TIMEOUT, REASONING_EFFORT, RETRY_ROUNDS)


# ------------------------------------------------------------ health check --
def health_check():
    """Test each moving part separately so you know which one is broken."""
    results = []

    def check(name, fn):
        t0 = time.time()
        try:
            ok, detail = fn()
        except Exception as e:
            ok, detail = False, f"{type(e).__name__}: {e}"
        secs = time.time() - t0
        log.info("HEALTH %-22s %s (%.1fs) %s", name, "OK" if ok else "FAIL", secs, detail)
        results.append((name, ok, f"{detail}  ({secs:.1f}s)"))

    def key_info():
        r = requests.get("https://openrouter.ai/api/v1/key",
                         headers={"Authorization": f"Bearer {API_KEY}"}, timeout=15)
        if r.status_code != 200:
            return False, f"HTTP {r.status_code} - key rejected? {r.text[:150]}"
        d = r.json().get("data", {})
        free = d.get("free_model_daily_requests") or {}
        msg = "key is valid"
        if free:
            msg += f"; free-model requests today: {free.get('used')} used / {free.get('limit')} limit"
        return True, msg

    def model_reply():
        # Try each model in order (like the agent does) and report what happened
        notes = []
        for m in MODELS:
            t0 = time.time()
            try:
                reply = build_llm(m, API_KEY, LLM_TIMEOUT, REASONING_EFFORT).invoke(
                    "Reply with just the word OK.")
            except Exception as e:
                notes.append(f"{m}: ✗ {short_error(e)}")
                continue
            notes.append(f"{m}: ✓ {time.time() - t0:.1f}s")
            return True, "; ".join(notes)
        return False, "all models failed - " + "; ".join(notes)

    def weather():
        out = get_weather.invoke({"city": "Bengaluru"})
        return not out.startswith(("Weather service error", "Could not")), out

    def stock():
        out = get_stock_price.invoke({"company": "INFY.NS"})
        return not out.startswith("Could not"), out

    check("OpenRouter API key", key_info)
    check("LLM model call", model_reply)
    check("Weather tool (Open-Meteo)", weather)
    check("Stock tool (Yahoo)", stock)
    return results


def free_tool_models():
    """Ask OpenRouter which free models currently support tool calling."""
    r = requests.get("https://openrouter.ai/api/v1/models",
                     params={"supported_parameters": "tools"}, timeout=20)
    r.raise_for_status()
    return sorted(m["id"] for m in r.json().get("data", []) if m["id"].endswith(":free"))


# ----------------------------------------------------------------- sidebar --
if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())
    st.session_state.chat = []  # [(role, text, tools_used)]

with st.sidebar:
    st.markdown("**Try asking**")
    st.markdown("- How's the weather in Indore?\n"
                "- What's HAL's share price today?\n"
                "- Compare Infosys and TCS stock prices\n"
                "- What was my first question?")
    if st.button("New conversation"):
        st.session_state.thread_id = str(uuid.uuid4())
        st.session_state.chat = []
        st.rerun()

    st.divider()
    st.markdown("**🩺 Diagnostics**")
    if st.button("Run health check"):
        with st.spinner("Checking..."):
            st.session_state.health = health_check()
    for name, ok, detail in st.session_state.get("health", []):
        (st.success if ok else st.error)(f"**{name}**: {detail}")
    if st.button("List free models that support tools"):
        try:
            st.session_state.free_models = free_tool_models()
        except Exception as e:
            st.session_state.free_models = [f"(could not load: {e})"]
    if st.session_state.get("free_models") is not None:
        st.caption("Copy one or more into MODEL in secrets (comma-separated):")
        st.code("\n".join(st.session_state.free_models) or "(none right now)", language=None)
    st.caption(f"{len(MODELS)} models · retry rounds: {RETRY_ROUNDS} · timeout: {LLM_TIMEOUT}s · reasoning: {REASONING_EFFORT or 'default'} · LangSmith tracing: " + ("ON" if os.environ.get("LANGSMITH_TRACING") else "off"))


def show_tools(tools_used):
    if tools_used:
        with st.expander(f"🔧 Used {len(tools_used)} tool call(s)"):
            for name, args, result in tools_used:
                st.markdown(f"**{name}**(`{args}`)")
                st.code(result, language=None)


for role, text, tools_used in st.session_state.chat:
    with st.chat_message(role):
        st.markdown(text)
        show_tools(tools_used)

# ------------------------------------------------------------------ chat ----
if question := st.chat_input("Ask Jarvis anything..."):
    st.session_state.chat.append(("user", question, []))
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        config = {"configurable": {"thread_id": st.session_state.thread_id}}
        tools_used, pending, answer, answered_by = [], {}, None, None
        t_start = step_start = time.time()
        log.info("QUESTION %r (thread %s)", question, st.session_state.thread_id[:8])

        # st.status shows every step live, so you can see exactly where it waits
        tid = st.session_state.thread_id
        with _ATTEMPTS_LOCK:
            ATTEMPTS[tid] = []
        attempts = ATTEMPTS[tid]
        shown = [0]

        def show_attempts():
            """Print failed model attempts (fallbacks / retries) not shown yet."""
            for name, ok, secs, err in attempts[shown[0]:]:
                if not ok:
                    st.write(f"⚠️ `{name}` failed after {secs:.1f}s ({err})")
            shown[0] = len(attempts)

        def last_ok_model():
            return next((n for n, ok, _, _ in reversed(attempts) if ok), None)
        with st.status("🧠 Asking the model...", expanded=True) as status:
            try:
                # stream_mode="updates" yields after each graph step (model or tools)
                for update in agent.stream({"messages": [("user", question)]},
                                           config=config, stream_mode="updates"):
                    secs = time.time() - step_start
                    step_start = time.time()
                    show_attempts()
                    for node, data in update.items():
                        for msg in (data or {}).get("messages", []):
                            served = ((getattr(msg, "response_metadata", None) or {}).get("model_name")
                                      or last_ok_model() or "?")
                            if isinstance(msg, AIMessage) and msg.tool_calls:
                                for tc in msg.tool_calls:
                                    pending[tc["id"]] = tc
                                    st.write(f"🧠 `{served}` decided to call **{tc['name']}**"
                                             f"(`{tc['args']}`) — {secs:.1f}s")
                                    log.info("MODEL %s -> tool %s %s (%.1fs)", served, tc["name"], tc["args"], secs)
                                status.update(label="🔧 Running tools...")
                            elif isinstance(msg, ToolMessage):
                                tc = pending.get(msg.tool_call_id, {"name": msg.name, "args": {}})
                                tools_used.append((tc["name"], tc["args"], msg.content))
                                st.write(f"🔧 **{tc['name']}** returned — {secs:.1f}s")
                                log.info("TOOL %s returned (%.1fs): %s", tc["name"], secs, msg.content[:120])
                                status.update(label="🧠 Model is writing the answer...")
                            elif isinstance(msg, AIMessage):
                                answer = msg.content
                                answered_by = served
                                st.write(f"🧠 `{served}` wrote the answer — {secs:.1f}s")
                                log.info("MODEL %s answered (%.1fs)", served, secs)
                total = time.time() - t_start
                status.update(label=f"✅ Done in {total:.1f}s" + (f" · answered by {answered_by}" if answered_by else ""),
                              state="complete", expanded=False)
                log.info("DONE in %.1fs", total)
            except Exception as e:
                show_attempts()
                total = time.time() - t_start
                log.exception("FAILED after %.1fs", total)
                status.update(label=f"❌ Failed after {total:.1f}s", state="error", expanded=True)
                st.error(f"`{type(e).__name__}: {e}`")
                hint = ""
                text = str(e).lower()
                if len(MODELS) > 1 and attempts and not any(ok for _, ok, _, _ in attempts):
                    hint = (f"All {len(MODELS)} models failed, even after {RETRY_ROUNDS} retry round(s). "
                            "Free providers are very busy right now - try again in a few minutes, "
                            "or add a low-cost paid model to MODEL.")
                elif "reasoning" in text and ("mandatory" in text or "not supported" in text or "none" in text):
                    hint = "This model doesn't accept that reasoning setting. Set REASONING_EFFORT = \"low\" (or remove it) in secrets."
                elif "toomanyrequests" in text or "provider returned error" in text:
                    hint = ("The free model's provider is overloaded (too many requests from all users). "
                            "Try again shortly; your fallback models are used automatically if listed in MODEL.")
                elif "error_type" in text and "timeout" in text:
                    hint = ("OpenRouter's server gave up waiting for the model (the model "
                            "provider timed out). Free models are often overloaded. Add fallback "
                            "models to MODEL (comma-separated) or pick another model.")
                elif "429" in text or "rate" in text:
                    hint = ("The free model is rate-limited (20 requests/min). "
                            "Wait a minute, or set a different MODEL in secrets.")
                elif "401" in text or "auth" in text:
                    hint = "Your OpenRouter key was rejected. Check OPENROUTER_API_KEY."
                elif "timeout" in text or "timed out" in text:
                    hint = f"The model didn't reply within {LLM_TIMEOUT}s. Free models can be slow; try again, raise LLM_TIMEOUT or switch MODEL."
                elif "404" in text or "no endpoints" in text:
                    hint = "That model id isn't available on OpenRouter. Change MODEL in secrets."
                answer = "Sorry, something went wrong. " + (hint or "See the error above and the terminal log.")

        st.markdown(answer or "(the model returned an empty answer)")
        show_tools(tools_used)

    st.session_state.chat.append(("assistant", answer or "", tools_used))
