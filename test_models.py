"""
Test OpenRouter models with your key before relying on them.

Usage (from the project folder, with .venv active):
    python test_models.py                              # each model in MODEL, one by one
    python test_models.py qwen/qwen3.8-27b:free        # just this id
    python test_models.py model-a:free model-b:free    # several ids
    python test_models.py --effort none                # with "thinking" switched off
    python test_models.py --chain                      # the whole MODEL list with
                                                       # fallback, exactly like the app

For each model it checks the two things Jarvis needs:
  1. Tool calling  - asked about Pune weather, does it call get_weather?
  2. Final answer  - given the real tool result, does it write an answer?
Reads OPENROUTER_API_KEY / MODEL / LLM_TIMEOUT / REASONING_EFFORT from
.streamlit/secrets.toml.
"""

import sys
import time
import tomllib
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage
from langchain_openrouter import ChatOpenRouter

from tools import get_stock_price, get_weather

secrets_file = Path(__file__).parent / ".streamlit" / "secrets.toml"
secrets = tomllib.loads(secrets_file.read_text(encoding="utf-8")) if secrets_file.exists() else {}
API_KEY = secrets.get("OPENROUTER_API_KEY")
TIMEOUT = int(secrets.get("LLM_TIMEOUT", 180))
if not API_KEY:
    sys.exit("OPENROUTER_API_KEY not found in .streamlit/secrets.toml")

args = sys.argv[1:]
EFFORT = str(secrets.get("REASONING_EFFORT", "low")).strip().lower()
if "--effort" in args:
    i = args.index("--effort")
    EFFORT = args[i + 1].lower()
    del args[i:i + 2]
CHAIN = "--chain" in args
if CHAIN:
    args.remove("--chain")

models = args or [m.strip() for m in str(secrets.get("MODEL", "")).split(",") if m.strip()]
if not models:
    sys.exit("No model ids given and MODEL is empty in secrets.toml")
print(f"Reasoning effort: {EFFORT or 'model default'}   timeout per call: {TIMEOUT}s")

TOOLS = {"get_weather": get_weather, "get_stock_price": get_stock_price}


def short_error(e: Exception) -> str:
    text = str(e)
    for marker, label in (("provider_overloaded", "provider overloaded"),
                          ("'timeout'", "provider timed out"),
                          ("TooManyRequests", "rate-limited (free provider busy)"),
                          ("only available on", "not allowed for this app - remove it"),
                          ("Forbidden", "not allowed for this app - remove it")):
        if marker in text or marker in type(e).__name__:
            return label
    return f"{type(e).__name__}: {text[:160]}"


def make_llm(model_id: str):
    return ChatOpenRouter(model=model_id, temperature=0, openrouter_api_key=API_KEY,
                          timeout=TIMEOUT * 1000, max_retries=0,
                          reasoning={"effort": EFFORT} if EFFORT else None
                          ).bind_tools(list(TOOLS.values()))


def call(model_ids: list[str], msgs):
    """Try the models in order (like the app's fallback). Returns (reply, model, secs)."""
    last = None
    for m in model_ids:
        t0 = time.time()
        try:
            return make_llm(m).invoke(msgs), m, time.time() - t0
        except Exception as e:
            last = e
            if len(model_ids) > 1:
                print(f"        {m}: failed after {time.time() - t0:.1f}s ({short_error(e)}) -> next")
    raise last


def test(model_ids: list[str]) -> bool:
    label = model_ids[0] if len(model_ids) == 1 else f"chain of {len(model_ids)} models"
    print(f"\n=== {label}")
    msgs = [SystemMessage("You are Jarvis. Use tools for weather and stock prices."),
            HumanMessage("How is the weather in Pune right now?")]
    t_start = time.time()

    # Step 1: does the model call the tool?
    try:
        ai, used, secs = call(model_ids, msgs)
    except Exception as e:
        print(f"  FAIL  step 1 (tool call): {short_error(e)}")
        return False
    if not ai.tool_calls:
        print(f"  FAIL  step 1: {used} answered without calling a tool ({secs:.1f}s): {ai.content[:150]!r}")
        return False
    tc = ai.tool_calls[0]
    print(f"  OK    step 1: {used} called {tc['name']}({tc['args']}) in {secs:.1f}s")

    # Run the real tool, then step 2: does the model write the answer?
    result = TOOLS[tc["name"]].invoke(tc["args"]) if tc["name"] in TOOLS else "unknown tool"
    msgs += [ai, ToolMessage(content=result, tool_call_id=tc["id"])]
    try:
        final, used, secs = call(model_ids, msgs)
    except Exception as e:
        print(f"  FAIL  step 2 (answer): {short_error(e)}")
        return False
    if not final.content:
        print(f"  FAIL  step 2: {used} returned an empty answer ({secs:.1f}s)")
        return False
    print(f"  OK    step 2: {used} answered in {secs:.1f}s: {final.content.strip()[:150]!r}")
    print(f"  PASS  total {time.time() - t_start:.1f}s")
    return True


if CHAIN:
    ok = test(models)
    print(f"\n=== Summary\n  {'PASS' if ok else 'FAIL'}  fallback chain ({len(models)} models)")
    sys.exit(0 if ok else 1)

results = {m: test([m]) for m in models}
print("\n=== Summary")
for m, ok in results.items():
    print(f"  {'PASS' if ok else 'FAIL'}  {m}")
passed = [m for m, ok in results.items() if ok]
print("\nNote: free models are often busy for a few minutes and then fine again.")
print("A FAIL marked 'rate-limited' or 'overloaded' is worth keeping as a fallback;")
print("a FAIL marked 'not allowed' or 'answered without calling a tool' is not.")
if passed:
    print('\nModels that passed right now:')
    print(f'MODEL = "{", ".join(passed)}"')
