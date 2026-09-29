# Jarvis – my first LangChain agent

A chat app for a LangChain agent with two real tools:

| Tool | Data source | Key needed? |
|---|---|---|
| `get_weather(city)` | [Open-Meteo](https://open-meteo.com) | No |
| `get_stock_price(company)` | Yahoo Finance via `yfinance` | No |

The LLM runs on [OpenRouter](https://openrouter.ai), so you need an **OpenRouter API key**.

```
jarvis-agent/
├── app.py                          # Streamlit chat UI + agent
├── tools.py                        # the two real tools
├── requirements.txt                # libraries the server installs
├── .gitignore                      # keeps your real key out of GitHub
└── .streamlit/secrets.toml.example # template for your key
```

---

## Step 1 – Run it on your laptop (5 min)

```bash
cd jarvis-agent
python -m venv .venv
# Windows:  .venv\Scripts\activate        Mac/Linux:  source .venv/bin/activate
pip install -r requirements.txt

# create your local secrets file from the template
# Windows:  copy .streamlit\secrets.toml.example .streamlit\secrets.toml
# Mac/Linux: cp .streamlit/secrets.toml.example .streamlit/secrets.toml
```

Open `.streamlit/secrets.toml`, paste your OpenRouter key, save, then:

```bash
python -m streamlit run app.py
```

Your browser opens at http://localhost:8501. Ask “How's the weather in Indore?”
and expand **🔧 Used 1 tool call(s)** to see the real tool output.

## Step 2 – Put the code on GitHub

1. Go to https://github.com/new → name it `jarvis-agent` → **Public** → **Create repository**.
2. Click **uploading an existing file** and drag in `app.py`, `tools.py`,
   `requirements.txt`, `.gitignore`, `README.md` → **Commit changes**.

   ⚠️ **Never upload `.streamlit/secrets.toml`** – that file holds your key.

   Or with git:
   ```bash
   git init && git add . && git commit -m "Jarvis agent"
   git branch -M main
   git remote add origin https://github.com/<your-username>/jarvis-agent.git
   git push -u origin main
   ```
   (`.gitignore` stops `secrets.toml` from being pushed.)

## Step 3 – Deploy on Streamlit Community Cloud (free)

1. Go to https://share.streamlit.io and **Continue with GitHub**; allow access.
2. Click **Create app** → **Yup, I have an app**.
3. Fill in:
   - **Repository:** `<your-username>/jarvis-agent`
   - **Branch:** `main`
   - **Main file path:** `app.py`
   - **App URL:** pick a name, e.g. `jarvis-harish`
4. Click **Advanced settings**:
   - **Python version:** 3.12
   - **Secrets:** paste
     ```toml
     OPENROUTER_API_KEY = "sk-or-v1-your-real-key"
     MODEL = "nvidia/nemotron-3.5-lightning:free"
     ```
   - **Save**
5. Click **Deploy**. The first build takes 2–5 minutes (it installs the libraries).

## Step 4 – Test it

- “How's the weather in Bengaluru?” → weather tool
- “What's HAL's share price today?” → stock tool
- “What was my first question?” → memory
- Share the URL: anyone can chat with Jarvis.

## Updating the app

Edit a file on GitHub (or `git push`) → the app redeploys automatically.
To change the key or model: app menu **⋮ → Settings → Secrets**.

## Is it working? How to monitor

1. **Live status box in the chat** – every answer shows the steps as they happen:
   `🧠 Model decided to call get_weather` → `🔧 get_weather returned` →
   `🧠 Model wrote the answer` → `✅ Done in 6.2s`. If it stops on a step, that
   step is the problem. A failure shows `❌ Failed` plus the error and a hint.
2. **Sidebar → Run health check** – tests the key (and shows how many free-model
   requests you have left today), the model, the weather tool and the stock tool
   separately. Green = OK, red = broken, with the reason.
3. **Logs** – one line per step, with timings:
   - local: the terminal where you ran `streamlit`
   - cloud: **Manage app** (bottom-right of the app) → logs
4. **LangSmith (optional)** – sign up free at https://smith.langchain.com, create
   an API key, add `LANGSMITH_API_KEY` to secrets. Every question then appears as
   a full trace (prompts, tool inputs/outputs, tokens, latency).

Each model call times out after `LLM_TIMEOUT` seconds (default 180; change it in secrets), so the app never sits on “Thinking…” forever.

## Testing models before using them

`python test_models.py` tests every model in `MODEL` on its own (tool call + final
answer, with timings) and prints a suggested `MODEL` line with only the passing ones.
`python test_models.py some/model:free` tests a single id.
`python test_models.py --chain` runs the whole `MODEL` list with fallback –
exactly what the app does – and shows which model answered.

### How Jarvis stays up when free models are busy

1. **Fallback:** models in `MODEL` are tried in order; if one fails (rate-limited,
   overloaded, timed out), the next one is used immediately.
2. **Retry rounds:** if *all* of them fail, Jarvis waits 5 s (then 10 s …) and tries
   the whole list again, `RETRY_ROUNDS` times (default 2).
3. **Timeout:** each single model call gets `LLM_TIMEOUT` seconds (default 180).

Every failed attempt is shown in the chat's status box (⚠️) and in the logs,
and the status line says which model finally answered.

Default order (health-checked on openrouter.ai and tested live, Sep 2026):

1. `inclusionai/ling-3.0-flash-sante:free` – Novita, 99.99% uptime, ~3 s per step in tests
2. `google/gemma-4-26b-a4b-it:free` – Google AI Studio, fast but often rate-limited
3. `nvidia/nemotron-3-super-120b-a12b:free` – Nvidia
4. `google/gemma-4-31b-it:free` – Google AI Studio (bigger Gemma)
5. `qwen/qwen3.8-27b:free` – ModelRun
6. `openrouter/free` – OpenRouter picks any free tool-capable model
7. `nvidia/nemotron-3.5-lightning:free` – slow (~2–3 min) but reliable, last resort

Free models come and go; re-check with the sidebar button *List free models that
support tools* and `test_models.py`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| “OPENROUTER_API_KEY is not set” | Secrets missing or misspelt; check **Settings → Secrets** and reboot the app |
| “couldn't reach the language model … 429” | Free OpenRouter models are rate-limited; wait a minute or set `MODEL` to another model |
| Stock tool says “Could not find a stock price” for every stock | Yahoo sometimes blocks cloud servers. The weather tool and the agent still work |
| Stuck/slow, then “didn't reply within …s” | The free model is overloaded; try again, raise `LLM_TIMEOUT`, or change `MODEL` |
| `ResponseValidationError … error_type: 'timeout'` | OpenRouter itself timed out waiting for the model. Add fallbacks: `MODEL = "model-a:free, model-b:free"` (sidebar → *List free models that support tools*) |
| Build fails | Open **Manage app** (bottom-right) and read the logs |
| App shows “This app has gone to sleep” | Free apps sleep when unused; click the button to wake it (about 30 s) |

Notes: chat memory lives in the server's RAM, so it resets when the app restarts
or sleeps. Everyone who opens the URL uses **your** OpenRouter key, so keep an
eye on usage at https://openrouter.ai/activity.
