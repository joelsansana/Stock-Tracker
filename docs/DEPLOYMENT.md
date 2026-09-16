# Deploying to Streamlit Community Cloud

This guide walks through deploying `python_stocks` to
[share.streamlit.io](https://share.streamlit.io) so the app is
publicly accessible.

## Prerequisites

1. **A public GitHub repo.** Community Cloud's free tier requires the
   repo to be public. Private repos need the **Teams** or **Enterprise**
   plan.

   This repo is `github.com:joelsansana/Stock-Tracker`. Confirm it's
   public in the repo Settings → General → "Danger Zone".

2. **A Streamlit Community Cloud account.** Sign in at
   <https://share.streamlit.io> with your GitHub account.

3. **Optional: API credentials.** The Twitter Sentiment page needs
   a v2 bearer token. Skip this if you don't plan to expose that
   page.

## One-time setup

### 1. Push your code

The repo already contains everything Cloud needs:

- `requirements.txt` — installs the local package plus streamlit/plotly
- `pyproject.toml` — declares the package and its runtime deps
- `app/Home.py` — Streamlit entry point (auto-discovered)
- `.streamlit/config.toml` — sensible Streamlit defaults

Confirm `main` is up to date:

```bash
git status        # should be clean
git push origin main
```

### 2. Create the app on Cloud

1. Visit <https://share.streamlit.io> and click **"Create app"**.
2. Fill in:
   - **Repository:** `joelsansana/Stock-Tracker`
   - **Branch:** `main`
   - **Main file path:** `app/Home.py`
   - **App URL:** pick a subdomain, e.g. `python-stocks.streamlit.app`
3. Under **Advanced settings** (optional):
   - **Python version:** 3.12 (matches the pinned `.python-version`)
   - **Secrets:** see next section
4. Click **Deploy**.

The first deploy takes 2–5 minutes — pip installs streamlit, plotly,
and the local package. Streamlit shows build logs in the UI; watch
for errors.

### 3. Configure secrets (optional)

Cloud stores secrets separately from code and injects them as
environment variables + `st.secrets`. For the Twitter page:

1. In your app's dashboard, click **⋮ → Settings → Secrets**.
2. Paste a TOML block:

   ```toml
   TWEET_BEARER_TOKEN = "your-twitter-v2-bearer-token"
   ```

3. Click **Save**. The app restarts automatically.

To get a token, sign up at <https://developer.twitter.com/> and create
a Project + App with **Read** access. The free tier caps at
~10,000 tweets/month.

The Hugging Face backend works without a token when using public
models (DistilBERT SST-2, FinBERT). Set `HF_TOKEN` only if you use
gated models.

## How the build works

Cloud runs roughly:

```bash
pip install -r requirements.txt   # installs python_stocks + UI deps
streamlit run app/Home.py
```

`requirements.txt` does `-e .` which installs `python_stocks` in
editable mode using `pyproject.toml`. The `[project]` dependencies
(numpy, pandas, requests, yfinance, textblob, tqdm) install
automatically; streamlit and plotly are listed explicitly.

## Persistence caveats

Community Cloud has **ephemeral storage** — anything written to disk
is lost on restart or redeploy. This affects:

- **`data/`** — ARK CSV and price caches are re-downloaded on each
  restart. This is slow on the first request but transparent
  afterwards.
- **Streamlit's in-memory `@st.cache_data`** — survives within a
  single process but not across redeploys.

For persistent storage, use an external database (Postgres, Supabase,
Turso, …) or object storage (S3, R2). Out of scope for this guide.

## Optional: enable Hugging Face

The Sentiment Lab uses TextBlob by default. To enable the neural
Hugging Face backend on Cloud:

1. In **Advanced settings** during (re)deploy, the build can be
   customized. Streamlit Cloud doesn't easily support extras, so
   the cleanest path is to add the dep to `requirements.txt`:

   ```text
   -e .
   streamlit>=1.30
   plotly>=5.18
   transformers>=4.20.0
   torch>=1.9.0
   ```

2. Commit, push, and Cloud rebuilds. Note this adds ~250 MB and
   significantly increases cold-start time.

If you'd rather not pay the cost on the free tier, leave
TextBlob as the only backend — it's already wired up and works
out of the box.

## Monitoring

- **Logs:** App page → **Manage app** → **Logs**. Useful for
  diagnosing import errors, network failures, and runtime issues.
- **Resource usage:** Visible under **Resource usage** in the same
  menu. Free tier caps at 1 GB RAM.
- **Reboot:** **⋮ → Reboot** restarts the app and clears in-memory
  caches without a code change.

## Updating

Push to `main` → Cloud rebuilds automatically. No manual deploy
needed.

```bash
git push origin main
# Cloud picks up the change within ~30 seconds
```

## Custom domain

Not supported on the free tier. The default URL is
`https://<your-subdomain>.streamlit.app`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| Build hangs forever | Check logs; usually pip can't resolve a dep. Pin versions in `requirements.txt`. |
| `ModuleNotFoundError: No module named 'python_stocks'` | The `-e .` line didn't run. Verify `pyproject.toml` is at the repo root and `[tool.setuptools.packages.find] where = ["src"]` is set. |
| `ModuleNotFoundError: No module named 'app'` | Page file is missing `import _bootstrap  # noqa: F401` as its first import. |
| Twitter page says "no token" | Secrets aren't saved correctly. Settings → Secrets → paste TOML → Save. |
| Memory exceeded | HF model loaded by accident. Skip the "Load model" button on Cloud; default to TextBlob. |
| Cold start > 30s | First request pays the cost of pip cache miss + Python startup. Subsequent requests are fast. |