# OrbitalNet

**Predicting Planetary Trajectories Using Physics-Informed Neural Networks**

A Physics-Informed Neural Network (PINN) that predicts two-body orbital trajectories by
embedding Newton's law of gravitation directly into its training loss via automatic
differentiation, benchmarked against a standard MLP baseline on accuracy, data efficiency,
and noise robustness — per the project synopsis.

---

## 1. What's in this project

```
OrbitalNet/
├── README.md                  <- you are here
├── requirements.txt
├── quick_test.py               <- run this FIRST to verify your setup (~1-2 min)
├── app.py                      <- Streamlit live-demo app
├── src/
│   ├── physics.py               two-body ODE + orbit sampling (SciPy solve_ivp)
│   ├── data_generation.py       builds the 500-trajectory dataset
│   ├── model.py                 the shared MLP architecture (5 -> 4x[64,Tanh] -> 4)
│   ├── losses.py                PINN physics-residual loss (torch.autograd)
│   ├── train.py                 shared training loop (PINN and baseline)
│   ├── evaluate.py              MSE/MAE, noise robustness, long-horizon tests
│   └── visualize.py             matplotlib + plotly plotting
├── scripts/
│   ├── generate_data.py         Step 1: build the dataset
│   ├── train_pinn.py            Step 2: train the PINN
│   ├── train_baseline.py        Step 3: train the baseline MLP
│   └── run_evaluation.py        Step 4: compare PINN vs baseline
├── data/                        generated dataset goes here (.npz)
├── models/                      trained model checkpoints go here (.pt)
├── results/                     evaluation plots + summary JSON go here
├── notebooks/
│   └── exploration.ipynb        optional notebook for the IEEE paper's figures
└── web/                         bonus interactive website (see section 5)
    ├── backend/
    │   ├── main.py               FastAPI server wrapping the SAME trained model
    │   └── requirements.txt
    └── frontend/
        ├── package.json
        ├── vite.config.js
        ├── index.html
        └── src/
            ├── main.jsx, App.jsx, App.css, index.css
            ├── api.js               calls the FastAPI backend
            └── components/
                ├── ControlsPanel.jsx   initial-condition sliders
                ├── OrbitPlot.jsx       custom SVG true-vs-predicted orbit plot
                └── MetricsPanel.jsx    MSE/MAE display
```

## 2. Design decisions worth knowing

The synopsis specifies a strict **5-input** network (`x0, y0, vx0, vy0, t`). Two decisions were
made to implement this precisely and defensibly:

1. **Central mass M is fixed (canonical units, GM = 1.0)** for every trajectory, rather than
   varied per-trajectory. In the two-body problem, varying GM is mathematically equivalent to
   rescaling length/time units — so fixing GM and instead varying the initial position and
   velocity (semi-major axis `a`, eccentricity `e`, orbital orientation `phi`) produces the same
   diversity of circular/elliptical orbit shapes, without needing a 6th network input. This is
   documented in `src/physics.py`.

2. **Noise robustness** (synopsis section 5.4) is implemented by adding Gaussian noise to the
   *initial-condition inputs* (`x0, y0, vx0, vy0`) only, simulating noisy real-world measurement
   of a spacecraft's initial state, and comparing predictions against the *true* noiseless
   trajectory. Documented in `src/evaluate.py`.

Both choices are deliberate engineering decisions to make the synopsis's methodology
precisely and unambiguously implementable — be ready to explain them if asked.

3. **One more thing worth flagging for your reviewer:** the synopsis's stated "Demo Interface"
   (section 6) is Streamlit, and `app.py` (already in this project) fulfils that exactly. The
   React + FastAPI website in `web/` (section 5 below) is an **additional, bonus interface** on
   top of that — not a replacement — built with the same trained model and the same physics
   code, for a more portfolio-polished presentation layer. If your evaluator is checking strictly
   against the synopsis, lead with the Streamlit demo; mention the website as extra work.

## 3. Setup in VS Code

### Step 1 — Install prerequisites
You need **Python 3.10, 3.11, or 3.12** and **VS Code** with the Python extension
(`ms-python.python`) installed from the Extensions marketplace.

Check your Python version in a terminal:
```bash
python3 --version
```

### Step 2 — Open the project and create a virtual environment
1. Unzip `OrbitalNet.zip` anywhere on your machine.
2. In VS Code: **File → Open Folder…** → select the unzipped `OrbitalNet` folder.
3. Open a terminal in VS Code: **Terminal → New Terminal** (or `` Ctrl+` ``).
4. Create and activate a virtual environment so this project's packages don't clash with
   anything else on your system:

   **Windows (PowerShell):**
   ```powershell
   python -m venv .venv
   .venv\Scripts\Activate.ps1
   ```
   **macOS / Linux:**
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   ```
5. VS Code will usually pop up a prompt asking *"Select this environment for the workspace?"*
   — click **Yes**. If it doesn't, press `Ctrl+Shift+P` (`Cmd+Shift+P` on Mac), type
   **Python: Select Interpreter**, and choose the one inside `.venv`.

   You'll know it worked because your terminal prompt now starts with `(.venv)`.

### Step 3 — Install dependencies
The project is CPU-only (no GPU required, per the synopsis). Installing plain `torch` from
PyPI on Windows/Linux pulls several GB of unnecessary CUDA/GPU libraries. Install the much
smaller, faster **CPU-only** build instead:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cpu
```

Then install everything else:
```bash
pip install -r requirements.txt
```
(pip will see torch is already installed and skip it.)

> If you're on macOS, the plain PyPI `torch` wheel is already CPU-only, so you can just run
> `pip install -r requirements.txt` directly and skip the command above.

### Step 4 — Verify your setup
Run the quick sanity check (~1-2 minutes, tiny dataset, tiny training run — this just proves
every stage of the pipeline works on your machine, it does **not** produce a usable model):

```bash
python quick_test.py
```

You should see `ALL CHECKS PASSED` at the end. If anything fails, re-read the error message —
it's almost always a missing/mismatched package from Step 3.

## 4. Running the full project

These four steps reproduce the exact methodology in the synopsis. Run them **in order**, from
the VS Code terminal, from the project's root folder (with `.venv` still activated).

### Step 1 — Generate the dataset
Simulates 500 two-body trajectories (200 time-steps each = 100,000 samples) via SciPy's
`solve_ivp`, and saves them to `data/orbitalnet_dataset.npz`.
```bash
python scripts/generate_data.py
```
Takes well under a minute on a normal laptop CPU.

### Step 2 — Train the PINN
Trains the physics-informed model: `L_total = L_data + 0.1 * L_physics`, Adam optimizer,
10,000 epochs, batch size 256 — exactly as specified in the synopsis.
```bash
python scripts/train_pinn.py
```
**This is the slow step.** On a typical laptop CPU, 10,000 epochs over 100,000 samples can take
a while (well over an hour — exact time depends heavily on your CPU). Progress prints every
200 epochs so you can watch training and validation loss converge. To do a much faster test
run first (not for your final results, just to see it work end-to-end quickly):
```bash
python scripts/train_pinn.py --epochs 500
```
Saves `models/pinn_model.pt`, `models/pinn_stats.npz`, `models/pinn_history.json`.

### Step 3 — Train the baseline
Trains the *identical* architecture with data loss only (no physics term), for a fair
comparison, exactly as the synopsis specifies:
```bash
python scripts/train_baseline.py
```
Saves `models/baseline_model.pt`, `models/baseline_stats.npz`, `models/baseline_history.json`.

### Step 4 — Evaluate and compare
Runs the full comparative evaluation (MSE/MAE, noise robustness at σ = 0.01/0.05/0.1,
long-horizon extrapolation) for both models, and saves plots + a JSON summary:
```bash
python scripts/run_evaluation.py
```
Look in `results/` for:
- `evaluation_summary.json` — all the numbers, for your report/paper
- `pinn_vs_true_orbit.png`, `baseline_vs_true_orbit.png` — trajectory plots
- `pinn_loss_curve.png`, `baseline_loss_curve.png` — loss convergence plots

### Step 5 (optional) — Launch the live demo
Once `models/pinn_model.pt` exists (after Step 2), you can launch the interactive Streamlit
demo mentioned in the synopsis's "Expected Outcomes":
```bash
streamlit run app.py
```
This opens a browser tab where you can drag sliders for the initial orbital conditions and
watch the PINN's predicted trajectory plotted live against the true orbit.

## 5. Interactive website (React + FastAPI) — bonus

A more polished, two-process web app: a **FastAPI backend** (`web/backend/`) that loads the
exact same trained PINN model and physics code from `src/`, and a **React frontend**
(`web/frontend/`, built with Vite) that lets you drag sliders for the initial orbital
conditions and see the predicted-vs-true orbit rendered live, with MSE/MAE metrics — the same
functionality as the Streamlit demo, as a standalone website you can deploy or show off
separately.

**Prerequisites for this section:** you must have already completed section 3 (Setup) and run
Step 1 + Step 2 of section 4 (`generate_data.py` and `train_pinn.py`), so that
`models/pinn_model.pt` exists — the backend needs a trained model to serve. You'll also need
**Node.js 18+** installed (check with `node --version`; get it from https://nodejs.org if
needed).

### Step 1 — Install the backend's extra dependencies
From the project root, with your `.venv` still active:
```bash
pip install -r web/backend/requirements.txt
```
(This just adds `fastapi` and `uvicorn` on top of the `torch`/`numpy`/`scipy` you already have.)

### Step 2 — Start the backend API server
Still from the project **root** (important — it needs to find `src/` and `models/`):
```bash
uvicorn web.backend.main:app --reload --port 8000
```
Leave this running. Visit http://127.0.0.1:8000/api/health in a browser — you should see
`{"status":"ok","model_loaded":true}`. If `model_loaded` is `false`, you haven't trained the
PINN yet (go back to section 4, Step 2).

### Step 3 — Install and start the frontend
Open a **second** VS Code terminal (**Terminal → Split Terminal**, or **Terminal → New
Terminal**), so the backend keeps running in the first one. Then:
```bash
cd web/frontend
npm install
npm run dev
```
This prints a local URL, normally **http://127.0.0.1:5173** — open it in your browser. You
should see the OrbitalNet website with sliders on the left and a live orbit plot on the right.

### Step 4 (optional) — Build a static production bundle
If you want a deployable static bundle (e.g. to host the frontend separately, or to hand in as
a build artifact) rather than the live dev server:
```bash
cd web/frontend
npm run build
```
This creates `web/frontend/dist/`, a static site you can preview locally with
`npm run preview`, or deploy to any static host. Note that the built site still needs the
FastAPI backend (Step 2) running and reachable at `http://127.0.0.1:8000` to actually make
predictions — it's a frontend build, not a self-contained bundle with the model baked in.

## 6. Customizing a run

Every script takes command-line flags so you can experiment without editing code, e.g.:
```bash
python scripts/generate_data.py --n_trajectories 800 --n_timesteps 250
python scripts/train_pinn.py --epochs 15000 --lam 0.05 --lr 5e-4
```
Run any script with `--help` to see all options, e.g. `python scripts/train_pinn.py --help`.

## 7. Troubleshooting

| Problem | Fix |
|---|---|
| `ModuleNotFoundError: No module named 'src'` | Make sure you're running scripts from the project **root** folder (where `README.md` lives), not from inside `scripts/`. |
| `(.venv)` doesn't appear in your terminal prompt | Re-run the activate command from Step 2, or reselect the interpreter via `Ctrl+Shift+P` → *Python: Select Interpreter*. |
| Training seems to hang / very slow | Normal on CPU for the full 10,000-epoch run — see Step 2's note above. Use `--epochs 500` for a quick check. |
| `FileNotFoundError: Dataset not found` | Run `python scripts/generate_data.py` first. |
| Streamlit app shows "No trained PINN model found" | Run `scripts/generate_data.py` then `scripts/train_pinn.py` first, then restart the app. |
| `pip install torch` downloads several GB | Use the CPU-only index URL from Setup Step 3 instead. |
| Website shows "Can't reach the backend API" | Make sure `uvicorn web.backend.main:app --reload --port 8000` (section 5, Step 2) is running in its own terminal and hasn't crashed. |
| Website shows "no trained PINN model was found" banner | Train the PINN first (section 4, Step 2), then restart the `uvicorn` process — it only loads the model at startup / first request. |
| `npm install` or `npm run dev` fails / `node: command not found` | Install Node.js 18+ from https://nodejs.org, then reopen the terminal so it's on your `PATH`. |
| Website loads but predictions never appear | Open your browser's dev console (F12) for errors — usually means the backend (Step 2) isn't running, or is running on a different port than `http://127.0.0.1:8000`. |

## 8. Tech stack (per synopsis section 6)

Python 3.11 · PyTorch 2.x (autograd) · SciPy (`solve_ivp`) · NumPy · Matplotlib · Plotly ·
Streamlit — all open-source, all CPU-only, zero cost to run. The bonus website
(section 5) additionally uses **FastAPI**, **Uvicorn**, **React 18**, and **Vite**.

## 9. References

[1] M. Raissi, P. Perdikaris, and G. E. Karniadakis, "Physics-informed neural networks: A deep
learning framework for solving forward and inverse problems involving nonlinear partial
differential equations," *Journal of Computational Physics*, vol. 378, pp. 686–707, 2019.

[2] I. E. Lagaris, A. Likas, and D. I. Fotiadis, "Artificial neural networks for solving
ordinary and partial differential equations," *IEEE Transactions on Neural Networks*, vol. 9,
no. 5, pp. 987–1000, 1998.

[3] SciPy documentation — `scipy.integrate.solve_ivp`, https://docs.scipy.org/

[4] PyTorch documentation — `torch.autograd`, https://pytorch.org/docs/

[5] Streamlit documentation, https://docs.streamlit.io/
