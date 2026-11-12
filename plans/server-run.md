# Run TookaBERT v2 on the GPU server

12 November 2026. This is the operator guide for `src/experiments/tookabert_v2.py`. The experiment itself is specified in `plans/new-experiments-plan.md`.

Run it from the repo directory on the server, inside `tmux`, so closing SSH does not stop the job.

After `sudo su`, `~` is `/root`, not `/home/ubuntu`. The repo and the virtualenv on this machine are:

- repo: `/home/ubuntu/digikala-products-recommender-systems`
- virtualenv: `/root/venvs/digikala`

Use those two paths. `cd ~/digikala-products-recommender-systems` as root looks in `/root` and fails.

## 1. Put the code on the server

From your Mac, in the repo directory:

```bash
scp -r src plans ubuntu@SERVER_IP:~/digikala-products-recommender-systems/
```

Use the real IP and the real SSH user. If that directory does not exist yet, `ssh` in first and run `mkdir -p ~/digikala-products-recommender-systems`.

## 2. Install the Python packages

On the server:

```bash
cd /home/ubuntu/digikala-products-recommender-systems
source /root/venvs/digikala/bin/activate
pip install gdown polars numpy transformers accelerate datasets
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"
```

You want `True` and `2`. If `pip` cannot reach `pypi.org`, add `-i https://mirror.arvancloud.ir/pypi/simple --trusted-host mirror.arvancloud.ir` after `pip install`.

## 3. Start the run

```bash
tmux new -s train
source /root/venvs/digikala/bin/activate
cd /home/ubuntu/digikala-products-recommender-systems
python -m src.experiments.tookabert_v2 run
```

Leave that window open. Detach with `Ctrl-b` then `d`. Attach again later with `tmux attach -t train`.

The same command is safe to repeat. A stage that already has a `DONE` file is skipped.

If the Google Drive download fails, copy these two things in and run the same command again:

- `experiments/tookabert-preproc-v2-2026-11-12/download/digikala-comments-preprocessed.csv`
- the saved `best/PartAI--TookaBERT-Base` folder, including `model.safetensors`, placed at `experiments/tookabert-preproc-v2-2026-11-12/model/start/`

## 4. Read the logs

Open a second SSH session. Do not stop the tmux session.

```bash
cd /home/ubuntu/digikala-products-recommender-systems
source /root/venvs/digikala/bin/activate
python -m src.experiments.tookabert_v2 status
```

That prints the current stage and the last 15 lines. For the live log:

```bash
tail -f experiments/tookabert-preproc-v2-2026-11-12/run.log
```

`Ctrl-c` stops `tail` only. The training keeps going. GPU use is `nvidia-smi`.

## 5. Where the files go

Everything is under `/home/ubuntu/digikala-products-recommender-systems/experiments/tookabert-preproc-v2-2026-11-12/`:

| Path | What it is |
| --- | --- |
| `status.json` | Current stage, step, and loss |
| `run.log` | Full text log |
| `kept.json` | Which epoch was kept, written at the end |
| `download/` | The comments CSV |
| `preprocess-letters/`, `preprocess-digits/`, `preprocess-emoji/` | Text after each stage |
| `sample/train_ids.parquet` | The 20% training ids |
| `model/start/` | Starting TookaBERT weights |
| `epoch-1/`, `epoch-2/` | Weights, `metrics.json` (full set), `heldout_metrics.json` (the 80%) |

## 6. Jupyter on the same server

Yes. Run it in a second tmux session so it sits beside the training job, and give Jupyter its own password. Keep it on localhost and open it through an SSH tunnel. The server is on the public internet, so do not publish port 8888.

On the server:

```bash
tmux new -s notebook
source /root/venvs/digikala/bin/activate
pip install jupyterlab
jupyter server password
jupyter lab --no-browser --ip 127.0.0.1 --port 8888
```

`jupyter server password` asks you to invent a notebook password and stores the hash. Detach with `Ctrl-b` then `d`.

On your Mac:

```bash
ssh -L 8888:127.0.0.1:8888 ubuntu@SERVER_IP
```

Then open `http://127.0.0.1:8888` in your browser and type the Jupyter password. That SSH login is the same one you already use. The notebook password is a second, separate password.

```bash
tmux ls

tmux list-windows -a

tmux switch-client -t train

tail -f /home/ubuntu/digikala-products-recommender-systems/experiments/tookabert-preproc-v2-2026-11-12/run.log

tmux attach -t train

tmux kill-session -t train2
```