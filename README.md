# vlm-robotics-training

Do LLMs know Franka Panda kinematics? 50 models, 30 joint configurations from DROID, no tools and no reasoning.

- `prompts.md`: the system prompts and example prompts
- `data/<run>/tasks.jsonl`: every prompt with its ground truth
- `data/<run>/rollouts/`: every model answer, one file per model (files with 12 answers are short probes)
- `plot/make_plots.py`: scores the rollouts and draws `figures/`

`panda` uses the "don't calculate" prompt; `panda_strict` adds "DO NOT USE ANY REASONING. JUST REPLY WITH AN ANSWER."

```
pip install -r requirements.txt
python plot/make_plots.py
```

Joint angles are from [DROID](https://droid-dataset.github.io/). Models were queried through OpenRouter in November 2026.
