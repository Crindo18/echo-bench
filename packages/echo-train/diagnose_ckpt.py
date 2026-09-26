import contextlib
import io

import espnet
import torch
import yaml
from espnet2.tasks.asr import ASRTask

from echo_train.embed import ENCODER_PREFIXES, _inside, fetch, find_checkpoint

root = fetch("hf:pyf98/librispeech_100_e_branchformer")
ckpt = find_checkpoint(root)
cfg = yaml.safe_load(ckpt.config.read_text())
print("checkpoint config :", ckpt.config.name)
print("checkpoint espnet :", cfg.get("version") or cfg.get("espnet_version") or "?")
print("installed espnet  :", espnet.__version__)
print("encoder type      :", cfg.get("encoder"))

quiet = io.StringIO()

with _inside(ckpt.root), contextlib.redirect_stdout(quiet), contextlib.redirect_stderr(quiet):
    model, args = ASRTask.build_model_from_file(ckpt.config, None, "cpu") if False else (None, None)

state = torch.load(ckpt.weights, map_location="cpu", weights_only=True)
own = model.state_dict()
needed = [k for k in own if k.startswith(ENCODER_PREFIXES)]
missing = [k for k in needed if k not in state]
reshaped = [k for k in needed if k in state and tuple(state[k].shape) != tuple(own[k].shape)]
unknown = [k for k in state if k.startswith(ENCODER_PREFIXES) and k not in own]

print("\nMISSING (model wants, checkpoint lacks):")
for k in missing:
    print("  ", k, tuple(own[k].shape))
print("\nUNKNOWN (checkpoint has, model doesn't):")
for k in unknown:
    print("  ", k, tuple(state[k].shape))
print("\nRESHAPED:")
for k in reshaped:
    print("  ", k, "model", tuple(own[k].shape), "ckpt", tuple(state[k].shape))
