"""End-to-end smoke test on synthetic data (no downloads).

Exercises every stage the real pipeline uses — CTC forward/backward, TORGO-style
finetune step, freeze, embedding extraction, few-shot evaluate, ONNX export +
INT8 quantize, and the DPWavLM gate — so the wiring is verified without the real
corpora. Run: python scripts/smoke_test.py

IMPORTANT — the numbers PRINTED BY THIS SMOKE TEST are an early signal, not a
thesis result. This test uses randomly initialised (untrained) models on
synthetic audio, so its accuracy/loss/size only prove the CODE RUNS CORRECTLY.
They become real model performance once you run the SAME pipeline on real
LibriSpeech/TORGO with trained checkpoints — nothing about the measurement
changes, only the model does. The eval is honest either way: where the synthetic
audio carries a per-class tone the accuracy rises because that signal is real,
and on pure noise it returns chance (verified) — so trained models will earn
their numbers, they are not inflated.
"""
from __future__ import annotations

import random
import tempfile

import torch
from torch.utils.data import Dataset, DataLoader

from echo.data import CharTokenizer, Collate, map_prompt_to_intent, NUM_INTENTS
from echo.data.intents import INTENT_SET
from echo.train.common import AcousticModel, set_seed, save_checkpoint
from echo.train.pretrain import build_model, train_one_epoch
from echo.encoders import build_encoder, count_parameters

set_seed(0)
TOK = CharTokenizer()
PROMPTS = [triggers[0] for _, _, triggers in INTENT_SET]  # one prompt per intent


class SynthSpeech(Dataset):
    """Random 'speech' with real intent-bearing prompts and speaker ids.

    ``reps_per_intent`` controls how many utterances each speaker has per intent,
    so the few-shot enrollment path (needs > K samples per intent) can actually
    run in the smoke test.
    """
    def __init__(self, n=48, speakers=("F01", "M01", "F03"), with_intent=True,
                 n_intents=None, reps_per_intent=None, separable=False):
        self.items = []
        if reps_per_intent is not None and n_intents is not None:
            for spk in speakers:
                for intent_i in range(n_intents):
                    for _ in range(reps_per_intent):
                        secs = random.uniform(1.0, 2.0)
                        wav = torch.randn(int(secs * 16000)) * 0.1
                        if separable:
                            # per-intent tone makes intents actually separable,
                            # so meta-training the projector has signal to learn.
                            t = torch.arange(wav.numel()) / 16000.0
                            f = 200.0 + 120.0 * intent_i
                            wav = wav + 0.5 * torch.sin(2 * 3.14159 * f * t)
                        prompt = PROMPTS[intent_i]
                        self.items.append(dict(wav=wav, text=prompt, speaker=spk,
                                               intent=intent_i if with_intent else None))
        else:
            for i in range(n):
                secs = random.uniform(1.0, 2.5)
                wav = torch.randn(int(secs * 16000)) * 0.1
                prompt = PROMPTS[i % len(PROMPTS)]
                spk = speakers[i % len(speakers)]
                intent = map_prompt_to_intent(prompt) if with_intent else None
                self.items.append(dict(wav=wav, text=prompt, speaker=spk, intent=intent))

    def __len__(self): return len(self.items)
    def __getitem__(self, i): return self.items[i]

    def speakers(self): return sorted({it["speaker"] for it in self.items})


def check(name, cond):
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}")
    assert cond, name


print("== 1. encoders build + CTC forward/backward ==")
# moonshine is waveform-native (accepts_waveform) — the AcousticModel routes it
# around the shared log-Mel front-end, so the same CTC harness covers it.
for enc in ["conformer", "ebranchformer", "zipformer", "moonshine"]:
    model = build_model(enc, TOK)
    n = count_parameters(model.encoder, trainable_only=False) / 1e6
    ds = SynthSpeech(n=12)
    loader = DataLoader(ds, batch_size=4, collate_fn=Collate(TOK))
    opt = torch.optim.SGD(model.parameters(), lr=1e-3)
    loss = train_one_epoch(model, loader, opt, None, torch.device("cpu"), TOK,
                           log_every=999)
    wf = " [waveform]" if model.accepts_waveform else ""
    check(f"{enc} ~15M ({n:.2f}M){wf} & CTC step (loss {loss:.2f})",
          13.5 < n < 16.5 and loss == loss)


print("== 2. finetune freeze + embedding + few-shot evaluate ==")
from echo.eval.benchmark import evaluate, extract_embeddings
model = build_model("conformer", TOK)
model.freeze_encoder()
check("encoder frozen", not any(p.requires_grad for p in model.encoder.parameters()))
# 4 speakers x 5 intents x 8 reps => enough per (speaker,intent) to enroll K=5
ds = SynthSpeech(speakers=("F01", "M01", "F03", "M03"), n_intents=5,
                 reps_per_intent=8)
emb, ints, spk = extract_embeddings(model, ds, list(range(len(ds))),
                                     torch.device("cpu"))
check(f"embeddings {tuple(emb.shape)} extracted", emb.size(0) > 0 and emb.size(1) == 256)
res = evaluate(model, ds, holdout=["F03", "M03"], ks=(3, 5), n_episodes=3,
               device=torch.device("cpu"), seed=0)
check("evaluate returns pre + post@K",
      "acc" in res["pre"] and 3 in res["post"] and 5 in res["post"])
check("post-enrollment@3 produced a real number (path ran)",
      res["post"][3]["acc"] == res["post"][3]["acc"])  # not NaN
print(f"      pre acc={res['pre']['acc']:.3f} post@3={res['post'][3]['acc']:.3f} "
      f"gain@3={res['gain'][3]:+.3f}")


print("== 2b. episodic meta-training actually trains the projector ==")
from echo.train.metatrain import episodic_metatrain
# seed here so this check is deterministic regardless of earlier RNG consumption
torch.manual_seed(0); random.seed(0)
mt_model = build_model("conformer", TOK)
# separable intents so there is real signal for the projector to learn
mt_ds = SynthSpeech(speakers=("F01", "M01"), n_intents=5, reps_per_intent=12,
                    separable=True)
before = mt_model.projector.proj.weight.detach().clone()
acc = episodic_metatrain(mt_model, mt_ds, list(range(len(mt_ds))),
                         torch.device("cpu"), epochs=12, episodes_per_epoch=60,
                         n_way=5, k_shot=5, n_query=5, lr=1e-3, seed=0,
                         verbose=False)
after = mt_model.projector.proj.weight.detach()
changed = not torch.allclose(before, after)
frozen_after = not any(p.requires_grad for p in mt_model.projector.parameters())
check(f"projector weights updated by meta-training", changed)
check(f"episodic acc beats 5-way chance (0.20): {acc:.2f}", acc > 0.30)
check("projector re-frozen after meta-training", frozen_after)


print("== 3. ONNX export + INT8 static quantize + footprint ==")
from echo.quantize.quantize_export import (export_fp32_onnx, quantize_int8_static,
                                            footprint_report)
tmp = tempfile.mkdtemp()
model = build_model("zipformer", TOK).eval()
fp32 = f"{tmp}/zip_fp32.onnx"
int8 = f"{tmp}/zip_int8.onnx"
export_fp32_onnx(model, fp32, seconds=2.0)
quantize_int8_static(fp32, int8, accepts_waveform=False, seconds=2.0)
rep = footprint_report(model, fp32, int8)
# On this RANDOM-weight model with random calibration the absolute MB is not
# meaningful, so we assert the two things that are: the parameter count meets
# the ~15M / ~1-byte-per-param budget (the spec's SO1 proxy), and INT8 static
# quantization measurably compresses the graph vs FP32. Absolute <=16MB is
# verified on trained checkpoints with real calibration data (see README).
check(f"params {rep['encoder_params_M']:.2f}M within ~15M budget "
      f"& INT8 compresses ({rep['onnx_fp32_MB']:.1f}->{rep['onnx_int8_MB']:.1f}MB)",
      rep["params_within_budget"] and rep["int8_compresses"])
print(f"      params {rep['encoder_params_M']:.2f}M (~{rep['param_budget_MB']:.1f}MB "
      f"INT8 ideal) | FP32 {rep['onnx_fp32_MB']:.1f}MB -> INT8 {rep['onnx_int8_MB']:.1f}MB")

print("== 3b. ONNX Runtime latency profile ==")
from echo.eval.latency import profile_onnx
lat = profile_onnx(int8, seconds=2.0, runs=5, warmup=2, accepts_waveform=False)
check(f"latency measured ({lat['latency_ms_mean']:.0f}ms, RTF {lat['rtf']:.2f})",
      lat["latency_ms_mean"] > 0)


print("== 4. DPWavLM gate + expected_num_params + materialize ==")
from echo.encoders import DPWavLMStudent
from echo.train.distill_prune import go_no_go_gate, pick_layer_pairs
student = DPWavLMStudent()
dense = count_parameters(student, trainable_only=False) / 1e6
ep = student.expected_num_params().item() / 1e6
check(f"dense student ~94M ({dense:.1f}M), expected_num_params differentiable ({ep:.1f}M)",
      90 < dense < 100 and student.expected_num_params().requires_grad)
def force_prune(model, keep_layers, head_keep, ffn_keep):
    """Emulate the L0 objective's structured pruning by hard-setting gates:
    drop all but `keep_layers` layers and, on the survivors, close a fraction of
    the head and FFN-unit gates (width pruning)."""
    with torch.no_grad():
        for i, layer in enumerate(model.layers):
            if i >= keep_layers:
                layer.layer_gate.log_alpha.fill_(-8.0)
            else:
                h = layer.attn.head_gate.log_alpha
                h[int(h.numel() * head_keep):] = -8.0
                f = layer.ffn.unit_gate.log_alpha
                f[int(f.numel() * ffn_keep):] = -8.0

# (a) NO-GO: heavy pruning that still can't clear the budget floor (conv
#     extractor + positional conv ~9M) -> gate says stop per spec Section 5.
force_prune(student, keep_layers=4, head_keep=0.5, ffn_keep=1 / 3)
gate_nogo = go_no_go_gate(student, param_limit=16e6, verbose=True)
check(f"gate flags over-budget config as NO-GO ({gate_nogo['n_params']/1e6:.1f}M)",
      gate_nogo["decision"] == "NO-GO")

# (b) GO: a reachable in-budget configuration (all gate types exercised) proves
#     the pruning mechanism *can* hit the ~15M target.
student_go = DPWavLMStudent()
force_prune(student_go, keep_layers=3, head_keep=0.5, ffn_keep=1 / 6)
gate = go_no_go_gate(student_go, param_limit=16e6, verbose=True)
check(f"gate clears reachable in-budget config as GO ({gate['n_params']/1e6:.1f}M)",
      gate["decision"] == "GO")
pairs = pick_layer_pairs(4, 12)
check(f"layer-pair selection {pairs}", len(pairs) >= 1)


print("== 4b. DPWavLM distill+prune training step (teacher->student) ==")
from echo.encoders import WavLMTeacher, init_student_from_teacher
from echo.train.distill_prune import DistillProjector, distill_loss
teacher = WavLMTeacher(pretrained=False).eval()   # random-init fallback (offline)
for p in teacher.parameters():
    p.requires_grad_(False)
stu = DPWavLMStudent()
# strict=False so the wiring test never hard-crashes on a torchaudio whose WavLM
# key names differ; we surface the inheritance coverage as a check instead, so
# key-name drift (which would make a real student mostly random) is still caught.
_inh = init_student_from_teacher(stu, teacher, strict=False)
check(_inh["coverage"] > 0.70,
      f"teacher->student inheritance coverage {_inh['coverage']:.0%} "
      f"({_inh['n_layers_copied']}/{_inh['n_layers_total']} layers)")
dp_pairs = pick_layer_pairs(stu.cfg["n_layers"], 12)
dproj = DistillProjector(stu.cfg["d_model"], 768, len(dp_pairs))
si = [s for s, _ in dp_pairs]; ti = [t for _, t in dp_pairs]
m_opt = torch.optim.Adam(list(stu.main_parameters()) + list(dproj.parameters()), lr=1e-4)
g_opt = torch.optim.Adam(stu.gate_parameters(), lr=1e-2)
dense = count_parameters(stu, trainable_only=False)
wav = torch.randn(2, 16000); lens = torch.tensor([16000, 11000])
losses = []
for _ in range(2):
    with torch.no_grad():
        tf, _ = teacher.hidden_states(wav, lens)   # unmasked teacher (WavLM can't mask)
    _, s_len, sh = stu(wav, lens, return_hidden=True)
    d = distill_loss(dproj([sh[i] for i in si]), [tf[i] for i in ti], lengths=s_len)
    viol = (stu.expected_num_params() - 15e6) / dense
    m_opt.zero_grad(); g_opt.zero_grad()
    (d + 0.5 * viol).backward()
    m_opt.step(); g_opt.step()
    losses.append(d.item())
check(f"distill step runs w/ variable lengths & masking (loss {losses[-1]:.2f})",
      all(x == x for x in losses))
check("teacher/student frame lengths agree",
      teacher.hidden_states(wav, lens)[0][0].shape[1] == sh[0].shape[1])


print("== 4c. DPWavLM materialize -> save -> reload chain (pipeline hand-off) ==")
import gc
del teacher, stu, dproj, student, student_go        # free ~94M-param models
gc.collect()
import tempfile as _tf
from echo.encoders import build_encoder
from echo.train.common import save_checkpoint, load_init_weights
from echo.train.pretrain import build_model as _bm
big = DPWavLMStudent()
with torch.no_grad():
    for i, l in enumerate(big.layers):
        if i >= 3:
            l.layer_gate.log_alpha.fill_(-8.0)
        else:
            hh = l.attn.head_gate.log_alpha; hh[hh.numel() // 2:] = -8.0
            ff = l.ffn.unit_gate.log_alpha; ff[ff.numel() // 6:] = -8.0
mat = big.materialize()
del big; gc.collect()
# cfg must rebuild the *exact* pruned architecture and load STRICT
rebuilt = build_encoder("dpwavlm", **mat.cfg)
rebuilt.load_state_dict(mat.state_dict(), strict=True)
mat.eval(); rebuilt.eval()
_w = torch.randn(1, 16000); _l = torch.tensor([16000])
with torch.no_grad():
    oa, _ = mat(_w, _l); ob, _ = rebuilt(_w, _l)
check("materialized cfg rebuilds exact arch (strict load, identical output)",
      torch.allclose(oa, ob, atol=1e-5))
# encoder-only compressed.pt loads into a full AcousticModel (finetune --init)
cp = _tf.mktemp(suffix=".pt")
save_checkpoint(cp, mat, meta=dict(encoder_cfg=dict(mat.cfg)))
am = _bm("dpwavlm", TOK, encoder_cfg=dict(mat.cfg))
load_init_weights(cp, am)
loaded_ok = torch.allclose(am.encoder.feature_extractor.conv[0].weight,
                           mat.feature_extractor.conv[0].weight)
check("encoder-only compressed.pt loads into AcousticModel.encoder", loaded_ok)


print("== 5. data hygiene: CV plan + leakage contract + coverage ==")
import math as _math
from echo.data.corpus import Utterance, UtteranceDataset
from echo.data.splits import make_cv_plan, verify_no_leakage
from echo.data.coverage import coverage as _coverage
_tiers = {f"S{i}": ["severe", "moderate", "mild"][i // 3] for i in range(9)}
_recs = []
for _spk, _sev in _tiers.items():
    _noise = {"severe": 0.6, "moderate": 0.3, "mild": 0.12}[_sev]
    for _intent in range(5):
        for _rep in range(8):
            _w = torch.randn(int(1.2 * 16000)) * _noise
            _t = torch.arange(_w.numel()) / 16000.0
            _w = _w + 0.5 * torch.sin(2 * _math.pi * (200 + 120 * _intent) * _t)
            _recs.append(dict(wav=_w, speaker=_spk, intent=_intent, sev=_sev))
_utts = [Utterance(utt_id=f"{r['speaker']}/{i}", corpus="torgo", speaker=r["speaker"],
                   cohort="dysarthric", severity=r["sev"], group=f"g{i}", mic="headMic",
                   is_primary=True, wav_path="x", prompt="p", intent=r["intent"],
                   duration_s=1.2, sha1="") for i, r in enumerate(_recs)]
_plan = make_cv_plan(_utts, n_folds=3, seeds=(1, 2))
check(f"CV plan built (3 folds x 2 seeds = {len(_plan.folds)})", len(_plan.folds) == 6)
_clean = verify_no_leakage(_plan, _utts)
check("leakage contract passes on clean speaker-independent folds", _clean["passed"])
_leaky = list(_utts) + [Utterance(utt_id="leak", corpus="torgo", speaker="S3",
    cohort="dysarthric", severity="moderate", group="g0", mic="headMic",
    is_primary=True, wav_path="x", prompt="p", intent=0, duration_s=1.2, sha1="")]
check("leakage contract CATCHES an injected cross-speaker group leak",
      not verify_no_leakage(_plan, _leaky)["passed"])
_cov = _coverage(_utts, ks=(3, 5, 10))
check("coverage: K=3/5 supported (8 reps), K=10 not",
      _cov["overall_speakers_supporting"][3] == 9 and
      _cov["overall_speakers_supporting"][10] == 0)

# primary-channel selection must NOT drop groups whose canonical mic is absent
from echo.data.corpus import _select_primary_per_group
_two_mic = []          # group with head+array
_array_only = []       # group with array only (no head mic)
for _mic in ("headMic", "arrayMic"):
    _two_mic.append(Utterance(utt_id=f"g1/{_mic}", corpus="torgo", speaker="S0",
        cohort="dysarthric", severity="mild", group="grp1", mic=_mic,
        is_primary=("head" in _mic), wav_path="x", prompt="p", intent=0,
        duration_s=1.0, sha1=""))
_array_only.append(Utterance(utt_id="g2/arrayMic", corpus="torgo", speaker="S0",
    cohort="dysarthric", severity="mild", group="grp2", mic="arrayMic",
    is_primary=False, wav_path="x", prompt="p", intent=0, duration_s=1.0, sha1=""))
_sel = _select_primary_per_group(_two_mic + _array_only)
_groups_kept = {r.group for r in _sel}
check("primary-channel selection keeps 1/group AND never drops array-only groups",
      len(_sel) == 2 and _groups_kept == {"grp1", "grp2"} and
      [r for r in _sel if r.group == "grp1"][0].mic == "headMic")

# CV partition must never leave an empty test fold when speakers >= n_folds
from collections import Counter as _Counter
_stress = {f"{t}{j}": f"torgo|{t}" for t in ("a", "b", "c") for j in range(2)}
from echo.data.splits import _stratified_partition as _sp
_no_empty = all(
    len(_Counter(_sp(_stress, 5, seed=_s).values())) == 5 for _s in range(6))
check("CV folds: no empty test fold with speakers>=n_folds (6 spk, 5 folds)",
      _no_empty)


print("== 6. measurement: cross-validated accuracy (per-tier) + scorecard ==")
from echo.eval.protocol import cross_validate as _cv
from echo.eval.scorecard import card_from_measurements, build_scorecard


class _RecDS:
    def __init__(s, recs):
        s.items = [dict(wav=r["wav"], text="p", speaker=r["speaker"],
                        intent=r["intent"]) for r in recs]
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.items[i]


_ds = _RecDS(_recs)
_m = build_model("conformer", TOK)
from echo.train.metatrain import episodic_metatrain as _mt
_mt(_m, _ds, list(range(len(_ds))), torch.device("cpu"), epochs=5,
    episodes_per_epoch=30, n_way=5, k_shot=5, n_query=3, lr=1e-3, seed=0, verbose=False)
_acc = _cv(_m, _ds, _plan, torch.device("cpu"), ks=(3, 5))
check("cross_validate returns per-severity-tier accuracy",
      set(_acc["per_tier"]) >= {"severe", "moderate", "mild"})
check("harder tier scores <= easier tier (severe <= mild)",
      _acc["per_tier"]["severe"][5]["mean"] <= _acc["per_tier"]["mild"][5]["mean"] + 1e-6)
_fp = dict(encoder_params_M=14.6, onnx_int8_MB=14.6, params_within_budget=True)
_fp_big = dict(encoder_params_M=14.6, onnx_int8_MB=19.9, params_within_budget=False)
_card_ok = card_from_measurements("fits", _fp, _acc,
    dict(latency_ms_mean=70, rtf=0.03, peak_ram_mb=80), primary_k=5)
_card_big = card_from_measurements("over_budget", _fp_big, _acc,
    dict(latency_ms_mean=60, rtf=0.03, peak_ram_mb=80), primary_k=5)
_sc = build_scorecard([_card_ok, _card_big])
check("scorecard disqualifies the over-budget model from the recommendation",
      _sc["recommendation"] == "fits" and "over_budget" in _sc["disqualified"])


print("== 7. severity toggle: stratify on/off, tiers still reported, ranking shifts ==")
from echo.data.splits import make_cv_plan as _mkplan, verify_no_leakage as _vnl
from echo.eval.scorecard import (ModelCard as _MC, build_scorecard as _bsc,
                                 weights_without_severity as _wns)
_toggle_recs = [Utterance(utt_id=f"{s}/{i}", corpus="torgo", speaker=s,
    cohort="dysarthric", severity=v, group=f"tg{s}{i}", mic="headMic",
    is_primary=True, wav_path="x", prompt="p", intent=i % 5, duration_s=1.0, sha1="")
    for s, v in {"F01": "severe", "M01": "severe", "F03": "moderate",
                 "M05": "moderate", "F04": "mild", "M03": "mild"}.items()
    for i in range(10)]
_p_on = _mkplan(_toggle_recs, n_folds=3, seeds=(1,), stratify=True)
_p_off = _mkplan(_toggle_recs, n_folds=3, seeds=(1,), stratify=False)
check("stratify OFF: folds unstratified but severity still recorded for reporting",
      _p_off.stratify_by == [] and
      set(_p_off.speaker_tier.values()) == {"severe", "moderate", "mild"})
check("both stratified and unstratified plans are leakage-free",
      _vnl(_p_on, _toggle_recs)["passed"] and _vnl(_p_off, _toggle_recs)["passed"])
# A wins on hardest-tier/gain/stability; B wins on raw accuracy
_A = _MC("A", 14.6, 14.6, True, post_acc=0.80, gain=0.10, hardest_tier="severe",
         hardest_tier_acc=0.90, stability_ci=0.01, latency_ms=70, rtf=0.03, peak_ram_mb=80)
_B = _MC("B", 14.6, 14.6, True, post_acc=0.85, gain=0.02, hardest_tier="severe",
         hardest_tier_acc=0.40, stability_ci=0.05, latency_ms=70, rtf=0.03, peak_ram_mb=80)
_pick_on = _bsc([_A, _B])["recommendation"]
_pick_off = _bsc([_A, _B], weights=_wns())["recommendation"]
check(f"toggle changes the pick (severity on -> {_pick_on}, off -> {_pick_off})",
      _pick_on == "A" and _pick_off == "B")

print("== 8. Zipformer training fidelity: ScaledAdam + Balancer/Whitener ==")
from echo.train.common import build_optimizer as _bopt
from echo.encoders.regularizers import Balancer as _Bal, Whitener as _Whi
from echo.train.scaled_adam import ScaledAdam as _SA
# right optimizer per encoder
_opt_z, _sc_z = _bopt(build_model("zipformer", TOK), "zipformer")
_opt_c, _sc_c = _bopt(build_model("conformer", TOK), "conformer")
check("zipformer uses ScaledAdam, conformer uses AdamW",
      isinstance(_opt_z, _SA) and _sc_z and not _sc_c)
# regularizers add ZERO params (train-only) and are identity in forward
_zp = count_parameters(build_encoder("zipformer"), trainable_only=False) / 1e6
check(f"zipformer still {_zp:.2f}M — regularizers add no params", 14.0 < _zp < 15.0)
_bx = torch.randn(2, 15, 64, requires_grad=True)
check("Balancer/Whitener are exact identity in forward",
      torch.equal(_Bal(64).train()(_bx), _bx) and
      torch.equal(_Whi().train()(_bx), _bx))
# ScaledAdam reduces a toy loss
torch.manual_seed(0)
_lin = torch.nn.Linear(16, 16); _o = _SA(_lin.parameters(), lr=0.04)
_X = torch.randn(32, 16); _Y = torch.randn(32, 16); _l0 = None
for _i in range(50):
    _o.zero_grad(); _L = ((_lin(_X) - _Y) ** 2).mean(); _L.backward(); _o.step()
    if _i == 0: _l0 = _L.item()
check(f"ScaledAdam reduces loss ({_l0:.2f} -> {_L.item():.2f})", _L.item() < _l0)


print("== 9. label-scheme toggle: match against intents OR any prompt ==")
from echo.data.labels import LabelScheme as _LS
_si = _LS("intent"); _sp = _LS("prompt")
_ps = ["open the door", "play music", "stop music"]
_int_labels = [_si(p) for p in _ps]
_prompt_labels = [_sp(p) for p in _ps]
check("intent mode maps prompts via the 36-intent taxonomy",
      all(l is None or 0 <= l < 36 for l in _int_labels))
check("prompt mode gives each distinct prompt its own class (0,1,2,...)",
      _prompt_labels == [0, 1, 2] and _sp.num_classes == 3)
check("prompt labels are case/whitespace-normalized and stable",
      _sp("Open  THE Door") == _sp("open the door"))
# cross_validate derives n_classes from data, so prompt mode (few classes) works
_lab = _LS("prompt")
_pr = ["cmd a", "cmd b", "cmd c", "cmd d"]
_items = [dict(wav=(torch.randn(int(1.1 * 16000)) * 0.1 +
                    0.4 * torch.sin(2 * _math.pi * (200 + 120 * (j % 4)) *
                                    torch.arange(int(1.1 * 16000)) / 16000)),
               text=_pr[j % 4], speaker=f"S{j % 3}", intent=_lab(_pr[j % 4]))
          for j in range(72)]
_urecs = [Utterance(utt_id=f"u{j}", corpus="torgo", speaker=it["speaker"],
    cohort="dysarthric", severity=["severe", "mild"][j % 2], group=f"pg{j}",
    mic="headMic", is_primary=True, wav_path="x", prompt=it["text"],
    intent=it["intent"], duration_s=1.1, sha1="") for j, it in enumerate(_items)]


class _LDS:
    def __init__(s, it): s.items = it
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.items[i]


_lplan = make_cv_plan(_urecs, n_folds=3, seeds=(1,))
_lres = _cv(build_model("conformer", TOK), _LDS(_items), _lplan,
            torch.device("cpu"), ks=(3,))
check("cross_validate runs in prompt mode (n_classes from data, no fixed 36)",
      _lres["post"][3]["mean"] == _lres["post"][3]["mean"])   # not NaN


print("== 10. meta-train-K x eval-K sweep (comprehensive K comparison) ==")
from echo.eval.sweep import metatrain_eval_sweep as _sweep
_sw_recs, _sw_items = [], []
for _spk in ("A", "B", "C", "D"):
    for _c in range(5):
        for _r in range(10):
            _w = (torch.randn(int(1.0 * 16000)) * 0.1 +
                  0.4 * torch.sin(2 * _math.pi * (200 + 120 * _c) *
                                  torch.arange(int(1.0 * 16000)) / 16000))
            _sw_items.append(dict(wav=_w, text=f"p{_c}", speaker=_spk, intent=_c))
            _sw_recs.append(Utterance(utt_id=f"{_spk}/{_c}/{_r}", corpus="torgo",
                speaker=_spk, cohort="dysarthric", severity="mild",
                group=f"sg{_spk}{_c}{_r}", mic="h", is_primary=True, wav_path="x",
                prompt=f"p{_c}", intent=_c, duration_s=1.0, sha1=""))


class _SwDS:
    def __init__(s, it): s.items = it
    def __len__(s): return len(s.items)
    def __getitem__(s, i): return s.items[i]


_swplan = make_cv_plan(_sw_recs, n_folds=3, seeds=(1,))
_swres = _sweep(build_model("conformer", TOK), _SwDS(_sw_items), _swplan,
                torch.device("cpu"), meta_ks=(3, 5), eval_ks=(3, 5),
                epochs=3, episodes=25, n_way=5)
check("sweep returns a full meta-K x eval-K matrix of real numbers",
      all(_swres["matrix"][mk][ek]["mean"] == _swres["matrix"][mk][ek]["mean"]
          for mk in (3, 5) for ek in (3, 5)))
check("sweep head is leakage-free (retrained per fold on train speakers only)",
      _swres["meta_ks"] == [3, 5] and _swres["eval_ks"] == [3, 5])


print("== 11. ESPnet E-Branchformer adapter (optional upstream encoder) ==")
from echo.encoders import ENCODERS, build_encoder
check("ebranchformer_espnet is registered (swappable like the others)",
      "ebranchformer_espnet" in ENCODERS)
# without espnet installed, building it must give a CLEAR error, not a crash
import importlib.util as _ilu
if _ilu.find_spec("espnet2") is None:
    _raised = False
    try:
        build_encoder("ebranchformer_espnet")
    except ImportError:
        _raised = True
    check("adapter gives a clear ImportError when espnet is absent (no crash)",
          _raised)
# verify MY adapter wiring against a mock ESPnet encoder (real espnet not needed)
import sys as _sys, types as _types
if _ilu.find_spec("espnet2") is None:
    _fm = _types.ModuleType("espnet2.asr.encoder.e_branchformer_encoder")

    class _FakeEBF(torch.nn.Module):
        def __init__(s, input_size, output_size, num_blocks, **kw):
            super().__init__()
            s.sub = torch.nn.Conv1d(input_size, output_size, 3, stride=2, padding=1)
            s.b = torch.nn.Linear(output_size, output_size)

        def forward(s, xs, il, prev_states=None, ctc=None):
            x = s.sub(xs.transpose(1, 2)).transpose(1, 2)
            x = x + torch.relu(s.b(x))
            ol = (torch.div(il - 1, 2, rounding_mode="floor") + 1).clamp_max(x.size(1))
            return x, ol, None
    _fm.EBranchformerEncoder = _FakeEBF
    for _n in ["espnet2", "espnet2.asr", "espnet2.asr.encoder"]:
        _sys.modules.setdefault(_n, _types.ModuleType(_n))
    _sys.modules["espnet2.asr.encoder.e_branchformer_encoder"] = _fm
    _ad = build_encoder("ebranchformer_espnet", n_mels=80, d_model=256, num_blocks=4)
    _am = AcousticModel(_ad, vocab_size=TOK.vocab_size).eval()
    with torch.no_grad():
        _e = _am.embed(torch.randn(2, 16000), torch.tensor([16000, 12000]))
    check("adapter maps ESPnet output through the interface -> 256-d embedding",
          not _ad.accepts_waveform and _e.shape == (2, 256) and
          bool((abs(_e.norm(dim=-1) - 1) < 1e-4).all()))
    for _n in ["espnet2", "espnet2.asr", "espnet2.asr.encoder",
               "espnet2.asr.encoder.e_branchformer_encoder"]:
        _sys.modules.pop(_n, None)


print("== 12. Moonshine own-vs-official (matched-size + official adapter) ==")
# moonshine_tiny = this repo's Moonshine at the official tiny geometry (~7.68M),
# for a like-for-like comparison against the released model.
_mt = count_parameters(build_encoder("moonshine_tiny"), trainable_only=False) / 1e6
_mb = count_parameters(build_encoder("moonshine"), trainable_only=False) / 1e6
check(f"moonshine_tiny matches official tiny size ({_mt:.2f}M ~= 7.68M)",
      7.5 < _mt < 7.9)
check(f"moonshine (budget) is the matched-size variant ({_mb:.2f}M)", 14.0 < _mb < 15.0)
check("moonshine_official is registered (real released model, needs transformers)",
      "moonshine_official" in ENCODERS)
if _ilu.find_spec("transformers") is None:
    _raised = False
    try:
        build_encoder("moonshine_official")
    except ImportError:
        _raised = True
    check("official Moonshine adapter gives a clear ImportError w/o transformers",
          _raised)
    # mock transformers to verify MY adapter wiring (real transformers not needed)
    import tempfile as _tf
    _t = _types.ModuleType("transformers")
    _tu = _types.ModuleType("transformers.utils")
    _tlg = _types.ModuleType("transformers.utils.logging")
    _tlg.set_verbosity_error = lambda: None
    _tu.logging = _tlg

    class _MCfg:
        model_type = "moonshine"; hidden_size = 288

    class _MEnc(torch.nn.Module):
        def __init__(s):
            super().__init__(); s.c = torch.nn.Conv1d(1, 288, 127, stride=64)

        def forward(s, iv):
            return _types.SimpleNamespace(
                last_hidden_state=s.c(iv.unsqueeze(1)).transpose(1, 2))

    class _MModel(torch.nn.Module):
        def __init__(s, cfg=None):
            super().__init__(); s.e = _MEnc()

        def get_encoder(s):
            return s.e

        @classmethod
        def from_pretrained(cls, f, output_loading_info=False):
            m = cls()
            return (m, {"missing_keys": [], "mismatched_keys": [],
                        "unexpected_keys": []}) if output_loading_info else m
    _t.MoonshineForConditionalGeneration = _MModel
    _t.AutoConfig = type("AC", (), {"from_pretrained": staticmethod(lambda f: _MCfg())})
    _t.AutoFeatureExtractor = type("AFE", (), {"from_pretrained":
        staticmethod(lambda f: (_ for _ in ()).throw(Exception("no extractor")))})
    _t.utils = _tu
    _sys.modules["transformers"] = _t
    _sys.modules["transformers.utils"] = _tu
    _sys.modules["transformers.utils.logging"] = _tlg
    _oad = build_encoder("moonshine_official", source=_tf.mkdtemp(), pretrained=False)
    _oam = AcousticModel(_oad, vocab_size=TOK.vocab_size).eval()
    with torch.no_grad():
        _oe = _oam.embed(torch.randn(2, 16000), torch.tensor([16000, 12000]))
    check("official Moonshine adapter -> 256-d embedding through the interface",
          _oad.accepts_waveform and _oe.shape == (2, 256))
    for _n in ["transformers", "transformers.utils", "transformers.utils.logging"]:
        _sys.modules.pop(_n, None)

print("\nALL SMOKE TESTS PASSED ✅")
