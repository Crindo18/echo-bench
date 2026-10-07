"""Quick TORGO data check (cross-platform, no bash needed).

Confirms the corpus extracted correctly and reports how many utterances were
found and which enrollment-K each speaker can support. Run from the project root:

    python scripts/check_torgo.py
    python scripts/check_torgo.py --root C:/Users/ASUS/Desktop/ECHO_Datasets/torgo --label-mode prompt
"""
import argparse
import os
import sys

# make `echo` importable whether or not the package was pip-installed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from echo.data import scan_corpus, coverage, print_coverage
from echo.data.labels import LabelScheme


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="C:/Users/ASUS/Desktop/ECHO_Datasets/torgo", help="path to the extracted TORGO corpus")
    ap.add_argument("--corpus", default="torgo", choices=["torgo", "uaspeech"])
    ap.add_argument("--speaker-table", default="configs/datasets/torgo_speakers.csv")
    ap.add_argument("--label-mode", default="prompt", choices=["prompt", "intent"])
    ap.add_argument("--ks", default="3,5,10")
    args = ap.parse_args()

    if not os.path.isdir(args.root):
        sys.exit(f"[check] corpus folder not found: {args.root}\n"
                 f"        extract the TORGO archives there first (see README Step 3).")

    recs = scan_corpus(args.root, args.corpus, args.speaker_table,
                       intent_map=LabelScheme(args.label_mode))
    speakers = sorted({r.speaker for r in recs})
    print(f"[check] {len(recs)} usable utterances scanned "
          f"from {len(speakers)} speakers ({args.label_mode} labels).")
    if not recs:
        sys.exit("[check] 0 utterances — check the folder layout: "
                 "C:/Users/ASUS/Desktop/ECHO_Datasets/torgo/<SPEAKER>/Session*/wav_*/*.wav + prompts/*.txt")
    print(f"[check] speakers: {' '.join(speakers)}")
    print_coverage(coverage(recs, ks=tuple(int(k) for k in args.ks.split(","))))


if __name__ == "__main__":
    main()
