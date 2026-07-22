"""Generate the fixed, deterministic detector-style ("context") validation
crops used by classifier_v2 training (val_context_macro_f1). Run once —
the result is saved to datasets/acnescu/classifier/val_context_crops.json
and reused by every training run/epoch, never re-randomized.

Uses the class-specific scale-sampling policy in
src.data.acnescu_crops.CLASS_SCALE_POLICY (deeper_inflammatory_like is
capped at 1.5x, 0% at 2.0x, due to severe cross-class contamination found
at 2.0x — see runs/crop_scale_contamination_audit/).

Usage:
    python3 src/scripts/build_val_context_crops.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.acnescu_crops import CLASS_SCALE_POLICY, VAL_CONTEXT_CROPS_PATH, build_val_context_crops


def main():
    summary = build_val_context_crops(seed=42)
    print(f"Generated {summary['num_entries']} fixed context validation crops.")
    print("\nAchieved scale distribution per class (target in parentheses):")
    for cls, policy in CLASS_SCALE_POLICY.items():
        target = dict(zip(policy["scale_choices"], policy["scale_weights"]))
        achieved = summary["achieved_distribution_per_class"].get(cls, {})
        print(f"\n  {cls}:")
        for scale in policy["scale_choices"]:
            achieved_frac = achieved.get(str(scale), 0.0)
            print(f"    {scale:.1f}x  achieved={achieved_frac:.1%}  (target={target[scale]:.0%})")
    print(f"\nSaved: {VAL_CONTEXT_CROPS_PATH}")


if __name__ == "__main__":
    main()
