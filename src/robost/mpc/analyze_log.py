"""Summarize an MPC_LOG CSV written by rs06_mpc.

    MPC_LOG=runs/mpc/debug.csv ./src/robost/mpc/build/rs06_mpc --headless --vx 0.2
    python src/robost/mpc/analyze_log.py runs/mpc/debug.csv
"""
import csv
import sys

LEGS = ("FR", "FL", "RR", "RL")
CONTACT_N = 5.0  # [N] foot normal force above this counts as real contact


def main(path: str, tail_s: float = 1.0) -> None:
    rows = [r for r in csv.DictReader(open(path)) if r["gait"] != "stand"]
    if not rows:
        print("no gait rows")
        return
    f = lambda r, k: float(r[k])

    print(f"{rows[0]['gait']} rows: {len(rows)}  ({f(rows[0], 't'):.2f}s .. {f(rows[-1], 't'):.2f}s)\n")
    print("leg | stance: no contact | swing: in contact | stance: joint saturated | MPC fz at f_max")
    for leg in LEGS:
        st = [r for r in rows if r[f"{leg}_sched"] == "1"]
        sw = [r for r in rows if r[f"{leg}_sched"] == "0"]
        miss = sum(f(r, f"{leg}_cn") < CONTACT_N for r in st)
        drag = sum(f(r, f"{leg}_cn") >= CONTACT_N for r in sw)
        sat = sum(int(r[f"{leg}_sat"]) > 0 for r in st)
        fzmax = max((f(r, f"{leg}_fz") for r in st), default=0.0)
        pct = lambda n, d: f"{100 * n / max(d, 1):5.1f}%"
        print(f"{leg}  | {pct(miss, len(st)):>18} | {pct(drag, len(sw)):>17} | "
              f"{pct(sat, len(st)):>23} | {fzmax:7.1f}N")
    other = sum(int(r["other_contacts"]) > 0 for r in rows)
    print(f"\nnon-foot body touching floor: {100 * other / len(rows):.1f}% of MPC ticks")

    t_end = f(rows[-1], "t")
    print(f"\nlast {tail_s:.1f}s (sched/contactN/sat per leg):")
    print("   t    roll  pitch |" + " | ".join(f"{l:^14}" for l in LEGS) + " | other")
    for r in rows:
        if f(r, "t") < t_end - tail_s:
            continue
        legs = " | ".join(
            f"{'S' if r[f'{l}_sched'] == '1' else 'w'} {f(r, f'{l}_cn'):6.1f} s{r[f'{l}_sat']}"
            .ljust(14) for l in LEGS)
        print(f"{f(r, 't'):5.2f} {f(r, 'roll'):6.3f} {f(r, 'pitch'):6.3f} | {legs} | {r['other_contacts']}")


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 1.0)
