"""
validate_tobler.py

Validates the Tobler hiking function against real SF geography scenarios.
Computes round-trip walking times: outbound (unloaded) + return (loaded, 1.2x penalty).

Tobler hiking function:
    speed (km/h) = 6 * exp(-3.5 * |grade + 0.05|)

where grade = rise / run (dimensionless slope).

The +0.05 offset reflects that humans walk slightly faster on gentle downslopes.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------

def tobler_speed(grade: float) -> float:
    """
    Tobler hiking function.
    Returns walking speed in km/h for a given slope (grade = rise/run).
    """
    return 6.0 * np.exp(-3.5 * abs(grade + 0.05))


def walking_time_minutes(distance_m: float, grade: float) -> float:
    """
    Compute walking time in minutes for a given one-way distance (metres)
    and slope grade (rise/run).
    """
    speed_kmh = tobler_speed(grade)
    distance_km = distance_m / 1000.0
    time_hours = distance_km / speed_kmh
    return time_hours * 60.0


def round_trip_time(distance_m: float, outbound_grade: float,
                    grocery_penalty: float = 1.2) -> dict:
    """
    Compute full round-trip stats.

    Parameters
    ----------
    distance_m       : one-way network distance in metres
    outbound_grade   : slope on the way TO the store (rise/run)
    grocery_penalty  : multiplicative time penalty for carrying groceries (return)

    Returns
    -------
    dict with all intermediate values and totals
    """
    return_grade = -outbound_grade  # reversed slope on the way back

    out_speed = tobler_speed(outbound_grade)
    ret_speed = tobler_speed(return_grade)

    out_time = walking_time_minutes(distance_m, outbound_grade)
    ret_time_base = walking_time_minutes(distance_m, return_grade)
    ret_time = ret_time_base * grocery_penalty

    total = out_time + ret_time

    return {
        "distance_m": distance_m,
        "outbound_grade": outbound_grade,
        "out_speed_kmh": out_speed,
        "out_time_min": out_time,
        "return_grade": return_grade,
        "ret_speed_kmh": ret_speed,
        "ret_time_min": ret_time,
        "total_min": total,
        "accessible": total <= 30.0,
    }


# ---------------------------------------------------------------------------
# Scenarios
# ---------------------------------------------------------------------------

SCENARIOS = [
    {
        "label": "Flat: SoMa -> Safeway Bryant St",
        "distance_m": 600,
        "grade": +0.01,
    },
    {
        "label": "Moderate uphill: Mission/24th -> Whole Foods Castro",
        "distance_m": 700,
        "grade": +0.06,
    },
    {
        "label": "Steep uphill: Noe Valley floor -> store atop hill",
        "distance_m": 500,
        "grade": +0.12,
    },
    {
        "label": "Very steep: base of Twin Peaks -> store uphill",
        "distance_m": 400,
        "grade": +0.20,
    },
    {
        "label": "Downhill store: resident at top -> store at bottom",
        "distance_m": 500,
        "grade": -0.15,
    },
]


# ---------------------------------------------------------------------------
# Grade sensitivity table values
# ---------------------------------------------------------------------------

SENSITIVITY_GRADES = [-0.20, -0.15, -0.10, -0.05, 0.00,
                       +0.05, +0.10, +0.15, +0.20, +0.25]


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def fmt_bool(val: bool) -> str:
    return "YES" if val else "NO "


def separator(width: int = 110) -> str:
    return "-" * width


# ---------------------------------------------------------------------------
# Main output
# ---------------------------------------------------------------------------

def main():
    W = 110  # total table width

    print()
    print("=" * W)
    print("  TOBLER HIKING FUNCTION — SF GEOGRAPHY VALIDATION")
    print("  Grocery trip model: outbound unloaded  |  return loaded (1.2x time penalty, reversed slope)")
    print("=" * W)

    # -----------------------------------------------------------------------
    # Scenario table header
    # -----------------------------------------------------------------------
    print()
    print("SCENARIO RESULTS")
    print(separator(W))

    col_w = [38, 6, 8, 8, 8, 8, 10, 8, 6]
    hdr = (
        f"{'Scenario':<{col_w[0]}}"
        f"{'Dist':>{col_w[1]}}"
        f"{'ObGrade':>{col_w[2]}}"
        f"{'ObSpd':>{col_w[3]}}"
        f"{'ObTime':>{col_w[4]}}"
        f"{'RetGrd':>{col_w[5]}}"
        f"{'RetSpd':>{col_w[6]}}"
        f"{'RetTime':>{col_w[7]}}"
        f"{'Total':>{col_w[8]}}"
        f"  {'<=30m?':>6}"
    )
    print(hdr)
    sub = (
        f"{'':>{col_w[0]}}"
        f"{'(m)':>{col_w[1]}}"
        f"{'(r/r)':>{col_w[2]}}"
        f"{'(km/h)':>{col_w[3]}}"
        f"{'(min)':>{col_w[4]}}"
        f"{'(r/r)':>{col_w[5]}}"
        f"{'(km/h)':>{col_w[6]}}"
        f"{'(min,x1.2)':>{col_w[7]+2}}"
        f"{'(min)':>{col_w[8]-2}}"
        f"  {'access?':>6}"
    )
    print(sub)
    print(separator(W))

    results = []
    for sc in SCENARIOS:
        r = round_trip_time(sc["distance_m"], sc["grade"])
        results.append((sc, r))

        row = (
            f"{sc['label']:<{col_w[0]}}"
            f"{r['distance_m']:>{col_w[1]}}"
            f"{r['outbound_grade']:>{col_w[2]+1}.2f} "
            f"{r['out_speed_kmh']:>{col_w[3]}.2f}"
            f"{r['out_time_min']:>{col_w[4]}.1f}"
            f"{r['return_grade']:>{col_w[5]+1}.2f} "
            f"{r['ret_speed_kmh']:>{col_w[6]}.2f}"
            f"{r['ret_time_min']:>{col_w[7]+2}.1f}"
            f"{r['total_min']:>{col_w[8]}.1f}"
            f"  {fmt_bool(r['accessible']):>6}"
        )
        print(row)

    print(separator(W))

    # -----------------------------------------------------------------------
    # Per-scenario narrative
    # -----------------------------------------------------------------------
    print()
    print("SCENARIO NOTES")
    print(separator(W))

    notes = [
        ("Flat: SoMa -> Safeway Bryant St",
         f"Total {results[0][1]['total_min']:.1f} min — nearly flat walking, well under 30 min. "
         "Highly accessible."),
        ("Moderate uphill: Mission/24th -> Whole Foods Castro",
         f"Total {results[1][1]['total_min']:.1f} min — modest grade adds noticeable time. "
         "Still accessible for most walkers."),
        ("Steep uphill: Noe Valley floor -> store atop hill",
         f"Total {results[2][1]['total_min']:.1f} min — steep grade significantly reduces speed. "
         f"{'Within' if results[2][1]['accessible'] else 'Exceeds'} 30-min threshold."),
        ("Very steep: base of Twin Peaks -> store uphill",
         f"Total {results[3][1]['total_min']:.1f} min — very steep grade makes even a short trip slow. "
         f"{'Accessible' if results[3][1]['accessible'] else 'Not accessible'} at 30-min cutoff."),
        ("Downhill store: resident at top -> store at bottom",
         f"Outbound {results[4][1]['out_time_min']:.1f} min (fast, downhill) "
         f"vs return {results[4][1]['ret_time_min']:.1f} min (slow, uphill + groceries). "
         f"Total {results[4][1]['total_min']:.1f} min — asymmetry clearly visible."),
    ]

    for label, note in notes:
        print(f"  {label}")
        print(f"    -> {note}")
        print()

    # -----------------------------------------------------------------------
    # Grade sensitivity table
    # -----------------------------------------------------------------------
    print("=" * W)
    print("GRADE SENSITIVITY — Tobler Speed at Various Slopes")
    print(separator(W))
    hdr2 = (
        f"{'Grade (rise/run)':>20}  "
        f"{'Speed (km/h)':>14}  "
        f"{'Time for 500m (min)':>22}  "
        f"{'Time for 1000m (min)':>22}  "
        f"{'% of flat speed':>16}"
    )
    print(hdr2)
    print(separator(W))

    flat_speed = tobler_speed(0.0)
    for g in SENSITIVITY_GRADES:
        spd = tobler_speed(g)
        t500 = walking_time_minutes(500, g)
        t1000 = walking_time_minutes(1000, g)
        pct = 100.0 * spd / flat_speed
        sign = "+" if g > 0 else (" " if g == 0 else "")
        row2 = (
            f"{sign}{g:>19.2f}  "
            f"{spd:>14.3f}  "
            f"{t500:>22.1f}  "
            f"{t1000:>22.1f}  "
            f"{pct:>15.1f}%"
        )
        print(row2)

    print(separator(W))

    # -----------------------------------------------------------------------
    # Asymmetry insight note
    # -----------------------------------------------------------------------
    print()
    print("=" * W)
    print("KEY INSIGHT: THE DOWNHILL DECEPTION")
    print(separator(W))
    print("""
  A store located downhill from a resident appears highly accessible on the outbound trip —
  gravity assists the walk, and Tobler speeds are close to or faster than flat-ground pace
  for gentle-to-moderate descents.

  However, the RETURN trip — uphill AND loaded with groceries (1.2x time penalty) — can be
  significantly harder than any purely flat round trip of the same distance. This asymmetry
  means that naive straight-line or even flat-network distance metrics will UNDERESTIMATE
  the effective access burden for residents living uphill from food stores.

  Example from the "Downhill store" scenario above:
    Outbound (downhill, grade {out_g:+.2f}):  {out_t:.1f} min
    Return   (uphill,  grade {ret_g:+.2f}):  {ret_t:.1f} min  (includes 1.2x grocery load)
    Ratio return/outbound:                  {ratio:.2f}x

  This ratio grows rapidly with steeper slopes, which is common in SF neighborhoods
  like the Mission, Noe Valley, and the Castro. Food access models that ignore slope
  and load asymmetry will systematically misclassify some of the city's most
  topographically disadvantaged residents as "well-served."
""".format(
        out_g=results[4][1]["outbound_grade"],
        out_t=results[4][1]["out_time_min"],
        ret_g=results[4][1]["return_grade"],
        ret_t=results[4][1]["ret_time_min"],
        ratio=results[4][1]["ret_time_min"] / results[4][1]["out_time_min"],
    ))
    print("=" * W)
    print()


if __name__ == "__main__":
    main()
