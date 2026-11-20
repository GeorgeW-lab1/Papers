
#!/usr/bin/env python3
"""
Interactive Logical Composure Questionnaire + Kripke Model Simulator

Run:
    python modal_sim.py

Features:
    - Interactive Y/N questionnaire
    - Review and edit answers
    - Run simulation
    - Kripke model search
    - Minimal repair branches
    - Year-10 outlook
"""

from itertools import combinations

QUESTIONS = [
    "Planning ahead usually beats improvising.",
    "Improvising usually beats planning.",
    "If I commit to a goal, I will reach it. (C -> G)",
    "If I fail to reach a goal, I never really committed. (~G -> ~C)",
    "In the next year I will commit to a major goal. (C)",
    "Within 10 years I will reach a major goal. (G)",
    "More than one very different future is possible. (diamond)",
    "Some parts of my 10-year future are certain. (box)"
]

WORLDS = {
    "w1": (1, 1),
    "w2": (1, 0),
    "w3": (0, 1),
    "w4": (0, 0)
}


# ==========================================================
# LOGICAL ENGINE
# ==========================================================

def composure(a):
    q = dict(enumerate(a, 1))

    return {
        "T1 Non-contradiction": not (q[1] and q[2]),
        "T2 Contraposition": q[3] == q[4],
        "T3 Modus ponens": not (q[3] and q[5]) or q[6],
        "T4 Modal coherence": q[7] or q[8]
    }


def satisfies(model, a):
    vals = [WORLDS[w] for w in model]

    all_ = lambda f: all(f(c, g) for c, g in vals)
    some = lambda f: any(f(c, g) for c, g in vals)

    q = dict(enumerate(a, 1))

    checks = [
        not (q[1] and q[2]),

        all_(lambda c, g: not c or g)
        if q[3] else some(lambda c, g: c and not g),

        all_(lambda c, g: g or not c)
        if q[4] else some(lambda c, g: not g and c),

        all_(lambda c, g: c)
        if q[5] else some(lambda c, g: not c),

        all_(lambda c, g: g)
        if q[6] else some(lambda c, g: not g),

        (len(model) >= 2) == bool(q[7]),

        (
            len({c for c, _ in vals}) == 1
            or len({g for _, g in vals}) == 1
        ) == bool(q[8])
    ]

    return all(checks)


def all_models():
    names = list(WORLDS)

    for r in range(1, len(names) + 1):
        for sub in combinations(names, r):
            yield frozenset(sub)


def find_models(a):
    return [m for m in all_models() if satisfies(m, a)]


def repairs(a, max_flips=3):
    found = []

    for k in range(1, max_flips + 1):
        for idx in combinations(range(8), k):
            b = list(a)

            for i in idx:
                b[i] ^= 1

            ms = find_models(b)

            if ms and all(composure(b).values()):
                found.append((idx, b, ms))

        if found:
            return found

    return found


# ==========================================================
# PROJECTION
# ==========================================================

def describe(model):
    vals = [WORLDS[w] for w in sorted(model)]

    cs = {c for c, _ in vals}
    gs = {g for _, g in vals}

    parts = []

    parts.append(
        "always committed" if cs == {1}
        else "never committed" if cs == {0}
        else "commitment uncertain"
    )

    parts.append(
        "goal reached" if gs == {1}
        else "goal missed" if gs == {0}
        else "success contingent"
    )

    return ", ".join(parts)


def fmt(model):
    return "{" + ", ".join(sorted(model)) + "}"


def bits(a):
    return " ".join("Y" if x else "N" for x in a)


# ==========================================================
# SIMULATION REPORT
# ==========================================================

def run(a):
    lines = []

    def out(s=""):
        lines.append(s)

    out("=" * 65)
    out("       LOGICAL COMPOSURE / KRIPKE MODEL SIMULATION")
    out("=" * 65)

    out("\nAnswers: " + bits(a))

    out("\n== COMPOSURE TESTS ==")

    tests = composure(a)

    for name, ok in tests.items():
        out(f"  [{'PASS' if ok else 'FAIL'}] {name}")

    score = sum(tests.values())

    out(f"\n  Composure score: {score}/4 ({score * 25}%)")

    out("\n== KRIPKE MODEL SEARCH ==")

    ms = find_models(a)

    out("  Candidate world-sets: 15")

    if ms:
        out(f"  CONSISTENT: {len(ms)} model(s) satisfy all answers.")

        for model in ms:
            out(f"    {fmt(model)} -> {describe(model)}")

    else:
        out("  INCONSISTENT: no world-set satisfies all answers.")

        out("\n== MINIMAL REPAIR BRANCHES ==")

        rs = repairs(a)

        if not rs:
            out("  No repair within 3 answer flips.")

        for idx, b, models in rs:
            changes = ", ".join(
                f"Q{i+1}: {'Y' if a[i] else 'N'}"
                f" -> {'Y' if b[i] else 'N'}"
                for i in idx
            )

            out(f"\n  Change {changes}")

            for model in models:
                out(f"    {fmt(model)} -> {describe(model)}")

    out("\n== YEAR-10 OUTLOOK ==")

    pool = ms if ms else [
        model
        for _, _, models in repairs(a)
        for model in models
    ]

    if pool:
        worlds_seen = sorted({
            world
            for model in pool
            for world in model
        })

        n_goal = sum(WORLDS[w][1] for w in worlds_seen)

        out("  Possible worlds: " + ", ".join(worlds_seen))
        out(f"  Goal reached in {n_goal}/{len(worlds_seen)} possible worlds.")

    else:
        out("  No outlook available.")

    out("\n" + "=" * 65)
    out("Formal consequences of the answers, not a real-world forecast.")
    out("=" * 65)

    return "\n".join(lines)


# ==========================================================
# INTERACTIVE TERMINAL
# ==========================================================

def ask(question, number, previous=None):
    while True:
        suffix = f" [current: {'Y' if previous else 'N'}]" if previous is not None else ""

        answer = input(
            f"\nQ{number}. {question}{suffix}\n"
            "     [Y] Yes  [N] No  [B] Back  [Q] Quit: "
        ).strip().upper()

        if answer in ("Y", "N"):
            return int(answer == "Y")

        if answer == "B":
            return "BACK"

        if answer == "Q":
            raise SystemExit("\nQuestionnaire terminated.")

        print("Invalid input. Enter Y, N, B, or Q.")


def questionnaire():
    answers = [None] * len(QUESTIONS)
    index = 0

    print("\n" + "=" * 65)
    print("          LOGICAL COMPOSURE QUESTIONNAIRE")
    print("=" * 65)

    print("\nAnswer each statement with Y or N.")
    print("You can enter B to return to the previous question.")
    print("You can enter Q to quit.")

    while index < len(QUESTIONS):
        answer = ask(
            QUESTIONS[index],
            index + 1,
            answers[index]
        )

        if answer == "BACK":
            if index > 0:
                index -= 1
            else:
                print("Already at the first question.")
            continue

        answers[index] = answer
        index += 1

    return answers


def review(answers):
    while True:
        print("\n" + "=" * 65)
        print("                  ANSWER REVIEW")
        print("=" * 65)

        for i, (question, answer) in enumerate(
            zip(QUESTIONS, answers), 1
        ):
            print(
                f"\n  Q{i}. {question}\n"
                f"      Answer: {'YES' if answer else 'NO'}"
            )

        print("\n[E] Edit answer")
        print("[R] Run simulation")
        print("[Q] Quit")

        choice = input("\nSelection: ").strip().upper()

        if choice == "E":
            while True:
                try:
                    number = int(input("Question number (1-8): "))
                    if 1 <= number <= 8:
                        break
                except ValueError:
                    pass

                print("Enter a number from 1 to 8.")

            answer = ask(QUESTIONS[number - 1], number)

            if answer != "BACK":
                answers[number - 1] = answer

        elif choice == "R":
            return answers

        elif choice == "Q":
            raise SystemExit("\nSimulation terminated.")

        else:
            print("Invalid selection.")


def main():
    while True:
        answers = questionnaire()
        answers = review(answers)

        print("\n" + run(answers))

        choice = input(
            "\n[R] Restart questionnaire\n"
            "[Q] Quit\n"
            "Selection: "
        ).strip().upper()

        if choice != "R":
            break

    print("\nSession ended.")


if __name__ == "__main__":
    main()
