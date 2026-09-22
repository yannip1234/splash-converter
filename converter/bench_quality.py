"""Quality benchmark: task accuracy and cross-engine agreement.

Perplexity is unavailable here (the Splash HTTP API exposes no logprobs and no
/v1/completions), so quality is measured two ways instead:

1. TASK ACCURACY — short questions with programmatically checkable answers,
   scored exact-match after normalisation. Absolute, engine-independent.

2. AGREEMENT WITH AN 8-BIT REFERENCE — the same Swift weights served at 8-bit
   (oMLX oQ8) are the closest thing to ground truth available on this machine.
   A better 4-bit quantization should agree with it more often. This is what
   makes the comparison fair between two *different* 4-bit schemes: neither is
   judged against the other, both are judged against higher precision.

Chat templates differ between engines, so agreement is measured on normalised
answer text, not token ids.

Usage:
  python -m converter.bench_quality --base-url http://127.0.0.1:8127/v1 \
      --model local/Swift-Qwen3.8-27B-Splash --label splash-q4 --out q.json
"""

from __future__ import annotations

import argparse
import json
import re
import time
import urllib.request

# (prompt, accepted answers). Answers are checked after normalisation, and the
# model is told to answer bare so that reasoning models still end with the value.
TASKS = [
    ("What is 17 * 23? Reply with only the number.", ["391"]),
    ("What is 2^10? Reply with only the number.", ["1024"]),
    ("What is 144 / 12? Reply with only the number.", ["12"]),
    ("What is 1000 - 377? Reply with only the number.", ["623"]),
    ("What is 15% of 240? Reply with only the number.", ["36"]),
    ("How many minutes are in 3 hours? Reply with only the number.", ["180"]),
    ("What is the 7th prime number? Reply with only the number.", ["17"]),
    ("What is the square root of 169? Reply with only the number.", ["13"]),
    ("What is the capital of Japan? Reply with only the city name.", ["tokyo"]),
    ("What is the capital of Australia? Reply with only the city name.", ["canberra"]),
    ("What is the capital of Canada? Reply with only the city name.", ["ottawa"]),
    ("What is the capital of Brazil? Reply with only the city name.", ["brasilia", "brasília"]),
    ("Which planet is closest to the Sun? Reply with only the planet name.", ["mercury"]),
    ("What is the chemical symbol for gold? Reply with only the symbol.", ["au"]),
    ("What is the chemical symbol for potassium? Reply with only the symbol.", ["k"]),
    ("How many bones are in the adult human body? Reply with only the number.", ["206"]),
    ("Who wrote the play 'Hamlet'? Reply with only the surname.", ["shakespeare"]),
    ("Who wrote '1984'? Reply with only the surname.", ["orwell"]),
    ("In which year did the Berlin Wall fall? Reply with only the year.", ["1989"]),
    ("In which year did the Titanic sink? Reply with only the year.", ["1912"]),
    ("How many continents are there? Reply with only the number.", ["7", "seven"]),
    ("What is the largest ocean on Earth? Reply with only the ocean name.",
     ["pacific", "pacific ocean"]),
    ("What language has the most native speakers? Reply with only the language name.",
     ["mandarin", "chinese", "mandarin chinese"]),
    ("Reverse the string 'stressed'. Reply with only the result.", ["desserts"]),
    ("How many letters are in the word 'extraordinary'? Reply with only the number.", ["13"]),
    ("What is the last letter of the alphabet? Reply with only the letter.", ["z"]),
    ("Sort these numbers ascending and reply with them comma-separated: 5, 2, 9, 1",
     ["1,2,5,9", "1, 2, 5, 9"]),
    ("What does the Python expression len('hello world') return? Reply with only the number.",
     ["11"]),
    ("In Python, what does 7 // 2 evaluate to? Reply with only the number.", ["3"]),
    ("In Python, what does bool([]) return? Reply with only True or False.", ["false"]),
    ("What is the time complexity of binary search? Reply with only the big-O notation.",
     ["o(log n)", "o(logn)", "o(log(n))"]),
    ("What HTTP status code means 'Not Found'? Reply with only the number.", ["404"]),
    ("How many bits are in a byte? Reply with only the number.", ["8"]),
    ("What is the default port for HTTPS? Reply with only the number.", ["443"]),
    ("If a train travels 60 km in 45 minutes, what is its speed in km/h? "
     "Reply with only the number.", ["80"]),
    ("A shirt costs $40 after a 20% discount. What was the original price in dollars? "
     "Reply with only the number.", ["50"]),
    ("If today is Wednesday, what day is it 10 days later? Reply with only the day name.",
     ["saturday"]),
    ("How many sides does a hexagon have? Reply with only the number.", ["6", "six"]),
    ("What is the boiling point of water in Celsius at sea level? Reply with only the number.",
     ["100"]),
    ("Translate 'thank you' into Spanish. Reply with only the translation.",
     ["gracias"]),
]


# A harder set. The 40 easy tasks saturate at 100% for every healthy
# configuration, which cannot distinguish "as good as the reference" from
# "slightly worse". These need multiple steps, so a damaged model degrades
# measurably before it becomes obviously broken.
HARD_TASKS = [
    # multi-step arithmetic
    ("What is 47 * 63? Reply with only the number.", ["2961"]),
    ("What is 17 cubed? Reply with only the number.", ["4913"]),
    ("What is 1234 + 5678? Reply with only the number.", ["6912"]),
    ("What is 15% of 1840? Reply with only the number.", ["276"]),
    ("What is (25 * 4) - (18 / 3)? Reply with only the number.", ["94"]),
    ("What is 7 factorial? Reply with only the number.", ["5040"]),
    ("What is the square root of 1024? Reply with only the number.", ["32"]),
    ("What is 3 to the power of 7? Reply with only the number.", ["2187"]),
    ("What is 999 * 999? Reply with only the number.", ["998001"]),
    ("What is 2 to the power of 16? Reply with only the number.", ["65536"]),
    # word problems
    ("A train travels 120 km in 1.5 hours. What is its speed in km/h? "
     "Reply with only the number.", ["80"]),
    ("A shirt costs $60 after a 25% discount. What was the original price in dollars? "
     "Reply with only the number.", ["80"]),
    ("If 3 workers finish a job in 6 days, how many days do 9 workers need at the same "
     "rate? Reply with only the number.", ["2"]),
    ("A car drives 40 mph for 2 hours then 60 mph for 3 hours. How many miles in total? "
     "Reply with only the number.", ["260"]),
    ("A book has 300 pages. After reading 2/5 of it, how many pages remain? "
     "Reply with only the number.", ["180"]),
    ("A rectangle has perimeter 36 and its length is twice its width. What is its area? "
     "Reply with only the number.", ["72"]),
    ("If 5 apples cost $3.75, what do 8 apples cost in dollars? Reply with only the number.",
     ["6"]),
    ("A meeting starts at 8:45 am and lasts 2 hours 40 minutes. What time does it end? "
     "Reply with only the time, like 3:15 pm.", ["11:25 am", "11:25am", "1125 am"]),
    ("What is a 20% tip on a $45 bill, in dollars? Reply with only the number.", ["9"]),
    ("What is the average of 12, 18, 24 and 30? Reply with only the number.", ["21"]),
    # sequences and logic
    ("What comes next: 2, 6, 12, 20, 30? Reply with only the number.", ["42"]),
    ("What comes next: 1, 1, 2, 3, 5, 8? Reply with only the number.", ["13"]),
    ("What comes next: 3, 6, 12, 24? Reply with only the number.", ["48"]),
    ("All Bloops are Razzies and all Razzies are Lazzies. Are all Bloops Lazzies? "
     "Reply with only yes or no.", ["yes"]),
    ("Alice is older than Bob. Bob is older than Carol. Who is youngest? "
     "Reply with only the name.", ["carol"]),
    ("If today is Friday, what day is it 100 days later? Reply with only the day name.",
     ["sunday"]),
    ("How many times does the digit 1 appear when writing the numbers 1 to 20? "
     "Reply with only the number.", ["12"]),
    ("A clock shows 3:00. What is the angle in degrees between the hands? "
     "Reply with only the number.", ["90"]),
    # strings and code
    ("Reverse the string 'algorithm'. Reply with only the result.", ["mhtirogla"]),
    ("How many characters are in the word 'quantization'? Reply with only the number.",
     ["12"]),
    ("In Python, what does sum(range(10)) return? Reply with only the number.", ["45"]),
    ("In Python, what does 2**10 // 3 return? Reply with only the number.", ["341"]),
    ("In Python, what does len(set([1,1,2,2,3])) return? Reply with only the number.",
     ["3"]),
    ("In Python, what does ''.join(sorted('dcba')) return? Reply with only the result.",
     ["abcd"]),
    ("How many vowels are in the word 'encyclopedia'? Reply with only the number.", ["5"]),
    ("In Python, what does bool('False') return? Reply with only True or False.", ["true"]),
    ("In Python, what does list(range(2, 10, 3)) return? Reply with only the list.",
     ["[2, 5, 8]", "[2,5,8]"]),
    ("In Python, what does 10 % 3 return? Reply with only the number.", ["1"]),
    # knowledge
    ("What is the chemical symbol for tungsten? Reply with only the symbol.", ["w"]),
    ("Which is the largest planet in the Solar System? Reply with only the planet name.",
     ["jupiter"]),
    ("Who developed the theory of general relativity? Reply with only the surname.",
     ["einstein"]),
    ("What is the capital of Switzerland? Reply with only the city name.", ["bern"]),
    ("What is the currency of Japan? Reply with only the currency name.", ["yen"]),
    ("Who wrote 'Pride and Prejudice'? Reply with only the surname.", ["austen"]),
    ("In which year did World War 2 end? Reply with only the year.", ["1945"]),
    ("Which element has atomic number 6? Reply with only the element name.", ["carbon"]),
    ("How many degrees are in a circle? Reply with only the number.", ["360"]),
    ("What is the smallest prime number? Reply with only the number.", ["2"]),
    ("What is the capital of New Zealand? Reply with only the city name.", ["wellington"]),
    ("How many players from one team are on a soccer field at kickoff? "
     "Reply with only the number.", ["11"]),
    # units
    ("How many grams are in 2.5 kilograms? Reply with only the number.", ["2500"]),
    ("How many feet are in one mile? Reply with only the number.", ["5280"]),
    ("How many days are in 72 hours? Reply with only the number.", ["3"]),
    ("What is 100 degrees Celsius in Fahrenheit? Reply with only the number.", ["212"]),
    ("How many minutes are in a week? Reply with only the number.", ["10080"]),
]

TASK_SETS = {"easy": TASKS, "hard": HARD_TASKS, "all": TASKS + HARD_TASKS}

# Open-ended prompts used only for cross-engine agreement, not scored for truth.
AGREEMENT_PROMPTS = [
    "In one sentence, what is photosynthesis?",
    "In one sentence, what does a compiler do?",
    "Name three primary colors, comma-separated.",
    "In one sentence, what is the greenhouse effect?",
    "In one short sentence, what is recursion?",
    "Name the four seasons, comma-separated.",
]


def normalise(text):
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[*_`#]", "", text)
    text = re.sub(r"[.!]+$", "", text.strip())
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def answer_matches(reply, accepted):
    """Exact match, tolerant of formatting but not of extra content.

    A reasoning model often answers with a short preamble ("The answer is: 391")
    or puts the value on its own final line, so the last line and the trailing
    words are checked too. Anything longer than the answer itself still fails --
    the point is to score correctness, not to hunt for the answer in an essay.
    """
    wanted = [normalise(a) for a in accepted]
    squash = lambda t: re.sub(r"[^a-z0-9(),]", "", t)
    # If the expected answer is a bare number, ignore units and currency the
    # model may attach ("100°C", "$50", "80 km/h") -- applied to every engine.
    if all(re.fullmatch(r"[0-9]+", w) for w in wanted):
        numeric = lambda t: (re.findall(r"-?\d+", t) or [""])[-1] if len(re.findall(r"-?\d+", t)) == 1 else ""
    else:
        numeric = lambda t: ""
    candidates = [normalise(reply)]
    lines = [ln for ln in (reply or "").splitlines() if ln.strip()]
    if lines:
        candidates.append(normalise(lines[-1]))
        candidates.append(normalise(lines[0]))   # answer first, explanation after
    for candidate in list(candidates):
        # trailing value after a colon or a short lead-in
        if ":" in candidate:
            candidates.append(normalise(candidate.rsplit(":", 1)[1]))
        words = candidate.split()
        if 1 < len(words) <= 8:
            candidates.append(words[-1])
    def as_number(text):
        found = re.findall(r"-?\d+(?:\.\d+)?", text.replace(",", ""))
        if len(found) != 1:
            return None
        try:
            return float(found[0])
        except ValueError:
            return None

    numeric_wanted = [as_number(w) for w in wanted]
    numeric_wanted = [v for v in numeric_wanted if v is not None]
    for candidate in candidates:
        if candidate in wanted or any(squash(candidate) == squash(w) for w in wanted):
            return True
        if numeric(candidate) and numeric(candidate) in wanted:
            return True
        # "6.00" and "6" are the same answer; applied to every engine alike
        value = as_number(candidate)
        if value is not None and any(abs(value - w) < 1e-9 for w in numeric_wanted):
            return True
    return False


def ask(base_url, model, prompt, api_key, max_tokens, timeout):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": prompt}],
                       "max_tokens": max_tokens, "temperature": 0}).encode()
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    request = urllib.request.Request(f"{base_url}/chat/completions", body, headers)
    started = time.time()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    choice = payload["choices"][0]
    return dict(content=choice["message"].get("content") or "",
                finish=choice.get("finish_reason"),
                seconds=round(time.time() - started, 2),
                usage=payload.get("usage", {}))


def run(base_url, model, api_key, max_tokens, timeout, log=print, tasks=None):
    tasks = TASKS if tasks is None else tasks
    correct = 0
    task_rows = []
    for prompt, accepted in tasks:
        r = ask(base_url, model, prompt, api_key, max_tokens, timeout)
        ok = answer_matches(r["content"], accepted)
        correct += ok
        task_rows.append(dict(prompt=prompt, answer=r["content"].strip()[:120],
                              accepted=accepted, correct=ok, finish=r["finish"],
                              seconds=r["seconds"]))
        log(f"  {'OK ' if ok else 'MISS'} {normalise(r['content'])[:44]:46s} {prompt[:40]}")
    agree_rows = []
    for prompt in AGREEMENT_PROMPTS:
        r = ask(base_url, model, prompt, api_key, max_tokens, timeout)
        agree_rows.append(dict(prompt=prompt, answer=r["content"].strip()))
    return dict(model=model, base_url=base_url, tasks=len(tasks), correct=correct,
                accuracy=round(correct / len(tasks), 4), task_rows=task_rows,
                agreement_rows=agree_rows)


def main(argv=None):
    p = argparse.ArgumentParser(prog="converter.bench_quality")
    p.add_argument("--base-url", required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--api-key-file", default=None)
    p.add_argument("--max-tokens", type=int, default=1400)
    p.add_argument("--timeout", type=float, default=600)
    p.add_argument("--label", default="run")
    p.add_argument("--out", default=None)
    p.add_argument("--task-set", default="easy", choices=sorted(TASK_SETS))
    args = p.parse_args(argv)
    key = None
    if args.api_key_file:
        with open(args.api_key_file) as f:
            key = f.read().strip()
    print(f"quality benchmark {args.label}: {args.model}")
    result = run(args.base_url, args.model, key, args.max_tokens, args.timeout,
                 tasks=TASK_SETS[args.task_set])
    result["task_set"] = args.task_set
    result["label"] = args.label
    print(f"\n{args.label}: {result['correct']}/{result['tasks']} correct "
          f"({result['accuracy'] * 100:.1f}%)")
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
