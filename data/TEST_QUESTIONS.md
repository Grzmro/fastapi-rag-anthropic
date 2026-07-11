# Manual test questions

Ingest all four sample documents first:

```bash
for f in data/samples/*.md; do curl -s -F "file=@$f" http://localhost:8000/ingest; echo; done
```

Then send each question to `POST /ask` (use `"include_chunks": true` to see
what was retrieved). Check three things every time:

1. **Correctness** — is the answer factually right per the source document?
2. **Citation validity** — does the cited `source` + quote actually contain
   the claim? (Open the file and confirm.)
3. **Right document retrieved** — for single-topic questions, the citation
   should point at the expected file, not a neighbour.

The interesting cases are the **trap** questions: the answer is not in any
document, so a good system says "I don't know" and returns an **empty
citations list** instead of inventing something.

| # | Question | Expected source | Expected behaviour |
|---|----------|-----------------|--------------------|
| 1 | What is the largest volcano in the Solar System? | solar_system.md | "Olympus Mons, ~22 km", cited |
| 2 | How much can a remote employee claim per year for home-office equipment? | remote_work_policy.md | "500 USD", cited |
| 3 | Who created Python and in what year? | python_language.md | "Guido van Rossum, 1991", cited |
| 4 | What water temperature is recommended for brewing coffee? | coffee_brewing.md | "90–96 °C", cited |
| 5 | Are contractors eligible to work remotely at Acme Corp? | remote_work_policy.md | No — contractors are excluded, cited |
| 6 | When did Python 2 reach end of life? | python_language.md | "January 1, 2020", cited |
| 7 | What grind size is used for espresso? | coffee_brewing.md | "Fine grind", cited |
| 8 | Which planet has the strongest winds? | solar_system.md | "Neptune, up to 2,100 km/h", cited |
| **T1** | **What is the capital of Australia?** | — | **Refuse — not in any document** |
| **T2** | **What is Acme Corp's parental leave policy?** | — | **Refuse — the policy doc covers remote work only, not parental leave** |
| **T3** | **How much caffeine is in a cup of coffee?** | — | **Refuse — the brewing guide never states caffeine content** |
| **T4** | **What programming language did Guido van Rossum create before Python?** | — | **Refuse — not stated (it was ABC, but the doc doesn't say so)** |

T2 and T4 are the sharpest tests: the *topic* is present (there IS a policy
doc; there IS a Python-history doc) so a weak system is tempted to fill the
gap from general knowledge. A well-grounded system must still refuse, because
that specific fact is not in the retrieved text.

## Quick one-liner (PowerShell)

```powershell
$body = @{ question = "Who created Python and in what year?"; include_chunks = $true } | ConvertTo-Json
Invoke-RestMethod -Uri http://localhost:8000/ask -Method Post -ContentType 'application/json' -Body $body | ConvertTo-Json -Depth 5
```
