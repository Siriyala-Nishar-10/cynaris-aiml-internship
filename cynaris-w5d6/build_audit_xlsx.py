"""
W5D6 (Deliverable 1): Build the token-cost audit spreadsheet
--------------------------------------------------------------
Reads the numbers you measured and writes token_cost_audit.xlsx with live
formulas, so changing any assumption recalculates the whole sheet.

Inputs (produced by the other W5D6 scripts):
  token_audit.csv            tokens per prompt size          (token_audit.py)
  compression_results.json   measured fraction of tokens kept (compress_prompt.py)
  cost_log.csv               real output token counts         (cost_tracker.py)
The last two are optional; sensible defaults are used if they are missing, and
the sheet says which values are measured and which are assumptions.

Sheets:
  Assumptions   prices, exchange rate, traffic, compression, cache hit rate, routing share
  Audit         per prompt size: baseline (every query to GPT-4o, full prompt)
                vs optimised (compression + semantic cache + cheap-model routing)

Optimised cost per query =
    (1 - cache_hit_rate) x [ cheap_share x cheap_model_cost + (1 - cheap_share) x strong_model_cost ]
with input tokens reduced by the compression ratio.

Colour code: blue text = input, black = formula, yellow fill = replace with your
own measured value.

Run:  python build_audit_xlsx.py
Needs: pip install openpyxl
"""

import csv
import json
from pathlib import Path

from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Font, PatternFill

OUTPUT_FILE = "token_cost_audit.xlsx"

FONT = "Arial"
BLUE = Font(name=FONT, color="0000FF")
BLACK = Font(name=FONT)
BOLD = Font(name=FONT, bold=True)
HEADER_FILL = PatternFill("solid", start_color="D9E1F2")
YELLOW = PatternFill("solid", start_color="FFFF00")

INR = '"₹"#,##0.00'
INR4 = '"₹"#,##0.0000'
INR_BIG = '"₹"#,##0'
PCT = "0.0%"
NUM = "#,##0"


def load_measured() -> dict:
    """Pull measured values from the other scripts' outputs, with defaults."""
    m = {"keep_ratio": 0.4, "keep_measured": False, "out_tokens": 150, "out_measured": False}

    if Path("compression_results.json").exists():
        with open("compression_results.json", encoding="utf-8") as f:
            m["keep_ratio"] = json.load(f)["summary"]["avg_keep_ratio"]
            m["keep_measured"] = True

    if Path("cost_log.csv").exists():
        with open("cost_log.csv", newline="", encoding="utf-8") as f:
            outs = [int(r["completion_tokens"]) for r in csv.DictReader(f) if int(r["completion_tokens"]) > 0]
        if outs:
            m["out_tokens"] = round(sum(outs) / len(outs))
            m["out_measured"] = True
    return m


def write_assumptions(ws, m: dict) -> None:
    ws["A1"] = "Assumptions (LLM cost audit)"
    ws["A1"].font = Font(name=FONT, bold=True, size=14)
    ws["A2"] = "Legend: blue = input you can edit | yellow fill = replace with your own measured value | black = formula"
    ws["A2"].font = Font(name=FONT, italic=True)

    for col, text in zip("ABC", ["Parameter", "Value", "Source / note"]):
        ws[f"{col}3"] = text
        ws[f"{col}3"].font = BOLD
        ws[f"{col}3"].fill = HEADER_FILL

    rows = [
        # row, label, value, number format, note, needs_replacing
        (4, "USD to INR rate", 88.0, "0.00", "Assumption: update to the current exchange rate", True),
        (5, "Output tokens per answer", m["out_tokens"], NUM,
         "Measured average from cost_log.csv" if m["out_measured"] else "Assumption: run cost_tracker.py to measure", not m["out_measured"]),
        (6, "Queries per day", 10000, NUM, "Scenario from the W5D6 lesson example; change to your expected traffic", True),
        (7, "Days per month", 30, NUM, "Assumption", False),
        (8, "Fraction of input tokens kept after compression", m["keep_ratio"], "0.000",
         "Measured by compress_prompt.py (LLMLingua rate=0.4)" if m["keep_measured"] else "Assumption: LLMLingua rate=0.4; run compress_prompt.py to measure", not m["keep_measured"]),
        (9, "Semantic cache hit rate", 0.30, PCT,
         "Assumption: replace with the hit rate from the semantic cache notebook", True),
        (10, "Share of queries routed to the cheap model", 0.70, PCT,
         "W5D6 lesson: about 70% of queries can be handled by the cheap model", True),
    ]
    for r, label, value, fmt, note, replace in rows:
        ws[f"A{r}"] = label
        ws[f"A{r}"].font = BLACK
        ws[f"B{r}"] = value
        ws[f"B{r}"].font = BLUE
        ws[f"B{r}"].number_format = fmt
        if replace:
            ws[f"B{r}"].fill = YELLOW
        ws[f"C{r}"] = note
        ws[f"C{r}"].font = BLACK

    ws["A12"] = "API prices (USD per 1M tokens)"
    ws["A12"].font = BOLD
    for col, text in zip("ABCD", ["Model", "Input", "Output", "Role in this audit"]):
        ws[f"{col}13"] = text
        ws[f"{col}13"].font = BOLD
        ws[f"{col}13"].fill = HEADER_FILL

    prices = [
        (14, "GPT-4o", 5.00, 15.00, "Strong model (baseline and escalation)"),
        (15, "Claude Sonnet 4", 3.00, 15.00, "Reference only"),
        (16, "Gemini 2.0 Flash", 0.10, 0.40, "Reference only"),
        (17, "GPT-4o mini", 0.15, 0.60, "Cheap model (routing target)"),
    ]
    for r, name, p_in, p_out, role in prices:
        ws[f"A{r}"], ws[f"B{r}"], ws[f"C{r}"], ws[f"D{r}"] = name, p_in, p_out, role
        for c in "ABCD":
            ws[f"{c}{r}"].font = BLUE if c in "BC" else BLACK
        ws[f"B{r}"].number_format = ws[f"C{r}"].number_format = "$0.00"
    ws["A18"] = "Source: W5D6 lesson table. Check current provider pricing before quoting these figures."
    ws["A18"].font = Font(name=FONT, italic=True)

    ws["B8"].comment = Comment("Applied to the whole prompt as a simplification; in practice only the "
                               "retrieved context is compressed.", "audit")
    ws.column_dimensions["A"].width = 48
    ws.column_dimensions["B"].width = 14
    ws.column_dimensions["C"].width = 14
    ws.column_dimensions["D"].width = 42


def write_audit(ws, rows: list[dict]) -> None:
    ws["A1"] = "Token-cost audit: baseline vs optimised (INR)"
    ws["A1"].font = Font(name=FONT, bold=True, size=14)
    ws["A2"] = ("Baseline = every query sent to GPT-4o with the full prompt. Optimised = compression + "
                "semantic cache + cheap-model routing. Monthly columns assume ALL queries use that prompt size.")
    ws["A2"].font = Font(name=FONT, italic=True)

    headers = ["Prompt size", "Chunks", "Input tokens", "Output tokens",
               "Baseline cost / query (USD)", "Baseline cost / query (INR)",
               "Optimised input tokens", "Cheap model cost (INR)", "Strong model cost, compressed (INR)",
               "Optimised cost / query (INR)", "Saving %", "Baseline / month (INR)",
               "Optimised / month (INR)", "Saving / month (INR)"]
    for i, h in enumerate(headers, start=1):
        c = ws.cell(row=4, column=i, value=h)
        c.font = BOLD
        c.fill = HEADER_FILL
        c.alignment = Alignment(wrap_text=True, vertical="center")
    ws.row_dimensions[4].height = 45

    A = "Assumptions!"
    first = 5
    for i, r in enumerate(rows):
        n = first + i
        ws[f"A{n}"] = r["label"]
        ws[f"B{n}"] = int(r["k_chunks"])
        ws[f"C{n}"] = int(r["tokens_tiktoken"])
        for col in "ABC":
            ws[f"{col}{n}"].font = BLUE
        ws[f"D{n}"] = f"={A}$B$5"
        ws[f"E{n}"] = f"=(C{n}*{A}$B$14+D{n}*{A}$C$14)/1000000"
        ws[f"F{n}"] = f"=E{n}*{A}$B$4"
        ws[f"G{n}"] = f"=C{n}*{A}$B$8"
        ws[f"H{n}"] = f"=(G{n}*{A}$B$17+D{n}*{A}$C$17)/1000000*{A}$B$4"
        ws[f"I{n}"] = f"=(G{n}*{A}$B$14+D{n}*{A}$C$14)/1000000*{A}$B$4"
        ws[f"J{n}"] = f"=(1-{A}$B$9)*({A}$B$10*H{n}+(1-{A}$B$10)*I{n})"
        ws[f"K{n}"] = f"=IF(F{n}=0,0,1-J{n}/F{n})"
        ws[f"L{n}"] = f"=F{n}*{A}$B$6*{A}$B$7"
        ws[f"M{n}"] = f"=J{n}*{A}$B$6*{A}$B$7"
        ws[f"N{n}"] = f"=L{n}-M{n}"
        for col in "DEFGHIJKLMN":
            ws[f"{col}{n}"].font = Font(name=FONT, color="008000") if col == "D" else BLACK

    last = first + len(rows) - 1
    avg = last + 1
    ws[f"A{avg}"] = "Average"
    ws[f"A{avg}"].font = BOLD
    for col in "CDEFGHIJK":
        ws[f"{col}{avg}"] = f"=AVERAGE({col}{first}:{col}{last})"
        ws[f"{col}{avg}"].font = BOLD
    ws[f"L{avg}"] = f"=AVERAGE(L{first}:L{last})"
    ws[f"M{avg}"] = f"=AVERAGE(M{first}:M{last})"
    ws[f"N{avg}"] = f"=AVERAGE(N{first}:N{last})"
    for col in "LMN":
        ws[f"{col}{avg}"].font = BOLD

    for n in range(first, avg + 1):
        for col, fmt in {"C": NUM, "D": NUM, "E": "$0.000000", "F": INR4, "G": NUM, "H": INR4,
                         "I": INR4, "J": INR4, "K": PCT, "L": INR_BIG, "M": INR_BIG, "N": INR_BIG}.items():
            ws[f"{col}{n}"].number_format = fmt

    ws[f"A{avg + 2}"] = ("Green = value linked from the Assumptions sheet. Input tokens come from token_audit.py "
                         "(tiktoken, GPT-4o tokenizer).")
    ws[f"A{avg + 2}"].font = Font(name=FONT, italic=True)

    widths = [14, 9, 12, 12, 16, 16, 14, 16, 18, 18, 10, 18, 18, 18]
    for i, w in enumerate(widths, start=1):
        ws.column_dimensions[ws.cell(row=4, column=i).column_letter].width = w
    ws.freeze_panes = "B5"


def main() -> None:
    if not Path("token_audit.csv").exists():
        raise SystemExit("token_audit.csv not found. Run 'python token_audit.py' first.")
    with open("token_audit.csv", newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    measured = load_measured()
    wb = Workbook()
    ws_a = wb.active
    ws_a.title = "Assumptions"
    write_assumptions(ws_a, measured)
    write_audit(wb.create_sheet("Audit"), rows)
    wb.save(OUTPUT_FILE)

    print(f"Saved {OUTPUT_FILE} with {len(rows)} prompt sizes")
    print(f"  compression kept fraction: {measured['keep_ratio']} ({'measured' if measured['keep_measured'] else 'assumed'})")
    print(f"  output tokens per answer:  {measured['out_tokens']} ({'measured' if measured['out_measured'] else 'assumed'})")
    print("VERIFY: open the file in Excel; yellow cells on the Assumptions sheet are placeholders to replace")


if __name__ == "__main__":
    main()
