# grammar_build/patch_anc_wo.py

"""
Patch target grammar ANC word-order rules.

Usage:
python -m grammar_build.patch_anc_wo \
  --grammar GRAMMAR_DIR
"""

from argparse import ArgumentParser, Namespace
from pathlib import Path
import re


TARGET_RULES = [
    "trans-erg-poss-lex-rule",
    "trans-poss-acc-lex-rule",
    "trans-nominal-lex-rule",
]


def find_tdl(grammar_dir: Path) -> Path:
    path = grammar_dir / f"{grammar_dir.name}.tdl"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def replace_rule_block(text: str, rule_name: str) -> tuple[str, int]:
    pattern = re.compile(
        rf"({re.escape(rule_name)}\s*:=.*?\n\n)",
        flags=re.DOTALL,
    )
    count = 0

    def replace(match: re.Match) -> str:
        nonlocal count
        block = match.group(1)
        new_block, n = re.subn(r"HEAD\.ANC-WO\s+\+", "HEAD.ANC-WO -", block)
        count += n
        return new_block

    return pattern.sub(replace, text), count


def patch_anc_head_opt_comp_phrase(text: str) -> tuple[str, int]:
    pattern = re.compile(
        r"(anc-head-opt-comp-phrase\s*:=.*?)"
        r"VAL\s*\[\s*SPR\s*<\s*>\s*,\s*SUBJ\s*<\s*>\s*\]",
        flags=re.DOTALL,
    )
    return pattern.subn(r"\1VAL.SPR < [ ] >", text, count=1)


def patch_tdl(path: Path) -> None:
    text = path.read_text(encoding="utf-8")

    if "ANC-WO" not in text:
        print(f"[skip] no ANC-WO: {path}")
        return

    new_text = text
    anc_wo_count = 0

    for rule in TARGET_RULES:
        new_text, count = replace_rule_block(new_text, rule)
        anc_wo_count += count

    new_text, opt_comp_count = patch_anc_head_opt_comp_phrase(new_text)

    if new_text == text:
        print(f"[skip] no changes: {path}")
        return

    path.with_suffix(path.suffix + ".bak").write_text(text, encoding="utf-8")
    path.write_text(new_text, encoding="utf-8")

    print(
        f"[patched] {path}: "
        f"ANC-WO replacements={anc_wo_count}, "
        f"anc-head-opt-comp-phrase replacements={opt_comp_count}"
    )


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--grammar", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    grammar_dir = Path(args.grammar)

    if not grammar_dir.is_dir():
        raise NotADirectoryError(grammar_dir)

    patch_tdl(find_tdl(grammar_dir))


if __name__ == "__main__":
    main()
