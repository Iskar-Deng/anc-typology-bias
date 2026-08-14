# grammar_build/generate_choices.py

"""
Generate the 96 target Grammar Matrix choices files.

Usage:
python -m grammar_build.generate_choices \
  --output CHOICES_DIR
"""

from argparse import ArgumentParser, Namespace
from itertools import product
from pathlib import Path
from typing import Any
import csv

from utils import (
    ALIGNMENTS,
    ANC_CHOICE_WORD_ORDER_TABLE,
    ANC_IV_ORDER_TABLE,
    ANC_MARK_TABLE,
    ANC_STRATEGIES,
    ANC_TV_ORDER_TABLE,
    CLAUSE_WORD_ORDERS,
    COMPLEMENT_SYSTEMS,
    FINITE_MARK_TABLE,
    NP_WORD_ORDERS,
    VERB_MARK_TABLE,
)


JsonDict = dict[str, Any]

MANIFEST_FIELDS = [
    "id",
    "language",
    "choices_file",
    "clause_wo",
    "np_wo",
    "alignment",
    "alignment_code",
    "comp_system",
    "comp_system_code",
    "strategy",
    "strategy_code",
    "anc_choice_word_order",
    "anc_iv_order",
    "anc_tv_order",
]

FINITE_P_CASE = {
    "nom-acc": "acc",
    "erg-abs": "abs",
}

POSSESSIVE_ORDER = {
    "gn": "head-final",
    "ng": "head-initial",
}

COMPLEMENT_FORM = {
    "balancing": "finite",
    "deranking": "nonfinite",
}

CASE_INVENTORY = {
    "nom-acc": {
        "case_lines": [
            "case-marking=nom-acc",
            "nom-acc-nom-case-name=nominative",
            "nom-acc-acc-case-name=accusative",
        ],
        "case_lrts": [
            ("nominative", "case", "nom", "no"),
            ("accusative", "case", "acc", "ca"),
        ],
    },
    "erg-abs": {
        "case_lines": [
            "case-marking=erg-abs",
            "erg-abs-erg-case-name=ergative",
            "erg-abs-abs-case-name=absolutive",
        ],
        "case_lrts": [
            ("ergative", "case", "erg", "ca"),
            ("absolutive", "case", "abs", "no"),
        ],
    },
}


def code_for(value: str, table: dict[str, str]) -> str:
    for code, table_value in table.items():
        if table_value == value:
            return code
    raise ValueError(value)


def iter_parameter_grid():
    return product(
        CLAUSE_WORD_ORDERS,
        NP_WORD_ORDERS,
        ALIGNMENTS.values(),
        COMPLEMENT_SYSTEMS.values(),
        ANC_STRATEGIES.values(),
    )


def grid_index(
    clause_wo: str,
    np_wo: str,
    alignment: str,
    comp_system: str,
    strategy: str,
) -> int:
    target = (clause_wo, np_wo, alignment, comp_system, strategy)
    for index, combo in enumerate(iter_parameter_grid()):
        if combo == target:
            return index
    raise ValueError(f"Unknown language parameter combination: {target}")


def make_language_id(
    clause_wo: str,
    np_wo: str,
    alignment: str,
    comp_system: str,
    strategy: str,
) -> str:
    index = grid_index(clause_wo, np_wo, alignment, comp_system, strategy)
    alignment_code = code_for(alignment, ALIGNMENTS)
    comp_system_code = code_for(comp_system, COMPLEMENT_SYSTEMS)
    strategy_code = code_for(strategy, ANC_STRATEGIES)

    return "_".join(
        [
            f"{index:02d}",
            clause_wo,
            np_wo,
            alignment_code,
            comp_system_code,
            strategy_code,
        ]
    )


def language_config(
    clause_wo: str,
    np_wo: str,
    alignment: str,
    comp_system: str,
    strategy: str,
) -> JsonDict:
    language = make_language_id(clause_wo, np_wo, alignment, comp_system, strategy)
    numeric_id, _, _, alignment_code, comp_system_code, strategy_code = language.split("_")
    key = (clause_wo, np_wo)

    return {
        "id": numeric_id,
        "language": language,
        "clause_wo": clause_wo,
        "np_wo": np_wo,
        "alignment": alignment,
        "alignment_code": alignment_code,
        "comp_system": comp_system,
        "comp_system_code": comp_system_code,
        "strategy": strategy,
        "strategy_code": strategy_code,
        "gen_mark": "ge",
        "anc_choice_word_order": ANC_CHOICE_WORD_ORDER_TABLE[key][strategy],
        "anc_iv_order": ANC_IV_ORDER_TABLE[key][strategy],
        "anc_tv_order": ANC_TV_ORDER_TABLE[key][strategy],
        **FINITE_MARK_TABLE[alignment],
        **ANC_MARK_TABLE[(strategy, alignment)],
        **VERB_MARK_TABLE[comp_system],
    }


def iter_language_configs():
    for combo in iter_parameter_grid():
        yield language_config(*combo)


def case_inventory(alignment: str) -> JsonDict:
    return CASE_INVENTORY[alignment]


def poss_order(np_wo: str) -> str:
    return POSSESSIVE_ORDER[np_wo]


def comp_section(comp_system: str) -> str:
    comp_form = COMPLEMENT_FORM[comp_system]
    return "\n".join(
        [
            "section=clausal-comp",
            "  comps1_clause-pos-same=on",
            "    comps1_feat1_name=form",
            f"    comps1_feat1_value={comp_form}",
        ]
    )


def verb_form_lrts(comp_system: str) -> list[str]:
    lines = [
        "  verb-pc1_name=finiteness",
        "  verb-pc1_obligatory=on",
        "  verb-pc1_order=suffix",
        "  verb-pc1_inputs=verb",
        "    verb-pc1_lrt1_name=finite",
        "      verb-pc1_lrt1_feat1_name=form",
        "      verb-pc1_lrt1_feat1_value=finite",
        "      verb-pc1_lrt1_feat1_head=verb",
        "      verb-pc1_lrt1_lri1_inflecting=yes",
        "      verb-pc1_lrt1_lri1_orth=s",
    ]

    if comp_system == "deranking":
        lines += [
            "    verb-pc1_lrt2_name=nonfinite",
            "      verb-pc1_lrt2_feat1_name=form",
            "      verb-pc1_lrt2_feat1_value=nonfinite",
            "      verb-pc1_lrt2_feat1_head=verb",
            "      verb-pc1_lrt2_lri1_inflecting=yes",
            "      verb-pc1_lrt2_lri1_orth=ing",
        ]

    return lines


def nominalclause_section(config: JsonDict) -> str:
    strategy = config["strategy"]

    if strategy == "sent":
        return "\n".join(
            [
                "section=nominalclause",
                "  ns1_name=sent",
                "  ns1_nmz_type=sentential",
                "  ns1_nmzRel=yes",
                "  ns1_intrans=on",
                "  ns1_trans=on",
            ]
        )

    gm_type = {
        "poss-acc": "poss-acc",
        "erg-poss": "erg-poss",
        "nomn": "nominal",
    }[strategy]

    return "\n".join(
        [
            "section=nominalclause",
            f"  ns1_name={strategy}",
            f"  ns1_nmz_type={gm_type}",
            "  ns1_det=imp",
            "  ns1_intrans=on",
            "  ns1_trans=on",
            "same-word-order=no",
            f"nmz-clause-word-order={config['anc_choice_word_order']}",
            "  nmz_poss_strat1_name=poss-strat1",
            "non_sent_sem=verb-only",
        ]
    )


def nominalization_lrt(strategy: str, alignment: str) -> list[str]:
    lines = [
        "  verb-pc2_name=nominalization",
        "  verb-pc2_order=suffix",
        "  verb-pc2_inputs=verb1, verb2",
        f"    verb-pc2_lrt1_name={strategy}-nmz",
        "      verb-pc2_lrt1_feat1_name=nominalization",
        f"      verb-pc2_lrt1_feat1_value={strategy}",
        "      verb-pc2_lrt1_feat1_head=verb",
    ]

    if strategy == "poss-acc":
        lines += [
            "      verb-pc2_lrt1_feat3_name=case",
            f"      verb-pc2_lrt1_feat3_value={FINITE_P_CASE[alignment]}",
            "      verb-pc2_lrt1_feat3_head=obj",
        ]
    elif strategy in {"erg-poss", "nomn"}:
        lines += [
            "      verb-pc2_lrt1_feat3_name=case",
            "      verb-pc2_lrt1_feat3_value=oblique",
            "      verb-pc2_lrt1_feat3_head=obj",
        ]

    lines += [
        "      verb-pc2_lrt1_lri1_inflecting=yes",
        "      verb-pc2_lrt1_lri1_orth=ing",
    ]

    return lines


def case_lrt_block(alignment: str, strategy: str) -> list[str]:
    info = case_inventory(alignment)
    lines = [
        "  noun-pc1_name=case",
        "  noun-pc1_obligatory=on",
        "  noun-pc1_order=suffix",
        "  noun-pc1_inputs=noun, verb-pc2",
    ]

    index = 1
    for name, feat_name, feat_value, orth in info["case_lrts"]:
        lines += [
            f"    noun-pc1_lrt{index}_name={name}",
            f"      noun-pc1_lrt{index}_feat1_name={feat_name}",
            f"      noun-pc1_lrt{index}_feat1_value={feat_value}",
            f"      noun-pc1_lrt{index}_feat1_head=itself",
        ]

        if orth == "no":
            lines.append(f"      noun-pc1_lrt{index}_lri1_inflecting=no")
        else:
            lines += [
                f"      noun-pc1_lrt{index}_lri1_inflecting=yes",
                f"      noun-pc1_lrt{index}_lri1_orth={orth}",
            ]

        index += 1

    lines += [
        f"    noun-pc1_lrt{index}_name=genitive",
        f"      noun-pc1_lrt{index}_feat1_name=poss-strat1",
        f"      noun-pc1_lrt{index}_feat1_value=possessor",
        f"      noun-pc1_lrt{index}_feat1_head=itself",
        f"      noun-pc1_lrt{index}_lri1_inflecting=yes",
        f"      noun-pc1_lrt{index}_lri1_orth=ge",
    ]
    index += 1

    if strategy in {"erg-poss", "nomn"}:
        lines += [
            f"    noun-pc1_lrt{index}_name=oblique",
            f"      noun-pc1_lrt{index}_feat1_name=case",
            f"      noun-pc1_lrt{index}_feat1_value=oblique",
            f"      noun-pc1_lrt{index}_feat1_head=itself",
            f"      noun-pc1_lrt{index}_lri1_inflecting=yes",
            f"      noun-pc1_lrt{index}_lri1_orth=ob",
        ]

    return lines


def lexicon_section(alignment: str) -> str:
    if alignment == "nom-acc":
        iv_valence = "nom"
        tv_valence = "nom-acc"
        cv_valence = "nom,comps1"
    elif alignment == "erg-abs":
        iv_valence = "abs"
        tv_valence = "erg-abs"
        cv_valence = "erg,comps1"
    else:
        raise ValueError(alignment)

    return f"""section=lexicon
  noun1_name=common_noun
  noun1_det=imp
    noun1_stem1_orth=n1
    noun1_stem1_pred=_n1_n_rel
    noun1_stem2_orth=n2
    noun1_stem2_pred=_n2_n_rel
    noun1_stem3_orth=n3
    noun1_stem3_pred=_n3_n_rel
  verb1_name=intran_verb
  verb1_valence={iv_valence}
    verb1_stem1_orth=iv1
    verb1_stem1_pred=_iv1_v_rel
    verb1_stem2_orth=iv2
    verb1_stem2_pred=_iv2_v_rel
  verb2_name=tran_verb
  verb2_valence={tv_valence}
    verb2_stem1_orth=tv1
    verb2_stem1_pred=_tv1_v_rel
    verb2_stem2_orth=tv2
    verb2_stem2_pred=_tv2_v_rel
  verb3_name=clausal_verb
  verb3_valence={cv_valence}
    verb3_stem1_orth=cv1
    verb3_stem1_pred=_cv1_v_rel
    verb3_stem2_orth=cv2
    verb3_stem2_pred=_cv2_v_rel"""


def generate_choice_text(config: JsonDict) -> str:
    case_info = case_inventory(config["alignment"])
    case_lines = case_info["case_lines"][:]
    if config["strategy"] in {"erg-poss", "nomn"}:
        case_lines.append("  case1_name=oblique")

    morphology_lines = []
    morphology_lines += case_lrt_block(config["alignment"], config["strategy"])
    morphology_lines += verb_form_lrts(config["comp_system"])
    morphology_lines += nominalization_lrt(config["strategy"], config["alignment"])

    return f"""version=35

section=general
language={config['language']}
punctuation-chars=keep-all
archive=yes

section=word-order
word-order={config['clause_wo']}
has-dets=no
has-aux=no
subord-word-order=same

section=number

section=person
person=none

section=gender

section=case
{chr(10).join(case_lines)}

section=adnom-poss
  poss-strat1_order={poss_order(config['np_wo'])}
  poss-strat1_mod-spec=spec
  poss-strat1_mark-loc=possessor
  poss-strat1_possessor-type=affix
  poss-strat1_possessor-affix-agr=non-agree

section=direct-inverse

section=tense-aspect-mood

section=evidentials

section=other-features
form-fin-nf=on

section=sentential-negation

section=coordination

section=matrix-yes-no

section=wh-q

section=info-str

section=arg-opt

{nominalclause_section(config)}

section=lvc

{comp_section(config['comp_system'])}

section=clausalmods

{lexicon_section(config['alignment'])}

section=morphology
{chr(10).join(morphology_lines)}

section=toolbox-import

section=test-sentences

section=gen-options

section=ToolboxLexicon
"""


def manifest_row(config: JsonDict) -> dict[str, str]:
    row = {field: str(config[field]) for field in MANIFEST_FIELDS if field != "choices_file"}
    row["choices_file"] = f"{config['language']}.choice"
    return row


def write_manifest(rows: list[dict[str, str]], path: Path) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", default="manifest.tsv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for config in iter_language_configs():
        choice_path = output_dir / f"{config['language']}.choice"
        choice_path.write_text(generate_choice_text(config), encoding="utf-8")
        rows.append(manifest_row(config))

    manifest_path = output_dir / args.manifest
    write_manifest(rows, manifest_path)

    print(f"Wrote {len(rows)} choices files to {output_dir}")
    print(f"Wrote manifest: {manifest_path}")


if __name__ == "__main__":
    main()
