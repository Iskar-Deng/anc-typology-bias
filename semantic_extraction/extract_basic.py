# semantic_extraction/extract_basic.py

"""
Extract controlled predicate-argument records.

Usage:
python -m semantic_extraction.extract_basic \
  --input INPUT_TXT \
  --output EXTRACT_JSONL \
  --stats-output
"""

from argparse import ArgumentParser, Namespace
from collections import Counter
from pathlib import Path
from typing import Any

from spacy.language import Language
from spacy.tokens import Doc, Span, Token
from tqdm import tqdm
import json
import spacy


JsonDict = dict[str, Any]


def first_child_by_dep(token: Token, deps: set[str]) -> Token | None:
    for child in token.children:
        if child.dep_ in deps:
            return child
    return None


def children_by_dep(token: Token, deps: set[str]) -> list[Token]:
    return [child for child in token.children if child.dep_ in deps]


def has_child_dep(token: Token, deps: set[str]) -> bool:
    return any(child.dep_ in deps for child in token.children)


def subtree_text(token: Token | None) -> str | None:
    if token is None:
        return None
    left = min(t.i for t in token.subtree)
    right = max(t.i for t in token.subtree)
    return token.doc[left:right + 1].text


def extract_head_lemma(token: Token | None) -> str | None:
    if token is None:
        return None
    return token.lemma_


def token_record(token: Token | None) -> JsonDict | None:
    if token is None:
        return None
    return {
        "lemma": token.lemma_,
        "head_text": token.text,
        "text": subtree_text(token),
        "token_index": token.i,
        "sent_token_index": token.i - token.sent.start,
        "dep": token.dep_,
    }


def extract_poss_modifiers(noun: Token) -> list[JsonDict]:
    return [
        {
            "relation": "poss",
            "text": subtree_text(child),
            "head_text": child.text,
            "head_lemma": child.lemma_,
            "dep": child.dep_,
        }
        for child in noun.children
        if child.dep_ == "poss"
    ]


def extract_pp_modifiers(noun: Token, prep_lemma: str) -> list[JsonDict]:
    results = []

    for child in noun.children:
        if child.dep_ != "prep" or child.lemma_.lower() != prep_lemma:
            continue

        pobjs = [grandchild for grandchild in child.children if grandchild.dep_ == "pobj"]
        if not pobjs:
            results.append(
                {
                    "relation": f"{prep_lemma}_pp",
                    "text": subtree_text(child),
                    "prep_text": child.text,
                    "prep_lemma": child.lemma_,
                    "object_text": None,
                    "object_head_text": None,
                    "object_head_lemma": None,
                }
            )
            continue

        for pobj in pobjs:
            results.append(
                {
                    "relation": f"{prep_lemma}_pp",
                    "text": subtree_text(child),
                    "prep_text": child.text,
                    "prep_lemma": child.lemma_,
                    "object_text": subtree_text(pobj),
                    "object_head_text": pobj.text,
                    "object_head_lemma": pobj.lemma_,
                }
            )

    return results


def extract_nominal_modifiers(sent: Span) -> list[JsonDict]:
    records = []

    for token in sent:
        if token.pos_ not in {"NOUN", "PROPN"}:
            continue

        records.append(
            {
                "noun_text": token.text,
                "noun_lemma": token.lemma_,
                "token_index": token.i,
                "sent_token_index": token.i - sent.start,
                "noun_dep": token.dep_,
                "noun_head": token.head.text,
                "modifiers": {
                    "poss": extract_poss_modifiers(token),
                    "of": extract_pp_modifiers(token, "of"),
                    "by": extract_pp_modifiers(token, "by"),
                },
            }
        )

    return records


def keep_record(
    id: int,
    sentence: str,
    construction: str,
    predicate: str,
    arguments: JsonDict,
    argument_info: JsonDict | None = None,
    object_info: JsonDict | None = None,
    complement: JsonDict | None = None,
    nominal_modifiers: list[JsonDict] | None = None,
) -> JsonDict:
    return {
        "id": id,
        "sentence": sentence,
        "status": "keep",
        "construction": construction,
        "predicate": predicate,
        "arguments": arguments,
        "argument_info": argument_info,
        "object_info": object_info,
        "complement": complement,
        "nominal_modifiers": nominal_modifiers if nominal_modifiers is not None else [],
    }


def extract_particle(pred: Token) -> JsonDict | None:
    particle = first_child_by_dep(pred, {"prt"})
    if particle is None:
        return None
    return {
        "particle_text": particle.text,
        "particle_lemma": particle.lemma_,
        "particle_dep": particle.dep_,
    }


def collect_object_candidates(pred: Token) -> list[JsonDict]:
    candidates = []

    direct_obj = first_child_by_dep(pred, {"dobj", "obj"})
    if direct_obj is not None:
        candidates.append(
            {
                "object_type": "direct_obj",
                "token": direct_obj,
                "adposition": None,
                "adposition_dep": None,
            }
        )

    for child in pred.children:
        if child.dep_ == "dative" and child.pos_ != "ADP":
            candidates.append(
                {
                    "object_type": "indirect_obj",
                    "token": child,
                    "adposition": None,
                    "adposition_dep": "dative",
                }
            )
            continue

        if child.dep_ in {"prep", "dative"}:
            pobj = first_child_by_dep(child, {"pobj"})
            if pobj is not None:
                candidates.append(
                    {
                        "object_type": "pp_obj",
                        "token": pobj,
                        "adposition": child.lemma_.lower(),
                        "adposition_dep": child.dep_,
                    }
                )

    return candidates


def build_object_info(pred: Token, obj_info: JsonDict) -> JsonDict:
    obj_token = obj_info["token"]
    return {
        "object_type": obj_info["object_type"],
        "adposition": obj_info["adposition"],
        "adposition_dep": obj_info["adposition_dep"],
        "object_form": extract_head_lemma(obj_token),
        "object_text": subtree_text(obj_token),
        "particle": extract_particle(pred),
    }


def extract_simple_clause(pred: Token | None, forced_subject: Token | None = None) -> JsonDict | None:
    if pred is None:
        return None

    if has_child_dep(pred, {"nsubjpass", "auxpass", "xcomp", "ccomp"}):
        return None

    if pred.pos_ != "VERB":
        return None

    subj = forced_subject if forced_subject is not None else first_child_by_dep(pred, {"nsubj"})
    if subj is None:
        return None

    obj_candidates = collect_object_candidates(pred)

    if len(obj_candidates) == 0:
        particle_info = extract_particle(pred)
        return {
            "construction": "iv",
            "predicate": pred.lemma_,
            "arguments": {"S": extract_head_lemma(subj)},
            "argument_info": {"S": token_record(subj)},
            "object_info": {
                "object_type": None,
                "adposition": None,
                "adposition_dep": None,
                "object_form": None,
                "object_text": None,
                "particle": particle_info,
            } if particle_info is not None else None,
            "complement": None,
        }

    if len(obj_candidates) == 1:
        obj_info = obj_candidates[0]
        obj_token = obj_info["token"]
        return {
            "construction": "tv",
            "predicate": pred.lemma_,
            "arguments": {
                "A": extract_head_lemma(subj),
                "P": extract_head_lemma(obj_token),
            },
            "argument_info": {
                "A": token_record(subj),
                "P": token_record(obj_token),
            },
            "object_info": build_object_info(pred, obj_info),
            "complement": None,
        }

    return None


def extract_cv(
    root: Token | None,
    forced_subject: Token | None = None,
    depth: int = 0,
    max_depth: int = 10,
) -> JsonDict | None:
    if root is None or root.pos_ != "VERB":
        return None

    if depth >= max_depth:
        return None

    if has_child_dep(root, {"nsubjpass", "auxpass"}):
        return None

    matrix_subj = forced_subject if forced_subject is not None else first_child_by_dep(root, {"nsubj"})
    if matrix_subj is None:
        return None

    comps = children_by_dep(root, {"xcomp", "ccomp"})
    if len(comps) != 1:
        return None

    comp = comps[0]
    embedded = extract_clause(
        comp,
        forced_subject=matrix_subj if comp.dep_ == "xcomp" else None,
        depth=depth + 1,
        max_depth=max_depth,
    )
    if embedded is None:
        return None

    particle_info = extract_particle(root)
    return {
        "construction": "cv",
        "predicate": root.lemma_,
        "arguments": {"A": extract_head_lemma(matrix_subj)},
        "argument_info": {"A": token_record(matrix_subj)},
        "object_info": {
            "object_type": None,
            "adposition": None,
            "adposition_dep": None,
            "object_form": None,
            "object_text": None,
            "particle": particle_info,
        } if particle_info is not None else None,
        "complement": {
            "comp_type": comp.dep_,
            "construction": embedded["construction"],
            "predicate": embedded["predicate"],
            "arguments": embedded["arguments"],
            "argument_info": embedded.get("argument_info"),
            "object_info": embedded.get("object_info"),
            "complement": embedded.get("complement"),
        },
    }


def extract_clause(
    pred: Token | None,
    forced_subject: Token | None = None,
    depth: int = 0,
    max_depth: int = 10,
) -> JsonDict | None:
    cv = extract_cv(pred, forced_subject=forced_subject, depth=depth, max_depth=max_depth)
    if cv is not None:
        return cv
    return extract_simple_clause(pred, forced_subject=forced_subject)


def extract_from_sentence(sent: Span, id: int, max_depth: int = 10) -> JsonDict | None:
    sentence = sent.text.strip()
    if not sentence:
        return None

    nominal_modifiers = extract_nominal_modifiers(sent)
    root = next((token for token in sent if token.dep_ == "ROOT"), None)
    if root is None:
        return None

    if has_child_dep(root, {"nsubjpass", "auxpass"}):
        return None

    subj = first_child_by_dep(root, {"nsubj"})
    attr = first_child_by_dep(root, {"attr"})
    acomp = first_child_by_dep(root, {"acomp"})

    if root.lemma_ == "be" and root.pos_ == "AUX":
        if acomp is not None:
            return None

        if subj is not None and attr is not None and attr.pos_ in {"NOUN", "PROPN", "PRON"}:
            return keep_record(
                id,
                sentence,
                "cop_n",
                "be",
                {
                    "A": extract_head_lemma(subj),
                    "PRED": extract_head_lemma(attr),
                },
                argument_info={
                    "A": token_record(subj),
                    "PRED": token_record(attr),
                },
                object_info=None,
                nominal_modifiers=nominal_modifiers,
            )

        return None

    if root.pos_ != "VERB" or len(collect_object_candidates(root)) > 1:
        return None

    result = extract_clause(root, forced_subject=None, depth=0, max_depth=max_depth)
    if result is None:
        return None

    return keep_record(
        id,
        sentence,
        result["construction"],
        result["predicate"],
        result["arguments"],
        argument_info=result.get("argument_info"),
        object_info=result.get("object_info"),
        complement=result.get("complement"),
        nominal_modifiers=nominal_modifiers,
    )


def iter_nonempty_lines(path: Path) -> list[str]:
    with path.open(encoding="utf-8") as infile:
        return [line.strip() for line in infile if line.strip()]


def load_nlp(model_name: str) -> Language:
    return spacy.load(model_name, disable=["ner"])


def get_complement_depth(comp: JsonDict | None) -> int:
    if comp is None:
        return 0
    return 1 + get_complement_depth(comp.get("complement"))


class ExtractionStats:
    def __init__(self) -> None:
        self.total = 0
        self.kept = 0
        self.skipped = 0
        self.construction_counts: Counter[str] = Counter()
        self.object_type_counts: Counter[str] = Counter()
        self.comp_type_counts: Counter[str] = Counter()
        self.comp_depth_counts: Counter[int] = Counter()
        self.nominal_modifier_totals: Counter[str] = Counter()
        self.sentences_with_modifier: Counter[str] = Counter()

    def update(self, rec: JsonDict | None) -> None:
        self.total += 1

        if rec is None:
            self.skipped += 1
            return

        self.kept += 1
        self.construction_counts[rec["construction"]] += 1

        obj = rec.get("object_info")
        if obj and obj.get("object_type"):
            self.object_type_counts[obj["object_type"]] += 1

        comp = rec.get("complement")
        if comp:
            self.comp_type_counts[comp["comp_type"]] += 1
            self.comp_depth_counts[get_complement_depth(comp)] += 1

        has_modifier = False
        for noun_rec in rec.get("nominal_modifiers", []):
            modifiers = noun_rec.get("modifiers", {})
            for key in ("poss", "of", "by"):
                count = len(modifiers.get(key, []))
                self.nominal_modifier_totals[key] += count
                if count:
                    self.sentences_with_modifier[key] += 1
                    has_modifier = True
        if has_modifier:
            self.sentences_with_modifier["any"] += 1

    def to_dict(self) -> JsonDict:
        return {
            "total_sentences": self.total,
            "kept": self.kept,
            "skipped": self.skipped,
            "kept_rate": self.kept / self.total if self.total else 0.0,
            "construction_counts": dict(self.construction_counts),
            "object_type_counts": dict(self.object_type_counts),
            "complement_type_counts": dict(self.comp_type_counts),
            "complement_depth_counts": dict(self.comp_depth_counts),
            "nominal_modifier_totals": dict(self.nominal_modifier_totals),
            "sentences_with_modifier_counts": dict(self.sentences_with_modifier),
        }


def default_stats_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.stem}_stats.json")


def parse_args() -> Namespace:
    parser = ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--stats-output", nargs="?", const=True, default=None)
    parser.add_argument("--model", default="en_core_web_md")
    parser.add_argument("--max-depth", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    stats_path = None
    if args.stats_output is True:
        stats_path = default_stats_path(output_path)
    elif args.stats_output:
        stats_path = Path(args.stats_output)

    nlp = load_nlp(args.model)
    lines = iter_nonempty_lines(input_path)
    stats = ExtractionStats()
    next_id = 1

    with output_path.open("w", encoding="utf-8") as outfile:
        for doc in tqdm(nlp.pipe(lines, batch_size=args.batch_size), total=len(lines), desc="Extracting"):
            assert isinstance(doc, Doc)
            for sent in doc.sents:
                rec = extract_from_sentence(sent, id=next_id, max_depth=args.max_depth)
                stats.update(rec)
                next_id += 1
                if rec is None:
                    continue
                outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if stats_path is not None:
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        with stats_path.open("w", encoding="utf-8") as outfile:
            json.dump(stats.to_dict(), outfile, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
