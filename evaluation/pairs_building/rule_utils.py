# evaluation/pairs_building/rule_utils.py

"""
Provide shared helpers for building evaluation minimal pairs.

Usage:
Imported by phenomenon rule modules and evaluation.pairs_building.apply_perturbation.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Any
import json

from utils import ANC_MARK_TABLE, FINITE_MARK_TABLE

JsonDict = dict[str, Any]

def surface_noun_markers() -> tuple[str, ...]:
    markers = set()

    for table in (FINITE_MARK_TABLE, ANC_MARK_TABLE):
        for config in table.values():
            markers.update(mark for mark in config.values() if mark)

    return tuple(sorted(markers, key=lambda mark: (-len(mark), mark)))

NOUN_MARKERS = surface_noun_markers()

@dataclass(frozen=True)
class NpSpan:
    tokens: list[str]
    start: int
    head_offset: int

    @property
    def head_index(self) -> int:
        return self.start + self.head_offset

    @property
    def genitive_index(self) -> int | None:
        if len(self.tokens) != 2:
            return None
        return self.start + (1 - self.head_offset)

    @property
    def possessor_index(self) -> int | None:
        return self.genitive_index

    @property
    def text(self) -> str:
        return " ".join(self.tokens)

@dataclass(frozen=True)
class TransitiveTemplateMatch:
    template_name: str
    a: NpSpan
    p: NpSpan
    verb_token: str
    verb_index: int

@dataclass(frozen=True)
class ClausalTemplateShape:
    base_name: str
    embedded_construction: str
    matrix_a_len: int
    embedded_s_len: int | None = None
    embedded_a_len: int | None = None
    embedded_p_len: int | None = None

    @property
    def embedded_len(self) -> int:
        if self.embedded_construction == "iv":
            if self.embedded_s_len is None:
                raise ValueError("IV clausal template is missing embedded_s_len")
            return self.embedded_s_len + 1

        if self.embedded_a_len is None or self.embedded_p_len is None:
            raise ValueError("TV clausal template is missing embedded argument lengths")
        return self.embedded_a_len + self.embedded_p_len + 1

    @property
    def name(self) -> str:
        if self.embedded_construction == "iv":
            return f"{self.base_name}_ma{self.matrix_a_len}_s{self.embedded_s_len}"
        return (
            f"{self.base_name}_ma{self.matrix_a_len}_"
            f"a{self.embedded_a_len}_p{self.embedded_p_len}"
        )

@dataclass(frozen=True)
class ClausalMatch:
    template_name: str
    matrix_a: NpSpan
    matrix_verb_index: int
    matrix_verb_token: str
    embedded_construction: str
    embedded_verb_index: int
    embedded_verb_token: str
    embedded_s: NpSpan | None = None
    embedded_a: NpSpan | None = None
    embedded_p: NpSpan | None = None

@dataclass(frozen=True)
class AncTemplate:
    name: str
    profile: str
    source_id_start: int
    source_id_end: int
    source_construction: str
    overt_arguments: list[str]
    external_role: str = ""

@dataclass(frozen=True)
class VerbToken:
    index: int
    token: str
    stem: str
    marker: str

def load_json_templates(path: Path) -> list[JsonDict]:
    with path.open(encoding="utf-8") as infile:
        return json.load(infile)

def finite_s_verb(token: str, allow_noun_markers: bool = True) -> bool:
    lower = token.lower()
    if not lower.endswith("s"):
        return False
    if allow_noun_markers:
        return True
    return not any(lower.endswith(marker) for marker in NOUN_MARKERS)

def token_has_suffix(token: str, suffix: str) -> bool:
    return token.lower().endswith(suffix)

def replace_token_suffix(token: str, old: str, new: str) -> str:
    if not token_has_suffix(token, old):
        raise ValueError(f"Token {token!r} does not end in {old!r}")
    return token[: -len(old)] + new

def finite_content_verb(token: str) -> bool:
    return finite_s_verb(token, allow_noun_markers=False)

def nonfinite_content_verb(token: str) -> bool:
    return token.endswith("ing") and not any(
        token.lower().endswith(marker) for marker in NOUN_MARKERS
    )

def complement_verb_form(comp_system: str) -> str:
    if comp_system == "balancing":
        return "finite_s"
    if comp_system == "deranking":
        return "nonfinite_ing"
    raise ValueError(f"Unsupported comp_system: {comp_system}")

def token_matches_verb_form(token: str, form: str) -> bool:
    if form == "finite_s":
        return finite_content_verb(token)
    if form == "nonfinite_ing":
        return nonfinite_content_verb(token)
    raise ValueError(f"Unsupported verb form: {form}")

def strip_nonempty_suffix(token: str, suffix: str) -> str:
    if not token.endswith(suffix):
        raise ValueError(f"Expected token ending in {suffix!r}, got: {token}")
    stem = token[: -len(suffix)]
    if not stem:
        raise ValueError(f"Could not strip suffix {suffix!r} from token: {token}")
    return stem

def head_matches_finite_marker(token: str, expected_marker: str | None) -> bool:
    marker = expected_marker or ""
    if marker == "":
        return True
    if marker in NOUN_MARKERS:
        return token.endswith(marker)
    raise ValueError(f"Unsupported finite marker: {marker!r}")

def parse_finite_np(
    tokens: list[str],
    start: int,
    np_wo: str,
    expected_head_marker: str | None,
) -> NpSpan | None:
    if len(tokens) == 1:
        if not head_matches_finite_marker(tokens[0], expected_head_marker):
            return None
        return NpSpan(tokens=tokens, start=start, head_offset=0)

    if len(tokens) != 2:
        return None

    if np_wo == "gn":
        if not tokens[0].endswith("ge"):
            return None
        if not head_matches_finite_marker(tokens[1], expected_head_marker):
            return None
        return NpSpan(tokens=tokens, start=start, head_offset=1)

    if np_wo == "ng":
        if not tokens[1].endswith("ge"):
            return None
        if not head_matches_finite_marker(tokens[0], expected_head_marker):
            return None
        return NpSpan(tokens=tokens, start=start, head_offset=0)

    raise ValueError(f"Unsupported np_wo: {np_wo}")

def intransitive_template_matches(
    template: JsonDict,
    tokens: list[str],
    clause_wo: str,
    np_wo: str,
) -> bool:
    if clause_wo not in template["clause_wo"]:
        return False
    if np_wo not in template["np_wo"]:
        return False
    if len(tokens) != template["token_count"]:
        return False
    if not finite_content_verb(tokens[template["verb_index"]]):
        return False

    subject_start = template.get("subject_start", 0)
    for requirement in template.get("required_suffixes", []):
        index = subject_start + requirement["relative_index"]
        if not tokens[index].endswith(requirement["suffix"]):
            return False

    for index in template.get("gen_indices", []):
        if not token_has_suffix(tokens[index], "ge"):
            return False

    return True

def find_intransitive_template_match(
    templates: list[JsonDict],
    tokens: list[str],
    clause_wo: str,
    np_wo: str,
) -> JsonDict | None:
    matches = [
        template
        for template in templates
        if intransitive_template_matches(template, tokens, clause_wo, np_wo)
    ]

    if len(matches) == 1:
        return matches[0]
    return None

def transitive_template_spans(
    template: JsonDict,
    clause_wo: str,
) -> tuple[int, int, int, int, int]:
    a_len = template["a_len"]
    p_len = template["p_len"]

    if clause_wo == "sov":
        return 0, a_len, a_len, p_len, a_len + p_len
    if clause_wo == "svo":
        return 0, a_len, a_len + 1, p_len, a_len
    if clause_wo == "vos":
        return 1 + p_len, a_len, 1, p_len, 0
    raise ValueError(f"Unsupported clause_wo: {clause_wo}")

def match_transitive_template(
    template: JsonDict,
    tokens: list[str],
    clause_wo: str,
    np_wo: str,
    a_marker: str | None,
    p_marker: str | None,
) -> TransitiveTemplateMatch | None:
    if clause_wo not in template["clause_wo"]:
        return None

    expected_len = template["a_len"] + template["p_len"] + 1
    if len(tokens) != expected_len:
        return None

    a_start, a_len, p_start, p_len, verb_index = transitive_template_spans(
        template,
        clause_wo,
    )
    verb_token = tokens[verb_index]
    if not finite_s_verb(verb_token, allow_noun_markers=False):
        return None

    a = parse_finite_np(tokens[a_start : a_start + a_len], a_start, np_wo, a_marker)
    if a is None:
        return None

    p = parse_finite_np(tokens[p_start : p_start + p_len], p_start, np_wo, p_marker)
    if p is None:
        return None

    return TransitiveTemplateMatch(
        template_name=template["name"],
        a=a,
        p=p,
        verb_token=verb_token,
        verb_index=verb_index,
    )

def find_transitive_template_match(
    templates: list[JsonDict],
    tokens: list[str],
    clause_wo: str,
    np_wo: str,
    a_marker: str | None,
    p_marker: str | None,
    expected_shape: tuple[int, int] | None = None,
    target_role: str | None = None,
) -> TransitiveTemplateMatch | None:
    matches = []

    for template in templates:
        if expected_shape is not None:
            expected_a_len, expected_p_len = expected_shape
            if template["a_len"] != expected_a_len or template["p_len"] != expected_p_len:
                continue
        if target_role == "A" and template["a_len"] != 2:
            continue
        if target_role == "P" and template["p_len"] != 2:
            continue

        match = match_transitive_template(
            template=template,
            tokens=tokens,
            clause_wo=clause_wo,
            np_wo=np_wo,
            a_marker=a_marker,
            p_marker=p_marker,
        )
        if match is not None:
            matches.append(match)

    if len(matches) == 1:
        return matches[0]
    return None

def transitive_shape_from_pseudo(row: JsonDict | None) -> tuple[int, int] | None:
    if row is None:
        return None

    pseudo = row.get("pseudo_english")
    if not isinstance(pseudo, str):
        return None

    tokens = pseudo.strip().split()
    if len(tokens) not in (3, 4, 5):
        return None

    a_len = 2 if len(tokens) >= 2 and tokens[0].endswith("ge") else 1
    p_len = len(tokens) - a_len - 1
    if p_len not in (1, 2):
        return None

    return a_len, p_len

def stable_row_index(row: JsonDict | None, fallback_index: int) -> int:
    if row is not None:
        for key in ("id", "source_id", "pseudo_index", "pair_index"):
            value = row.get(key)
            try:
                return int(value)
            except (TypeError, ValueError):
                pass
    return fallback_index

def expand_clausal_templates(
    templates: list[JsonDict],
    allowed_constructions: set[str] | None = None,
) -> list[ClausalTemplateShape]:
    shapes = []

    for template in templates:
        base = template["name"]
        construction = template["embedded_construction"]
        if allowed_constructions is not None and construction not in allowed_constructions:
            raise ValueError(f"Unsupported embedded construction: {construction}")

        for matrix_a_len in template["matrix_a_lens"]:
            if construction == "iv":
                for s_len in template["embedded_s_lens"]:
                    shapes.append(
                        ClausalTemplateShape(
                            base_name=base,
                            embedded_construction="iv",
                            matrix_a_len=matrix_a_len,
                            embedded_s_len=s_len,
                        )
                    )
            elif construction == "tv":
                for a_len in template["embedded_a_lens"]:
                    for p_len in template["embedded_p_lens"]:
                        shapes.append(
                            ClausalTemplateShape(
                                base_name=base,
                                embedded_construction="tv",
                                matrix_a_len=matrix_a_len,
                                embedded_a_len=a_len,
                                embedded_p_len=p_len,
                            )
                        )
            else:
                raise ValueError(f"Unsupported embedded construction: {construction}")

    return shapes

def clausal_matrix_positions(
    shape: ClausalTemplateShape,
    clause_wo: str,
) -> tuple[int, int, int]:
    embedded_len = shape.embedded_len
    if clause_wo == "sov":
        return 0, shape.matrix_a_len, shape.matrix_a_len + embedded_len
    if clause_wo == "svo":
        return 0, shape.matrix_a_len + 1, shape.matrix_a_len
    if clause_wo == "vos":
        return 1 + embedded_len, 1, 0
    raise ValueError(f"Unsupported clause_wo: {clause_wo}")

def match_clausal_shape(
    shape: ClausalTemplateShape,
    tokens: list[str],
    clause_wo: str,
    np_wo: str,
    comp_form: str,
    matrix_a_marker: str | None,
    embedded_s_marker: str | None,
    embedded_a_marker: str | None,
    embedded_p_marker: str | None,
) -> ClausalMatch | None:
    expected_len = shape.matrix_a_len + shape.embedded_len + 1
    if len(tokens) != expected_len:
        return None

    matrix_a_start, embedded_start, matrix_verb_index = clausal_matrix_positions(
        shape,
        clause_wo,
    )
    matrix_verb = tokens[matrix_verb_index]
    if not finite_content_verb(matrix_verb):
        return None

    matrix_a = parse_finite_np(
        tokens[matrix_a_start : matrix_a_start + shape.matrix_a_len],
        matrix_a_start,
        np_wo,
        matrix_a_marker,
    )
    if matrix_a is None:
        return None

    if shape.embedded_construction == "iv":
        if shape.embedded_s_len is None:
            return None
        if clause_wo in {"sov", "svo"}:
            s_start = embedded_start
            verb_index = embedded_start + shape.embedded_s_len
        elif clause_wo == "vos":
            verb_index = embedded_start
            s_start = embedded_start + 1
        else:
            raise ValueError(f"Unsupported clause_wo: {clause_wo}")

        embedded_verb = tokens[verb_index]
        if not token_matches_verb_form(embedded_verb, comp_form):
            return None

        embedded_s = parse_finite_np(
            tokens[s_start : s_start + shape.embedded_s_len],
            s_start,
            np_wo,
            embedded_s_marker,
        )
        if embedded_s is None:
            return None

        return ClausalMatch(
            template_name=shape.name,
            matrix_a=matrix_a,
            matrix_verb_index=matrix_verb_index,
            matrix_verb_token=matrix_verb,
            embedded_construction="iv",
            embedded_verb_index=verb_index,
            embedded_verb_token=embedded_verb,
            embedded_s=embedded_s,
        )

    if shape.embedded_a_len is None or shape.embedded_p_len is None:
        return None

    if clause_wo == "sov":
        a_start = embedded_start
        p_start = embedded_start + shape.embedded_a_len
        verb_index = embedded_start + shape.embedded_a_len + shape.embedded_p_len
    elif clause_wo == "svo":
        a_start = embedded_start
        verb_index = embedded_start + shape.embedded_a_len
        p_start = embedded_start + shape.embedded_a_len + 1
    elif clause_wo == "vos":
        verb_index = embedded_start
        p_start = embedded_start + 1
        a_start = embedded_start + 1 + shape.embedded_p_len
    else:
        raise ValueError(f"Unsupported clause_wo: {clause_wo}")

    embedded_verb = tokens[verb_index]
    if not token_matches_verb_form(embedded_verb, comp_form):
        return None

    embedded_a = parse_finite_np(
        tokens[a_start : a_start + shape.embedded_a_len],
        a_start,
        np_wo,
        embedded_a_marker,
    )
    if embedded_a is None:
        return None

    embedded_p = parse_finite_np(
        tokens[p_start : p_start + shape.embedded_p_len],
        p_start,
        np_wo,
        embedded_p_marker,
    )
    if embedded_p is None:
        return None

    return ClausalMatch(
        template_name=shape.name,
        matrix_a=matrix_a,
        matrix_verb_index=matrix_verb_index,
        matrix_verb_token=matrix_verb,
        embedded_construction="tv",
        embedded_verb_index=verb_index,
        embedded_verb_token=embedded_verb,
        embedded_a=embedded_a,
        embedded_p=embedded_p,
    )

def clausal_shapes_from_pseudo(
    row: JsonDict | None,
    allowed_constructions: set[str] | None = None,
) -> list[ClausalTemplateShape] | None:
    if row is None:
        return None

    pseudo = row.get("pseudo_english")
    if not isinstance(pseudo, str):
        return None

    pseudo_tokens = pseudo.strip().lower().split()
    if "that" not in pseudo_tokens:
        return None

    that_index = pseudo_tokens.index("that")
    matrix_a_len = that_index - 1
    if matrix_a_len not in (1, 2):
        return None

    embedded = pseudo_tokens[that_index + 1 :]
    if len(embedded) < 2:
        return None

    candidates = []
    allow_iv = allowed_constructions is None or "iv" in allowed_constructions
    allow_tv = allowed_constructions is None or "tv" in allowed_constructions

    if allow_iv and len(embedded) - 1 in (1, 2) and finite_content_verb(embedded[-1]):
        candidates.append(
            ClausalTemplateShape(
                base_name="cv_embedded_iv",
                embedded_construction="iv",
                matrix_a_len=matrix_a_len,
                embedded_s_len=len(embedded) - 1,
            )
        )

    if allow_tv:
        for a_len in (1, 2):
            verb_index = a_len
            if verb_index >= len(embedded):
                continue
            if not finite_content_verb(embedded[verb_index]):
                continue
            p_len = len(embedded) - a_len - 1
            if p_len not in (1, 2):
                continue
            candidates.append(
                ClausalTemplateShape(
                    base_name="cv_embedded_tv",
                    embedded_construction="tv",
                    matrix_a_len=matrix_a_len,
                    embedded_a_len=a_len,
                    embedded_p_len=p_len,
                )
            )

    return candidates or None

def candidate_clausal_shapes(
    row: JsonDict | None,
    shapes: list[ClausalTemplateShape],
    allowed_constructions: set[str] | None = None,
) -> list[ClausalTemplateShape]:
    return clausal_shapes_from_pseudo(row, allowed_constructions) or shapes

def find_clausal_match(
    shapes: list[ClausalTemplateShape],
    tokens: list[str],
    language_config: JsonDict,
    row: JsonDict | None,
    allowed_constructions: set[str] | None = None,
) -> ClausalMatch | None:
    clause_wo = language_config["clause_wo"]
    np_wo = language_config["np_wo"]
    comp_form = complement_verb_form(language_config["comp_system"])

    matches = []
    for shape in candidate_clausal_shapes(row, shapes, allowed_constructions):
        match = match_clausal_shape(
            shape=shape,
            tokens=tokens,
            clause_wo=clause_wo,
            np_wo=np_wo,
            comp_form=comp_form,
            matrix_a_marker=language_config["FIN_A_MARK"],
            embedded_s_marker=language_config["FIN_S_MARK"],
            embedded_a_marker=language_config["FIN_A_MARK"],
            embedded_p_marker=language_config["FIN_P_MARK"],
        )
        if match is not None:
            matches.append(match)

    if len(matches) == 1:
        return matches[0]
    return None

def load_anc_templates(path: Path) -> list[AncTemplate]:
    with path.open(encoding="utf-8") as infile:
        raw_templates = json.load(infile)

    return [
        AncTemplate(
            name=raw["name"],
            profile=raw["profile"],
            source_id_start=int(raw["source_id_start"]),
            source_id_end=int(raw["source_id_end"]),
            source_construction=raw["source_construction"],
            overt_arguments=list(raw["overt_arguments"]),
            external_role=raw.get("external_role", ""),
        )
        for raw in raw_templates
    ]

def template_for_source_id(
    templates: list[AncTemplate],
    source_id: int,
) -> AncTemplate | None:
    for template in templates:
        if template.source_id_start <= source_id <= template.source_id_end:
            return template
    return None

def marker_value(marker: str | None) -> str:
    return marker or "0"

def strip_marker(token: str) -> tuple[str, str]:
    lower = token.lower()
    for marker in NOUN_MARKERS:
        if lower.endswith(marker) and len(token) > len(marker):
            return token[: -len(marker)], token[-len(marker) :]
    return token, ""

def is_anc_verb_token(token: str) -> bool:
    base, _ = strip_marker(token)
    lower = base.lower()
    return lower.endswith("ing") and len(lower) > 3

def marker_for_expected_head(token: str, expected_head: str) -> str | None:
    lower = token.lower()
    expected = expected_head.lower()

    if lower == expected:
        return ""

    for marker in NOUN_MARKERS:
        if lower == expected + marker:
            return marker

    return None

def replace_expected_head_marker(
    token: str,
    expected_head: str,
    good_marker: str,
    bad_marker: str,
) -> str:
    marker = marker_for_expected_head(token, expected_head)
    if marker != good_marker:
        raise ValueError(
            f"Expected marker {good_marker!r} on token {token!r}, got {marker!r}"
        )

    base = token[: len(expected_head)]
    if bad_marker in {"", "0"}:
        return base
    if bad_marker in NOUN_MARKERS:
        return base + bad_marker
    raise ValueError(f"Unsupported bad marker: {bad_marker!r}")

def find_unique_expected_head_index(tokens: list[str], expected_head: str) -> int | None:
    matches = []

    for index, token in enumerate(tokens):
        if is_anc_verb_token(token):
            continue
        if marker_for_expected_head(token, expected_head) is not None:
            matches.append(index)

    if len(matches) == 1:
        return matches[0]
    return None

def find_anc_verb(tokens: list[str]) -> VerbToken | None:
    candidates = []

    for index, token in enumerate(tokens):
        base, marker = strip_marker(token)
        lower = base.lower()
        if not lower.endswith("ing"):
            continue

        stem = base[:-3]
        if not stem:
            continue

        candidates.append(VerbToken(index=index, token=token, stem=stem, marker=marker))

    if len(candidates) == 1:
        return candidates[0]
    return None

def finite_from_anc_verb(verb: VerbToken, keep_marker: bool = False) -> str:
    finite = verb.stem + "s"
    if verb.token[0].isupper():
        finite = finite[:1].upper() + finite[1:]
    if keep_marker:
        return finite + verb.marker
    return finite

# ANC rule contexts

@dataclass(frozen=True)
class PseudoAncTvArgs:
    a_head: str
    p_head: str

@dataclass(frozen=True)
class AncBaseContext:
    template: AncTemplate
    pseudo_english: str
    tokens: list[str]
    anc_verb: VerbToken

@dataclass(frozen=True)
class AncExternalMarkerContext:
    base: AncBaseContext
    good_marker: str

@dataclass(frozen=True)
class AncIvContext:
    base: AncBaseContext
    s_head: str
    s_index: int
    s_token: str
    actual_marker: str
    good_marker: str

@dataclass(frozen=True)
class AncTvContext:
    base: AncBaseContext
    pseudo_args: PseudoAncTvArgs
    a_index: int
    p_index: int
    a_token: str
    p_token: str
    actual_a_marker: str | None
    actual_p_marker: str | None

def phenomenon_key(phenomenon_id: str) -> str:
    return phenomenon_id.replace(".", "_")

def row_stable_index(row: JsonDict | None, fallback_index: int) -> int:
    value: Any = fallback_index
    if row is not None:
        value = row.get("id", row.get("source_id", fallback_index))

    try:
        return int(value)
    except (TypeError, ValueError):
        return fallback_index

def alternating_noun_marker(row: JsonDict | None, fallback_index: int) -> str:
    return "ca" if row_stable_index(row, fallback_index) % 2 else "ge"

def anc_order_metadata(language_config: JsonDict) -> JsonDict:
    return {
        "anc_wo": language_config["anc_wo"],
        "anc_wo_choice": language_config.get("anc_wo_choice", language_config["anc_wo"]),
        "anc_iv_order": language_config.get("anc_iv_order", ""),
        "anc_tv_order": language_config.get("anc_tv_order", ""),
    }

def anc_template_metadata(
    template: AncTemplate,
    language_config: JsonDict,
    empty_overt_arguments: str = "none",
    include_alignment: bool = True,
    include_external_role: bool = False,
) -> JsonDict:
    metadata = {
        "anc_profile": template.profile,
        "anc_source_construction": template.source_construction,
        "anc_overt_arguments": (
            ",".join(template.overt_arguments)
            if template.overt_arguments
            else empty_overt_arguments
        ),
        "template": template.name,
        **anc_order_metadata(language_config),
        "strategy": language_config["strategy"],
    }
    if include_alignment:
        metadata["alignment"] = language_config["alignment"]
    if include_external_role:
        metadata["anc_external_role"] = template.external_role
    return metadata

def anc_skip_metadata(
    template: AncTemplate,
    source_index: int,
    pseudo_english: Any,
    include_external_role: bool = False,
) -> JsonDict:
    metadata = {
        "source_index": source_index,
        "profile": template.profile,
        "pseudo_english": pseudo_english,
    }
    if include_external_role:
        metadata["external_role"] = template.external_role
    return metadata

def prepare_anc_base_context(
    templates: list[AncTemplate],
    phenomenon_id: str,
    good_sentence: str,
    source_index: int,
    row: JsonDict | None,
    include_external_role: bool = False,
) -> tuple[AncBaseContext | None, JsonDict | None]:
    template = template_for_source_id(templates, source_index)
    if template is None:
        return None, {
            "skip": True,
            "skip_reason": f"source_index_outside_{phenomenon_key(phenomenon_id)}_profile_ranges",
            "good": good_sentence,
            "source_index": source_index,
        }

    pseudo_english = row.get("pseudo_english") if row is not None else None
    if not isinstance(pseudo_english, str) or "nmz" not in pseudo_english:
        return None, {
            "skip": True,
            "skip_reason": "pseudo_english_missing_anc_nmz",
            "good": good_sentence,
            **anc_skip_metadata(
                template,
                source_index,
                pseudo_english,
                include_external_role=include_external_role,
            ),
        }

    tokens = good_sentence.strip().split()
    anc_verb = find_anc_verb(tokens)
    if anc_verb is None:
        return None, {
            "skip": True,
            "skip_reason": "anc_ing_token_match_count_not_one",
            "good": good_sentence,
            "tokens": tokens,
            **anc_skip_metadata(
                template,
                source_index,
                pseudo_english,
                include_external_role=include_external_role,
            ),
        }

    return AncBaseContext(
        template=template,
        pseudo_english=pseudo_english,
        tokens=tokens,
        anc_verb=anc_verb,
    ), None

def replace_marker_on_anc_head(token: str, expected_marker: str, bad_marker: str) -> str:
    base, marker = strip_marker(token)
    if marker != expected_marker:
        raise ValueError(
            f"Expected marker {expected_marker!r} on token {token!r}, got {marker!r}"
        )
    if bad_marker and bad_marker not in NOUN_MARKERS:
        raise ValueError(f"Unsupported bad marker: {bad_marker!r}")
    return base + bad_marker

def prepare_anc_external_marker_context(
    templates: list[AncTemplate],
    phenomenon_id: str,
    good_sentence: str,
    source_index: int,
    row: JsonDict | None,
    good_marker: str,
    marker_role: str,
    marker_mismatch_reason: str | None = None,
) -> tuple[AncExternalMarkerContext | None, JsonDict | None]:
    base, skip = prepare_anc_base_context(
        templates,
        phenomenon_id,
        good_sentence,
        source_index,
        row,
    )
    if skip is not None or base is None:
        return None, skip

    if base.anc_verb.marker != good_marker:
        return None, {
            "skip": True,
            "skip_reason": marker_mismatch_reason
            or f"anc_external_{marker_role.lower()}_marker_does_not_match_expected_value",
            "good": good_sentence,
            "tokens": base.tokens,
            "target_index": base.anc_verb.index,
            "target_token": base.anc_verb.token,
            "actual_marker": marker_value(base.anc_verb.marker),
            "expected_marker": marker_value(good_marker),
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    return AncExternalMarkerContext(base=base, good_marker=good_marker), None

def extract_pseudo_anc_s_head(pseudo_english: str) -> str | None:
    pseudo_tokens = pseudo_english.strip().lower().split()
    for index, token in enumerate(pseudo_tokens):
        base, _ = strip_marker(token)
        if not base.endswith("nmz"):
            continue
        if index > 0:
            s_token = pseudo_tokens[index - 1]
            if s_token.endswith("ge") and len(s_token) > 2:
                return s_token[:-2]
    return None

def extract_pseudo_anc_tv_args(pseudo_english: str) -> PseudoAncTvArgs | None:
    pseudo_tokens = pseudo_english.strip().lower().split()
    for index, token in enumerate(pseudo_tokens):
        base, _ = strip_marker(token)
        if not base.endswith("nmz"):
            continue
        if index <= 0 or index + 1 >= len(pseudo_tokens):
            continue

        a_token = pseudo_tokens[index - 1]
        p_token = pseudo_tokens[index + 1]
        if not a_token.endswith("ge") or len(a_token) <= 2:
            continue
        if not p_token.endswith("ob") or len(p_token) <= 2:
            continue
        return PseudoAncTvArgs(a_head=a_token[:-2], p_head=p_token[:-2])
    return None

def expected_internal_anc_marker(role: str, language_config: JsonDict) -> str:
    strategy = language_config["strategy"]
    role = role.upper()

    if role == "S":
        if strategy == "sent":
            return language_config["FIN_S_MARK"] or ""
        if strategy in {"poss-acc", "erg-poss", "nomn"}:
            return "ge"
    elif role == "A":
        if strategy == "sent":
            return language_config["FIN_A_MARK"] or ""
        if strategy in {"poss-acc", "nomn"}:
            return "ge"
        if strategy == "erg-poss":
            return "ob"
    elif role == "P":
        if strategy == "sent":
            return language_config["FIN_P_MARK"] or ""
        if strategy == "poss-acc":
            return language_config["FIN_P_MARK"] or ""
        if strategy == "erg-poss":
            return "ge"
        if strategy == "nomn":
            return "ob"

    raise ValueError(f"Unsupported ANC {role} marker strategy: {strategy}")

def bad_internal_anc_marker(role: str, language_config: JsonDict, good_marker: str) -> str:
    strategy = language_config["strategy"]
    role = role.upper()

    if role == "S":
        if strategy == "sent":
            return "ge"
        if strategy in {"poss-acc", "erg-poss", "nomn"}:
            return marker_value(language_config["FIN_S_MARK"] or "")
    elif role == "A":
        if strategy == "sent":
            return "ge"
        if strategy in {"poss-acc", "erg-poss", "nomn"}:
            return marker_value(language_config["FIN_A_MARK"] or "")
    elif role == "P":
        if strategy in {"sent", "poss-acc"}:
            return "ge"
        if strategy in {"erg-poss", "nomn"}:
            return marker_value(language_config["FIN_P_MARK"] or "")

    raise ValueError(
        f"Unsupported ANC {role} marker perturbation: {strategy=} {good_marker=}"
    )

def anc_s_is_before_verb(anc_order: str) -> bool:
    order = anc_order.lower()
    if order in {"sv", "sov", "svo"}:
        return True
    if order in {"vs", "vos", "ovs"}:
        return False
    raise ValueError(f"Unsupported ANC intransitive order: {anc_order}")

def find_target_anc_s_index(
    tokens: list[str],
    anc_verb: VerbToken,
    anc_order: str,
    expected_head: str,
) -> int | None:
    if anc_s_is_before_verb(anc_order):
        candidate_indices = range(anc_verb.index - 1, -1, -1)
    else:
        candidate_indices = range(anc_verb.index + 1, len(tokens))

    for index in candidate_indices:
        token = tokens[index]
        if is_anc_verb_token(token):
            continue
        if marker_for_expected_head(token, expected_head) is not None:
            return index
    return None

def prepare_anc_iv_context(
    templates: list[AncTemplate],
    phenomenon_id: str,
    good_sentence: str,
    source_index: int,
    row: JsonDict | None,
    language_config: JsonDict,
) -> tuple[AncIvContext | None, JsonDict | None]:
    base, skip = prepare_anc_base_context(
        templates,
        phenomenon_id,
        good_sentence,
        source_index,
        row,
    )
    if skip is not None or base is None:
        return None, skip

    s_head = extract_pseudo_anc_s_head(base.pseudo_english)
    if s_head is None:
        return None, {
            "skip": True,
            "skip_reason": "pseudo_english_missing_overt_anc_s",
            "good": good_sentence,
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    s_index = find_target_anc_s_index(
        base.tokens,
        base.anc_verb,
        language_config.get("anc_iv_order", language_config["anc_wo"]),
        s_head,
    )
    if s_index is None:
        return None, {
            "skip": True,
            "skip_reason": "anc_s_target_not_adjacent_to_anc_verb",
            "good": good_sentence,
            "tokens": base.tokens,
            "anc_verb_index": base.anc_verb.index,
            **anc_order_metadata(language_config),
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    s_token = base.tokens[s_index]
    actual_marker = marker_for_expected_head(s_token, s_head)
    if actual_marker is None:
        return None, {
            "skip": True,
            "skip_reason": "anc_s_target_does_not_match_pseudo_head",
            "good": good_sentence,
            "tokens": base.tokens,
            "target_index": s_index,
            "target_token": s_token,
            "expected_s_head": s_head,
            "anc_verb_index": base.anc_verb.index,
            "anc_verb_token": base.anc_verb.token,
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    good_marker = expected_internal_anc_marker("S", language_config)
    if actual_marker != good_marker:
        return None, {
            "skip": True,
            "skip_reason": "anc_s_marker_does_not_match_expected_value",
            "good": good_sentence,
            "tokens": base.tokens,
            "target_index": s_index,
            "target_token": s_token,
            "actual_marker": marker_value(actual_marker),
            "expected_marker": marker_value(good_marker),
            "expected_s_head": s_head,
            "anc_verb_index": base.anc_verb.index,
            "anc_verb_token": base.anc_verb.token,
            "strategy": language_config["strategy"],
            "alignment": language_config["alignment"],
            **anc_order_metadata(language_config),
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    return AncIvContext(
        base=base,
        s_head=s_head,
        s_index=s_index,
        s_token=s_token,
        actual_marker=actual_marker,
        good_marker=good_marker,
    ), None

def prepare_anc_tv_context(
    templates: list[AncTemplate],
    phenomenon_id: str,
    good_sentence: str,
    source_index: int,
    row: JsonDict | None,
    a_missing_reason: str,
    p_missing_reason: str,
) -> tuple[AncTvContext | None, JsonDict | None]:
    base, skip = prepare_anc_base_context(
        templates,
        phenomenon_id,
        good_sentence,
        source_index,
        row,
    )
    if skip is not None or base is None:
        return None, skip

    pseudo_args = extract_pseudo_anc_tv_args(base.pseudo_english)
    if pseudo_args is None:
        return None, {
            "skip": True,
            "skip_reason": "pseudo_english_missing_overt_anc_a_or_p",
            "good": good_sentence,
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    a_index = find_unique_expected_head_index(base.tokens, pseudo_args.a_head)
    if a_index is None:
        return None, {
            "skip": True,
            "skip_reason": a_missing_reason,
            "good": good_sentence,
            "tokens": base.tokens,
            "expected_a_head": pseudo_args.a_head,
            "expected_p_head": pseudo_args.p_head,
            "anc_verb_index": base.anc_verb.index,
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    p_index = find_unique_expected_head_index(base.tokens, pseudo_args.p_head)
    if p_index is None:
        return None, {
            "skip": True,
            "skip_reason": p_missing_reason,
            "good": good_sentence,
            "tokens": base.tokens,
            "expected_a_head": pseudo_args.a_head,
            "expected_p_head": pseudo_args.p_head,
            "anc_verb_index": base.anc_verb.index,
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    a_token = base.tokens[a_index]
    p_token = base.tokens[p_index]
    return AncTvContext(
        base=base,
        pseudo_args=pseudo_args,
        a_index=a_index,
        p_index=p_index,
        a_token=a_token,
        p_token=p_token,
        actual_a_marker=marker_for_expected_head(a_token, pseudo_args.a_head),
        actual_p_marker=marker_for_expected_head(p_token, pseudo_args.p_head),
    ), None

def anc_tv_marker_mismatch_skip(
    context: AncTvContext,
    role: str,
    good_marker: str,
    language_config: JsonDict,
    source_index: int,
    good_sentence: str,
    use_target_fields: bool = False,
) -> JsonDict | None:
    role = role.upper()
    if role == "A":
        actual_marker = context.actual_a_marker
        index = context.a_index
        token = context.a_token
    elif role == "P":
        actual_marker = context.actual_p_marker
        index = context.p_index
        token = context.p_token
    else:
        raise ValueError(f"Unsupported ANC TV marker role: {role}")

    if actual_marker == good_marker:
        return None

    index_key = "target_index" if use_target_fields else f"{role.lower()}_index"
    token_key = "target_token" if use_target_fields else f"{role.lower()}_token"
    return {
        "skip": True,
        "skip_reason": f"anc_{role.lower()}_marker_does_not_match_expected_value",
        "good": good_sentence,
        "tokens": context.base.tokens,
        index_key: index,
        token_key: token,
        "actual_marker": marker_value(actual_marker or ""),
        "expected_marker": marker_value(good_marker),
        "expected_a_head": context.pseudo_args.a_head,
        "expected_p_head": context.pseudo_args.p_head,
        "anc_verb_index": context.base.anc_verb.index,
        "anc_verb_token": context.base.anc_verb.token,
        "strategy": language_config["strategy"],
        "alignment": language_config["alignment"],
        **anc_order_metadata(language_config),
        "source_index": source_index,
        "profile": context.base.template.profile,
        "pseudo_english": context.base.pseudo_english,
    }

@dataclass(frozen=True)
class PseudoAncSingleOvertArg:
    head: str

@dataclass(frozen=True)
class AncSingleOvertTvContext:
    base: AncBaseContext
    role: str
    head: str
    index: int
    token: str
    actual_marker: str | None
    good_marker: str
    bad_marker: str

def pseudo_has_bare_anc(pseudo_english: str) -> bool:
    pseudo_tokens = pseudo_english.strip().lower().split()
    nmz_indices = []

    for index, token in enumerate(pseudo_tokens):
        base, _ = strip_marker(token)
        if base.endswith("nmz"):
            nmz_indices.append(index)

    if len(nmz_indices) != 1:
        return False

    index = nmz_indices[0]
    for arg_index in (index - 1, index + 1):
        if arg_index < 0 or arg_index >= len(pseudo_tokens):
            continue
        arg_token = pseudo_tokens[arg_index]
        if arg_token.endswith("ge") or arg_token.endswith("ob"):
            return False

    return True

def extract_pseudo_anc_single_overt_tv_arg(
    pseudo_english: str,
    role: str,
) -> PseudoAncSingleOvertArg | None:
    role = role.upper()
    pseudo_tokens = pseudo_english.strip().lower().split()
    candidates = []

    for index, token in enumerate(pseudo_tokens):
        base, _ = strip_marker(token)
        if not base.endswith("nmz"):
            continue

        if role == "P" and index > 0 and pseudo_tokens[index - 1].endswith("ge"):
            continue

        adjacent_a = []
        adjacent_p = []
        for arg_index in (index - 1, index + 1):
            if arg_index < 0 or arg_index >= len(pseudo_tokens):
                continue
            arg_token = pseudo_tokens[arg_index]
            if arg_token.endswith("ge") and len(arg_token) > 2:
                adjacent_a.append(arg_token[:-2])
            if arg_token.endswith("ob") and len(arg_token) > 2:
                adjacent_p.append(arg_token[:-2])

        if role == "A" and len(adjacent_a) == 1 and not adjacent_p:
            candidates.append(PseudoAncSingleOvertArg(head=adjacent_a[0]))
        elif role == "P" and len(adjacent_p) == 1:
            candidates.append(PseudoAncSingleOvertArg(head=adjacent_p[0]))

    if len(candidates) == 1:
        return candidates[0]
    return None

def prepare_anc_single_overt_tv_context(
    templates: list[AncTemplate],
    phenomenon_id: str,
    good_sentence: str,
    source_index: int,
    row: JsonDict | None,
    language_config: JsonDict,
    role: str,
    pseudo_missing_reason: str,
    head_missing_reason: str,
) -> tuple[AncSingleOvertTvContext | None, JsonDict | None]:
    role = role.upper()
    base, skip = prepare_anc_base_context(
        templates,
        phenomenon_id,
        good_sentence,
        source_index,
        row,
    )
    if skip is not None or base is None:
        return None, skip

    pseudo_arg = extract_pseudo_anc_single_overt_tv_arg(base.pseudo_english, role)
    if pseudo_arg is None:
        return None, {
            "skip": True,
            "skip_reason": pseudo_missing_reason,
            "good": good_sentence,
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    arg_index = find_unique_expected_head_index(base.tokens, pseudo_arg.head)
    if arg_index is None:
        return None, {
            "skip": True,
            "skip_reason": head_missing_reason,
            "good": good_sentence,
            "tokens": base.tokens,
            f"expected_{role.lower()}_head": pseudo_arg.head,
            "anc_verb_index": base.anc_verb.index,
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    arg_token = base.tokens[arg_index]
    actual_marker = marker_for_expected_head(arg_token, pseudo_arg.head)
    good_marker = language_config[f"ANC_{role}_MARK"] or ""
    if actual_marker != good_marker:
        return None, {
            "skip": True,
            "skip_reason": f"anc_{role.lower()}_marker_does_not_match_expected_value",
            "good": good_sentence,
            "tokens": base.tokens,
            f"{role.lower()}_index": arg_index,
            f"{role.lower()}_token": arg_token,
            "actual_marker": marker_value(actual_marker or ""),
            "expected_marker": marker_value(good_marker),
            f"expected_{role.lower()}_head": pseudo_arg.head,
            "anc_verb_index": base.anc_verb.index,
            "anc_verb_token": base.anc_verb.token,
            "strategy": language_config["strategy"],
            "alignment": language_config["alignment"],
            **anc_order_metadata(language_config),
            **anc_skip_metadata(base.template, source_index, base.pseudo_english),
        }

    return AncSingleOvertTvContext(
        base=base,
        role=role,
        head=pseudo_arg.head,
        index=arg_index,
        token=arg_token,
        actual_marker=actual_marker,
        good_marker=good_marker,
        bad_marker=language_config[f"FIN_{role}_MARK"] or "",
    ), None

def foil_marker_for_external_role(
    role: str,
    good_marker: str,
    row: JsonDict | None,
    source_index: int,
) -> tuple[str, str]:
    role = role.lower()
    index = row_stable_index(row, source_index)

    if good_marker == "":
        bad_marker = "ca" if index % 2 else "ge"
        return bad_marker, f"add_{bad_marker}_to_anc_external_{role}_head"

    if good_marker == "ca":
        if index % 2:
            return "", f"delete_anc_external_{role}_ca"
        return "ge", f"replace_anc_external_{role}_ca_with_ge"

    bad_marker = "ca" if good_marker != "ca" else "ge"
    return bad_marker, f"replace_anc_external_{role}_{good_marker}_with_{bad_marker}"

