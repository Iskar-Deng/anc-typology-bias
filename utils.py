# utils.py

"""
Store shared project paths and constants.
"""

# Paths
ARTIFACT_ROOT = "artifact"

DATA_DIR = f"{ARTIFACT_ROOT}/data"
TOOLS_DIR = f"{ARTIFACT_ROOT}/tools"
GRAMMARS_DIR = f"{ARTIFACT_ROOT}/grammars"
MODELS_DIR = f"{ARTIFACT_ROOT}/models"
EVAL_MATERIALS_DIR = f"{ARTIFACT_ROOT}/eval_materials"
RESULTS_DIR = f"{ARTIFACT_ROOT}/results"


# Model parameters

TRAINING_CONFIG = {
    "model_name": "gpt2",
    "text_field": "sent",
    "block_size": 32,
    "seed": 42,
    "max_steps": 70000,
    "per_device_train_batch_size": 16,
    "per_device_eval_batch_size": 16,
    "gradient_accumulation_steps": 2,
    "learning_rate": 5e-4,
    "warmup_steps": 590,
    "weight_decay": 0.01,
    "logging_steps": 100,
    "eval_steps": 5000,
    "save_steps": 5000,
    "save_total_limit": 1,
    "dataloader_num_workers": 0,
}


# Grammar configuration

CLAUSE_WORD_ORDERS = ["sov", "svo", "vos"]
NP_WORD_ORDERS = ["gn", "ng"]

ALIGNMENTS = {
    "ac": "nom-acc",
    "er": "erg-abs",
}

COMPLEMENT_SYSTEMS = {
    "b": "balancing",
    "d": "deranking",
}

ANC_STRATEGIES = {
    "se": "sent",
    "pa": "poss-acc",
    "ep": "erg-poss",
    "no": "nomn",
}

ANC_CHOICE_WORD_ORDER_TABLE = {
    ("sov", "gn"): {
        "sent": "sov",
        "poss-acc": "sov",
        "erg-poss": "sov",
        "nomn": "sov",
    },
    ("svo", "gn"): {
        "sent": "svo",
        "poss-acc": "svo",
        "erg-poss": "sov",
        "nomn": "svo",
    },
    ("vos", "gn"): {
        "sent": "vos",
        "poss-acc": "svo",
        "erg-poss": "sov",
        "nomn": "svo",
    },
    ("sov", "ng"): {
        "sent": "sov",
        "poss-acc": "ovs",
        "erg-poss": "vos",
        "nomn": "ovs",
    },
    ("svo", "ng"): {
        "sent": "svo",
        "poss-acc": "vos",
        "erg-poss": "vos",
        "nomn": "vos",
    },
    ("vos", "ng"): {
        "sent": "vos",
        "poss-acc": "vos",
        "erg-poss": "vos",
        "nomn": "vos",
    },
}

ANC_IV_ORDER_TABLE = {
    ("sov", "gn"): {
        "sent": "SV",
        "poss-acc": "SV",
        "erg-poss": "SV",
        "nomn": "SV",
    },
    ("svo", "gn"): {
        "sent": "SV",
        "poss-acc": "SV",
        "erg-poss": "SV",
        "nomn": "SV",
    },
    ("vos", "gn"): {
        "sent": "VS",
        "poss-acc": "SV",
        "erg-poss": "SV",
        "nomn": "SV",
    },
    ("sov", "ng"): {
        "sent": "SV",
        "poss-acc": "VS",
        "erg-poss": "VS",
        "nomn": "VS",
    },
    ("svo", "ng"): {
        "sent": "SV",
        "poss-acc": "VS",
        "erg-poss": "VS",
        "nomn": "VS",
    },
    ("vos", "ng"): {
        "sent": "VS",
        "poss-acc": "VS",
        "erg-poss": "VS",
        "nomn": "VS",
    },
}

ANC_TV_ORDER_TABLE = {
    ("sov", "gn"): {
        "sent": "APV",
        "poss-acc": "APV",
        "erg-poss": "APV",
        "nomn": "APV",
    },
    ("svo", "gn"): {
        "sent": "AVP",
        "poss-acc": "AVP",
        "erg-poss": "PVA",
        "nomn": "AVP",
    },
    ("vos", "gn"): {
        "sent": "VPA",
        "poss-acc": "AVP",
        "erg-poss": "PVA",
        "nomn": "AVP",
    },
    ("sov", "ng"): {
        "sent": "APV",
        "poss-acc": "PVA",
        "erg-poss": "AVP",
        "nomn": "PVA",
    },
    ("svo", "ng"): {
        "sent": "AVP",
        "poss-acc": "VPA",
        "erg-poss": "VPA",
        "nomn": "VPA",
    },
    ("vos", "ng"): {
        "sent": "VPA",
        "poss-acc": "VPA",
        "erg-poss": "VPA",
        "nomn": "VPA",
    },
}

FINITE_MARK_TABLE = {
    "nom-acc": {
        "FIN_S_MARK": "",
        "FIN_A_MARK": "",
        "FIN_P_MARK": "ca",
    },
    "erg-abs": {
        "FIN_S_MARK": "",
        "FIN_A_MARK": "ca",
        "FIN_P_MARK": "",
    },
}

ANC_MARK_TABLE = {
    ("sent", "nom-acc"): {
        "ANC_S_MARK": "",
        "ANC_A_MARK": "",
        "ANC_P_MARK": "ca",
    },
    ("sent", "erg-abs"): {
        "ANC_S_MARK": "",
        "ANC_A_MARK": "ca",
        "ANC_P_MARK": "",
    },
    ("poss-acc", "nom-acc"): {
        "ANC_S_MARK": "ge",
        "ANC_A_MARK": "ge",
        "ANC_P_MARK": "ca",
    },
    ("poss-acc", "erg-abs"): {
        "ANC_S_MARK": "ge",
        "ANC_A_MARK": "ge",
        "ANC_P_MARK": "",
    },
    ("erg-poss", "nom-acc"): {
        "ANC_S_MARK": "ge",
        "ANC_A_MARK": "ob",
        "ANC_P_MARK": "ge",
    },
    ("erg-poss", "erg-abs"): {
        "ANC_S_MARK": "ge",
        "ANC_A_MARK": "ob",
        "ANC_P_MARK": "ge",
    },
    ("nomn", "nom-acc"): {
        "ANC_S_MARK": "ge",
        "ANC_A_MARK": "ge",
        "ANC_P_MARK": "ob",
    },
    ("nomn", "erg-abs"): {
        "ANC_S_MARK": "ge",
        "ANC_A_MARK": "ge",
        "ANC_P_MARK": "ob",
    },
}

VERB_MARK_TABLE = {
    "balancing": {
        "fin_v_mark": "s",
        "comp_v_mark": "s",
        "anc_v_mark": "ing",
    },
    "deranking": {
        "fin_v_mark": "s",
        "comp_v_mark": "ing",
        "anc_v_mark": "ing",
    },
}

MRS_REWRITE_RULES = [
    ("SF: prop-or-ques", "SF: iforce"),
    ("COG-ST: uniq-id", "COG-ST: cog-st"),
    ("COG-ST: in-foc", "COG-ST: cog-st"),
]
