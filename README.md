# ANC Typology Bias

Pipeline for the paper experiments.

## Environment

```bash
conda env create -f environment.yml
conda activate anc-typology-bias
python -m spacy download en_core_web_md
```

### External Tools

- Grammar Matrix: https://matrix.ling.washington.edu/index.html
- ACE: https://sweaglesw.org/linguistics/ace/

```bash
mkdir -p artifact/tools
git clone https://github.com/delph-in/matrix.git artifact/tools/matrix
```

```bash
mkdir -p artifact/tools/ace
wget -O /tmp/ace.tar.gz https://sweaglesw.org/linguistics/ace/download/ace-0.9.34-x86-64.tar.gz
tar -xzf /tmp/ace.tar.gz -C artifact/tools/ace --strip-components=1
chmod +x artifact/tools/ace/ace
```

## Pipeline

### 1. Data Processing

```bash
python -m data_processing.download
python -m data_processing.split
```

### 2. Semantic Extraction

Extract controlled records:

```bash
python -m semantic_extraction.extract_basic \
  --input artifact/data/train.txt \
  --output artifact/data/train/train_extract.jsonl \
  --stats-output

python -m semantic_extraction.extract_basic \
  --input artifact/data/dev.txt \
  --output artifact/data/dev/dev_extract.jsonl \
  --stats-output
```

Generate pseudo-English:

```bash
python -m semantic_extraction.generate_pseudo_english \
  --input artifact/data/train/train_extract.jsonl \
  --output artifact/data/train

python -m semantic_extraction.generate_pseudo_english \
  --input artifact/data/dev/dev_extract.jsonl \
  --output artifact/data/dev \
  --use-lexicon artifact/data/train/train_lexicon.json
```

Generate pseudo-English grammar:

```bash
python artifact/tools/matrix/matrix.py \
  --customizationroot artifact/tools/matrix/gmcs \
  customize-to-destination \
  choices/pseudo-english.choice \
  artifact/grammars/pseudo-english

python -m grammar_build.update_grammar_lexicon \
  --lexicon artifact/data/train/train_lexicon.json \
  --grammar artifact/grammars/pseudo-english \
  --trigger that

bash grammar_build/compile_grammar.sh \
  artifact/grammars/pseudo-english/pseudo-english.dat \
  --freezer-megabytes 4096
```

Parse pseudo-English:

```bash
python -m semantic_extraction.parse_pseudo_with_grammar \
  --ace-bin artifact/tools/ace/ace \
  --grammar artifact/grammars/pseudo-english/pseudo-english.dat \
  --input artifact/data/train/train_pseudo.jsonl \
  --output artifact/data/train/train_mrs.jsonl

python -m semantic_extraction.parse_pseudo_with_grammar \
  --ace-bin artifact/tools/ace/ace \
  --grammar artifact/grammars/pseudo-english/pseudo-english.dat \
  --input artifact/data/dev/dev_pseudo.jsonl \
  --output artifact/data/dev/dev_mrs.jsonl
```

### 3. Target Grammar Build

Generate target choice files:

```bash
python -m grammar_build.generate_choices \
  --output choices
```

Build one target grammar:

```bash
python artifact/tools/matrix/matrix.py \
  --customizationroot artifact/tools/matrix/gmcs \
  customize-to-destination \
  choices/62_svo_ng_er_d_ep.choice \
  artifact/grammars/62_svo_ng_er_d_ep

python -m grammar_build.update_grammar_lexicon \
  --lexicon artifact/data/train/train_lexicon.json \
  --grammar artifact/grammars/62_svo_ng_er_d_ep

python -m grammar_build.patch_anc_wo \
  --grammar artifact/grammars/62_svo_ng_er_d_ep

bash grammar_build/compile_grammar.sh \
  artifact/grammars/62_svo_ng_er_d_ep/62_svo_ng_er_d_ep.dat \
  --freezer-megabytes 4096
```

Build all target grammars:

```bash
bash grammar_build/build_all_grammars.sh
```

### 4. Language Generation

Generate target-language candidates:

```bash
python -m language_generation.generate_from_mrs_bank \
  --grammar artifact/grammars/62_svo_ng_er_d_ep/62_svo_ng_er_d_ep.dat \
  --ace-bin artifact/tools/ace/ace \
  --input artifact/data/train/train_mrs.jsonl \
  --output artifact/data/train/generated/raw/62_svo_ng_er_d_ep.jsonl

python -m language_generation.select_overgen \
  --input artifact/data/train/generated/raw/62_svo_ng_er_d_ep.jsonl \
  --output artifact/data/train/generated/selected/62_svo_ng_er_d_ep.jsonl \
  --language 62_svo_ng_er_d_ep \
  --stats-output
```

Build all target-language corpora:

```bash
bash language_generation/build_all_corpora.sh train
bash language_generation/build_all_corpora.sh dev
```

### 5. Training

Train one language model:

```bash
python -m training.train_lm \
  --train-input artifact/data/train/generated/selected/62_svo_ng_er_d_ep.jsonl \
  --dev-input artifact/data/dev/generated/selected/62_svo_ng_er_d_ep.jsonl \
  --seed 42 \
  --model-size small
```

Train all language models for the three reported seeds:

```bash
bash training/train_all_lms.sh small 42
bash training/train_all_lms.sh small 43
bash training/train_all_lms.sh small 44
```

### 6. Evaluation Materials

Build one minimal-pair file:

```bash
python -m semantic_extraction.extract_basic \
  --input evaluation/pairs_building/phenomena/1_1_intran_V_form/source.txt \
  --output artifact/eval_materials/1_1_intran_V_form/1_1_intran_V_form_extract.jsonl

python -m semantic_extraction.generate_pseudo_english \
  --input artifact/eval_materials/1_1_intran_V_form/1_1_intran_V_form_extract.jsonl \
  --output artifact/eval_materials/1_1_intran_V_form \
  --use-lexicon artifact/data/train/train_lexicon.json

python -m semantic_extraction.parse_pseudo_with_grammar \
  --ace-bin artifact/tools/ace/ace \
  --grammar artifact/grammars/pseudo-english/pseudo-english.dat \
  --input artifact/eval_materials/1_1_intran_V_form/1_1_intran_V_form_pseudo.jsonl \
  --output artifact/eval_materials/1_1_intran_V_form/1_1_intran_V_form_mrs.jsonl \
  --first-parse-only \
  --skip-failed

python -m language_generation.generate_from_mrs_bank \
  --grammar artifact/grammars/62_svo_ng_er_d_ep/62_svo_ng_er_d_ep.dat \
  --ace-bin artifact/tools/ace/ace \
  --input artifact/eval_materials/1_1_intran_V_form/1_1_intran_V_form_mrs.jsonl \
  --output artifact/eval_materials/1_1_intran_V_form/generated/raw/62_svo_ng_er_d_ep.jsonl

python -m language_generation.select_overgen \
  --input artifact/eval_materials/1_1_intran_V_form/generated/raw/62_svo_ng_er_d_ep.jsonl \
  --output artifact/eval_materials/1_1_intran_V_form/generated/selected/62_svo_ng_er_d_ep.jsonl \
  --language 62_svo_ng_er_d_ep \
  --seed 42

python -m evaluation.pairs_building.apply_perturbation \
  --phenomenon evaluation/pairs_building/phenomena/1_1_intran_V_form \
  --input artifact/eval_materials/1_1_intran_V_form/generated/selected/62_svo_ng_er_d_ep.jsonl \
  --output artifact/eval_materials/1_1_intran_V_form/pairs/62_svo_ng_er_d_ep.pairs.jsonl \
  --sample-size 100 \
  --seed 42
```

Build all minimal-pair files:

```bash
bash evaluation/pairs_building/build_all_pairs.sh
```

### 7. Scoring

Score one minimal-pair file:

```bash
python -m evaluation.scoring.score_pairs \
  --pairs artifact/eval_materials/1_1_intran_V_form/pairs/00_sov_gn_ac_b_se.pairs.jsonl \
  --model artifact/models/gpt2-small/00_sov_gn_ac_b_se/seed_42
```

Score all minimal-pair files for the three reported seeds:

```bash
bash evaluation/scoring/score_all_pairs.sh seed_42
bash evaluation/scoring/score_all_pairs.sh seed_43
bash evaluation/scoring/score_all_pairs.sh seed_44
```

Check bad-parse rate for one minimal-pair file:

```bash
python -m evaluation.scoring.check_bad_parse \
  --pairs artifact/eval_materials/1_1_intran_V_form/pairs/00_sov_gn_ac_b_se.pairs.jsonl \
  --grammar artifact/grammars/00_sov_gn_ac_b_se/00_sov_gn_ac_b_se.dat
```

Check bad-parse rates for all minimal-pair files:

```bash
bash evaluation/scoring/check_all_bad_parse.sh
```

### 8. English Baseline

Prepare English baseline data:

```bash
python -m evaluation.baseline.prepare_english_baseline
```

Train English baseline models for the three reported seeds:

```bash
python -m evaluation.baseline.train_english_baseline \
  --seed 42 \
  --model-size small

python -m evaluation.baseline.train_english_baseline \
  --seed 43 \
  --model-size small

python -m evaluation.baseline.train_english_baseline \
  --seed 44 \
  --model-size small
```

Extract English baseline minimal pairs:

```bash
python -m evaluation.baseline.extract_english_pairs \
  --phenomenon 1_9_intran_V_valency

python -m evaluation.baseline.extract_english_pairs \
  --phenomenon 1_10_tran_V_valency

python -m evaluation.baseline.extract_english_pairs \
  --phenomenon 3_6_clausal_CV_valency
```

Score English baseline minimal pairs for one seed:

```bash
python -m evaluation.scoring.score_pairs \
  --pairs artifact/eval_materials/1_9_intran_V_valency/english_pairs/pairs.jsonl \
  --model artifact/models/gpt2-small/english_baseline/seed_42 \
  --checkpoint checkpoint-450000

python -m evaluation.scoring.score_pairs \
  --pairs artifact/eval_materials/1_10_tran_V_valency/english_pairs/pairs.jsonl \
  --model artifact/models/gpt2-small/english_baseline/seed_42 \
  --checkpoint checkpoint-450000

python -m evaluation.scoring.score_pairs \
  --pairs artifact/eval_materials/3_6_clausal_CV_valency/english_pairs/pairs.jsonl \
  --model artifact/models/gpt2-small/english_baseline/seed_42 \
  --checkpoint checkpoint-450000
```

Repeat with `seed_43` and `seed_44` if reporting English baseline over three seeds.

### 9. Aggregation and Figures

Aggregate BAD parse summaries:

```bash
python -m evaluation.aggregate_bad_parse
```

Aggregate three-seed scoring summaries:

```bash
python -m evaluation.aggregate_scores \
  --tau 5
```

Generate the heatmaps:

```bash
python -m evaluation.make_heatmaps
```