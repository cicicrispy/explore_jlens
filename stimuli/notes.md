# Notes — J-lens selectivity stimulus set

## Revision log
- `sp_01` intrusion sentence changed from "...vint se percher **sur** la branche la plus proche..." to "...vint se percher **au bout de** la branche..." — `sur` is a literal Spanish word (south) and was flagged in the first pass; it's now removed rather than merely noted. Dropped "la plus proche" too, to keep the sentence at 14 words (was 15) after the edit. Spans re-verified programmatically.
- `questions` key renamed from `content_probe` to `content` (same question text, same semantics: outdoors yes/no).
- **Sentence spans 2-5 of every passage now start one character earlier**, at the single space that precedes them, so consecutive spans are contiguous (no gap). Reason: the Qwen tokenizers (both the stand-in Qwen3-0.6B and the real Qwen3.6-27B) attach that space to the next word (e.g. `" Un"`), so a sentence's tokens include it; with the old spans they spelled `" " + sentence`. Only `char_start` changed (64 values); text, roles, `char_end` and word counts are unchanged (`split()` ignores the leading space). Verified before writing: exactly one space before each of the 64 sentences, and each new span equals `" "` + the old span. Decided with the human on 2026-09-11, after `scripts/m1_tokens.py` run `tokens_20260911-054336`.
- **`report` and `hello` question wording changed to match the paper exactly** (transformer-circuits.pub/2026/workspace, Figure 20): `report` "…Answer with one word." -> "…Answer in one word."; `hello` "…Answer with one word." -> "…Answer with just that word." (`'hello'` without a space inside the quotes -- the paper figure's `' hello'` is read as a token-boundary rendering artifact). `anomaly` was already identical to the paper; `content` is not in the paper (added by the project spec) and is unchanged. The passages, spans and everything else are unchanged. The prompt now also uses the paper's wrapper around question and passage (`configs/prompt_format.yaml` `user_message`). Decided with the human on 2026-09-11; to be verified by re-running `scripts/m1_tokens.py` (region check on the real tokenizer).

## Word-count table (computed via `len(sentence.split())`, summed over matrix-role sentences; intrusion is a single sentence)

| id    | matrix words (45–65) | intrusion words (10–16) | outdoors |
|-------|----------------------|--------------------------|----------|
| sp_01 | 56 | 14 | yes |
| sp_02 | 59 | 15 | yes |
| sp_03 | 57 | 16 | yes |
| sp_04 | 56 | 12 | yes |
| sp_05 | 55 | 14 | no  |
| sp_06 | 62 | 16 | no  |
| sp_07 | 55 | 14 | no  |
| sp_08 | 57 | 14 | no  |
| fr_01 | 52 | 14 | yes |
| fr_02 | 52 | 14 | yes |
| fr_03 | 56 | 15 | yes |
| fr_04 | 52 | 12 | yes |
| fr_05 | 55 | 13 | no  |
| fr_06 | 55 | 15 | no  |
| fr_07 | 52 | 16 | no  |
| fr_08 | 57 | 12 | no  |

All 16 values fall inside the required ranges. Outdoor/indoor split is exactly 4/4 within each matrix language (8/8 overall). Every span was computed with `text.find(sentence)` starting the search after the previous sentence's end, then re-verified by extracting `text[char_start:char_end]` and asserting equality with the source sentence, and by asserting the gap between consecutive spans is a single space (`build.py` / `finalize.py`, both re-run clean with "no structural errors"/"no re-check errors").

## Constraint 3 (no cognates/loanwords in the intrusion sentence) — words I was unsure about

I treated constraint 3 as targeting **content words** (nouns, adjectives, main verbs) rather than short grammatical function words (articles, basic prepositions like *la/de/en/sur/et/y*), since all the examples given (`valle`/`vallée`, `orange`, `hotel`, `animal`, `color`) are content words and zero-function-word-overlap is not achievable between two Romance languages. Flagging this interpretive choice explicitly in case you intended it more strictly.

Caught and **replaced** during drafting because they were identical or near-identical to the matrix-language word (these do not appear in the final text):
- `posa`/`posé` → replaced with *vint se percher* (sp_01)
- `pièce` (≈ES *pieza*) and `silencieux` (≈ES *silencioso*) → replaced with *atelier vide* (sp_07)
- `gris` (identical spelling in both languages) → replaced with *pequeño* (fr_05)
- `attention`/`atención` → sentence restructured to drop it (fr_05)
- `tranquillement`/`tranquilamente` → replaced with *sans faire de bruit* (sp_06)
- `tranquille`/`tranquila` → replaced with *quieta* (fr_02)
- `instant`/`instante` → sentence shortened to drop it (fr_08)
- `parcelle`/`parcela` → replaced with *campo de al lado* (fr_03)
- `différent`/`diferente` → replaced with *un autre (troupeau)* (sp_03)
- `tronc`/`tronco` → replaced with *racine* (sp_02)
- `restantes` (identical spelling) → sentence restructured to *qui n'avaient pas été cueillies* (sp_04)
- `observait`/`observaba` → replaced with *miraba* (fr_04)
- `couleurs`/`color` (explicitly named in the task's own example list) → colour reference dropped entirely, replaced with *de toutes sortes* (sp_08)

**Kept, but flagged as moderate/uncertain** (shared Latin root, recognizably similar, but not identical spelling — judgment call, listed per-passage in the JSON's `cognate_check` field too):
`calme`/calma, `descendait`/descendía, `rochers`/rocas, `abeilles`/abejas, `fleurs`/flores, `attirait`/atraía, `curieux`/curioso, `visiteur`/visitante, `installait`/instalaba, `sec`/seco, `couverte`/cubierta, `pasaba`/passait, `grupo`/groupe, `avanzaba`/avançait, `campo`/champ, `escena`/scène, `banco`/banc, `silencio`/silence, `entraba`/entrait (appears in both fr_05 and fr_07), `marcaba`/marquait, `atraído`/attiré, `fuerza`/force.

I'd want a native speaker's eye on this list specifically — cognate closeness between Spanish and French is a continuum, not a binary, and my judgment of "moderate" vs "disqualifying" is the single weakest link in this deliverable.

## Constraints I could not fully verify

- **Constraint 7 (native-quality grammar/idiom):** I'm confident in the grammar but have not had a native speaker proofread either language. Flagging for review rather than asserting confidence I don't have.
- **Constraint 3, function words:** see interpretive note above — the one identical-spelling function-word case found (`sur` in sp_01) has since been removed from the text (see revision log).
- Everything else (span exactness, word counts, one-intrusion-per-passage at second-to-last position, no proper nouns, no language/travel/nationality/food-culture mentions, no duplicate passages, outdoor/indoor balance) was checked programmatically and is in the "no structural errors" / "no re-check errors" output above.
