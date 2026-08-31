# Character semantic rollback review

## Scope

This review compares the records immediately before `1b0fa61` (`7c2f22ab`)
with the current safe runtime seed. It restores the material that was
compressed during that transfer without rolling back its runtime contract.

The migration keeps every existing `ActorRecord.facts` entry, then appends
the missing durable pre-transfer material directly (apart from surrounding
whitespace):

| Former source | Current owner-bounded representation |
| --- | --- |
| `backstory` | `ActorFact(origin="lived")` |
| `personality` | `ActorFact(origin="lived")` |
| `known_context` | `ActorFact(origin="told")` |
| `private_state.goals`, `current_objectives`, and `secrets` | individual `ActorFact(origin="lived")` entries |
| `descriptions.private` | `ActorFact(origin="lived")` |

`descriptions.public` was already preserved byte-for-byte in
`public_sheet.public_context` for every seed. The retired
`intentions_enabled` flag remains absent: its scheduling meaning stays in the
current `may_act_offstage` field, while no free-form intent/carry is restored.

Most added facts are the original source text. The small exception is retired
runtime-control language: the old Iselle/Master opening and handoff scripts,
the promotion playtest's control transcript, and author-facing phrases such as
"do not play" or "the engine should render" are not actor knowledge. Their
character-level behavior is retained as self facts where needed; their
obsolete control instructions are not. This keeps the semantic rollback from
reintroducing a parallel prompt/runtime contract.

The One-Star promotion fixture inherits each restored source record as its
prefix, then retains its scenario-specific facts. The player-authored
`one_star_newcomer` remains actor-free, as does the non-social Warden.
Its post-source fixture facts remain `told`, preserving the established
contract that later-state additions are learned or selected in play.

## Before and after

| Checkpoint | Restored actor records | Facts before | Facts after | Added direct source payloads |
| --- | ---: | ---: | ---: | ---: |
| `dating_villa_s1` | 14 | 75 | 284 | 209 / 38,471 characters |
| `one_star_ascension_s1` | 14 | 58 | 235 | 177 / 52,729 characters |
| `one_star_ascension_s1_promotion_playtest` | 15 | 67 | 265 | 177 inherited source facts plus 21 fixture-specific semantic facts |
| `spring_rain_second_chorus` | 11 | 63 | 216 | 153 / 21,760 characters |
| `the_unblessed_summon` | 42 | 218 | 873 | 655 / 264,066 characters |

Each added payload remains actor-owned: it renders in that character's
`<you>` packet and is absent from the exterior-only visible-self packet. No
production dialogue prompt changed.

## Record coverage

- `dating_villa_s1`: `jordan_reeves`, `marcus_chen`, `elena_vasquez`,
  `priya_kapoor`, `marisol_ortega`, `tessa_ng`, `omar_haddad`, `noah_ellis`,
  `avery_brooks`, `vik_morrison`, `dante_royale`, `lena_sato`,
  `pietro_adler`, `britney_spears`.
- `one_star_ascension_s1`: `edren_marr`, `renna_holt`,
  `iselle_the_guide`, `the_master`, `halcyon_of_the_gilded_march`,
  `soren_ironvow`, `castor_valebrand`, `wren_thelantern`, `rowan_kest`,
  `liora_fen`, `mirelle_voss`, `seris_nightglass`, `aveline_morcant`,
  `veil_the_unnumbered`.
- `one_star_ascension_s1_promotion_playtest`: the preceding One-Star actor
  records plus `promotion_playtest_faceless`; the common records retain the
  source-record prefix.
- `spring_rain_second_chorus`: `ren_sato`, `mio_tachibana`,
  `hanae_morikawa`, `yui_kisaragi`, `kenta_fujiwara`, `naomi_kurata`,
  `sora_minazuki`, `professor_aya_shimizu`, `daichi_okumura`,
  `chika_enomoto`, `subaru_amari`.
- `the_unblessed_summon`: `player_protagonist`, `player_protagonist_2`,
  `liriel_vaerien`, `korva_sahl`, `sariel_marenne`, `vaella_coldspire`,
  `quill`, `sylira_vesh`, `marennis_vale`, `sora_kageyama`, `mika_aoyama`,
  `riku_tsumura`, `daichi_nikaido`, `mei_iwasaki`, `yuna_kiyose`,
  `tatsuya_hozumi`, `crown_prince_aldemar`, `king_halric`,
  `cardinal_vespera`, `archon_selivar`, `sage_wencel`, `demon_lord`,
  `guild_master_bren`, `court_mage_selen`, `lady_aoi`, `mira`,
  `master_ovrec`, `household_physician_orvell`, `crown_liaison_lerin`,
  `princess_nirvel`, `anelle_aubin`, `ambassador_sashina`, `sevarin`,
  `faulker`, `halen`, `veranne`, `yuto_arai`, `asami_kuroda`,
  `kenta_morimura`, `yui_sasahara`, `hiroshi_kasai`, `kei_sugino`.

## Deliberate non-restorations

- No legacy `backstory`, `personality`, `known_context`, `private_state`, or
  `private_carry` schema reader/writer returned.
- No history-derived personality synthesis, private footer, or cross-actor
  carry returned.
- No retired opening/handoff, Master-interface, or playtest-control prose
  returned through actor facts.
- No content moved into public perception or other characters' packets.
- No dialogue-prompt prose was changed.
