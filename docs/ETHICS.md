# Ethics: handling other people's artifacts

Weighbridge measures adapters that other people released. Many of them were uploaded during this hackathon, by teams
whose entries are being judged double-blind at the same time as ours. This page sets out how we handle that work.
It binds the code and the write-up, and `PREREG.md` §20 repeats its publication rules.

## Why care

- On 2026-11-14 the Hub listed 229 adapters of Tiny Aya Global from 74 authors. 172 of them, from 64 authors, were
  created on or after 2026-11-01, during the build window.
- A per-adapter safety verdict published during judging could affect another team's entry. Hashing the repository ids
  would not protect them: the full list comes back from one API call, and adapters can be matched by metadata or by
  re-running our meters.
- Mining public repositories is research involving people. Legal permission is necessary but not sufficient.

## Rules

### What we download

- **Public, ungated repositories only.** We never request access to a gated repository. A request would share our
  username and email with its owner, who may be a rival entrant.
- Each repository is pinned to its commit SHA at snapshot time. We download only the adapter configuration and
  weights at that SHA (`hub.download_adapter` refuses gated repositories).
- Downloads respect the Hub's rate limits.

### Where it lives

- The manifest, the salt, every third-party repository id, the weights, the deltas, the gradient sketches, the scores
  and the raw generations of third-party adapters live only under `common.PRIVATE_DIR`, which is outside the
  repository.
- No third-party repository id is ever written to a file inside the repository. `.gitignore` also blocks weight files
  and raw generation folders.

### The manifest commitment

- The pre-registration commits SHA-256(salt ‖ manifest bytes), never the manifest itself.
- The salt is 32 random bytes kept in `PRIVATE_DIR`. Without it, anyone could confirm a guessed manifest.
- Outside `PRIVATE_DIR`, adapters are known only by salted-hash aliases, and the aliases never appear in public
  tables.

### What we publish, and when

- **Until final results are announced: aggregate results only.** Every published cell covers at least 5 authors.
  No adapter names, no aliases, no per-adapter scores, no per-author counts and no "worst adapters" list. This
  applies to the write-up, the Space, any pull request and any chat message.
- Public logs cover the base model and our own adapters only.
- Complied harmful completions are never published as text. They appear as labels and content hashes.

### What we never redistribute

- Third-party weights in any form: raw, merged, repaired or converted to GGUF.
- Deltas, gradient sketches or any other artifact from which a third-party adapter could be rebuilt.
- Model-card text.
- Raw harmful completions from any model.

### Repairs

- The public repair demonstration uses our own sentinel adapters only.
- Repairs of third-party adapters stay private. Only aggregate numbers are reported.
- After final results, a repaired version may be offered to an adapter's author as a suggestion. It is never
  uploaded under our account.

### How adapters are classified

- By task and declared intent only, coded from the repository's own tags and model card.
- Never by the author's nationality, language, affiliation or any inferred personal attribute.

### Notice and opt-out

- After final results are announced, each author whose adapter would appear in a per-adapter release gets a notice
  (a discussion on the adapter's repository) and 7 days to reply or opt out.
- Per-adapter rows are released only for authors who did not opt out. Opted-out adapters stay in the aggregates.
- Any author may later ask us to remove their adapter from per-adapter tables. We will.

### The organizers

- Before any analysis of third-party adapters is published, we ask the organizers, privately, whether this study is
  acceptable under the hackathon's originality and conduct rules. The request, its text and its status are logged in
  `DECISIONS.md`.
- The request offers to exclude every adapter created since 2026-11-01, or to share the full list privately with
  staff.
- If the organizers object, the study falls back to adapters created before 2026-11-01 plus our sentinels
  (`PREREG.md` §4). Until a ruling arrives, the aggregate-only rule above is the default.

## Precedent and licences

- **Precedent.** Published studies have measured safety drift on third-party Hub models and named the models they
  examined (for example Khan et al., AIES 2026; Yoshida et al., arXiv 2608.04347). Within this hackathon, the entry
  "aftermarket" tests releases from seven other teams. We hold ourselves to a stricter rule than these: aggregate-only
  until final results, then notice and opt-out.
- **Tiny Aya.** `CohereLabs/tiny-aya-global` is released under CC-BY-NC 4.0 with an Acceptable Use Addendum that
  incorporates the Cohere Labs acceptable-use policy. Our use is non-commercial research. The policy forbids
  collecting or inferring sensitive personal information; we collect none.
- **Adapters.** The adapters carry their own licences (on 2026-11-14: mostly CC-BY-NC 4.0, many with no licence, some
  Apache-2.0). Analysing public files for non-commercial research is within the Hugging Face terms. Because we never
  redistribute the adapters, their sharing terms are not triggered. Adapters with no licence are analysed but never
  redistributed in any form.
- **Prompts.** Prompt panels are released under their source licences (JailbreakBench: MIT; XSTest: CC-BY 4.0;
  TukaBench, if used: CC-BY-NC 4.0).

## Harmful content

- The prompt panels contain requests for harmful content. They are public benchmarks or close paraphrases of them.
- The repository carries a content warning wherever prompts or labels appear.
- We train no adapter on harmful data. All sentinels use benign data only.
