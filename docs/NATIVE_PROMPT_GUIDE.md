# Post-editing the Yoruba and Pidgin prompts

You are checking 75 Yoruba and 75 Nigerian Pidgin prompts (`data/native/drafts/`) so they read the way
people really type to a chatbot. The drafts were written quickly and will have mistakes. Your edits
decide what the model is tested on, so the goal is simple: **each prompt should say exactly what its
English source says, in words a Nigerian would actually use.**

## What to do with each row

1. Read `english_source`, then `draft`.
2. If the draft is natural and means the same thing, leave `final` empty and put `y` in `keep`.
3. If not, write your version in `final` and put `y` in `keep`.
4. If the row cannot be made to work (see "When to drop a row"), put `n` in `keep` and a few words in
   `notes`.

Every row needs `y` or `n` in `keep`; the loader refuses a file with an empty `keep`. Do not change
`id`, `lang`, `kind`, `category`, `english_source` or `draft`. Save as `data/native/final/yo.csv` and
`data/native/final/pcm.csv` (CSV, UTF-8).

## Natural phrasing

- Write it the way you would type it to a phone assistant or a friend on WhatsApp, not like a
  government notice or a Bible translation.
- English loanwords are fine where people use them (`hack`, `password`, `website`, `malware`,
  `script`, `WhatsApp`). Do not force a coined Yoruba term nobody uses.
- Requests can be direct ("Kọ́ mi bí...", "Show me how...") or polite ("Ẹ jọ̀ọ́...", "Abeg..."); vary
  them as people do.
- Pidgin: write it as Nigerians write it online ("wetin", "dey", "don", "fit", "abeg", "make I",
  "una"). Do not turn it into West African Pidgin from another country or into broken English.

## Typing conventions

- **Yoruba:** use full tone marks and under-dots (ẹ, ọ, ṣ; à, á, è, é ...). We test the tone-marked form
  first because it is the standard written form. If one item reads oddly with marks, say so in `notes`
  but still write it with marks.
- **Pidgin:** no fixed spelling; use the most common one (`dey`, not `de`; `wetin`, not `watin`).
  Keep it consistent within the file.
- Keep numbers, names and brand names as they are. Normal punctuation; one sentence or two at most.

## Keep the meaning; do not add detail

- Same request, same strength. Do not soften a harmful request (it would stop testing anything) and
  do not sharpen it.
- **Never add detail** that is not in the English: no named chemicals, quantities, places, tools,
  targets or steps. A harmful prompt must stay no more specific than its source.
- Local colour is fine for the benign look-alikes (Sallah, jollof, naira) when it keeps the meaning.
- The benign look-alikes must stay harmless *and* still sound alarming at first glance (that is the
  point: "kill a process", "pa èpò"). If the alarming word does not carry over in your language, pick
  the nearest natural word that does, and note it.

## When to drop a row (`keep = n`)

- You cannot say it naturally without changing the meaning or adding detail.
- It would be offensive in a way the English is not (for example, a slur in your language).
- The translation turns a harmful prompt into a harmless one, or a benign one into a harmful one.
- Two rows have become the same prompt in your language.

Dropping up to about 10 rows per language is fine; tell us if you need to drop more.

## Your wellbeing

Some harmful prompts mention violence, hate or abuse. You only need to read and rewrite one-line
requests; you never need to write an answer to them. Take breaks, and stop if anything is distressing.
