Nummers tussen haakjes verwijzen naar de stappen uit "Get to work".

## Al gedaan

- **Embedding model:** GloVe 2024 wikigiga 50d geladen (in notebook) in embeddings.py staan ook andere die we nog kunnen testen

- **Tokenizers:** `bow` (zonder leestekens/stopwoorden) en `rnn` (alles behouden, koppeltekens gesplitst)

- **Coverage:** 99.3% (bow) en 99.7% (rnn) token coverage (~95% type coverage - unique token coverage) - Bij rnn werden tokens met koppeltekens als <unk> gezien -> een tokenizer gemaakt die met regular expressions de koppeltekens vervangt voor spaties (99.3 -> 99.7)

- **Similariteit en buren:** emoties liggen dicht bij elkaar (0,53–0,87), tegengestelden ook.

- **Pipeline (tutorialinstellingen, 3 seeds × 5 folds):**
  | classifier (MLP) | 2 verborgen lagen van 50, dropout 0,5 (embed-and-pool)| 2 verborgen lagen van 50, dropout 0,1 (lstm)
  | optimizer | AdamW, lr 1e-3, weight decay 1e-4
  | batch size / epochs | 64 / 20
  | modelselectie | epoch met de laagste validatie-loss

  Met standaard instellingen (frozen embeddings):

  embed-and-pool mean=0.3871 std=0.0125 (n=15)
  lstm mean=0.4586 std=0.0118 (n=15)

  Met standaard instellingen (trainable embeddings):

  embed-and-pool mean=0.5205 std=0.0170 (n=15)
  lstm mean=0.4582 std=0.0228 (n=15)

  trainable params - dropout 0.1 -> 0.5, hidden size 50 -> 200, epochs 20 -> 35

  embed-and-pool mean=0.5545 std=0.0198 (n=15)

  MLP ------- dropout 0.1 -> 0.5, hidden size 50 -> 200, epochs 20 -> 60
  encoder --- dropout -> 0,2, hidden dim 50 -> 100

  lstm mean=0.5259 std=0.0247 (n=15) -- trainable params
  lstm mean=0.5581 std=0.0156 (n=15) -- frozen params

  (project 1: ~0,55)

## let op

- in project2.ipynb gebruiken we trained_emb_model voor analyse aan het begin maar gebruik later isear_emb_model om te trainen (isear_emb_model is het emb tabel met alleen isear tokens)

## Goede punten voor discussie

- De tokenizer aangepast zodat koppeltekens '-' vervangen worden door spaties ' ' zodat woorden met koppeltekens niet de embedding <unk> krijgen. Verder meer dan 99 procent coverage!

- In onze embedding tabel zijn alle tokens verwijderd die niet in de ISEAR dataset voorkomen. (embed and pool nu 3x sneller ong over 3 seeds 5 folds en lstm 15% sneller)
  -> resultaten zijn overigens natuurlijk hetzelfde

## TO DO

- Smalle tweaks maken om te zien wat een beetje potentie heeft om effect te hebben (zoals rnn trainen, lijkt nog niet te zijn geconvergeerd)

- Vervolgens een groot trainingsscript maken zodat we alle configuraties op de gpu kunnen trainen
