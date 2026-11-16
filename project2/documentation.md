Nummers tussen haakjes verwijzen naar de stappen uit "Get to work".

## Al gedaan

- **(1) Embedding model:** GloVe 2024 wikigiga 50d geladen (in notebook) in embeddings.py staan ook andere die we nog kunnen testen

- **(1) Tokenizers:** `bow` (zonder leestekens/stopwoorden) en `rnn` (alles behouden, koppeltekens gesplitst)

- **(1) Coverage:** 99.3% (bow) en 99.7% (rnn) token coverage (~95% type coverage - unique token coverage) - Bij rnn werden tokens met koppeltekens als <unk> gezien -> een tokenizer gemaakt die met regular expressions de koppeltekens vervangt voor spaties (99.3 -> 99.7)

- **(2) Similariteit en buren:** emoties liggen dicht bij elkaar (0,53–0,87), tegengestelden ook.

- **Pipeline (tutorialinstellingen, 3 seeds × 5 folds):**
  Met standaard instellingen:

  embed-and-pool mean=0.3871 std=0.0125 (n=15)
  lstm mean=0.4586 std=0.0118 (n=15)

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
