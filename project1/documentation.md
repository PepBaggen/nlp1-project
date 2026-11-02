# Documentation

## Dataset: Isear (emotions, 7 classes)

## Tokenizer

Train BPE on ISEAR (wss is de corpus te klein en kunnen we dit in eerste instantie doen
en dan vergelijken met een tokenizer die op een soortgelijke (dus een teks over hetzelfde 
onderwerp bijvoorbeeld) maar veel grotere corpus getraind is.)


## Current Issues:

Heel erg overfitten door kleine corpus. Moeten we wat op verzinnen. Bijv zie ## tokenizer

W

-----------------------------------------------------------------------------------------------------------
## Checklist

What **we expect you to do**:

- [] Compare different systems of text classification for your choice of data/task.
- Your comparisons should cover:
    - [ ] the two representation spaces in the tutorial (token indicators and bag-of-words) and an additional one of your choice.
    - [ ] the two kinds of architectures in the tutorial (log-linear and single-hidden-layer MLP) and an additional one of your choice (such as a modification of the MLP).
- [ ] Your comparisons must be based on cross-validation and a relevant metric (accuracy or else).
- [ ] You should make use of Bayesian estimation to help you analyse results in relation to a region of practical equivalence (ROPE), and you can motivate the ROPE size yourself.

What you need to **bring to the oral assessment**:
* You will be asked questions about the points above, you can bring a slide deck with up to five slides (one for each of the following topics _data and metrics_, _features_, _architectures_, _results_, _extra_); you _will not_ give a presentation, rather, these slides are to be thought of as "backup slides" that you can use to guide your answers to the questions the TA will ask you.
* You may also be asked questions about the tutorial, for those, you don't need to prepare any deliverable (a good way to prepare is to work through the tutorial and reflect on the questions there).

What you should **keep in mind** given that you will eventually write a blog post about your work throughout the course:
* Organise your code;
* Summarise your motivations and findings;
* Document your sources and any form of AI-assistance you use.