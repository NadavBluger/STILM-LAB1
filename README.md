# Lab 1-1: Byte-pair encoding tokenizers

## Goal

In this lab you build a simple tokenizer, to act as the first pre-processing step of text before it goes into a language model. You will implement a byte-level BPE tokenizer from scratch.

Before the lab, you should introduce yourself to tokenization in general, and specifically to BPE. 
Read one of these sources:
1. The introduction and pages 5--15 from [this tokenization survey](https://www.alphaxiv.org/pdf/2609.tokenization-survey-modern-nlp).
2. Read chapter 2 (up to 2.5) in Dan Jurafsky's ["Speech and Language Processing"](https://web.stanford.edu/~jurafsky/slp3/2.pdf). This book (available through https://web.stanford.edu/~jurafsky/slp3/) is a great resource for basic and more advanced content in NLP and language modeling. When the course will align with content from the book, we will point to relevant chapters and highly recommend reading them.

## Initial Environment Setup

This course uses Python 3.11.9. Create the environment with either conda:

```
conda env create -f environment.yml
conda activate lm-course
```

or uv:

```
uv venv         
source .venv/bin/activate
uv pip install -r requirements.txt
```

Either way, check what you got before starting:

```
python --version        # should print 3.11.9
```


## Setup
In addition to the environmnet installation files, you have been given multiple files:
```
 agents.md 		        # A guidelines file for coding agents - keep it included with the code at all times
 README.md              # This file, containing instructions and questions
 lab-1-1-tokenizers.py  # The code file which you will edit
 test_1_1.py            # A test file to run after each TODO for validation
```

To run the test file, run it as a python script, either with a TODO number or without arguments to run all tests:

```
python test_1_1.py 1      # check TODO 1
python test_1_1.py        # check everything
python lab-1-1-tokenizers.py
```

The tests in this lab **run in a few seconds** and train only on tiny in-memory corpora. 
Note that these tests are basic and non-comprehensive. The exercise will be graded using unseen tests. 

No GPU is needed for this lab.


## Coding agents usage guidelines
AI is able to solve many parts of this and following labs autonomously. 
Doing this **will** make it harder for you to deeply engage with and learn from the course material (and will likely lead to lower grades on the quizzes and exam). 
Thus, we expect you to follow these guidelines regarding AI usage in this course:

1. The use of AI tools is permitted for answering high-level conceptual questions or for providing low-level programming documentation like function signatures and library APIs.
2. You should not use AI tools (including coding agents, chat interfaces, and AI autocomplete) to generate any piece of code, and all your code should be handcrafted by you directly.

Concretely, students are expected to do the following:
* To do so, each assignment includes an `agents.md` file. Make sure your coding agents reads this file at the beginning of the session (if it doesn't do so automatically) and follows the instructions in it. This file should not be modified in any way.
* If you are using online chat interfaces for assistance, copy the contents of the `agents.md` file to the system prompt or to the start of **each** conversation.
* AI autocomplete should be disabled in your IDE (non-AI autocomplete, e.g., autocompleting function names, is permitted). 
For example, in VSCode, open the user settings JSON (Ctrl/Cmd + Shift + P --> `Open User Settings (JSON)`) and paste the following:

```
"[python]": {
    // Kill universal AI inline text ghosting
    "editor.inlineSuggest.enabled": false,

    // Disable GitHub Copilot for Python specifically
    "github.copilot.enable": {
        "python": false
    },
    "github.copilot.nextEditSuggestions.enabled": false,

    // Keep Pylance standard autocomplete active
    "python.languageServer": "Pylance",
        "editor.quickSuggestions": {
        "other": "on",
        "comments": "off",
        "strings": "on"
    }
},

"[markdown]": {
    // Kill AI text completions
    "editor.inlineSuggest.enabled": false,

    // Disable GitHub Copilot for Markdown specifically
    "github.copilot.enable": {
        "markdown": false
    },
    "github.copilot.nextEditSuggestions.enabled": false,
}
      
```

A good rule of thumb is to ask yourself whether your request would be complied with in office hours (and asking to get the solution in office hours does not comply). 
If not, then the request is likely not allowed under the AI policy.



## TODOs

TODOs 1-8 are marked `# TODO COMPLETE (N)` in `lab-1-1-tokenizers.py`. Each one
is specified by the docstring of the function it sits in. Do them in order: the
later ones build on the earlier ones.

### TODO 1: Running the pipeline by hand

Before you write any actual tokenization code, you will *manually* perform some BPE steps, to make sure you understand the algorithm.

TODO 1 is the three functions `manual_pretokenization`, `manual_toy_bpe` and `manual_encode`. Each one covers a fixed set of examples, given inside the function.

These three functions are different from every other TODO in the lab:

* You work the answer out **by hand**, and return it from the function as a literal value (a list you type out). Do not write code that computes it.
* The answer covers **only the given examples**, not the general case. `manual_pretokenization` returns a list of lists: the first inner list is the pre-tokens of the first example, the second inner list is the pre-tokens of the second example, and so on. It does not pre-tokenize anything else.

For `manual_toy_bpe`, run BPE training on the small corpus given in the function: split the corpus into pre-tokens, split each pre-token into bytes, count the adjacent pairs across the whole corpus, merge the most frequent pair, and repeat. The returned output should be a vocabulary of $n=10$ tokens, ordered by insertion order.

For `manual_encode`, go the other way: given a text, a vocabulary and a list of merges (all given in the function, and unrelated to the corpora of the other two functions), encode the text into token IDs. Split the text into pre-tokens, split each pre-token into single characters, apply the merges in the given order, and map the resulting pieces to their IDs. Note that a merge is applied everywhere it occurs before the next merge is considered, and that a merge may end up not applying at all.

```
python test_1_1.py 1
```

### TODO 2: Counting pretokens

The first step of the tokenization process is to split a text to its pre-tokens (which are just words).
You are given `bpe_pretokenize`, a function that does exactly that: it splits a text into pre-tokens (words and single punctuation characters).

The next step would be to count how many times each pre-token appears, for more efficient counting of the frequency of each byte-pair in later steps.
For that, you will implement `count_pretokens`, using `bpe_pretokenize` to convert a text to its pre-tokens and `str_to_bytes` to properly split a pre-token to bytes. 

Note: the function `count_pretokens` returns a dictionary whose keys are **split** pre-tokens (i.e. tuples of bytes). This is just our implementation choice to pass on **tuples of bytes** to the next steps.

Once you're done implementing, test your code:
```
python test_1_1.py 2
```

Our pre-tokenization rule is especially simple. Look at the regex used by GPT-4: [`tiktoken_ext/openai_public.py`](https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py#L106). Note that the pre-tokenization pattern gets complex in a real tokenizer, and which edge cases it handles that our pre-tokenizer does not (e.g. contractions, digit grouping, trailing whitespaces).

### TODO 3: Counting pairs

Now that we have an appearance count for each pre-token, we want to find the most frequent pairs (BPE == byte-**pair**, remember?). 
Implement `get_pair_counts`. To test the function:
```
python test_1_1.py 3
```

### TODO 4 & 5: Merging pairs

The next step in BPE is to merge the most frequent byte pairs to a single, new token.
Implement `merge_pair_in_token` and `merge_pair`.

```
python test_1_1.py 4 5
```

### TODO 6: BPE merge learning

Now let's connect all of the previous steps together.
Implement the full procedure to learn the pair merging from a corpus, in `SimpleBPETokenizer.train`.
Use the functions implemented in previous steps, as well as `add_piece`.

```
python test_1_1.py 6
```

### TODO 7 & 8: Encoding a text as tokens and decoding it back

After training your simple BPE tokenizer, you'll want to encode a given text to a list of tokens, just like you did in `manual_encode` (but not manually, of course); you'll also want to be able to decode a list of tokens to a string.

Implement `SimpleBPETokenizer.encode` to convert a text to a list of token IDs and `SimpleBPETokenizer.decode` for the reverse process of turning a list of token IDs back into text.

```
python test_1_1.py 7 8
```

### TODO 9: Vocabulary-size tradeoff

**Question:** The size of the tokenizer vocabulary inherently has a trade-off. What is the tradeoff between? Which advantages and disadvantages are there for larger vocabulary sizes?

**Answer:** 


After writing your answer, run the full script:

```
python lab-1-1-tokenizers.py
```

Read the `# Vocabulary-size tradeoff` section in the output. Was your hypothesis correct? Did you miss anything? 


## TODO 10: Arithmetic tokenization
**Question:** In some tokenization schemes, numbers that are written in digits (e.g. "74691") get tokenized in a special way. 
Some tokenize per-digit while others tokenize entire numbers up to a certain limit. How do you think this design choice affects follow-up training of the language model? Why?

**Answer:**


## Lab debrief
**Question:** We’re interested in **any** thoughts you have about this lab so that we can improve this lab for later years, and to inform later labs for this year. Please list any issues that arose or comments you have to improve the lab. Useful things to comment on include the following, but you’re not restricted to these:
* Was the lab too long or too short?
* Were the instructions clear?
* Was it clear (at least after you completed the lab) what the points of the exercises were?

**Answer:**

## Submission
Collect all of the files in the given directory into a **flat** zip file (**NO** internal directories) and submit it for grading via Gradescope.

Note: The zip file **MUST** be in the following format:
```
lab_1_1_name1_name2.zip
├── lab-1-1-tokenizers.py
└── README.md
```

If the zip won't be in this exact format, the autograder will fail and you will get **no points** for this lab.

## Further reading

- Additional tokenization algorithms: [WordPiece](https://huggingface.co/learn/llm-course/en/chapter6/6), [SentencePiece](https://arxiv.org/abs/1808.06226)
- Fast tokenizers: [openai/tiktoken](https://github.com/openai/tiktoken), [marcelroed/gigatoken](https://github.com/marcelroed/gigatoken)