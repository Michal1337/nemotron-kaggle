# huikang's reasoner style guide

Patterns the narrator must match so new (Alice/v3) CoT rationales blend with the existing 65 rule_found cryptarithm rationales in the training corpus.

## Universal scaffolding (every category)

### Opening — line 1 is category-specific, line 2 is constant

| Category | Line 1 |
|---|---|
| cryptarithm | `We need to infer the transformation rule from the examples.` |
| equation_numeric | `We need to infer the transformation rule from the examples.` |
| bit_manipulation | `We need to deduce the transformation by matching the example outputs.` |
| cipher | `We need to find the encryption mapping from the examples. It looks like a substitution cipher.` |
| gravity | `We need to determine the falling distance using d = k*t^2. Let me find k from the examples.` |
| unit_conversion | `We need to find a conversion rule that maps the inputs to outputs. Let me check if it's a linear factor.` |
| numeral | `This is an Arabic to Roman numeral conversion.` (no line 2) |

Line 2 (skip for numeral):

```
I will put my final answer inside \boxed{}.
```

Then a blank line.

### Closing — verbatim, every category

```
                                                ← blank line
I will now return the answer in \boxed{}
The answer in \boxed{–} is \boxed{<ANSWER>}
```

Note the **em-dash inside `\boxed{}`** — `\boxed{–}`. That's a huikang signature. Always present. Use `–` (`–`), not `-`.

## Bracket convention 【】

huikang wraps *every referenced character or expression* in **Chinese black-lenticular brackets** `【` and `】` (U+3010 and U+3011).

Examples from real traces:
- Full expression: `【:}-^}】 = 【:<】`
- Per-character: `【:】【}】【-】【^】【}】`
- Operator name: `【-】concatenation`
- Operator with classification: `The question operator is 【*】, which is concatenation.`
- Final intermediate: `Result: 【-92】`

**Rule**: any cipher symbol, any computed value, any operator character, any final intermediate gets wrapped. When showing per-character breakdown, wrap each character individually.

## Cryptarithm-specific structure (closest match to what our narrator outputs)

From `reasoning/60ed3f31.txt` (a real rule_found cryptarithm_deduce):

```
We need to infer the transformation rule from the examples.
I will put my final answer inside \boxed{}.

【:}-^}】 = 【:<】
  input: 【:】【}】【-】【^】【}】
  left:【:】【}】
  operator: 【-】
  right:【^】【}】
  output: 【:】【<】
  concatenation: 【:】【}】【^】【}】 mismatch
  reverse concatenation: 【^】【}】【:】【}】 mismatch
  operator: 【-】unknown

[... one section per example ...]

Question【@%*:\】
  input: 【@】【%】【*】【:】【\】
  left:【@】【%】
  operator:【*】
  right:【:】【\】

The question operator is 【*】, which is concatenation.

  concatenation(【@】【%】, 【:】【\】) = 【@】【%】【:】【\】
  output: 【@%:\】-> 【{@%:\}】

I will now return the answer in \boxed{}
The answer in \boxed{–} is \boxed{@%:\}
```

Indentation: **2 spaces** for sub-items. No tabs. Section breaks: single blank line.

## equation_numeric-style structure (we'll need this for ops beyond concat)

For cryptarithm problems whose op is `add/sub/mul/etc.` (most of v3's 142 new solves), borrow equation_numeric.py's style — since after cipher cracking, the underlying problem IS equation_numeric.

From `reasoning/b655eee9.txt` (a real rule_found equation_numeric_deduce):

```
We need to infer the transformation rule from the examples.
I will put my final answer inside \boxed{}.

Examples:
  16-71 = -44
  47-64 = -82
  37+79 = 171

The inputs are 16, 71, 47, 64, 37, 79

The outputs are -44, -82, 171
Some outputs have the operator symbol as prefix 【-】.
We now consider the outputs to be -44, -82, 171
We will add back the operator prefix if our answer is negative.

Looking at the input of the examples
16-71 -> -
47-64 -> -
37+79 -> +

The operators
-
+

Looking at the question
76-83 -> -
The question operator is found in the examples.

Looking at operator 【-】 [16-71 = -44, 47-64 = -82]:
  Trying common operations reversed operands [16->61 71->17, 47->74 64->46] and reversed result [expected (61,17)->-44, (74,46)->-82]:
    addition f(61, 17) = 61 + 17 = 78 wrong
    [... many candidate operations tried per line ...]
    negated absolute difference f(61, 17) = -|61 - 17| = -44 correct, actions: reversed operands, reversed result, negated absolute difference

[... more "Trying X operations" paragraphs until match ...]

Applying to 76-83:
  reversed operands [76->67, 83->38] and reversed result
  negated absolute difference f(67, 38) = -|67 - 38| = -29 -rev-> -92
  Result is negative - we add back the operator prefix 【-】: -92 -> 【-92】
  Result: 【-92】

I will now return the answer in \boxed{}
The answer in \boxed{–} is \boxed{-92}
```

### Key sub-patterns

**Per-candidate verification line**:
```
<op_name> f(<a>, <b>) = <expr> = <intermediate> = <raw_result> -rev-> <final> <status>
```
where:
- `<status>` ∈ `match` / `wrong` / `correct, actions: ...`
- `<expr>` is symbolic, e.g. `4*7 + 6*4`
- `<intermediate>` is the evaluated form, e.g. `28 + 24`
- `-rev->` only present if `rev_res=True`

**Transform header line**:
```
Trying common operations reversed operands [16->61 71->17, ...] and reversed result [expected (61,17)->-44, ...]:
```
4 transform combos in order: `(rev_ops=T, rev_res=T)`, `(F, F)`, `(T, F)`, `(F, T)` per equation_numeric.py lines 457-462.

**Format-handling lines** (when output has sign prefix/suffix):
```
Some outputs have the operator symbol as prefix 【X】.
We now consider the outputs to be ...
We will add back the operator prefix if our answer is negative.
```
And on application:
```
Result is negative - we add back the operator prefix 【X】: -92 -> 【-92】
Result is non-negative, no prefix needed: 【47】
```

## What our narrator needs to emit (for cryptarithm via Alice/v3)

Since cryptarithm = encrypted equation_numeric, the structure should be:

```
We need to infer the transformation rule from the examples.
I will put my final answer inside \boxed{}.

[CIPHER CRACK SECTION]
Examples in cipher:
  【cipher_input_1】 = 【cipher_output_1】
  ...

Inferred symbol-to-digit mapping:
  【sym1】 = 1
  【sym2】 = 2
  ...

Decoded examples:
  16-71 = -44
  47-64 = -82
  ...

[FROM HERE, USE equation_numeric.py STYLE ON DECODED DIGITS]
The inputs are 16, 71, 47, 64
The outputs are -44, -82
...
[etc — full equation_numeric trace on decoded form]

[RE-ENCODE SECTION]
Applying to 【ciphered_question】 (decoded: 76-83):
  [equation_numeric application steps]
  Result: 【-92】  (using digits)

Re-encoding to cipher symbols:
  -92 -> 【cipher_sym_a】【cipher_sym_b】【cipher_sym_c】
  (using inverse mapping)

I will now return the answer in \boxed{}
The answer in \boxed{–} is \boxed{<cipher_answer>}
```

## Concrete narrator implementation checklist

When converting one `investigations/cryptarithm_*/correct/<pid>.txt` (terse v3 or Alice format) to `reasoning/<pid>.txt` (huikang-style CoT), produce:

- [ ] Line 1: `We need to infer the transformation rule from the examples.`
- [ ] Line 2: `I will put my final answer inside \boxed{}.`
- [ ] Blank line.
- [ ] "Examples in cipher:" section listing each example with `【】`-wrapped expressions
- [ ] "Inferred symbol-to-digit mapping:" with each `【sym】 = digit` (sorted by symbol)
- [ ] "Decoded examples:" showing bare-digit form
- [ ] Equation_numeric-style operator analysis on decoded form (we have it from Alice's `details` dict)
- [ ] "Applying to <question>:" section with substitutions and computation
- [ ] "Result: 【<decoded_answer>】"
- [ ] "Re-encoding to cipher symbols:" with per-digit translation back via the inverse mapping
- [ ] Blank line.
- [ ] `I will now return the answer in \boxed{}`
- [ ] `The answer in \boxed{–} is \boxed{<cipher_answer>}` (em-dash!)

## Anti-checklist (don't do these)

- ❌ Don't use ASCII brackets `[` `]` for character wrapping. Use `【` `】`.
- ❌ Don't use regular dash `-` in the literal `\boxed{–}`. Use em-dash `–`.
- ❌ Don't add markdown headers (`##`, `###`) — huikang uses plain text + indentation.
- ❌ Don't add closing remarks ("I hope this helps", "let me know if...") — end immediately after the `\boxed{}` line.
- ❌ Don't bold/italicize anything.
- ❌ Don't number examples ("1.", "2.") in the main body — show them as `【input】 = 【output】`.
- ❌ Don't use "confidence:" sections — these appear only in LLM-narrated investigations, never in huikang's reasoner output.
- ❌ Don't mix categories — cryptarithm reasonings only show cryptarithm-format work, not generic prose.

## Token-length expectations

For our 197-problem cryptarithm extension, target rationale length similar to existing rule_found cryptarithms:

| Category | Median length (chars) | p95 |
|---|---:|---:|
| cryptarithm_deduce (current rule_found) | ~1,800 | ~2,200 |
| equation_numeric_deduce | ~10,500 | ~11,500 |

Our narrated cryptarithm will sit between these — probably ~3,000-8,000 chars because we have the cipher crack section + equation_numeric body. Under the 8,192-token TOKEN_LIMIT cutoff at corpus.py is the target.
