# Introduction

Codealign is a tool for evaluating neural decompilers that computes equivalence between two input functions at the instruction level.
The intended use case is comparing the predictions of a neural decompiler with a reference correct answer like the original source code.

This work was introduced in [Fast, Fine-Grained Equivalence Checking for Neural Decompilers](https://arxiv.org/abs/2501.04811)

# Installation

Codealign is a python package.
```
git clone https://github.com/squaresLab/codealign.git
cd codealign
pip install .
```
then optionally
```
python -m unittest
```

# Usage

```python
from codealign import align, Alignment

prediction = """
int write_response(int fd, char *buf, int len) {
	int	i;
	for (i = 0; i < len; i += len) {
		if ((i = write(fd, buf + i, len - i)) <= 0)
			return 0;
	}
    return 1;
}
"""

reference = """
int write_response(int fd, char *response, int len) {
	int	retval;
	int	byteswritten = 0;
	while (byteswritten < len) {
		retval = write(fd, response + byteswritten, len - byteswritten);
		if (retval <= 0) {
			return 0;
		}
        byteswritten += retval;
	}
    return 1;
}
"""

alignment: Alignment = align(prediction, reference, 'c', partial_loops=True)

print(alignment)
```
Will yield
```
Alignment(candidate=write_response, reference=write_response)

  %1 = phi 0 %7
  %1 = phi 0 %8

  %2 = < %1 len
  %3 = < %1 len

  loop %2
  loop %3

  %7 = + %5 len

  %3 = + buf %1
  %4 = + response %1

  %4 = - len %1
  %5 = - len %1

  %5 = write(fd, %3, %4)
  %6 = write(fd, %4, %5)

  %6 = <= %5 0
  %7 = <= %6 0

  if %6
  if %7

  return 0
  return 0

  return 1
  return 1

  %8 = + %1 %6
```
Equivalent instructions are grouped together.

Alignment objects and be interacted with programmatically via several methods.
#### IR representations
```python
alignment.candidate_ir
alignment.reference_ir
```
These allow for access to individual functions in terms of codealign's internal representation.

#### Alignment List Representation
```python
alignment.alignment_list
```
Represents the alignment as pairs of instructions in the order `(candidate_instruction, reference_instruction)`.
If an instruction does not align with anything the corresponding value will be `None`.
Except in injective mode, an instruction can occur in more than one pair if it aligns with multiple other instructions.

#### Alignment Lookup Representation
```python
alignment[instruction] # read-only
```
Returns the instruction(s) with which a given instruction is aligned.
Instructions in codealign IR can be found in the `.candidate_ir` and `.reference_ir` attributes.


## Object Model

Codealign describes code in term of an internal object model.
These can be imported from `codealign.ir`.
The codealign object model includes, but is not limited to
- `Function`
- `BasicBlock`
- `SSAOperator`
- `VarOperator`
- `Variable`

#### Accessing the Original AST
Where possible, codealign provides references to the `tree-sitter` AST nodes from which a given instruction was derived.
To access this, use
```
from codealign.ir import SSAOperator
instruction: SSAOperator
instruction.ast_node
```


