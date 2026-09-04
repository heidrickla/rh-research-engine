from __future__ import annotations

import re

import sympy as sp
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

from .functions import APPLIED_FUNCTIONS, STANDALONE_CONSTANTS
from .models import EquationKind, ExtractedEquation

_TRANSFORMS = standard_transformations + (convert_xor, implicit_multiplication_application)

#: LaTeX commands with a faithful algebraic meaning. Anything outside this set
#: is refused rather than normalized away.
#:
#: The previous normalizer deleted every backslash and rewrote braces to
#: parentheses, so command names and binders degraded into ordinary identifier
#: text that implicit multiplication then split into products. `\frac{1}` became
#: the literal constant 0; `\lim_{X\to\infty} R_q(X) = 0` became the algebraic
#: equation `R_q*X*lim_Xtooo = 0`. Both were reported as clean parses.
_ALGEBRAIC_COMMANDS = {
    r"\cdot": "*",
    r"\times": "*",
    r"\div": "/",
    r"\infty": "oo",
    r"\pi": "pi",
    r"\left": "",
    r"\right": "",
    r"\,": " ",
    r"\;": " ",
    r"\!": "",
    r"\quad": " ",
    r"\qquad": " ",
    # Relations. `_split_relation` already understands the ASCII forms.
    r"\le": "<=",
    r"\leq": "<=",
    r"\ge": ">=",
    r"\geq": ">=",
    r"\neq": "!=",
    # Euler-Mascheroni, spelled unambiguously -- see STANDALONE_CONSTANTS.
    r"\EulerGamma": "EulerGamma",
    # Quantities this engine defines because no standard notation exists
    # for them, spelled out for the same reason `\EulerGamma` is: a
    # one-letter name would collide with something, and a formula whose
    # terms the corpus cannot define is a formula nothing can check.
    r"\RedhefferDet": "RedhefferDet",
    r"\FareyCount": "FareyCount",
    r"\FareyDeviation": "FareyDeviation",
    r"\arg": "arg",
    # Real and imaginary parts are ordinary SymPy functions.
    r"\Re": "re",
    r"\Im": "im",
    r"\operatorname{Re}": "re",
    r"\operatorname{Im}": "im",

    # Functions with an exact SymPy counterpart. Safe only because applied
    # names now become Functions rather than Symbols -- see `_smart_locals`.
    # Without that, `\log(x)` would have parsed as `log*(x)`.
    r"\log": "log",
    r"\ln": "log",
    r"\exp": "exp",
    r"\sqrt": "sqrt",
    r"\sin": "sin",
    r"\cos": "cos",
    r"\tan": "tan",
}

#: Commands that bind a variable or denote an operator with a domain. These are
#: refused outright: their bounds are exactly the information that separates a
#: finite check from an asymptotic theorem, and this parser has no way to carry
#: them into a SymPy expression faithfully.
#: Commands that bind a variable and are NOT rewritten anywhere above. Each
#: one that gains a rewrite leaves this set.
_BINDER_COMMANDS = {
    r"\oint", r"\iint", r"\limsup", r"\liminf",
    r"\exists", r"\notin",
    r"\subset", r"\subseteq", r"\mapsto", r"\bigcup", r"\bigcap",
}

#: Greek letters and common symbol names, mapped to bare identifiers.
_SYMBOL_COMMANDS = {
    r"\alpha", r"\beta", r"\gamma", r"\delta", r"\epsilon", r"\varepsilon", r"\zeta",
    r"\eta", r"\theta", r"\vartheta", r"\iota", r"\kappa", r"\lambda", r"\mu", r"\nu",
    r"\xi", r"\rho", r"\varrho", r"\sigma", r"\tau", r"\upsilon", r"\phi", r"\varphi",
    r"\chi", r"\psi", r"\omega", r"\Gamma", r"\Delta", r"\Theta", r"\Lambda", r"\Xi",
    r"\Pi", r"\Sigma", r"\Upsilon", r"\Phi", r"\Psi", r"\Omega",
}

#: Greek names that collide with a SymPy callable (``beta``, ``gamma``,
#: ``zeta``, ``Lambda`` ...) or with a Python keyword (``lambda``). Left
#: unmapped, `\beta` parses as SymPy's Beta *function* and raises on a missing
#: argument -- an accidental rejection that would silently become an accidental
#: acceptance the moment the surrounding expression changed shape.
#: Names renamed on the way in, and the only reason each one is.
#:
#: This table used to hold every Greek letter that collided with a SymPy name,
#: which meant `\zeta(s)` was stored as an undefined `zeta_` -- the right shape
#: attached to no meaning. The collisions are settled by whether the source
#: wrote the name APPLIED, so the only entries left are the ones with a real
#: reason: `lambda` is a Python keyword and cannot be an identifier at all, and
#: SymPy's `Lambda` and `Chi` mean something else entirely from the constant
#: and the character this literature writes with those letters.
_SYMBOL_ALIASES = {
    "lambda": "lambda_",
    "Lambda": "Lambda_",
    "chi": "chi_",
    "Chi": "Chi_",
    "Order": "Order_",
}


#: Names that must reach SymPy's own object rather than a stand-in.
#:
#: `pi` was in the forced-Symbol set with every other Greek letter, so it
#: parsed as an undefined variable that happened to be spelled "pi". Every
#: formula containing it was quietly wrong: `N(T) = (T/2pi)log(T/2pi) - T/2pi`
#: is not the Riemann-von Mangoldt formula if pi is a free variable, and
#: nothing downstream could tell -- the parse was clean and the printed form
#: was identical.
#:
#: `oo` is the same defect for infinity, which matters the moment a bounded sum
#: runs to it, and `Sum`/`Product` are the same again for the binder rewrite:
#: shadowed, they became opaque functions named "Sum" over a symbol named "oo".
#: Constants. These double as function names in this domain -- `\pi(x)` is the
#: prime-counting function -- so an APPLIED occurrence is treated as a function
#: and a standalone one as the number.
_SYMPY_CONSTANTS = frozenset({"pi", "oo", "zoo", "nan"})

#: Classes the binder rewrite emits. Always applied, and never a function of
#: this package's invention -- shadowing them turned a real summation into an
#: opaque `Function('Sum')` that printed identically.
_SYMPY_CLASSES = frozenset(
    {
        "Sum", "Product", "Derivative", "Integral", "Limit", "NthPrime",
        "Mod", "Implies", "Eq", "Ne", "Lt", "Le", "Gt", "Ge", "Subs",
    }
)

#: Relational constructors. A normalized equation is re-read downstream, and
#: `Eq` in that string has to stay SymPy's `Eq` -- shadowed, the whole relation
#: collapses to an opaque two-argument function and every fingerprint taken
#: from it describes a different object than the one that was validated.
_SYMPY_RELATIONS = frozenset(
    {
        "Eq", "Ne", "Lt", "Le", "Gt", "Ge",
        "Equality", "Unequality",
        "StrictLessThan", "LessThan",
        "StrictGreaterThan", "GreaterThan",
    }
)

_SYMPY_PROVIDED = _SYMPY_CONSTANTS | _SYMPY_CLASSES | _SYMPY_RELATIONS

#: Names SymPy defines that this package must NOT let it supply.
#:
#: `O` is SymPy's `Order`, a germ at a point -- and at *zero* by default. Every
#: asymptotic bound in the literature is a statement about x -> infinity, so
#: resolving `O` to `Order` does not merely lose information, it asserts the
#: opposite regime. It is also absorbing: `Add.flatten` asks whether the other
#: terms are dominated by the O-term and folds them in, so
#: `pi(x) = li(x) + O(sqrt(x)log x)` would have SWALLOWED `li(x)` -- von Koch's
#: theorem, reduced to a statement about nothing. And `O(x**(epsilon + 1/2))`
#: took `epsilon` for a second limit variable, making the Mertens criterion a
#: germ as (x, epsilon) -> (0, 0).
#:
#: An undefined `Function('O')` asserts nothing. That is the honest reading:
#: this package does not implement asymptotic comparison, and a term it cannot
#: interpret must stay inert rather than rewrite the formula around it.
_SYMPY_REFUSED = frozenset({"O"})


def _symbol_locals() -> dict[str, sp.Symbol]:
    """Force every Greek identifier to be a plain Symbol, never a function.

    Except the ones SymPy defines as constants -- see `_SYMPY_PROVIDED`. A
    Symbol named `pi` is not pi.
    """
    names = {command[1:] for command in _SYMBOL_COMMANDS}
    names |= set(_SYMBOL_ALIASES.values())
    names -= _SYMPY_PROVIDED
    names -= set(APPLIED_FUNCTIONS)
    names -= set(STANDALONE_CONSTANTS)
    return {name: sp.Symbol(name) for name in names}


_WRAPPER_RE = re.compile(r"\\(?:operatorname|mathrm|mathscr|mathcal|mathbf|text)\{([^{}]+)\}")
_FRAC_RE = re.compile(r"\\frac\{([^{}]+)\}\{([^{}]+)\}")
#: `e^{...}` or `e^(...)` -- Euler's number raised to a group.
_EULER_POW_RE = re.compile(r"(?<![A-Za-z0-9_])e\^\{([^{}]+)\}|(?<![A-Za-z0-9_])e\^\(([^()]+)\)")
_SQRT_RE = re.compile(r"\\sqrt\{([^{}]+)\}")
_BARE_FRAC_RE = re.compile(r"\\d?frac(?!\{[^{}]+\}\{[^{}]+\})")
_COMMAND_RE = re.compile(r"\\[A-Za-z]+|\\.")


class LatexRejected(ValueError):
    """Raised when LaTeX cannot be normalized without losing meaning."""


def _strip_math_delimiters(text: str) -> str:
    text = text.strip()
    pairs = [("$$", "$$"), ("\\[", "\\]"), ("\\(", "\\)"), ("$", "$")]
    for left, right in pairs:
        if text.startswith(left) and text.endswith(right):
            return text[len(left) : -len(right)].strip()
    return text


def _check_braces_balanced(text: str) -> None:
    depth = 0
    for ch in text:
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                raise LatexRejected("unbalanced brace: unexpected '}'")
    if depth != 0:
        raise LatexRejected(f"unbalanced brace: {depth} group(s) left open")


#: `\sum_{i=a}^{b}` / `\prod_{i=a}^{b}`, with the braces optional on the bound.
_BOUNDED_BINDER_RE = re.compile(
    r"\\(sum|prod)"
    r"_\{([A-Za-z][A-Za-z0-9_]*)\s*=\s*([^{}]+)\}"      # _{i = lower}
    r"\^(?:\{([^{}]+)\}|([^\s{}]+))"                     # ^{upper} or ^upper
)

#: A binder whose index carries no bounds at all -- `\sum_p`, `\prod_{p}`.
_UNBOUNDED_BINDER_RE = re.compile(r"\\(sum|prod)_\{?([A-Za-z][A-Za-z0-9_]*)\}?(?!\s*=)")


def _top_level_additive(text: str) -> bool:
    """True if `text` has a `+` or `-` outside every bracket."""
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char in "+-" and depth == 0 and index > 0:
            previous = text[index - 1]
            if previous not in "*/^(+-eE":
                return True
    return False


#: Relation commands, which end an operator's body the way `=` does.
_RELATION_COMMANDS = (r"\le", r"\leq", r"\ge", r"\geq", r"\neq", r"\ne")


def _body_end(text: str) -> int:
    r"""Where an operator's body stops, by ordinary precedence.

    `\sum_{n=1}^{N} a_n + b` is `(\sum a_n) + b`. A binder, an integral, a
    limit and a derivative all bind tighter than addition, and every source
    that writes them relies on it -- this is the reading, not a guess between
    two readings. The body runs to the first top-level `+` or `-`, or to the
    relation separating the two sides of the statement, since no body spans
    an `=`.
    """
    depth = 0
    index = 0
    while index < len(text):
        char = text[index]
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
            if depth < 0:
                return index
        elif depth == 0:
            if char == "\\":
                for command in _RELATION_COMMANDS:
                    tail = text[index + len(command) : index + len(command) + 1]
                    if text.startswith(command, index) and not tail.isalpha():
                        return index
            elif char in "=<>,":
                # A comma too: a binder nested inside a call cannot span the
                # comma separating that call's arguments. Without this,
                # `Eq(Sum(f, (n,1,oo)), g)` had its body run on through the
                # comma and SymPy was handed `g` as a limit.
                return index
            elif char in "+-" and index > 0 and text[index - 1] not in "*/^(+-eE":
                return index
        index += 1
    return len(text)


def _split_leading_term(text: str) -> tuple[str, str]:
    """`(body, rest)` split where `_body_end` says the body stops."""
    end = _body_end(text)
    return text[:end].strip(), text[end:]


def _rewrite_bounded_binders(text: str) -> str:
    r"""Turn `\sum_{n=1}^{\infty} f(n)` into `Sum(f(n), (n, 1, oo))`.

    A bound written in the source is exact and SymPy carries it faithfully.
    The body is the following term, by the precedence every source assumes --
    see `_body_end`.
    """
    while True:
        match = _BOUNDED_BINDER_RE.search(text)
        if match is None:
            return text
        kind, index, lower, braced_upper, bare_upper = match.groups()
        upper = braced_upper if braced_upper is not None else bare_upper
        body, trailing = _split_leading_term(text[match.end() :].strip())
        if not body:
            raise LatexRejected(f"\\{kind} has bounds but no body to sum over")
        head = "Sum" if kind == "sum" else "Product"
        text = (
            text[: match.start()]
            + f"{head}({body}, ({index}, {lower.strip()}, {upper.strip()}))"
            + trailing
        )


#: `\prod_p`, `\sum_{p}`, `\prod_{p \in \mathbb{P}}`: a binder over the primes.
_PRIME_BINDER_RE = re.compile(
    r"\\(sum|prod)\s*_\s*(?:\{\s*([A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:\\(?:text|mathrm)\s*\{[^{}]*\}|\\in\s*(?:\\mathbb\s*\{\s*P\s*\}|P))?\s*\}"
    r"|([A-Za-z][A-Za-z0-9_]*))(?!\s*[\^_])"
)

#: The index variable a prime-indexed binder is rewritten onto.
_PRIME_POSITION = "k_"


def _rewrite_prime_binders(text: str) -> str:
    r"""Turn `\prod_p f(p)` into `Product(f(NthPrime(k_)), (k_, 1, oo))`.

    The product over the primes is the product over POSITION of a term in the
    k-th prime, which is the same statement written with an index SymPy can
    carry. `NthPrime` is a real function -- it evaluates to 2, 3, 5, ... on
    concrete indices -- so the Euler product is stored as something checkable
    rather than as an opaque token.
    """
    while True:
        match = _PRIME_BINDER_RE.search(text)
        if match is None:
            return text
        kind, braced_index, bare_index = match.groups()
        index = braced_index if braced_index is not None else bare_index
        body, trailing = _split_leading_term(text[match.end() :].strip())
        if not body:
            raise LatexRejected(f"\\{kind} over the primes has no body")
        over_position = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(index) + r"(?![A-Za-z0-9_])",
            f"NthPrime({_PRIME_POSITION})",
            body,
        )
        head = "Sum" if kind == "sum" else "Product"
        text = (
            text[: match.start()]
            + f"{head}({over_position}, ({_PRIME_POSITION}, 1, oo))"
            + trailing
        )


#: The unambiguous absolute-value delimiters. `\left|` and `\right|` say which
#: end they are; a bare `|` is classified by position instead.
_ABS_OPEN_RE = re.compile(r"\\left\s*\||\\lvert")
_ABS_CLOSE_RE = re.compile(r"\\right\s*\||\\rvert")

#: Characters after which an expression is still expected, so a bar there is
#: an opening delimiter rather than an operator.
_OPERAND_POSITION = set("([{,+-*/^=<>|&")


def _bar_marks(text: str) -> list[tuple[int, str]]:
    """Classify every bare `|` as an opener, a closer, or a divides sign.

    A bar in operand position -- at the start, or after an operator or an
    opening bracket -- can only begin something. A bar in operator position
    closes the delimiter it is inside, and if there is nothing to close it is
    the divisibility sign: `p | n` is "p divides n", which is a statement
    about the two operands and not a delimiter at all.

    Position is what distinguishes them, so `|a| + |b|` and `p | n` both read
    correctly and neither has to be refused.
    """
    marks: list[tuple[int, str]] = []
    depth = 0
    for index, char in enumerate(text):
        if char in "([{":
            depth += 1
        elif char in ")]}":
            depth -= 1
        elif char == "|":
            head = text[:index].rstrip()
            operand = not head or head[-1] in _OPERAND_POSITION or head.endswith("\\")
            open_count = sum(1 for _, mark in marks if mark == "open")
            close_count = sum(1 for _, mark in marks if mark == "close")
            if operand:
                marks.append((index, "open"))
            elif open_count > close_count:
                marks.append((index, "close"))
            else:
                marks.append((index, "divides"))
    return marks


def _rewrite_bare_bars(text: str) -> str:
    """Rewrite bare `|` pairs to `Abs(...)`, and a lone bar to divisibility."""
    while True:
        marks = _bar_marks(text)
        if not marks:
            return text
        divides = next((i for i, mark in marks if mark == "divides"), None)
        if divides is not None:
            left = text[:divides].strip()
            right = text[divides + 1 :].strip()
            if not left or not right:
                raise LatexRejected("a `|` with nothing on one side of it")
            # `p | n` is exactly "n leaves no remainder mod p".
            return f"Eq(Mod({right}, {left}), 0)"
        stack: list[int] = []
        pair: tuple[int, int] | None = None
        for index, mark in marks:
            if mark == "open":
                stack.append(index)
            else:
                start = stack.pop()
                if not any(
                    other > start and other < index for other, _ in marks
                ):
                    pair = (start, index)
                    break
        if pair is None:
            if stack:
                raise LatexRejected("an absolute-value bar is never closed")
            return text
        start, end = pair
        body = text[start + 1 : end].strip()
        if not body:
            raise LatexRejected("empty absolute value")
        text = text[:start] + f"Abs({body})" + text[end + 1 :]


def _rewrite_absolute_value(text: str) -> str:
    r"""Turn absolute-value bars into `Abs(...)`.

    `\left|...\right|` and `\lvert...\rvert` name their own ends and nest, so
    they are matched innermost-first. Bare bars are classified by position --
    see `_bar_roles`.
    """
    out = text
    while True:
        close = _ABS_CLOSE_RE.search(out)
        if close is None:
            break
        opens = list(_ABS_OPEN_RE.finditer(out[: close.start()]))
        if not opens:
            raise LatexRejected(
                "absolute-value bar closes without opening: found "
                f"{close.group(0)!r} with no matching opener"
            )
        open_ = opens[-1]
        body = out[open_.end() : close.start()].strip()
        if not body:
            raise LatexRejected("empty absolute value")
        out = out[: open_.start()] + f"Abs({body})" + out[close.end() :]
    leftover = _ABS_OPEN_RE.search(out)
    if leftover is not None:
        raise LatexRejected(
            f"absolute-value bar opens without closing: {leftover.group(0)!r}"
        )
    return _rewrite_bare_bars(out)


#: `[` and `]` used as grouping. Sources write `d/ds [f(s)]` for the operand of
#: an operator, and the brackets carry no meaning beyond the grouping.
_BRACKET_TRANSLATION = str.maketrans({"[": "(", "]": ")"})


def _rewrite_brackets(text: str) -> str:
    """Square brackets group. Floor and ceiling have their own commands.

    `[x]` did mean the floor of x in older literature, which is why this is a
    rewrite and not a guess: `\\lfloor`/`\\lceil` are unambiguous and are
    handled separately, so a bracket that survives to here is grouping.
    """
    depth = 0
    for char in text:
        if char == "[":
            depth += 1
        elif char == "]":
            depth -= 1
            if depth < 0:
                raise LatexRejected("unbalanced bracket: unexpected ']'")
    if depth:
        raise LatexRejected(f"unbalanced bracket: {depth} group(s) left open")
    return text.translate(_BRACKET_TRANSLATION)


#: The operator half of a derivative: `\frac{d^n}{ds^n}`, `\frac{d}{ds}`, and
#: the `\partial` forms. The order is optional and defaults to one.
_DERIVATIVE_RE = re.compile(
    r"\\frac\s*\{\s*(?:d|\\partial)\s*(?:\^\s*(?:\{([^{}]*)\}|(\w+)))?\s*\}"
    r"\s*\{\s*(?:d|\\partial)\s*([A-Za-z][A-Za-z0-9_]*)"
    r"\s*(?:\^\s*(?:\{[^{}]*\}|\w+))?\s*\}"
)


def _balanced_group(text: str, start: int) -> tuple[str, int] | None:
    """The parenthesised group at `start`, and the index just past it."""
    if start >= len(text) or text[start] != "(":
        return None
    depth = 0
    for index in range(start, len(text)):
        if text[index] == "(":
            depth += 1
        elif text[index] == ")":
            depth -= 1
            if depth == 0:
                return text[start + 1 : index], index + 1
    return None


#: Any derivative operator, used only to catch what the rewrite missed.
_DIFFERENTIAL_RE = re.compile(
    r"\\frac\s*{\s*(?:d|\\partial)"
    r"|\\partial"
    r"|(?<![A-Za-z])d\s*/\s*d(?![A-Za-z])"
)


def _rewrite_derivatives(text: str) -> str:
    r"""Turn `\frac{d^n}{ds^n} f` into `Derivative(f, (s, n))`.

    A derivative is an operator and SymPy has one, so this is exact. The body
    is the following term, by the same precedence the binders use: a
    derivative binds tighter than addition, so `d/dx f + g` differentiates `f`
    alone. Li's `lambda_n` is defined by this notation and is why it exists.
    """
    while True:
        match = _DERIVATIVE_RE.search(text)
        if match is None:
            return text
        braced_order, bare_order, variable = match.groups()
        order = braced_order if braced_order is not None else bare_order
        body, trailing = _split_leading_term(text[match.end() :].strip())
        if not body:
            raise LatexRejected(
                f"\\frac{{d}}{{d{variable}}} has nothing to differentiate"
            )
        wrt = f"({variable}, {order})" if order else variable
        text = text[: match.start()] + f"Derivative({body}, {wrt})" + trailing


def _refuse_unrepresentable_notation(text: str) -> None:
    """Catch a derivative that no rewrite above consumed.

    Read as a quotient, `d^n/ds^n` becomes a product of symbols named `d` and
    `ds` -- a clean parse of nonsense. Anything still matching here has to
    stop rather than reach the algebra.
    """
    if _DIFFERENTIAL_RE.search(text):
        raise LatexRejected(
            "differential operator this parser did not rewrite. Reading it as a "
            "quotient would yield a product of symbols named `d` and `ds`"
        )



#: `\int` with optional bounds. Bounds may be braced or a single token.
_INTEGRAL_RE = re.compile(
    r"\\int\s*(?:_\s*(?:\{([^{}]*)\}|(\\[A-Za-z]+|\w+)))?"
    r"\s*(?:\^\s*(?:\{([^{}]*)\}|(\\[A-Za-z]+|\w+)))?"
)

#: The `dx` that closes an integrand, and names the variable it runs over.
_INTEGRATION_VARIABLE_RE = re.compile(r"\\?,?\s*\bd\s*([A-Za-z][A-Za-z0-9_]*)")

#: `\lim_{x \to a}`.
_LIMIT_RE = re.compile(
    r"\\lim\s*_\s*\{\s*([A-Za-z][A-Za-z0-9_]*)"
    r"\s*(?:\\to|\\rightarrow)\s*([^{}]*?)\s*\}"
)


def _top_level_spans(text: str):
    """Yield `(index, char)` for characters outside any bracket or brace."""
    depth = 0
    for index, char in enumerate(text):
        if char in "({":
            depth += 1
        elif char in ")}":
            depth -= 1
        elif depth == 0:
            yield index, char


def _find_integration_variable(text: str) -> tuple[int, int, str] | None:
    """Locate the `dx` closing an integrand, at the top level only.

    `\\int_0^1 \\frac{dt}{t}` has a `dt` inside the fraction that does not
    close anything, so nesting has to be respected or the integrand is cut in
    the wrong place.
    """
    for index, char in _top_level_spans(text):
        if char != "d":
            continue
        if index and (text[index - 1].isalnum() or text[index - 1] == "_"):
            continue
        match = _INTEGRATION_VARIABLE_RE.match(text, index - 1 if index else 0)
        match = _INTEGRATION_VARIABLE_RE.match(text[index:])
        if match is None:
            continue
        return index, index + match.end(), match.group(1)
    return None


def _sole_variable(body: str) -> str:
    """The one variable an integrand runs over, when it names exactly one.

    Applied names are excluded: in `f(u)` the integral is over `u`, not over
    `f`. When the integrand names several, the source really has left the
    variable to context and there is nothing to recover it from.
    """
    applied = {m.group(1) for m in _APPLICATION_RE.finditer(body)}
    names = [
        name
        for name in dict.fromkeys(_IDENTIFIER_RE.findall(body))
        if name not in applied and name not in _SYMPY_PROVIDED
    ]
    if len(names) == 1:
        return names[0]
    if not names:
        raise LatexRejected(
            "integral with no differential and a constant integrand, so nothing "
            "names the variable it runs over"
        )
    raise LatexRejected(
        "integral with no differential over an integrand naming "
        f"{', '.join(names)}. Which one it runs over is exactly what the `dx` "
        "would have said"
    )


def _rewrite_integrals(text: str) -> str:
    r"""Turn `\int_a^b f dx` into `Integral(f, (x, a, b))`.

    The `dx` is not decoration: it names the variable and marks where the
    integrand ends, so an integral written without one has no determinate
    body. That refuses rather than guessing.
    """
    while True:
        match = _INTEGRAL_RE.search(text)
        if match is None:
            return text
        braced_low, bare_low, braced_high, bare_high = match.groups()
        lower = braced_low if braced_low is not None else bare_low
        upper = braced_high if braced_high is not None else bare_high
        if (lower is None) != (upper is None):
            raise LatexRejected(
                "integral with only one bound. A definite integral needs both, "
                "and an indefinite one needs neither"
            )
        rest = text[match.end() :]
        found = _find_integration_variable(rest)
        if found is None:
            # No differential written. The integrand still names its variable
            # whenever exactly one identifier occurs in it, and that is the
            # reading every source intends -- `\int_0^1 x` is over x. The
            # extent is then the following term, as with every other operator.
            body, trailing = _split_leading_term(rest.strip())
            variable = _sole_variable(body)
            limits = (
                f"({variable}, {lower}, {upper})" if lower is not None else variable
            )
            text = text[: match.start()] + f"Integral({body}, {limits})" + trailing
            continue
        start, end, variable = found
        body = rest[:start].strip()
        if not body:
            raise LatexRejected("integral with an empty integrand")
        limits = f"({variable}, {lower}, {upper})" if lower is not None else variable
        text = text[: match.start()] + f"Integral({body}, {limits})" + rest[end:]


def _rewrite_limits(text: str) -> str:
    r"""Turn `\lim_{x \to a} f` into `Limit(f, x, a)`.

    The body is the following term, by the precedence every source assumes.
    `\lim_{n \to \infty} d_n = 0` is the Baez-Duarte criterion, and the body
    there is `d_n` -- a limit does not span the `=`.
    """
    while True:
        match = _LIMIT_RE.search(text)
        if match is None:
            return text
        variable, target = match.groups()
        if not target.strip():
            raise LatexRejected(r"\lim has no target to approach")
        body, trailing = _split_leading_term(text[match.end() :].strip())
        if not body:
            raise LatexRejected(r"\lim has a target but no body")
        text = (
            text[: match.start()]
            + f"Limit({body}, {variable}, {target})"
            + trailing
        )



#: `\sum_{n \ge 2}` / `\sum_{n > 1}`: a lower bound with the upper one left
#: implicit at infinity, which is how most sources write a tail sum.
_HALF_OPEN_BINDER_RE = re.compile(
    r"\\(sum|prod)\s*_\s*\{\s*([A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:\\ge|\\geq|>=|\\gt|>)\s*([^{}]+?)\s*\}"
)


def _rewrite_half_open_binders(text: str) -> str:
    r"""Turn `\sum_{n \ge 2} f` into `Sum(f, (n, 2, oo))`.

    `n >= 2` states the lower bound and leaves the upper one at infinity. That
    is a bound, not a missing bound: writing it out changes nothing about what
    the source says.
    """
    while True:
        match = _HALF_OPEN_BINDER_RE.search(text)
        if match is None:
            return text
        kind, index, lower = match.groups()
        strict = match.group(0).rstrip("} ").rstrip()
        bound = lower.strip()
        if strict.endswith(bound) and (
            r"\gt" in match.group(0) or ">" in match.group(0).replace(">=", "")
        ):
            pass
        body, trailing = _split_leading_term(text[match.end() :].strip())
        if not body:
            raise LatexRejected(rf"\{kind} has a bound but no body")
        head = "Sum" if kind == "sum" else "Product"
        text = (
            text[: match.start()]
            + f"{head}({body}, ({index}, {bound}, oo))"
            + trailing
        )


#: `\forall x, P: Q` and `\forall x: Q`. A universally quantified implication is
#: written as the implication itself over a free variable, which is the ordinary
#: reading and the one SymPy can hold.
_FORALL_RE = re.compile(
    r"\\forall\s+[A-Za-z][A-Za-z0-9_]*\s*(?:,\s*(?P<guard>[^:]+?))?\s*:\s*"
)


#: LaTeX and ASCII spellings of each relation, longest first so `<=` is not
#: read as `<`.
_RELATION_TOKENS = (
    ("\\leq", "Le"), ("\\geq", "Ge"), ("\\neq", "Ne"),
    ("\\le", "Le"), ("\\ge", "Ge"), ("\\ne", "Ne"),
    ("<=", "Le"), (">=", "Ge"), ("!=", "Ne"),
    ("<", "Lt"), (">", "Gt"), ("=", "Eq"),
)


def _relation_to_call(text: str) -> str:
    """Rewrite `a < b` as `Lt(a, b)`.

    A relation nested inside another construct cannot stay infix: Python reads
    `Implies(P, a = b)` as an assignment and refuses it, and reads `a == b` as
    a boolean it evaluates immediately. The call form is what survives being
    nested.
    """
    for token, head in _RELATION_TOKENS:
        depth = 0
        for index in range(len(text)):
            char = text[index]
            if char in "([{":
                depth += 1
            elif char in ")]}":
                depth -= 1
            elif depth == 0 and text.startswith(token, index):
                left = text[:index].strip()
                right = text[index + len(token) :].strip()
                if left and right:
                    return f"{head}({left}, {right})"
        continue
    return text.strip()


def _rewrite_quantifiers(text: str) -> str:
    r"""Turn `\forall s, \Re(s)>1: \zeta(s)=1` into `Implies(re(s)>1, Eq(...))`.

    A statement about every `s` satisfying a condition is an implication whose
    variable happens to be free. Dropping the quantifier and keeping the
    implication loses nothing; dropping the GUARD would lose the domain the
    claim is restricted to, which is why the guard becomes the antecedent
    rather than being discarded.
    """
    match = _FORALL_RE.search(text)
    if match is None:
        return text
    guard = match.group("guard")
    body = text[match.end() :].strip()
    if not body:
        raise LatexRejected(r"\forall introduces a variable but states nothing")
    rest = text[: match.start()].strip()
    if rest:
        raise LatexRejected(
            r"\forall is not at the start of the statement, so what it ranges "
            "over is unclear"
        )
    if guard is None or not guard.strip():
        return _relation_to_call(body)
    return f"Implies({_relation_to_call(guard.strip())}, {_relation_to_call(body)})"


#: `f'(x)`, `f''(x)`: Lagrange's notation. The quote count is the order.
#: The name may still carry its command backslash: this runs before the
#: command table, so `\zeta'(s)` is matched as `\zeta` and stays that way
#: until the substitution that turns it into `zeta`.
_PRIME_NOTATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])(\\?[A-Za-z][A-Za-z0-9_]*)('+)\s*\("
)

#: A single identifier, so `f'(s)` differentiates with respect to s itself.
_PLAIN_IDENTIFIER_RE = re.compile(r"^[A-Za-z][A-Za-z0-9_]*$")

#: The placeholder a derivative at a compound point is taken with respect to.
_PRIME_POINT = "u_"


def _rewrite_prime_notation(text: str) -> str:
    r"""Turn `\zeta'(s)` into `Derivative(zeta(s), s)`.

    Lagrange's notation is how every source writes `-\zeta'(s)/\zeta(s)`, and
    a quote is not an operator in Python -- left alone it opens a string
    literal and the whole formula fails to tokenize.

    `f'(g)` where the argument is compound means the derivative of f evaluated
    AT g, not the derivative with respect to g: `\zeta'(1-s)` differentiates
    zeta and then substitutes. `Subs` is exactly that and says so.
    """
    while True:
        match = _PRIME_NOTATION_RE.search(text)
        if match is None:
            return text
        name, quotes = match.group(1), match.group(2)
        order = len(quotes)
        group = _balanced_group(text, match.end() - 1)
        if group is None:
            raise LatexRejected(f"{name}{quotes} has no closing parenthesis")
        argument, end = group
        argument = argument.strip()
        if not argument:
            raise LatexRejected(f"{name}{quotes} is applied to nothing")
        if _PLAIN_IDENTIFIER_RE.match(argument):
            replacement = f"Derivative({name}({argument}), ({argument}, {order}))"
        else:
            replacement = (
                f"Subs(Derivative({name}({_PRIME_POINT}), ({_PRIME_POINT}, {order}))"
                f", {_PRIME_POINT}, {argument})"
            )
        text = text[: match.start()] + replacement + text[end:]


#: The closing half of an evaluation bar, with the point it evaluates at.
_EVALUATION_BAR_RE = re.compile(r"\\right\s*\|\s*_\s*(?:\{([^{}]*)\}|(\S+))")


def _rewrite_evaluation_bar(text: str) -> str:
    r"""Turn `\left. X \right|_{s=1}` into `Subs(X, s, 1)`.

    The bar is where a derivative is TAKEN AT, and for Li's coefficients it is
    the whole content: `\lambda_n` is the n-th derivative of
    `s^{n-1} log xi(s)` evaluated at s = 1, and without the point the right
    side is a function of s while the left is a number. The formula sat in the
    corpus that way -- parsed, indexed, and not a definition of anything.

    Runs before the absolute-value pass, which would otherwise read the
    closing `\right|` as a delimiter that never opened.
    """
    while True:
        start = text.find("\\left.")
        if start < 0:
            return text
        match = _EVALUATION_BAR_RE.search(text, start)
        if match is None:
            raise LatexRejected(
                r"\left. with no \right|_{...} to close it, so nothing says "
                "where the expression is evaluated"
            )
        body = text[start + len("\\left.") : match.start()].strip()
        if not body:
            raise LatexRejected("evaluation bar around nothing")
        point = match.group(1) if match.group(1) is not None else match.group(2)
        if "=" not in point:
            raise LatexRejected(
                f"evaluation bar subscript {point!r} does not say what equals "
                "what; write it as `_{s=1}`"
            )
        variable, value = point.split("=", 1)
        if not variable.strip() or not value.strip():
            raise LatexRejected(f"malformed evaluation point {point!r}")
        text = (
            text[:start]
            + f"Subs({body}, {variable.strip()}, {value.strip()})"
            + text[match.end() :]
        )


#: `\binom{k}{j}` and `\dbinom`/`\tbinom`, which differ only in how they print.
_BINOM_RE = re.compile(r"\\[dt]?binom\s*\{([^{}]*)\}\s*\{([^{}]*)\}")


def _rewrite_binomials(text: str) -> str:
    r"""Turn `\binom{k}{j}` into `binomial(k, j)`.

    Baez-Duarte's coefficients are an alternating binomial sum, so without
    this the whole criterion is unreachable. Two brace groups exactly, like
    `\frac` -- a `\binom` with anything else is malformed and is left to the
    unsupported-command check rather than being guessed at.
    """
    return _BINOM_RE.sub(r"binomial(\1, \2)", text)


def _basic_latex_to_sympy(text: str) -> str:
    """Normalize LaTeX to SymPy input, refusing anything it cannot represent."""
    _check_braces_balanced(text)
    # Brackets become groups first: a derivative's operand is usually written
    # `[...]`, and the rewrite below needs to see it as a group.
    out = _rewrite_binomials(_rewrite_evaluation_bar(_rewrite_brackets(text)))
    out = _rewrite_prime_notation(out)
    out = _rewrite_derivatives(out)
    _refuse_unrepresentable_notation(out)

    # Bounded sums and products are representable exactly, so they are turned
    # into SymPy Sum/Product here -- before the refusal below, which now covers
    # only the binders that genuinely cannot be carried.
    out = _rewrite_quantifiers(out)
    out = _rewrite_bounded_binders(out)
    out = _rewrite_half_open_binders(out)
    # After the bounded rewrite: a binder WITH bounds is an ordinary Sum or
    # Product and must not be captured as a prime-indexed one.
    out = _rewrite_prime_binders(out)
    out = _rewrite_integrals(out)
    out = _rewrite_limits(out)

    # Before the command table strips the delimiter commands to nothing:
    # that substitution erases the only thing telling an opening bar from
    # a closing one, and the pairing has to happen while it is still there.
    out = _rewrite_absolute_value(out)

    # A \frac that did not match the two-argument form is malformed. Catching
    # it here is what stops `\frac{1}` from parsing as the number 0.
    if _BARE_FRAC_RE.search(out):
        raise LatexRejected("malformed \\frac: expected exactly two non-empty brace groups")

    # Before the command table strips the backslashes: after it, `theta`
    # and `xy` are both bare runs of letters and nothing separates them.
    out = _split_imaginary_literals(_split_juxtaposed_letters(out))
    out = _WRAPPER_RE.sub(r"\1", out)
    # `e^{x}` is exp(x). Left as a Symbol, Robin's inequality parsed cleanly and
    # meant nothing: `e` was an undefined name, so `e^{\gamma}` was some symbol
    # raised to another. Only the exponentiated form is rewritten -- a bare `e`
    # is genuinely ambiguous (it is a perfectly ordinary variable name), and
    # guessing there would trade one silent error for another.
    previous = None
    while previous != out:
        previous = out
        out = _EULER_POW_RE.sub(r"exp(\1)", out)
    # `\sqrt{x}` -> `sqrt(x)`. Done before the generic brace check, which would
    # otherwise reject the group as unconsumed.
    previous = None
    while previous != out:
        previous = out
        out = _SQRT_RE.sub(r"sqrt(\1)", out)
    previous = None
    while previous != out:
        previous = out
        out = _FRAC_RE.sub(r"((\1)/(\2))", out)
    if _BARE_FRAC_RE.search(out) or r"\frac" in out:
        raise LatexRejected("malformed \\frac: expected exactly two non-empty brace groups")

    for command, replacement in sorted(_ALGEBRAIC_COMMANDS.items(), key=lambda kv: -len(kv[0])):
        if replacement and replacement[-1].isalnum():
            # A separator when the replacement would abut an identifier.
            # Plain string replacement fused `\log\log` into the single
            # undefined name `loglog`, so Robin's inequality parsed as a product
            # with a nonsense symbol in it. The command boundary is real
            # structure and has to survive substitution.
            out = re.sub(
                re.escape(command) + r"(?![A-Za-z])",
                lambda _m, r=replacement: r + " ",
                out,
            )
        else:
            out = out.replace(command, replacement)
    for command in sorted(_SYMBOL_COMMANDS, key=len, reverse=True):
        name = command[1:]
        replacement = _SYMBOL_ALIASES.get(name, name)
        # Insert a space when the preceding character is part of an
        # identifier. Without it `i\gamma` collapses to the single symbol
        # `igamma_` -- one name where the source wrote two, so the
        # imaginary unit silently disappears. `i gamma_` is read correctly
        # by implicit multiplication.
        out = re.sub(
            r"(?<=[A-Za-z0-9_])" + re.escape(command) + r"(?![A-Za-z])",
            " " + replacement,
            out,
        )
        out = re.sub(re.escape(command) + r"(?![A-Za-z])", replacement, out)

    leftover = [m.group(0) for m in _COMMAND_RE.finditer(out)]
    if leftover:
        raise LatexRejected(
            f"unsupported LaTeX command(s): {', '.join(sorted(set(leftover)))}"
        )

    out = re.sub(r"\^\{([^{}]*)\}", r"^(\1)", out)
    out = re.sub(r"_\{([^{}]*)\}", r"_\1", out)
    if "{" in out or "}" in out:
        raise LatexRejected("unconsumed brace group; structure would be lost")

    out = _apply_bare_functions(out)
    return out.strip()


#: `name(` -- an identifier applied to an argument.
#:
#: A DIGIT may precede: `8\pi` is a coefficient times pi, and treating the
#: digit as part of the name hid the standalone reading entirely. Schoenfeld's
#: bound uses `\pi` both ways -- applied on the left, the constant on the
#: right -- and with `8pi` invisible, `pi` looked purely applied and the
#: constant became the prime-counting FUNCTION used as a value.
_APPLICATION_RE = re.compile(r"(?<![A-Za-z_])([A-Za-z][A-Za-z0-9_]*)\s*\(")
_IDENTIFIER_RE = re.compile(r"[A-Za-z][A-Za-z0-9_]*")

#: A name NOT immediately followed by `(` -- i.e. used as a value.
_STANDALONE_RE = re.compile(r"(?<![A-Za-z_])([A-Za-z][A-Za-z0-9_]*)(?!\s*\()")

#: Names SymPy already defines correctly. Left out of the locals map so the
#: real implementation is used rather than an opaque placeholder.
_SYMPY_FUNCTIONS = frozenset(
    {
        "Abs", "binomial", "ceiling", "cos", "exp", "factorial", "floor",
        "im", "log", "re", "sin", "sqrt", "tan",
    }
)


#: A name bound as a binder index: the `n` in `Sum(f(n), (n, 1, oo))`, the `x`
#: in `Limit(f, x, a)`, the `s` in `Derivative(f, (s, k))`.
_BOUND_TUPLE_RE = re.compile(r",\s*\(\s*([A-Za-z][A-Za-z0-9_]*)\s*,")
_BOUND_LIMIT_RE = re.compile(r"\bLimit\s*\([^,]*,\s*([A-Za-z][A-Za-z0-9_]*)\s*,")
_BOUND_SIMPLE_RE = re.compile(
    r"\b(?:Integral|Derivative)\s*\([^,]*,\s*([A-Za-z][A-Za-z0-9_]*)\s*\)"
)


def _bound_indices(text: str) -> set[str]:
    """Names a binder has bound, which are therefore variables and not values.

    `i` is the imaginary unit everywhere in this literature -- `1/2 + it` is a
    point on the critical line, and reading `i` as a free symbol makes it a
    product of two unknowns. But `i` is also the commonest index letter there
    is, and `Sum(f(i), (i, 1, n))` binds it. A bound name is a variable, and
    that is decidable from the binder rather than guessed at.
    """
    bound: set[str] = set()
    for pattern in (_BOUND_TUPLE_RE, _BOUND_LIMIT_RE, _BOUND_SIMPLE_RE):
        bound.update(match.group(1) for match in pattern.finditer(text))
    return bound


#: A run of two or more plain letters, not a command and not applied.
_LETTER_RUN_RE = re.compile(r"(?<![A-Za-z_\\])([A-Za-z]{2,})(?![A-Za-z0-9_]*\s*\()(?![A-Za-z0-9_])")

#: Multi-letter tokens the rewrites above emit, which are names and not runs.
_EMITTED_TOKENS = frozenset({"oo", "zoo", "nan", "dir"})


#: A digit glued to `j` or `J`, which Python's LEXER reads as an imaginary
#: literal before any transformation gets a chance at it.
_IMAGINARY_LITERAL_RE = re.compile(r"(?<![A-Za-z0-9_.])(\d+)([jJ])(?![A-Za-z0-9_])")


def _split_imaginary_literals(text: str) -> str:
    r"""Rewrite `2j` as `2*j`.

    `\zeta(2j+2)` in Baez-Duarte's coefficients came out as `zeta(2 + 2*I)` --
    the summation index read as the imaginary unit, giving a complex argument
    and a clean parse of a different formula. Implicit multiplication cannot
    fix it because the damage happens in the tokenizer: `2j` is one token.

    `i` is this corpus's imaginary unit -- see `STANDALONE_CONSTANTS` -- so a
    `j` here is always an index and never the unit.
    """
    return _IMAGINARY_LITERAL_RE.sub(r"\1*\2", text)


def _split_juxtaposed_letters(text: str) -> str:
    r"""Rewrite `4xy` as `4*x*y` and `1/2 + it` as `1/2 + i*t`.

    Juxtaposition is multiplication in mathematics and a single identifier in
    syntax -- the same mismatch as `\log n`, and the one that made
    `(x+y)^2 - (x-y)^2 = 4xy` an equation about a variable named "xy", and
    `\zeta(1/2 + it)` a function of an unknown named "it".

    Only bare runs. A name reached by a LaTeX command (`\theta`, `\epsilon`)
    still carries its backslash at this point in the pipeline and is skipped,
    which is why this runs BEFORE the command table rather than after: `theta`
    is one name and `xy` is two, and the backslash is what tells them apart.
    Anything applied is a function name, and anything with an underscore or a
    digit is a subscripted symbol; neither is a run.
    """

    def split(match: re.Match[str]) -> str:
        run = match.group(1)
        if run in _EMITTED_TOKENS or run in _SYMPY_FUNCTIONS or run in _SYMPY_PROVIDED:
            return run
        if run in APPLIED_FUNCTIONS or run in STANDALONE_CONSTANTS:
            return run
        return "*".join(run)

    return _LETTER_RUN_RE.sub(split, text)


def _smart_locals(text: str) -> dict[str, object]:
    """Applied names become Functions; every other name stays a Symbol.

    This is what makes `f(x)` mean *f applied to x*. The module used to force
    every Greek name to a Symbol -- so that a beta/gamma/zeta command did not
    resolve to the SymPy *function* and raise on a missing argument -- and
    implicit multiplication then read `xi(s)` as `xi*(s)`, reporting a clean
    parse of the wrong thing. `xi(s) = xi(1-s)`, which is true, came out as
    `Eq(s*xi, xi*(1-s))`, which is false.

    Deciding by *syntax* rather than by name keeps that fix from depending on a
    list of known functions: whatever is written applied is treated as applied,
    and a name that never touches a parenthesis is still a plain Symbol. So
    `2(x+1)` is untouched -- the digit is not an identifier.

    Names with an exact SymPy counterpart (`log`, `sqrt`, `exp`, the trig
    functions) are left to SymPy so they carry their real meaning; everything
    else becomes an undefined `Function`, which is the honest representation of
    a symbol this package knows nothing about.
    """
    applied = {m.group(1) for m in _APPLICATION_RE.finditer(text)}
    # A name that ALSO appears on its own is a variable, and `s(s-1)` is a
    # product. Deciding on the applied form alone made `s` a function in
    #     xi(s) = (1/2)s(s-1)pi^(-s/2)Gamma(s/2)zeta(s)
    # and the whole definition failed to parse. Names that are only ever
    # applied -- xi, Gamma, zeta here -- are the genuine functions.
    standalone = {
        m.group(1)
        for m in _STANDALONE_RE.finditer(text)
    }
    applied -= standalone
    locals_: dict[str, object] = {}
    reserved = _SYMPY_FUNCTIONS | _SYMPY_PROVIDED
    for name in applied:
        real = APPLIED_FUNCTIONS.get(name)
        if real is not None:
            # `\zeta(s)` is the Riemann zeta, not a symbol spelled zeta.
            # SymPy implements almost all of these; storing a stub instead
            # kept the shape and threw away everything that made it checkable.
            locals_[name] = real
            continue
        if name in _SYMPY_CONSTANTS:
            # A constant that is APPLIED is not that constant. `\pi(x)` is the
            # prime-counting function and `2\pi` is the number, and the applied
            # form is what tells them apart -- reserving the name outright made
            # von Koch's theorem parse as pi TIMES x.
            locals_[name] = sp.Function(name)
            continue
        if name in reserved and name not in _SYMPY_REFUSED:
            continue  # let SymPy supply the real one
        locals_[name] = sp.Function(name)
    bound = _bound_indices(text)
    for name in set(_IDENTIFIER_RE.findall(text)) - applied:
        constant = None if name in bound else STANDALONE_CONSTANTS.get(name)
        if constant is not None:
            locals_[name] = constant
            continue
        if name in reserved and name not in _SYMPY_REFUSED:
            continue
        locals_[name] = sp.Symbol(name)
    return locals_


#: Suffix for the applied reading of a name that is also used as a value.
_APPLIED_ALIAS = "__applied_"


def prepare_for_parsing(text: str) -> tuple[str, dict[str, object]]:
    """Text and locals for one `parse_expr` call, with both readings kept.

    A name can be a function and a value in the SAME formula. Schoenfeld's
    bound is the case that matters:

        |pi(x) - li(x)| < (1/(8*pi))*sqrt(x)*log(x)

    `pi` is the prime-counting function on the left and the number on the
    right, and `local_dict` holds one entry per name, so no single mapping is
    right for both. Resolving it either way silently corrupts half the formula:
    as a value, `pi(x)` becomes pi TIMES x.

    The extractor never hit this because it parses each side separately, which
    happens to separate the two readings. Anything reading a whole relation --
    the fingerprint path -- gets them at once, and quietly hashed `pi*x`.

    Applied occurrences of such a name are renamed to a private alias that maps
    back to `Function(name)`, so the parsed object is exactly what the
    per-side parse builds and the alias never escapes into it.
    """
    applied = {m.group(1) for m in _APPLICATION_RE.finditer(text)}
    standalone = {m.group(1) for m in _STANDALONE_RE.finditer(text)}
    # Only names with a KNOWN applied meaning. `\pi` is the prime-counting
    # function applied and the number standing alone, so both readings are
    # real and the applied one needs an alias. `s` has no applied meaning, so
    # `s(s-1)` is multiplication -- and aliasing it turned the xi definition
    # into a function called s, which is the misreading this whole mechanism
    # exists to prevent.
    ambiguous = applied & standalone & set(APPLIED_FUNCTIONS)
    locals_ = _smart_locals(text)
    out = text
    for name in sorted(ambiguous):
        alias = name + _APPLIED_ALIAS
        out = re.sub(
            r"(?<![A-Za-z0-9_])" + re.escape(name) + r"\s*\(",
            alias + "(",
            out,
        )
        locals_[alias] = APPLIED_FUNCTIONS.get(name) or sp.Function(name)
    return out, locals_


def sympy_locals(text: str) -> dict[str, object]:
    """The name-resolution policy, for every module that parses this package's
    expressions.

    It is exported because the policy has to be applied everywhere or it is
    applied nowhere. The fingerprint path used to re-read a normalized equation
    with a bare `parse_expr`, so `Eq(M(x), O(x**(epsilon + 1/2)))` came back as
    `Mul(Symbol('M'), Symbol('x'))` against an `Order` germ -- three separate
    misreadings of a string this module had just parsed correctly. Nothing
    downstream could notice: the re-parse succeeded, and what got hashed was
    simply not the formula.
    """
    return _smart_locals(text)


#: A mapped function name NOT followed by a parenthesis. In mathematics
#: `log n` means log applied to n; syntactically it is juxtaposition, which
#: implicit multiplication turns into `log * n`.
_BARE_APPLICATION_RE = re.compile(
    r"(?<![A-Za-z0-9_])(" + "|".join(sorted(_SYMPY_FUNCTIONS)) + r")(?!\s*\()"
)


def _apply_bare_functions(text: str) -> str:
    r"""Rewrite `log n` as `log(n)`, and `\log\log n` as `log(log(n))`.

    In mathematics `\log n` is application; in syntax it is juxtaposition,
    which implicit multiplication turns into `log * n`. Refusing it was safe
    and wrong-headed -- `\log n` is ordinary notation, and making a document
    add parentheses is fitting the mathematics to the parser.

    A function command applies to the atom that follows it: a braced group, an
    already-applied name (so `\log H(n)` takes all of `H(n)`), a bare name, or
    a number.

    Scanning is by POSITION and right-to-left, not by consuming regex matches.
    A consuming match on `log log n` swallowed both names at once and produced
    `log(log) n`, leaving the inner application without its argument -- so
    Robin's inequality still came out wrong, just differently. Wrapping the
    innermost first and re-scanning lets each name find its own atom.
    """
    names = "|".join(sorted(_SYMPY_FUNCTIONS, key=len, reverse=True))
    name_re = re.compile(r"(?<![A-Za-z0-9_])(" + names + r")(?![A-Za-z0-9_])")
    atom_re = re.compile(
        r"\s*(\{[^{}]+\}"
        r"|[A-Za-z][A-Za-z0-9_]*\s*\([^()]*\)"
        r"|[A-Za-z][A-Za-z0-9_]*"
        r"|\d+(?:\.\d+)?)"
    )
    for _ in range(64):  # bounded: each pass wraps exactly one application
        changed = False
        for match in reversed(list(name_re.finditer(text))):
            rest = text[match.end() :]
            if rest.lstrip().startswith("("):
                continue  # already applied
            atom_match = atom_re.match(rest)
            if not atom_match:
                continue
            atom = atom_match.group(1)
            if atom.startswith("{"):
                atom = atom[1:-1]
            text = text[: match.end()] + f"({atom})" + rest[atom_match.end() :]
            changed = True
            break
        if not changed:
            break
    return text


def _split_relation(text: str) -> tuple[EquationKind, str | None, str | None, str | None]:
    """Split on the relation, returning WHICH relation it is.

    The operator is part of the statement. Robin's criterion is the strict
    inequality and its non-strict form is a different claim, so an extractor
    that reports only "an inequality" has not read the formula.
    """
    for token in ("<=", ">=", "<", ">"):
        if token in text:
            left, right = text.split(token, 1)
            return EquationKind.INEQUALITY, token, left.strip(), right.strip()
    if "=" in text and "==" not in text:
        left, right = text.split("=", 1)
        return EquationKind.EQUATION, "=", left.strip(), right.strip()
    return EquationKind.EXPRESSION, None, None, None


def _parse_side(text: str | None) -> sp.Expr:
    if text is None or not text.strip():
        raise LatexRejected("relation has an empty side")
    # `_smart_locals` decides per-name from the text itself, so it must be
    # built from THIS side's source rather than from a fixed table.
    #
    # Through `prepare_for_parsing`, which also disambiguates a name used BOTH
    # ways in one expression. Splitting a top-level relation into two sides
    # happens to separate the two readings of `\pi` in Schoenfeld's bound --
    # applied on the left, the constant on the right -- so this looked correct
    # for as long as every such formula was a top-level relation. Wrapped in a
    # `\forall`, the whole statement is one expression and the readings
    # collide.
    source, locals_ = prepare_for_parsing(text)
    locals_ = {**_symbol_locals(), **locals_}
    expr = parse_expr(
        source, local_dict=locals_, transformations=_TRANSFORMS, evaluate=False
    )
    if isinstance(expr, (tuple, sp.Tuple)):
        raise LatexRejected(
            "expression parsed as a tuple; a stray comma has changed the structure"
        )
    return expr


def parse_math(text: str) -> ExtractedEquation:
    source = _strip_math_delimiters(text)
    lhs_text: str | None = None
    rhs_text: str | None = None
    normalized = source
    try:
        normalized = _basic_latex_to_sympy(source)
        kind, operator, lhs_text, rhs_text = _split_relation(normalized)
        if kind == EquationKind.EQUATION:
            lhs = _parse_side(lhs_text)
            rhs = _parse_side(rhs_text)
            obj = sp.Eq(lhs, rhs, evaluate=False)
            return ExtractedEquation(
                source=source, normalized=str(obj), kind=kind,
                lhs=str(lhs), rhs=str(rhs), sympy_srepr=sp.srepr(obj),
            )
        if kind == EquationKind.INEQUALITY:
            lhs = _parse_side(lhs_text)
            rhs = _parse_side(rhs_text)
            # A real relational, not a `Relation(lhs, rhs)` stand-in. The
            # stand-in dropped the operator, so `sigma(n) < B` and
            # `sigma(n) <= B` produced the SAME srepr -- and the fingerprint
            # taken from it could not tell Robin's criterion from a statement
            # that is not equivalent to RH.
            obj = sp.Rel(lhs, rhs, operator, evaluate=False)
            return ExtractedEquation(
                source=source, normalized=str(obj), kind=kind,
                lhs=str(lhs), rhs=str(rhs), sympy_srepr=sp.srepr(obj),
            )
        expr = _parse_side(normalized)
        return ExtractedEquation(
            source=source, normalized=str(expr), kind=kind, sympy_srepr=sp.srepr(expr)
        )
    except Exception as exc:
        return ExtractedEquation(
            source=source, normalized=normalized, kind=EquationKind.UNKNOWN,
            lhs=lhs_text, rhs=rhs_text,
            parse_error=f"{type(exc).__name__}: {exc}",
        )


_BLOCK_PATTERNS = [
    re.compile(r"\$\$(.+?)\$\$", re.DOTALL),
    re.compile(r"\\\[(.+?)\\\]", re.DOTALL),
    re.compile(r"\\\((.+?)\\\)", re.DOTALL),
]
_INLINE_PATTERN = re.compile(r"(?<!\$)\$([^\n$]+?)\$(?!\$)")


def extract_equations(markdown: str) -> list[ExtractedEquation]:
    matches: list[tuple[int, str]] = []
    occupied: list[tuple[int, int]] = []
    for pattern in _BLOCK_PATTERNS:
        for match in pattern.finditer(markdown):
            matches.append((match.start(), match.group(1)))
            occupied.append((match.start(), match.end()))
    for match in _INLINE_PATTERN.finditer(markdown):
        if any(start <= match.start() < end for start, end in occupied):
            continue
        matches.append((match.start(), match.group(1)))
    matches.sort(key=lambda pair: pair[0])
    return [parse_math(text) for _, text in matches]
