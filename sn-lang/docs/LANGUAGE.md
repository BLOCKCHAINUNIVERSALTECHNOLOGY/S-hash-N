# S#N Language Reference

## 1. Program structure

A program is a list of statements separated by newlines (or `;`). Blocks use braces `{ ... }`.
Comments start with `#`. Source files use UTF-8 and the extension `.sn`.

A statement continues on the next line if that line starts with `|>` or `.method`, or if the
line break is inside `(...)` / `[...]`, or directly after a binary operator or comma.

```sn
let total = [1, 2, 3]
    |> map(x => x * 2)      # continues because the line starts with |>
    |> sum
```

Two statements on one line need `;` between them: `let a = 1; let b = 2`.

## 2. Values and types

| Type | Examples | `type()` |
|---|---|---|
| nil | `nil` (`शून्य`) | `"nil"` |
| bool | `true`, `false` (`सच`, `झूठ`) | `"bool"` |
| int / float | `42`, `1_000`, `3.14`, `2e-3` | `"int"`, `"float"` |
| string | `"text"`, `'text'`, escapes `\n \t \\ \"` | `"string"` |
| list | `[1, "a", [2]]` | `"list"` |
| map | `{name: "Asha", "age": 30, [key]: 1}` | `"map"` |
| function | `fn(x) { ... }`, `x => x + 1` | `"function"` |
| tensor | `tensor([[1, 2], [3, 4]])` | `"tensor"` |
| dataframe | `read_csv("f.csv")` | `"dataframe"` |
| model | `LinearRegression()` ... | `"LinearRegression"` etc. |

*Truthiness:* `nil`, `false`, `0`, `0.0`, `""`, `[]` and `{}` are false; everything else is true.
A tensor with more than one element has no truth value (use `.any()` / `.all()`).

Map keys written as a bare word are strings: `{a: 1}` is `{"a": 1}`. Use `[expr]: value` for computed keys.

## 3. Variables

```sn
let x = 10            # declare
x = x + 1             # assign (must be declared first)
x += 5                # also -=, *=, /=
let a, b = [1, 2]     # unpack; also  let [a, b] = [1, 2]
```

Blocks create a new scope. Assigning to an undeclared name is an error, which catches typos.

## 4. Operators

From lowest to highest precedence:

| Level | Operators |
|---|---|
| pipe | `|>` |
| logic | `or`, `and`, `not` |
| conditional (lowest) | `a if cond else b` |
| comparison | `== != < > <= >=`, `in`, `not in` |
| additive | `+ -` |
| multiplicative | `* / // % @` |
| unary | `-x +x` |
| power (right-assoc.) | `**` |
| postfix | `f(x)`, `x[i]`, `x.name` |

* `/` always produces a float; `//` floors; `%` is the remainder.
* Conditional expression: `"big" if x > 3 else "small"` (chains: `a if c1 else b if c2 else c`).
* `+` joins strings and lists; `*` repeats them (`"ab" * 3`).
* `and` / `or` short-circuit and return one of their operands.
* On tensors every arithmetic and comparison operator is element-wise with broadcasting, and `@` is matrix multiplication.
  Comparisons produce a mask tensor that can index: `x[x > 0]`.

## 5. Strings and f-strings

```sn
let name = "Asha"; let score = 91.456
print(f"{name} scored {score:.1f} ({score / 100:.1%})")   # Asha scored 91.5 (91.5%)
print(f"braces: {{literal}}")
```

Format specs after `:` follow Python's mini-language (`.2f`, `8.3f`, `<10`, `>5d`, `,`...).
String methods: `upper lower title capitalize strip lstrip rstrip split join replace starts_with
ends_with contains find count chars lines is_numeric pad_left pad_right`.

## 6. Lists and maps

```sn
let xs = [5, 3, 8]
xs.push(1); xs[0] = 9
print(xs[1:3], xs[::-1], xs[-1], len(xs))
print([x * 2 for x in xs if x > 2])

let m = {a: 1}
m.b = 2; m["c"] = 3
print(m.keys(), m.has("a"), m.get("z", 0), "a" in m)
```

List methods: `push append pop insert extend remove index contains count reverse sort copy clear join map filter reduce`.
Map methods: `keys values items has get remove update copy clear`.
Lists and maps are mutable and shared by reference.

## 7. Control flow

```sn
if x > 10 { ... } elif x > 5 { ... } else { ... }     # `else if` also works; `else` may start a new line
while cond { ... }
for item in list { ... }
for i, item in enumerate(list) { ... }                  # also: for [i, item] in ...
for key, value in items(map) { ... }
break; continue
```

`for` iterates lists, strings (characters), maps (keys), tensors (rows) and data frames (row maps).

### match

`match` (`मिलान`) compares a value against patterns, top to bottom, and evaluates the first arm that fits.
It is an expression, so it can be returned or assigned. Arms are separated by a newline or a comma.

```sn
let text = match value {
    0 => "zero"
    1 or 2 or 3 => "a few"                       # `or` joins alternatives
    n if n < 0 => "negative"                     # a name binds the value; `if` adds a guard
    "hi" => "greeting"
    [] => "empty list"
    [first, ...rest] => f"{first}, then {len(rest)} more"
    {kind: "rect", w, h} => w * h                # map pattern: listed keys must match; `w` binds map["w"]
    [x, {y}] => x + y                            # patterns nest
    _ => "something else"                        # `_` matches anything
}
```

* An arm body is an expression, or a `{ ... }` block whose last expression is the value (`return` inside a block
  returns from the enclosing function). To produce a map, wrap it in parentheses.
* Variables bound by a pattern exist only inside that arm.
* `true` never matches `1` (and vice versa). If no arm matches, a runtime error is raised, so end with `_ => ...`
  when the value can be anything.

## 8. Functions

```sn
fn area(w, h = 1) { return w * h }       # default values
area(3)  area(3, 4)  area(h = 2, w = 5)  # keyword arguments
let double = x => x * 2                  # lambda: x => expr, (a, b) => expr
let make = fn(n) { return n + 1 }        # anonymous function with a block body
```

Functions are first-class and capture their defining scope (closures). Recursion depth is limited to
about 900 nested S#N calls; deeper recursion raises a normal error.

### The pipe operator

`a |> f(b)` means `f(a, b)`, and `a |> f` means `f(a)`. Because the piped value becomes the **first** argument,
`map`, `filter`, `reduce`, `sorted` and all data-frame methods are written to take the data first.

```sn
read_csv("data.csv") |> (df => df.filter(r => r.age > 18)) |> len
```

## 9. Classes

```sn
class Account {
    fn init(owner, balance = 0) {        # called by Account("Asha", 100)
        self.owner = owner
        self.balance = balance
    }
    fn deposit(amount) { self.balance += amount; return self }
    fn str() { return f"{self.owner}: {self.balance}" }     # used by print and f-strings
}

class Savings extends Account {           # single inheritance
    fn init(owner, balance = 0, rate = 0.04) {
        super.init(owner, balance)        # call the parent's method
        self.rate = rate
    }
    fn deposit(amount) { return super.deposit(amount * 1.01) }   # override
}

let s = Savings("Ravi", 100)
s.deposit(50).deposit(10)                 # methods that `return self` can be chained
print(s, is_a(s, Account), type(s))       # Ravi: 160.6 true Savings
```

* Only methods (`fn`) go inside a class body; fields are created by assigning to `self.name`.
  Hindi: `वर्ग`, `विस्तार`, `खुद` (for `self`).
* A class without `init` takes no arguments. `type(obj)` is the class name; `is_a(obj, Class)` checks inheritance.
* **Operator methods** make your own types behave like built-in ones:

| Method | Enables | Method | Enables |
|---|---|---|---|
| `add sub mul div floordiv mod pow matmul` | `a + b  a - b  a * b ...` | `eq` | `a == b`, `a != b` |
| `radd rsub rmul ...` | `3 * obj` (left operand is not an object) | `lt gt le ge` | `a < b ...` |
| `len` | `len(obj)` | `getitem setitem` | `obj[i]`, `obj[i] = v` |
| `iter` | `for x in obj` (return a list) | `contains` | `x in obj` |
| `str` | `print(obj)`, `f"{obj}"` | `bool` | `if obj { ... }` |

Without a `str` method an object prints as `Name(field=value, ...)`.

## 10. Errors

```sn
try {
    risky()
} catch e {
    print("failed:", e)         # e is the message text for built-in errors
}
throw {code: 404}                # throw any value; catch receives that value
```

Uncaught errors print the message, the source line, and the chain of function calls, then exit with status 1.

## 11. Modules

```sn
import "lib/geometry.sn" as geo   # geo.circle_area(2)
import "lib/geometry.sn"          # names are added to the current scope
```

Paths are resolved relative to the importing file, then the working directory, then the folders in `SN_PATH`.
Top-level names starting with `_` are private. A module runs once and is cached.

## 12. Tensors, data, ML and charts

See [STDLIB.md](STDLIB.md). Highlights: indexing `t[i, j]`, `t[:, 1]`, `t[mask]`, assignment `t[i] = ...`,
`t.shape`, `t.T`, `t.sum(axis=0)`; DataFrames index with `df["col"]`, `df[["a", "b"]]`, `df[mask]`, `df[3]`.

## 13. Command line

```
sn file.sn [args]      run (args are available as the list `args`)
sn check file.sn       parse only
sn eval "code"         run a snippet
sn                     REPL
exit(code)             stop the program with a status code
```

Set the environment variable `SN_DEBUG=1` to see Python tracebacks for errors inside built-in functions.

## 14. Grammar sketch

```
program    := statement*
statement  := let | match | fn_def | class | if | while | for | return | break | continue | try | throw | import | assign | expr
class      := "class" NAME ["extends" postfix] "{" fn_def* "}"
let        := "let" names ["=" expr]
fn_def     := "fn" NAME "(" params ")" block
if         := "if" expr block ("elif" expr block)* ["else" block]
for        := "for" names "in" expr block
try        := "try" block "catch" [NAME] block
expr       := pipe ["if" pipe "else" expr]
pipe       := or ("|>" or)*
or         := and ("or" and)*          and := not ("and" not)*        not := "not" not | comparison
comparison := additive (("=="|"!="|"<"|">"|"<="|">="|"in"|"not in") additive)*
additive   := term (("+"|"-") term)*   term := unary (("*"|"/"|"//"|"%"|"@") unary)*
unary      := ("-"|"+") unary | power  power := postfix ["**" unary]
postfix    := primary ( "(" args ")" | "[" index "]" | "." NAME )*
match      := "match" expr "{" (pattern ["if" expr] "=>" (expr | block) [","])+ "}"
pattern    := atom ("or" atom)*     atom := NUMBER | STRING | "true" | "false" | "nil" | NAME | "_" | "[" patterns ["..." [NAME]] "]" | "{" (KEY [":" pattern])* "}"
primary    := match | NUMBER | STRING | FSTRING | NAME | "(" expr ")" | list | map | lambda | "true" | "false" | "nil"
```
