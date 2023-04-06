---
title: "PyTorch と in-place operation のエラー"
emoji: "🍌"
type: "tech" # tech 技術記事 / idea: アイデア
topics: ["python", "PyTorch"]
published: true
---

# 本記事で伝えたいこと
- PyTorchで `RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.` という Error が出たら、in-place operationが原因。
- in-place operation とは、`x.add_()` , `x += 0.5` , `x[mask] = 0.5` のようなテンソルの値を直接書き換える演算。
- 上記のErrorは以下の方法で解決

  - `x.add_()` は `x = x.add()`とする。

  - `x += 1` は `x = x + 1` とする。

  - mask を使用したい場合は、例えば下の式に示すように$a$ と $b$ を新しい値 $a'$ と $b'$ で置き換えたい場合、変更後の値から元の値を引いた値に対して mask との要素積をとり、変更前の値に足せば良い。（ $?$ と示したのはどうでも良い値）

    $$
    \begin{align}
    \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}
    &\leftarrow\begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}
    +
    \begin{bmatrix}
    1 & 1\\
    0 & 0
    \end{bmatrix}
    \otimes
    \left(
    \begin{bmatrix}
    a' & b'\\
    ? & ?
    \end{bmatrix}
    - \begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}
    \right)
    \\
    %
    %
    &\leftarrow\begin{bmatrix}
    a & b\\
    c & d
    \end{bmatrix}
    +
    \begin{bmatrix}
    a'-a & b'-b\\
    0 & 0
    \end{bmatrix}
    \\
    %
    %
    &\leftarrow\begin{bmatrix}
    a' & b'\\
    c & d
    \end{bmatrix}
    \end{align}
    $$


    ```python
    >>> import torch
    >>> x = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)
    >>> mask = x > 0.5
    >>> # x[mask] = 0.0 # Error
    >>> x = x + mask * (torch.zeros_like(x) - x)
    >>> print(x)

    tensor([0.2000, 0.4000, 0.0000, 0.0000], grad_fn=<AddBackward0>)
    ```

    上のコードは、テンソルの値が $0.5$ よりも大きい値を、$0$ に置き換えたい場合の例。

    ```python
    # indice -> mask
    indice = torch.tensor([2, 3], dtype=torch.long)
    mask = torch.zeros_like(x).bool().scatter_(0, indice, torch.ones_like(indice).bool())

    # slice -> mask
    mask = torch.zeros_like(x).bool()
    mask[2:] = True
    ```

    slice や index を使用したい場合は、slice や index から mask を作成する。



# in-place operation とは

> in-place operation とは、新しくコピーを作ることなく、オブジェクトの中身を変更する演算のことである。in-place operation を実行するための演算子を in-place operator という。（意訳）

https://www.tutorialspoint.com/inplace-operator-in-python

![](/images/in-place-error/compare.png)
*通常の代入演算 `x = x + 1` と、in-place operator による演算 `x += 1` の比較。*

python では、`iadd()` や `+=` などを使うと in-place operation になる。

# PyTorch で in-place operation

PyTorch で in-place operation をする場合は以下のような方法がある。（他にもあるかも。）

1. `x.add_()` , `x.mul_()` などの通常のメソッドに `_` を付けたメソッドを使用する。
2. `x.data` を使う。（正確には in-place operation とは異なりそう。）
3. indexやmaskを使用する
4. `+=` , `*=` などを使う

## 1. `x.add_()` を使う
```python
>>> import torch
>>> x = torch.tensor([1.])
>>> x.add(1)
>>> print(x)
tensor([1.])
```
通常の `add` を使用した場合は、`x` は変更されない。
`x.add(1)` という新しい変数が確保されているイメージ。

```python
>>> import torch
>>> x = torch.tensor([1.])
>>> x.add_(1) # in-place operation
>>> print(x)
tensor([2.])
```

`add_` （ハイフン有り）を使用した場合は、`x` の値が変更される。

## 2. `x.data`を使う（正確にはin-placeではなさそう）

:::message
`x.data` を使う方法は PyTorchでエラーが発生しないことから、正確には in-place operation ではなさそうですが、似たようなことができるためまとめました。
:::


```python
import torch

def my_add_one(input):
    input = input + 1

x = torch.tensor([1.])
my_add_one(x)

print(x)
# tensor([1.])
```

pythonの関数の引数は参照渡しなので、`input` と `x` は `id` が同じだが、
`input = input + 1` の左側の変数 `input` は `id` は上で述べた2つと異なる。

```python
import torch

def my_add_one(input):
    input.data = input + 1

x = torch.tensor([1.])
my_add_one(x)

print(x)
# tensor([2.])
```

`input.data = input + 1` とすると `x` の中身が書き換えられる。

:::message
data field は使うべきでなないと言及されている。（以下のページを参考）
:::

https://discuss.pytorch.org/t/the-difference-between-torch-tensor-data-and-torch-tensor/25995
https://zenn.dev/chickenta2ta/articles/05763114fc0fe4

## 3. indexやmaskを使用する

```python
>>> import torch
>>> x = torch.tensor([1, 2, 3])
>>> mask = x <= 2
>>> x[mask] = 10
>>> print(x)
tensor([10, 10,  3])
```

`x[0] = 2` や `x[mask] = 2` のように、index、slice、maskなどを使用して選択したテンソルの一部分に代入すると、値が置き換えられる。

# in-place operation を勾配計算では避ける

## 発生するエラー

PyTorchで `requires_grad=True` としたテンソルに対して、勾配計算のために何らかの in-place な計算処理を行うと以下のようなエラーが発生する。

:::message alert
RuntimeError: a view of a leaf Variable that requires grad is being used in an in-place operation.
:::

例えば、以下のようにして `requires_grad=True` としたテンソルに in-place operation をすると、上述のエラーが発生する。

```python
import torch

x = torch.rand(20, requires_grad=True)

x.add_(1) # Error
x[1] = 0  # Error
x += 1    # Error

x.data = x + 1 # これはErrorがでない
```

PyTorchの公式によると、勾配計算のためのforwardの計算が壊れてしまうため（意訳）in-place operation による Error が発生すると書いてある。また、Errorが発生しない場合もあるとも書かれている（よくわからない。。。）。

https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd

## 対処法（再掲）

- `x.add_()` は `x = x.add()`とする。

- `x += 1` は `x = x + 1` とする。

- mask を使用したい場合は、例えば下の式に示すように$a$ と $b$ を新しい値 $a'$ と $b'$ で置き換えたい場合、変更後の値から元の値を引いた値に対して mask との要素積をとり、変更前の値に足せば良い。（ $?$ と示したのはどうでも良い値）

$$
\begin{align}
\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
&\leftarrow\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
+
\begin{bmatrix}
1 & 1\\
0 & 0
\end{bmatrix}
\otimes
\left(
\begin{bmatrix}
a' & b'\\
? & ?
\end{bmatrix}
- \begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
\right)
\\
%
%
&\leftarrow\begin{bmatrix}
a & b\\
c & d
\end{bmatrix}
+
\begin{bmatrix}
a'-a & b'-b\\
0 & 0
\end{bmatrix}
\\
%
%
&\leftarrow\begin{bmatrix}
a' & b'\\
c & d
\end{bmatrix}
\end{align}
$$


```python
>>> import torch
>>> x = torch.tensor([0.2, 0.4, 0.6, 0.8], requires_grad=True)
>>> mask = x > 0.5
>>> # x[mask] = 0.0 # Error
>>> x = x + mask * (torch.zeros_like(x) - x)
>>> print(x)

tensor([0.2000, 0.4000, 0.0000, 0.0000], grad_fn=<AddBackward0>)
```

上のコードは、テンソルの値が $0.5$ よりも大きい値を、$0$ に置き換えたい場合の例。

```python
# indice -> mask
indice = torch.tensor([2, 3], dtype=torch.long)
mask = torch.zeros_like(x).bool().scatter_(0, indice, torch.ones_like(indice).bool())

# slice -> mask
mask = torch.zeros_like(x).bool()
mask[2:] = True
```

slice や index を使用したい場合は、slice や index から mask を作成し、上述の方法で対処すれば良い。


# 参考文献
https://www.tutorialspoint.com/inplace-operator-in-python
https://pytorch.org/docs/stable/notes/autograd.html#in-place-operations-with-autograd
https://discuss.pytorch.org/t/the-difference-between-torch-tensor-data-and-torch-tensor/25995
https://discuss.pytorch.org/t/what-is-in-place-operation/16244
https://www.kaggle.com/code/aleksandradeis/in-place-operations-in-pytorch
https://zenn.dev/chickenta2ta/articles/05763114fc0fe4
https://qiita.com/mathlive/items/3dcb46af2e2f0eca559a
