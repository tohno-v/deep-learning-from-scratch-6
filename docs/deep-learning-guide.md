# Deep Learning / LLM 学習ガイド

このファイルは、このリポジトリを使って Deep Learning、LLM、関連する数学、Python コーディングを順番に学ぶための学習コンテンツです。コードの実行方法や環境構築は `README.md` を参照してください。

## このガイドの目的

このリポジトリは、Tokenizer、Attention、GPT、事前学習、生成、SFT、DPO、GRPO、KV Cache などを、小さな Python ファイルから段階的に確認できる構成になっています。

このガイドでは次の4点を重視します。

1. Deep Learning の基本概念を、テンソルの shape と計算グラフで理解する。
2. LLM を、Tokenizer、Embedding、Transformer、損失関数、生成処理に分解して理解する。
3. 数式をただ眺めるのではなく、コードに落ちる形まで丁寧に変形する。
4. Python / PyTorch のコードを、入力、出力、shape、パラメータ、学習ループの観点で読めるようにする。

## リポジトリ全体の読み方

最初に `README.md`、`pyproject.toml`、`requirements.txt` を読み、実行環境と依存ライブラリを確認します。このリポジトリでは、主に PyTorch、NumPy、Matplotlib、tqdm、regex を使います。

章はおおむね次の流れで読むと理解しやすいです。

| 対象 | 学ぶ内容 | 主な観点 |
|:--|:--|:--|
| `ch01` | Tokenizer | 文字列を整数 ID 列に変換する |
| `ch02` | Attention と GPT の部品 | softmax、mask、multi-head、正規化、FFN |
| `codebot/model.py` | 基本的な GPT 実装 | モデル全体の forward を読む |
| `ch03` | 事前学習、生成、SFT、chat、GRPO | 学習データ、損失、生成ループ |
| `ch04` | BPE の高速化 | キャッシュ、chunk、並列化 |
| `ch05` | 近年の LLM 改善 | RoPE、SwiGLU、RMSNorm、KV Cache |
| `ch06` | 学習の実用要素 | AdamW、LR scheduler、mixed precision、DPO、Judge |
| `ch07` / `storybot` / `ch09` / `webbot` | 統合実装 | 小さいモデルから実用寄りの構成へ進む |

`ch08` はこのリポジトリにはありません。

## 学習ロードマップ

### Stage 1: Python とテンソル操作に慣れる

まず `ch01` と `ch02` の短いファイルを読みます。この段階では、モデルの性能よりも「何が入力され、何が出力されるか」を追うことが重要です。

確認すること:

- `list[int]` と `torch.Tensor` の違い。
- `shape` がどこで変わるか。
- `view`、`transpose`、`matmul`、`softmax` が何をしているか。
- class の `__init__` と `forward` の役割。

実行例:

```bash
uv run python ch01/01_char_tokenizer.py
uv run python ch02/02_attn_math.py
uv run python ch02/06_attn_mask.py
```

### Stage 2: LLM の最小構成を理解する

次に `codebot/model.py` を読みます。このファイルには GPT の基本要素がまとまっています。

読む順番:

1. `GPT.forward`
2. `Block.forward`
3. `MultiHeadAttention.forward`
4. `FFN`
5. `LayerNorm`

`GPT.forward` は大きく次の流れです。

```text
ids
  -> token embedding
  -> position embedding を足す
  -> Transformer Block を n_layer 回通す
  -> LayerNorm
  -> unembed
  -> logits
```

ここで `logits` の shape は `(B, C, V)` です。

- `B`: batch size
- `C`: context length
- `V`: vocabulary size

### Stage 3: 学習ループを理解する

`ch03/01_pretrain.py` と `ch06/05_pretrain.py` を読みます。Deep Learning の学習は、ほとんどの場合次の形です。

```python
logits = model(x)
loss = loss_fn(logits, y)
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

この順番を必ず押さえます。

見るべきポイント:

- `x` と `y` はどう作られているか。
- `logits.view(-1, logits.size(-1))` はなぜ必要か。
- `batch_y.view(-1)` は何を表すか。
- `loss.backward()` の前に `optimizer.zero_grad()` を呼ぶ理由。
- `model.train()` と `model.eval()` の違い。

### Stage 4: 生成とファインチューニングを理解する

`ch03/02_generate.py`、`ch03/03_sft.py`、`ch03/04_chat.py`、`ch03/09_grpo.py`、`ch06/08_dpo.py` を読みます。

生成では、モデルを学習するのではなく、次トークンを1つずつ選びます。

```text
prompt
  -> tokenize
  -> model
  -> logits の最後の位置を取り出す
  -> softmax
  -> sample
  -> token を追加
  -> 繰り返す
```

SFT、DPO、GRPO は、事前学習したモデルの振る舞いを目的に合わせて調整する方法です。

### Stage 5: 現代的な LLM の改善を理解する

`ch05` と `webbot/model.py` を読みます。

主な改善点:

- RoPE: 位置情報を回転で表現する。
- RMSNorm: 平均を引かずにスケールだけ整える。
- SwiGLU: FFN の表現力を高める。
- KV Cache: 生成時に過去の Key / Value を再計算しない。
- GQA: Query head 数より Key / Value head 数を減らして計算量を抑える。

## LLM を構成要素に分けて理解する

### 1. Tokenizer

LLM は文字列を直接処理しません。まず Tokenizer が文字列を整数 ID 列に変換します。

```text
"hello" -> [104, 101, 108, 108, 111]
```

Tokenizer の段階で重要なのは、可逆性と語彙サイズです。

- 文字単位 Tokenizer は単純だが、系列が長くなりやすい。
- byte 単位 Tokenizer は未知文字に強い。
- BPE は頻出する byte / 文字ペアをまとめて、系列長を短くする。

`ch01/03_bpe_train.py` では、BPE の基本操作である「最頻ペアを探す」「そのペアを新しい ID に置き換える」を確認できます。

### 2. Embedding

整数 ID は、そのままでは意味のあるベクトルではありません。Embedding 層は ID をベクトルに変換します。

数式では、語彙サイズを `V`、埋め込み次元を `E` とすると、Embedding 行列は次の形です。

```math
W_{emb} \in \mathbb{R}^{V \times E}
```

トークン ID が `i` のとき、埋め込みベクトルは `W_emb` の `i` 行目です。

```math
x = W_{emb}[i]
```

複数トークンをまとめると、入力 `ids` の shape が `(B, C)` のとき、Embedding 後の shape は `(B, C, E)` になります。

### 3. Transformer Block

Transformer Block は、だいたい次の構成です。

```text
x
  -> norm
  -> attention
  -> residual add
  -> norm
  -> FFN / SwiGLU
  -> residual add
```

`codebot/model.py` では次の形です。

```python
x = x + self.attn(self.norm1(x))
x = x + self.ffn(self.norm2(x))
```

これは Pre-LN Transformer と呼ばれる構成です。正規化を attention や FFN の前に置くことで、深いモデルでも学習を安定させやすくなります。

### 4. logits と確率分布

モデルの最後は `unembed` によって、各位置のベクトルを語彙サイズ分のスコアに変換します。

```math
logits = h W_{out} + b
```

shape で見ると次の通りです。

```text
h      : (B, C, E)
W_out  : (E, V)
logits : (B, C, V)
```

`logits` は確率ではありません。softmax を通すと確率になります。

```math
p_i = \frac{\exp(z_i)}{\sum_{j=1}^{V}\exp(z_j)}
```

ここで `z_i` は語彙 `i` に対する logit です。

## 数学: softmax を丁寧に追う

### softmax の目的

softmax は任意の実数ベクトルを確率分布に変換します。

入力:

```math
z = (z_1, z_2, \dots, z_V)
```

出力:

```math
p = (p_1, p_2, \dots, p_V)
```

各成分は次のように定義されます。

```math
p_i = \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
```

確率分布になる理由は2つあります。

まず、指数関数は常に正です。

```math
e^{z_i} > 0
```

したがって、

```math
p_i > 0
```

次に、全成分の和を取ると1になります。

```math
\sum_{i=1}^{V} p_i
= \sum_{i=1}^{V} \frac{e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
```

分母は `i` に依存しないので外に出せます。

```math
= \frac{\sum_{i=1}^{V} e^{z_i}}{\sum_{j=1}^{V} e^{z_j}}
```

分子と分母は同じ量なので、

```math
= 1
```

### softmax の数値安定化

実装では、`exp` の値が大きくなりすぎることを避けるため、最大値を引くことがあります。

```math
p_i = \frac{e^{z_i - m}}{\sum_j e^{z_j - m}}
```

ここで、

```math
m = \max_j z_j
```

これは元の softmax と同じ値になります。理由を変形で確認します。

```math
\frac{e^{z_i - m}}{\sum_j e^{z_j - m}}
= \frac{e^{z_i}e^{-m}}{\sum_j e^{z_j}e^{-m}}
```

分母の `e^{-m}` は全ての項に共通なので、

```math
= \frac{e^{z_i}e^{-m}}{e^{-m}\sum_j e^{z_j}}
```

分子と分母の `e^{-m}` が打ち消されます。

```math
= \frac{e^{z_i}}{\sum_j e^{z_j}}
```

したがって、最大値を引いても softmax の結果は変わりません。

## 数学: cross entropy と next-token prediction

### one-hot 正解との cross entropy

正解トークンが `t` であるとします。正解分布 `y` は one-hot ベクトルです。

```math
y_i =
\begin{cases}
1 & (i = t) \\
0 & (i \ne t)
\end{cases}
```

予測分布を `p` とすると、cross entropy は次の通りです。

```math
L = -\sum_{i=1}^{V} y_i \log p_i
```

one-hot の性質を使うと、`y_t = 1` で、それ以外は0です。

```math
L = -(0 \cdot \log p_1 + \dots + 1 \cdot \log p_t + \dots + 0 \cdot \log p_V)
```

したがって、

```math
L = -\log p_t
```

つまり、正解トークンの確率 `p_t` が高いほど loss は小さくなります。

### logits から loss までの変形

softmax を代入します。

```math
p_t = \frac{e^{z_t}}{\sum_j e^{z_j}}
```

cross entropy は、

```math
L = -\log p_t
```

なので、

```math
L = -\log \left(\frac{e^{z_t}}{\sum_j e^{z_j}}\right)
```

log の性質 `log(a / b) = log a - log b` を使います。

```math
L = -\left(\log e^{z_t} - \log \sum_j e^{z_j}\right)
```

`log e^{z_t} = z_t` なので、

```math
L = -(z_t - \log \sum_j e^{z_j})
```

符号を分配します。

```math
L = \log \sum_j e^{z_j} - z_t
```

PyTorch の `F.cross_entropy` は、この softmax と negative log likelihood をまとめて安定に計算します。

### なぜ `view(-1, V)` にするのか

LLM の logits は `(B, C, V)` です。正解 `y` は `(B, C)` です。

PyTorch の `F.cross_entropy` は、分類対象の class 次元を `(N, V)` のような形で受け取ることが多いため、コードでは次のように平坦化します。

```python
loss = F.cross_entropy(
    logits.view(-1, logits.size(-1)),
    batch_y.view(-1),
)
```

shape の変化:

```text
logits : (B, C, V) -> (B*C, V)
target : (B, C)    -> (B*C)
```

これは「batch 内の全位置を、`B*C` 個の分類問題としてまとめて扱う」という意味です。

## 数学: cross entropy の勾配

softmax と cross entropy を組み合わせたとき、logit `z_k` に対する勾配は非常にきれいな形になります。

```math
\frac{\partial L}{\partial z_k} = p_k - y_k
```

この式を丁寧に追います。

先ほどの loss は、

```math
L = -\sum_i y_i \log p_i
```

softmax を含めた形では、

```math
L = \log \sum_j e^{z_j} - \sum_i y_i z_i
```

one-hot なら `-\sum_i y_i z_i = -z_t` ですが、ここでは一般形のまま扱います。

`z_k` で微分します。

```math
\frac{\partial L}{\partial z_k}
= \frac{\partial}{\partial z_k}\log \sum_j e^{z_j}
- \frac{\partial}{\partial z_k}\sum_i y_i z_i
```

まず第1項です。`S = \sum_j e^{z_j}` と置きます。

```math
\frac{\partial}{\partial z_k}\log S
= \frac{1}{S}\frac{\partial S}{\partial z_k}
```

`S` のうち `z_k` に依存するのは `e^{z_k}` だけです。

```math
\frac{\partial S}{\partial z_k} = e^{z_k}
```

したがって、

```math
\frac{\partial}{\partial z_k}\log S
= \frac{e^{z_k}}{\sum_j e^{z_j}}
```

これは softmax の `p_k` です。

```math
= p_k
```

次に第2項です。

```math
\frac{\partial}{\partial z_k}\sum_i y_i z_i
```

`z_k` を含む項は `y_k z_k` だけなので、

```math
= y_k
```

よって、

```math
\frac{\partial L}{\partial z_k} = p_k - y_k
```

この式の意味は直感的です。

- 予測確率 `p_k` が正解 `y_k` より大きすぎる class は logit を下げる。
- 予測確率 `p_k` が正解 `y_k` より小さすぎる class は logit を上げる。

## 数学: Attention を丁寧に追う

### Query、Key、Value

Attention は「各トークンが、どの過去トークンをどれだけ参照するか」を計算する仕組みです。

入力を `X` とします。

```math
X \in \mathbb{R}^{C \times E}
```

線形変換で Query、Key、Value を作ります。

```math
Q = XW_Q
```

```math
K = XW_K
```

```math
V = XW_V
```

ここで、

```math
Q, K, V \in \mathbb{R}^{C \times D}
```

### 類似度スコア

位置 `a` の Query と、位置 `b` の Key の類似度は内積で計算します。

```math
score_{a,b} = q_a \cdot k_b
```

ベクトルの内積は、成分で書くと次の通りです。

```math
q_a \cdot k_b = \sum_{d=1}^{D} q_{a,d} k_{b,d}
```

全位置の組み合わせをまとめて計算すると、

```math
Scores = QK^T
```

shape は次のようになります。

```text
Q    : (C, D)
K^T  : (D, C)
QK^T : (C, C)
```

PyTorch では次の形です。

```python
scores = torch.matmul(Q, K.transpose(-2, -1))
```

### なぜ `sqrt(D)` で割るのか

Attention では次のようにスケーリングします。

```math
Scores = \frac{QK^T}{\sqrt{D}}
```

理由は、内積の分散が次元 `D` に比例して大きくなるからです。

各成分が平均0、分散1の独立な値だと仮定します。

```math
q_d k_d
```

の平均は0、分散はおおよそ1です。内積は `D` 個の和です。

```math
q \cdot k = \sum_{d=1}^{D} q_d k_d
```

独立な変数の和の分散は分散の和になります。

```math
\mathrm{Var}(q \cdot k)
= \sum_{d=1}^{D} \mathrm{Var}(q_d k_d)
```

各項の分散を1と見ると、

```math
= D
```

標準偏差は分散の平方根なので、

```math
\mathrm{Std}(q \cdot k) = \sqrt{D}
```

そこで `sqrt(D)` で割ると、スコアのスケールをおおよそ一定に保てます。

```math
\mathrm{Std}\left(\frac{q \cdot k}{\sqrt{D}}\right)
= \frac{\sqrt{D}}{\sqrt{D}}
= 1
```

スコアが大きすぎると softmax が極端に尖り、勾配が流れにくくなります。スケーリングは学習を安定させるための工夫です。

### softmax で重みにする

各 query 位置ごとに、参照先 key 方向へ softmax を取ります。

```math
A = \mathrm{softmax}\left(\frac{QK^T}{\sqrt{D}}\right)
```

shape は `(C, C)` です。`A_{a,b}` は「位置 `a` が位置 `b` をどれだけ見るか」を表します。

最後に Value の重み付き和を取ります。

```math
O = AV
```

shape:

```text
A : (C, C)
V : (C, D)
O : (C, D)
```

`ch02/02_attn_math.py` と `codebot/model.py` の `MultiHeadAttention.forward` は、この流れをそのまま実装しています。

## 数学: causal mask

言語モデルは、未来のトークンを見て次トークンを予測してはいけません。たとえば位置3を予測するとき、位置4以降を参照できると答えを見ていることになります。

そこで causal mask を使います。

長さ `C = 4` の場合、使ってよい位置を1、見てはいけない位置を0で書くと次の行列です。

```text
1 0 0 0
1 1 0 0
1 1 1 0
1 1 1 1
```

これは `torch.tril` で作れます。

```python
mask = torch.tril(torch.ones(C, C, device=scores.device))
```

見てはいけない位置の score を `-inf` にします。

```python
scores = scores.masked_fill(mask == 0, float("-inf"))
```

なぜ `-inf` なのかを softmax で確認します。

```math
\exp(-\infty) = 0
```

したがって、softmax 後の確率は0になります。

```math
\frac{\exp(-\infty)}{\sum_j \exp(z_j)} = 0
```

これにより、未来位置への attention weight が0になります。

## 数学: Multi-Head Attention

1つの attention だけでは、文脈の見方が1種類に固定されます。Multi-Head Attention では、複数の head が別々の射影で attention を計算します。

`codebot/model.py` では、入力 `x` の shape は `(B, C, E)` です。

線形変換:

```python
Q = self.W_q(x)
K = self.W_k(x)
V = self.W_v(x)
```

この時点では、

```text
Q, K, V : (B, C, H*D)
```

`H` 個の head に分けます。

```python
Q = Q.view(B, C, H, D).transpose(1, 2)
```

shape は、

```text
(B, C, H, D) -> (B, H, C, D)
```

head ごとに attention を計算すると、

```text
scores  : (B, H, C, C)
weights : (B, H, C, C)
hidden  : (B, H, C, D)
```

最後に head を結合します。

```python
hidden = hidden.transpose(1, 2).contiguous()
hidden = hidden.view(B, C, H * D)
output = self.W_o(hidden)
```

shape:

```text
(B, H, C, D)
-> (B, C, H, D)
-> (B, C, H*D)
-> (B, C, E)
```

読むときは、常に `B`、`C`、`H`、`D`、`E` のどれがどこにあるかを追います。

## 数学: LayerNorm と RMSNorm

### LayerNorm

LayerNorm は、最後の次元ごとに平均と分散を計算して正規化します。

入力ベクトルを次のように置きます。

```math
x = (x_1, x_2, \dots, x_E)
```

平均:

```math
\mu = \frac{1}{E}\sum_{i=1}^{E} x_i
```

分散:

```math
\sigma^2 = \frac{1}{E}\sum_{i=1}^{E}(x_i - \mu)^2
```

正規化:

```math
\hat{x}_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}
```

最後に学習可能なスケールとバイアスを適用します。

```math
y_i = \gamma_i \hat{x}_i + \beta_i
```

`codebot/model.py` の `LayerNorm` はこの式に対応しています。

### RMSNorm

RMSNorm は平均を引かず、二乗平均平方根でスケールだけ整えます。

```math
\mathrm{RMS}(x) = \sqrt{\frac{1}{E}\sum_{i=1}^{E}x_i^2 + \epsilon}
```

正規化:

```math
\hat{x}_i = \frac{x_i}{\mathrm{RMS}(x)}
```

学習可能な重みをかけます。

```math
y_i = w_i \hat{x}_i
```

RMSNorm は LayerNorm より計算が少なく、近年の LLM でよく使われます。`ch05/03_rmsnorm.py`、`storybot/model.py`、`webbot/model.py` を見ると確認できます。

## 数学: GELU、SiLU、SwiGLU

### FFN の役割

Attention はトークン間の情報を混ぜます。一方、FFN は各位置ごとの表現を非線形に変換します。

基本的な FFN は次の形です。

```math
\mathrm{FFN}(x) = W_2 f(W_1x + b_1) + b_2
```

`f` は GELU などの活性化関数です。

### GELU

GELU は入力を滑らかにゲートする活性化関数です。`codebot/model.py` では近似式が実装されています。

```math
\mathrm{GELU}(x)
= 0.5x\left(1 + \tanh\left(\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right)\right)
```

ReLU は負の値をばっさり0にしますが、GELU は確率的に残すような滑らかな形になります。

### SwiGLU

SwiGLU は `ch05/02_swiglu.py` で確認できます。基本形は次の通りです。

```math
\mathrm{SwiGLU}(x) = (\mathrm{SiLU}(xW) \odot xV)O
```

ここで `\odot` は要素ごとの積です。

SiLU は次の関数です。

```math
\mathrm{SiLU}(x) = x \cdot \sigma(x)
```

sigmoid は、

```math
\sigma(x) = \frac{1}{1 + e^{-x}}
```

したがって、

```math
\mathrm{SiLU}(x) = \frac{x}{1 + e^{-x}}
```

SwiGLU は、片方の線形変換をゲートとして使い、もう片方の情報を制御します。

## 数学: RoPE

RoPE は Rotary Position Embedding の略です。位置情報をベクトルに足すのではなく、Query と Key の成分を回転させることで位置を表現します。

2次元の回転行列は次の通りです。

```math
R_\theta =
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
```

ベクトル `(x_1, x_2)` に適用すると、

```math
\begin{bmatrix}
x'_1 \\
x'_2
\end{bmatrix}
=
\begin{bmatrix}
\cos\theta & -\sin\theta \\
\sin\theta & \cos\theta
\end{bmatrix}
\begin{bmatrix}
x_1 \\
x_2
\end{bmatrix}
```

行列積を展開します。

```math
x'_1 = x_1\cos\theta - x_2\sin\theta
```

```math
x'_2 = x_1\sin\theta + x_2\cos\theta
```

RoPE では、位置 `m` に応じて角度を変えます。

```math
\theta_m = m \cdot \theta
```

これにより、Query と Key の内積に相対的な位置差が反映されます。

コードでは `cos_cache` と `sin_cache` を事前に作り、forward 時に必要な長さだけ取り出します。`ch05/01_rope.py` の `RoPE` と `MultiHeadAttention` を読むと、数式と実装の対応が追えます。

## 数学: AdamW

`ch06/02_adamw.py` では optimizer の考え方を学べます。

通常の勾配降下法は、

```math
\theta_{t+1} = \theta_t - \eta g_t
```

です。

- `\theta_t`: 現在のパラメータ
- `\eta`: learning rate
- `g_t`: 勾配

Adam は勾配の移動平均と二乗勾配の移動平均を使います。

1次モーメント:

```math
m_t = \beta_1 m_{t-1} + (1 - \beta_1)g_t
```

2次モーメント:

```math
v_t = \beta_2 v_{t-1} + (1 - \beta_2)g_t^2
```

初期値が0なので、序盤は小さく偏ります。その補正をします。

```math
\hat{m}_t = \frac{m_t}{1 - \beta_1^t}
```

```math
\hat{v}_t = \frac{v_t}{1 - \beta_2^t}
```

Adam の更新は、

```math
\theta_{t+1}
= \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
```

AdamW は weight decay を勾配更新から分離します。

```math
\theta_{t+1}
= \theta_t
- \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
- \eta \lambda \theta_t
```

最後の項 `-\eta \lambda \theta_t` が weight decay です。大きすぎる重みを抑え、過学習を防ぎやすくします。

## 数学: learning rate scheduler

`ch06/03_lr_scheduler.py` と `ch06/lr_graph.py` では、learning rate を一定にせず、学習の進行に合わせて変える方法を学べます。

よく使われる構成は warmup + cosine decay です。

warmup 中:

```math
\eta_t = \eta_{\max}\frac{t}{T_{warmup}}
```

これは `t = 0` で0、`t = T_warmup` で `eta_max` になります。

warmup 後の cosine decay:

```math
\eta_t
= \eta_{\min}
+ \frac{1}{2}(\eta_{\max} - \eta_{\min})
\left(1 + \cos\left(\pi \frac{t - T_{warmup}}{T_{max} - T_{warmup}}\right)\right)
```

開始時は learning rate を小さくして不安定な更新を避け、後半は徐々に小さくして収束しやすくします。

## 数学: DPO の読み方

`ch06/08_dpo.py` では DPO を確認できます。DPO は、選ばれた応答 `chosen` と拒否された応答 `rejected` の log probability を比較します。

モデルの差分:

```math
\Delta_\theta
= \log \pi_\theta(y_w|x) - \log \pi_\theta(y_l|x)
```

参照モデルの差分:

```math
\Delta_{ref}
= \log \pi_{ref}(y_w|x) - \log \pi_{ref}(y_l|x)
```

DPO の loss は次の形です。

```math
L
= -\log \sigma\left(\beta(\Delta_\theta - \Delta_{ref})\right)
```

`y_w` は好ましい応答、`y_l` は好ましくない応答です。

この式の意味:

- 現在のモデルが、参照モデルよりも `chosen` を相対的に高く評価すれば loss は小さくなる。
- `rejected` を高く評価しすぎると loss は大きくなる。
- `beta` は調整の強さを決める係数。

sigmoid の中身を `r` と置きます。

```math
r = \beta(\Delta_\theta - \Delta_{ref})
```

loss は、

```math
L = -\log \sigma(r)
```

sigmoid は、

```math
\sigma(r) = \frac{1}{1 + e^{-r}}
```

したがって、

```math
L = -\log \frac{1}{1 + e^{-r}}
```

log の性質より、

```math
L = \log(1 + e^{-r})
```

`r` が大きいほど `e^{-r}` は小さくなり、loss も小さくなります。つまり、chosen を rejected より十分高く評価できるほど、DPO loss は下がります。

## Python / PyTorch コーディングの学び方

### class と `nn.Module`

PyTorch のモデルは `nn.Module` を継承して作ります。

```python
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(10, 3)

    def forward(self, x):
        return self.linear(x)
```

`__init__` では、学習する層やパラメータを定義します。`forward` では、入力から出力までの計算を定義します。

読むときは次のように分けます。

- `__init__`: どんな部品を持つか。
- `forward`: どんな順番で計算するか。
- `save` / `load_from`: パラメータをどう保存・復元するか。

### `view`、`transpose`、`contiguous`

LLM 実装では shape 変換が頻繁に出ます。

```python
Q = Q.view(B, C, H, D).transpose(1, 2)
```

これは `(B, C, H*D)` を `(B, H, C, D)` に変えています。

`transpose` 後のテンソルはメモリ上で連続とは限りません。そのため、`view` の前に `contiguous()` を呼ぶことがあります。

```python
hidden = hidden.transpose(1, 2).contiguous()
hidden = hidden.view(B, C, H * D)
```

shape だけでなく、メモリ配置も意識する練習になります。

### `@torch.no_grad()`

生成時は学習しないので、勾配を保存する必要がありません。

```python
@torch.no_grad()
def generate(...):
    ...
```

これにより、メモリ使用量が減り、推論が速くなります。

### `model.train()` と `model.eval()`

学習時:

```python
model.train()
```

評価・生成時:

```python
model.eval()
```

Dropout などは学習時と評価時で挙動が変わります。評価時に `model.eval()` を忘れると、出力が不安定になることがあります。

### optimizer の基本

学習ループでは、次の順番を崩さないようにします。

```python
optimizer.zero_grad()
loss.backward()
optimizer.step()
```

PyTorch は勾配を累積します。そのため、毎 step の前に `zero_grad()` で前回の勾配を消します。

### `Dataset` と batch

`ch03/01_pretrain.py` の `TokenDataset` や `ch03/03_sft.py` の `SFTDataset` を読むと、学習データを PyTorch に渡す方法が分かります。

Dataset で重要なメソッド:

```python
def __len__(self):
    ...

def __getitem__(self, idx):
    ...
```

`__getitem__` は、1つのサンプルを返します。DataLoader はそれらをまとめて batch にします。

### コードを読むときのチェックリスト

各ファイルを読むときは、次の順番で確認します。

1. 入力データの型と shape は何か。
2. 出力データの型と shape は何か。
3. 学習可能なパラメータはどこにあるか。
4. 乱数が使われている箇所はどこか。
5. 学習時だけ必要な処理はどこか。
6. 生成時だけ必要な処理はどこか。
7. loss は何と何を比較しているか。

## 章別の学習メモ

### `ch01`: Tokenizer

学ぶこと:

- 文字単位 Tokenizer。
- byte 単位 Tokenizer。
- BPE の学習。
- 特殊トークン。
- pretokenize。

読むファイル:

- `ch01/01_char_tokenizer.py`
- `ch01/02_byte_tokenizer.py`
- `ch01/03_bpe_train.py`
- `ch01/04_bpe_tokenizer.py`
- `ch01/05_special_token.py`
- `ch01/06_pretokenize.py`
- `ch01/09_bpe_encode.py`

確認問題:

1. 文字単位 Tokenizer で未知文字が来たらどうなるか。
2. byte 単位 Tokenizer が未知文字に強い理由は何か。
3. BPE の `merge` は、リストのどの部分を置き換えているか。
4. 特殊トークンは通常の BPE merge と何が違うか。

### `ch02`: Attention と GPT の部品

学ぶこと:

- soft dictionary。
- attention の数式。
- scaling。
- causal mask。
- multi-head attention。
- LayerNorm / GELU / FFN。
- GPT の最小構成。

読むファイル:

- `ch02/01_soft_dict.py`
- `ch02/02_attn_math.py`
- `ch02/03_attn_scaling.py`
- `ch02/06_attn_mask.py`
- `ch02/08_multi_head.py`
- `ch02/09_norm_gelu.py`
- `ch02/10_gpt2.py`

確認問題:

1. `Q @ K.T` の shape はどうなるか。
2. `scores / sqrt(D)` がないと何が起きやすいか。
3. causal mask で `-inf` を使う理由は何か。
4. Multi-Head Attention で `transpose(1, 2)` する理由は何か。

### `codebot`: 基本 GPT

学ぶこと:

- Tokenizer と GPT モデルの統合。
- weight tying。
- save / load。
- 生成関数。

読むファイル:

- `codebot/model.py`
- `codebot/tokenizer.py`
- `codebot/utils.py`

確認問題:

1. `self.embed.weight = self.unembed.weight` は何を共有しているか。
2. `GPT.forward` の戻り値 `logits` の shape は何か。
3. `generate` で最後の位置の logits だけを使う理由は何か。

### `ch03`: 学習、生成、SFT、chat、GRPO

学ぶこと:

- pretraining。
- next-token prediction。
- sampling。
- SFT。
- chat prompt。
- GRPO。

読むファイル:

- `ch03/01_pretrain.py`
- `ch03/02_generate.py`
- `ch03/03_sft.py`
- `ch03/04_chat.py`
- `ch03/09_grpo.py`

確認問題:

1. `batch_x` と `batch_y` は何トークンずれているか。
2. 生成時に `temperature` を大きくすると何が起きるか。
3. SFT のデータは pretraining のデータと何が違うか。
4. GRPO の reward は何を評価しているか。

### `ch04`: BPE の高速化

学ぶこと:

- pretokenize。
- ペアカウントのキャッシュ。
- chunk 分割。
- 並列化。
- encode の最適化。

読むファイル:

- `ch04/01_bpe_optimize.py`
- `ch04/02_bpe_cache.py`
- `ch04/03_bpe_chunk.py`
- `ch04/04_bpe_parallel.py`
- `ch04/06_encode_optimize.py`
- `ch04/07_encode_parallel.py`

確認問題:

1. 全テキストを毎回数え直すと何が遅いか。
2. キャッシュすると、どの計算を省略できるか。
3. chunk 分割するとき、特殊トークン境界を意識する理由は何か。
4. 並列化しても結果を同じにするには何に注意するか。

### `ch05`: 現代的 LLM 改善

学ぶこと:

- RoPE。
- SwiGLU。
- RMSNorm。
- 改良版 GPT。
- KV Cache。

読むファイル:

- `ch05/01_rope.py`
- `ch05/02_swiglu.py`
- `ch05/03_rmsnorm.py`
- `ch05/04_improved_gpt.py`
- `ch05/05_kvcache.py`

確認問題:

1. RoPE は絶対位置埋め込みと何が違うか。
2. RMSNorm は LayerNorm から何を省いているか。
3. SwiGLU の要素ごとの積は何をゲートしているか。
4. KV Cache によって、生成時のどの再計算が減るか。

### `ch06`: 実用的な学習

学ぶこと:

- AdamW。
- learning rate scheduler。
- mixed precision。
- pretraining の改善。
- LLM-as-a-Judge。
- DPO。

読むファイル:

- `ch06/02_adamw.py`
- `ch06/03_lr_scheduler.py`
- `ch06/04_mixed_precision.py`
- `ch06/05_pretrain.py`
- `ch06/07_llm_judge.py`
- `ch06/08_dpo.py`
- `ch06/09_llm_judge.py`

確認問題:

1. AdamW は SGD と何が違うか。
2. warmup はなぜ必要か。
3. mixed precision で BF16 を使う利点は何か。
4. DPO は chosen と rejected をどう比較しているか。

### `storybot` / `webbot`

学ぶこと:

- より統合された LLM 実装。
- tokenizer、model、utils の分離。
- RoPE、KV Cache、GQA などの組み合わせ。
- 生成ユーティリティ。

読むファイル:

- `storybot/model.py`
- `storybot/tokenizer.py`
- `storybot/utils.py`
- `webbot/model.py`
- `webbot/utils.py`

確認問題:

1. `storybot/model.py` と `webbot/model.py` の違いは何か。
2. `webbot` の `n_kv_head` は何のためにあるか。
3. `clear_cache()` はいつ呼ぶべきか。
4. 生成時に `use_cache=True` にすると forward の入力はどう変わるか。

## 実践演習

### 演習1: shape コメントを追加する

`codebot/model.py` を読み、各テンソルの shape をノートに書き出します。

対象:

- `ids`
- `emb`
- `pos_emb`
- `Q`
- `K`
- `V`
- `scores`
- `weights`
- `hidden`
- `logits`

ゴール:

`GPT.forward` と `MultiHeadAttention.forward` の全 shape を説明できる。

### 演習2: cross entropy を手計算する

語彙サイズ3の logits を考えます。

```math
z = (2.0, 1.0, 0.0)
```

正解 class は0とします。

1. softmax の分母を計算する。
2. `p_0` を計算する。
3. `L = -log(p_0)` を計算する。
4. `p - y` を計算し、各 logit の勾配の向きを確認する。

### 演習3: causal mask を手で作る

`C = 5` の causal mask を手で書きます。

その後、PyTorch で確認します。

```python
import torch

C = 5
mask = torch.tril(torch.ones(C, C))
print(mask)
```

ゴール:

位置 `i` が参照できる位置と参照できない位置を説明できる。

### 演習4: 生成ループを読む

`codebot/utils.py` または `storybot/utils.py` の `generate` を読みます。

確認すること:

1. prompt はどこで token ID に変換されるか。
2. logits のどの位置を使っているか。
3. temperature はどこに入るか。
4. sampling された token はどこで文脈に追加されるか。
5. 最後に ID 列はどこで文字列に戻るか。

### 演習5: 小さな改造をする

次のどれかを試します。

- `temperature` を変えて生成結果を比較する。
- `max_new_tokens` を変えて生成長を比較する。
- `n_head` や `embed_dim` を変えてパラメータ数や実行時間を比較する。
- `dropout_rate` を変えて学習 loss の動きを比較する。

実験したら、次の形式で記録します。

```text
変更内容:
期待した結果:
実際の結果:
理由の仮説:
次に試すこと:
```

## 数式とコードを対応させる読み方

数式を読むときは、必ず対応するコードを探します。

| 数式 | コードで探す語 |
|:--|:--|
| `Q = XW_Q` | `self.W_q(x)` |
| `QK^T` | `K.transpose(-2, -1)` / `matmul` |
| `/ sqrt(D)` | `scores / (D ** 0.5)` |
| `softmax` | `F.softmax(..., dim=-1)` |
| causal mask | `torch.tril` / `masked_fill` |
| cross entropy | `F.cross_entropy` |
| embedding | `nn.Embedding` |
| unembedding | `nn.Linear(embed_dim, vocab_size)` |
| residual connection | `x = x + ...` |
| optimizer update | `loss.backward()` / `optimizer.step()` |

この対応表を使うと、数式が抽象的なまま止まらず、実装上どこで何をしているかを追えます。

## 学習記録テンプレート

各ファイルを読んだら、次のテンプレートで記録します。

```markdown
## 読んだファイル

- ファイル:
- 実行コマンド:
- 入力:
- 出力:

## 理解したこと

- 

## shape メモ

| 変数 | shape | 意味 |
|:--|:--|:--|
| | | |

## 数式との対応

- 

## 疑問

- 

## 次に読むファイル

- 
```

## 最短の学習順

時間が限られている場合は、次の順で進めると全体像を早くつかめます。

1. `ch01/01_char_tokenizer.py`
2. `ch01/03_bpe_train.py`
3. `ch02/02_attn_math.py`
4. `ch02/06_attn_mask.py`
5. `ch02/08_multi_head.py`
6. `codebot/model.py`
7. `ch03/01_pretrain.py`
8. `ch03/02_generate.py`
9. `ch05/01_rope.py`
10. `ch05/05_kvcache.py`
11. `ch06/02_adamw.py`
12. `ch06/08_dpo.py`
13. `webbot/model.py`

## まとめ

このリポジトリを使うと、LLM を「巨大で複雑なもの」としてではなく、次の部品の組み合わせとして学べます。

```text
Tokenizer
  -> token IDs
  -> Embedding
  -> Transformer Blocks
  -> logits
  -> cross entropy で学習
  -> softmax と sampling で生成
```

学習の中心は、常に次の3点です。

1. shape を追う。
2. 数式をコードに対応させる。
3. 小さく実行して結果を確認する。

この3点を繰り返すと、Deep Learning、LLM、数学、Python コーディングを同時に積み上げられます。
