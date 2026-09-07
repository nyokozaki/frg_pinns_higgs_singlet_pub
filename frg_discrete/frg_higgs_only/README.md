# frg_discrete4_main

PINNsとの比較のためのmain解析フォルダ (旧 frg_discrete4)。

`frg_discrete` (shooting: t=0からforward積分) の代わりに、リラクゼーション法
(Numerical Recipes 17.3節の意味: 空間+時間をまとめて離散化し、1つの大域的な
連立方程式として解く) で tree-only (rloop=eta=0) の自由フローを解けるか試す。

## 実装

`du/dt = -4u + 2*rho*u_rho + 2*sigma*u_sigma` は線形PDEなので、Newton反復は
不要 (1回の線形ソルブで厳密収束する)。

- 空間微分: 中心差分 (`frg_discrete/grid_fd.py` の "central" スキームと同じ
  2次精度ステンシル。内部は中心差分、境界は2次精度片側差分)
- 時間積分: Crank-Nicolson (中心差分、2次精度)
- 全時刻ステップをまとめた1つのブロック双対角線形系
  `(I - dt/2 A) U^n = (I + dt/2 A) U^{n-1}` (n=1..Nt, U^0=UV境界条件) を、
  LU分解1回 + ブロック消去で解く。tree-onlyでは演算子Aがtに依らないため
  LHS行列は全ステップ共通で使い回せる。

線形かつ下三角ブロック構造なので、この「まとめて解く」ことは数学的には
同じ離散化でのCNマーチングと厳密に同値。したがって tree-only の場合、
リラクゼーション法固有の御利益 (Newtonで全軌道を自己無撞着に解くことで
forward shootingの発散を回避できること) はまだ現れない。ここで見ているのは
**中心差分(2次精度) + 非適応的な大域線形ソルブ** の効果のみ。

## 検証結果

厳密解 (特性曲線 `rho(0)=rho(t_end)*exp(2*t_end)` で逆引き) との比較:

### Case A: 純粋なtree多項式のseed (`u_tree_exact(0,...)`, thermal項なし)

```
python relax_tree.py --n-rho 41 --n-sigma 41 --n-t 100
```

相対誤差 **0.0198%** (41x41格子)。`frg_discrete` upwind shooting (81x81格子)
の27%から劇的に改善。ただしChebyshev (機械精度近く, 2e-10) には及ばない
— 中心差分は多項式に対して厳密ではなく、あくまで2次精度の代数的収束。

### Case B: フルseed (`u_seed` = tree+thermal)、rloop=eta=0の自由フロー

| 手法 | 格子 | 相対誤差 |
|---|---|---|
| FD upwind + shooting (frg_discrete) | 81x81 | 27% |
| 中心差分 relaxation (このフォルダ) | 41x41, Nt=100 | 0.57% |
| 中心差分 relaxation | 61x61, Nt=200 | 0.33% |
| 中心差分 relaxation | 81x81, Nt=200 | 0.24% |
| 中心差分 relaxation | 121x121, Nt=200 | 0.13% |
| 中心差分 relaxation | 121x121, Nt=400 | 0.12% (時間刻みを倍にしてもほぼ不変 → 空間打ち切り誤差が支配的) |
| Chebyshev (frg_discrete3) | 31x31 | 0.073% |

## 結論

1. **shooting (forward marching) 自体よりも、空間差分の精度(upwind 1次 vs
   中心差分2次)が主な問題だった** ことがはっきり分かる。中心差分に変えて
   大域的に解くだけで、tree-onlyの誤差は27%→0.1〜0.6%まで落ちる。
   (ただし中心差分を素朴にexplicit/adaptiveなshootingマーチングに使うと
   移流項に対して数値不安定になりやすい、というのが `frg_discrete` が
   upwindを採用した元々の理由 — `grid_fd.py` 冒頭のコメント参照。
   今回のように陰的CN+固定刻みの大域ソルブにすることで、その不安定性を
   気にせず中心差分を使えている。)
2. それでもChebyshevの指数収束にはまだ届かない (121x121 という
   Chebyshevの31x31よりずっと多い点数でも、Chebyshevの0.073%を上回れない)
   — 中心差分はあくまで多項式に対して厳密ではなく代数的収束なので、
   tree多項式部分に対する「Chebyshevは多項式に厳密」という御利益
   (frg_discrete3/README.md 参照) は持たない。
3. tree-only (線形) の場合、「まとめて解く」こと自体は数学的にCN
   マーチングと同値なので、リラクゼーション法の**本当の**御利益
   (発散しがちなforward shootingを、全軌道の自己無撞着なNewton解に
   置き換えることで回避する) はまだ試せていない。それが効いてくるのは
   rloopを入れて非線形になったとき — 今後の課題。

## 使い方 (tree-only)

```bash
# Case A: 純粋なtree多項式seed
python relax_tree.py --n-rho 41 --n-sigma 41 --n-t 100

# Case B: フルseed (tree+thermal)、まだrloopなし
python relax_tree.py --n-rho 121 --n-sigma 121 --n-t 200 --full-seed
```

## フル方程式 (rloop+eta込み) を Newton-Krylov で解く (relax_full.py)

tree-onlyは線形なので「まとめて解く」ことがCNマーチングと数学的に同値
だったが (上記参照)、rloopを入れると非線形になり、ここで初めて
リラクゼーション法固有の御利益 (全軌道を1つの自己無撞着解としてNewtonで
同時に解くことで、forward shooting特有の"誤差が時間方向に一方向に
伝播・増幅していく"構造そのものを回避できるかもしれないこと) を試せる。

実装: 空間中心差分+CN (または backward Euler) で全時刻ステップの残差を
`R^n(U^1,...,U^Nt) = 0` (n=1..Nt, U^0=UV境界条件で固定) とまとめ、
`scipy.optimize.newton_krylov` (行列を陽に組まないJacobian-free
Newton-Krylov法) で解く。初期推定値には tree-only中心差分リラクゼーション
解 (rloop=eta=0) を使う。

```bash
python relax_full.py --n-rho 21 --n-sigma 21 --n-t 40 --maxiter 50
```

### 結果: 収束しない (stall)

- CN (`--scheme cn`, デフォルト): 残差ノルムが初回でwarm start比98%減
  (0.36 → 0.0057) と急速に下がるが、その後 ~0.0052 前後で頭打ちになり
  50反復回しても f_tol=1e-8 に到達しない (最終残差ノルム 0.175、
  最大点別残差 5.2e-3)。
- backward Euler (`--scheme be`): さらに悪く、残差ノルムが単調減少
  すらせず反復8, 18, 27回目付近で跳ね上がる不安定な挙動 (最終残差ノルム
  0.27)。CNの方がまだ頑健。

### 原因: Higgs/singlet質量固有値のほぼ縮退点で sqrt(disc) がほぼ特異になる

rloopの質量固有値は `m1_sq, m2_sq = 0.5*(M11+M22 ∓ sqrt(disc))`,
`disc=(M11-M22)^2+4*M12^2 >= 0` で計算される (`flow_equation.py`)。
`disc -> 0` は2つの質量固有値が縮退する点で、`sqrt(disc)` は
(clip(disc,EPS,None)のEPS floorに引っかかっていなくても) その近傍で
微分がほぼ発散する平方根分岐点になる — 古典的な固有値交差の非平滑性。

warm start (tree-onlyリラクゼーション解) 上で `disc` の最小値を全格子・
全時刻について調べると **4.7e-7** (EPSの clip floor である1e-10には
達していないのでclip()自体は発火していないが、それでもsqrt微分は
ほぼ発散するくらい0に近い)。Newton-Krylovをstallさせた最終状態では
この最小値がさらに **4.5e-8** まで縮み (rho≈0.09, sigma=0,
t≈-1.85付近)、Newton反復がこの縮退点に向かって"引き寄せられ"ながら
そこで足踏みしていることを示している。

**結論**: フル方程式でのNewtonリラクゼーションの困難は、単なる
tachyonicクランプ(clip)ではなく、Higgs-singlet質量固有値のほぼ縮退
という、rloopの定義そのものに内在する非平滑性 (平方根分岐点) に
起因する可能性が高い。Jacobian-freeなNewton-Krylov (差分近似で
Jacobian-vector積を取る) は、この近傍でのほぼ発散する曲率を捉えられず
局所線形モデルが崩れ、収束が止まる。次に試すべき方向としては、
(a) sqrt(disc)を`sqrt(disc+delta^2)`のように少しだけ正則化する
(質量固有値の交差を人為的に避ける)、(b) 縮退点近傍だけ細かい
trust-regionを使う準Newton法に切り替える、(c) 解析的Jacobianを
組んで真のNewton法 (Jacobian-free近似に頼らない) を使う、などが
考えられる。

### 追試: domain restrict / disc_cut / mass_floor の効果検証

上記の「Higgs-singlet質量固有値のほぼ縮退」仮説を、3つの独立した正則化・回避策で
検証した (`--n-rho 21 --n-sigma 21 --n-t 40 --maxiter 100`、CN scheme)。

`relax_full.py` に `--rho-min`/`--sigma-min` (Grid2Dの下限、既存の`rho_min=0.0`
デフォルトを露出しただけ) と `--mass-floor` (新規、下記) を追加した。

**disc縮退はsigma=0/rho=0の軸上で起きやすい**: `M12 = 2*sqrt(rho*sigma)*u_rhosigma`
は rho=0 か sigma=0 の軸上で恒等的に0になるため、そこでは
disc=(M11-M22)^2+4*M12^2=0 という縮退条件が「M11=M22」だけ(codim-2→codim-1)に
弱まり、達成されやすくなる。実際、report済みの縮退点 (rho≈0.09, **sigma=0**,
t≈-1.85) はこの軸上だった。そこで rho,sigma を0から離す(`--rho-min --sigma-min`)
ことで、この軸自体を回避できるはず、という仮説を立てて検証した。

warm start (tree+thermal) 上で全格子・全時刻でのdiscの最小値:

| rho_min=sigma_min | min(disc) |
|---|---|
| 0 (デフォルト) | 3.7e-7 (原点で最小) |
| 1e-3 | 2.3e-5 (60倍改善) |
| 1e-2 | 2.3e-3 (さらに100倍改善) |

Newton-Krylovを実際に回した結果 (最終残差ノルム):

| 設定 | 最終残差ノルム |
|---|---|
| 何もなし (baseline, `full_n21.npz`) | 0.175 |
| disc_cut=1e-2 のみ | 0.1145 |
| **domain restrict (rho,sigma≥1e-3) のみ** | **0.0618 (最良)** |
| domain restrict + disc_cut=1e-2 | 0.0825 (restrictのみより悪化) |
| domain restrict + mass_floor=0.05 | 0.1587 (さらに悪化) |

**disc_cutはdomain restrictと併用すると無意味 (むしろ逆効果)**。domain restrict
だけで最終解のdiscは2.8e-5前後で高止まりし、baseline (制限なし) で見られた
「Newton反復がdisc→4.5e-8まで引き寄せられ続ける」現象自体が起きなくなる。
disc_cut=1e-2 (floor=1e-4) を追加してもこの2.8e-5をさらに切り上げるだけで
線形モデルの改善にはほぼ寄与せず、人為的な非平滑クリップを追加した分だけ
収束がわずかに悪化した。

**mass_floor (新規): 見せかけのtachyonic出現は原因ではなく症状だった。**
domain restrictのみで解いた最終解 (収束はしていない) を診断すると、disc縮退とは
無関係な場所 (rho=sigma=1.75の角、IR深部 t≈-1.85〜-2.0) で `1+m1_sq` が
0.02〜0.12までtachyonicに近づいていた。ところが同じ点をtree-only
(`u_tree_derivs`) や warm start (tree+thermal) で評価すると `1+m1_sq` は
0.58〜0.78で全く問題ない。さらにNewton停滞解のUの値そのものが同じ点で
warm startから6〜11も乖離しており (例: t=-2.0で Newton解U=-7.23 vs
warm start U=3.90)、**この見かけのtachyonicはNewtonが暴れて発散した先の
症状であって、独立した物理的障害ではない**ことが確認できた。

それでも「物理的に1+m^2がtachyonicになるのは非現実的」という理由で
`flow_equation.py::_rloop_from_derivs` に `mass_floor` パラメータを追加した
(disc_delta/disc_cutと同じパターン。`mass_floor>0`のとき
`_inv_sqrt`と有限温度項のEの両方で `clip(1+m^2, EPS, None)` の代わりに
`clip(1+m^2, mass_floor, None)` を使う。mG2, m1_sq, m2_sqにのみ適用、
gauge/topは1+m^2>=1で floor に触れないため無関係)。ただし
`--mass-floor 0.05` で試したところ収束はむしろ悪化した (0.0618→0.1587)。
症状(tachyonic clip)への対症療法は効かず、新たな非平滑クリップの折れ目を
追加しただけに終わった — Newtonが域の角で暴れる**根本原因**(境界の
finite-differenceステンシル、GMRES内部反復数、ステップ長制御など)は
未特定のまま。

**結論**: domain restrictはdisc縮退という「本物の非平滑性」の回避に有効
(disc_cutは重複的で不要)。一方、フル方程式のNewton-Krylovリラクゼーションは
どの組み合わせでも `f_tol=1e-8` まで収束せず、この非線形問題が
"そう簡単には解けない"ことのデモンストレーションとして機能している
(PINNsによる同じ問題の解法との比較対象として使用)。

### 未収束解の比較 (`plots/case_a_tree_slices_cw_thermal.png`)

![full flow eq. (未収束) の断面比較](plots/case_a_tree_slices_cw_thermal.png)

t_end=-2での断面 (左: rho=0固定、右: sigma=0固定)。tree/relax_tree/RGE-run
tree+CW (青丸・黒破線・緑) はほぼ一致する物理的な参照カーブ。**赤点線
(RGE-run tree+CW+thermal、解析的近似) が rloop+eta のフル物理に一番近いと
期待される参照**である一方、オレンジ (素のNewton-Krylov, 未収束) は
左図(rho=0)ではsigma=1.75にかけて赤点線に近づくが、右図(sigma=0)では
逆に赤点線から大きく外れてrho=1.75で-0.45まで突っ込む — 2つの断面を
同時には説明できていない。coupling prior併用/central4スキームの解
(紫・茶・ピンク) はどちらの断面でもtree系(~0.35 / ~-0.18)寄りに留まり、
赤点線の熱補正による持ち上がりをあまり再現できていない。

正則化・離散化スキームを変えるだけで未収束解が(赤点線に近づいたり遠ざかったり)
大きく揺れるという事実自体が、「フル方程式のNewton-Krylovリラクゼーションは
初期値・スキームに敏感で、単純にはtrueな解へ収束しない」ことを裏付けている。

## ファイル構成

```
_pathsetup.py    frg_discrete (兄弟フォルダ) を sys.path に追加する
relax_tree.py    tree-only: 中心差分+Crank-Nicolsonのブロック双対角"線形"系をLU分解+ブロック消去で解く
relax_full.py    フル方程式 (rloop+eta込み): 中心差分+CN/backward Eulerの"非線形"残差をnewton_krylovで解く
```

`perturbation/`, `running_couplings.py`, `seed_potential.py`,
`flow_equation.py` (eta_rho/eta_sigma, flow_rhs, EPS等) は frg_discrete
のものをそのまま import して使う。
