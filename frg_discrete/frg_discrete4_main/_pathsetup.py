"""
frg_discrete (兄弟フォルダ) の検証済みモジュール群
(perturbation/, running_couplings.py, seed_potential.py, flow_equation.py)
を import できるよう sys.path に追加する。

frg_discrete4 はUVマッチング・種ポテンシャルの中身を frg_discrete から
そのまま再利用し、"shooting" (t=0からの初期値問題としてforward積分) の
代わりに、空間+時間をまとめて離散化してグローバルに解く
リラクゼーション法 (Numerical Recipes 17.3節の意味での relaxation method)
を試す。
"""

import os
import sys

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_FRG_DISCRETE_DIR = os.path.normpath(os.path.join(_THIS_DIR, "..", "frg_discrete"))

if _FRG_DISCRETE_DIR not in sys.path:
    sys.path.insert(0, _FRG_DISCRETE_DIR)
