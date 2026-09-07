#!/bin/bash
set -e
cd "$(dirname "$0")"

run() {
  local tag=$1; shift
  echo "Launching $tag"
  python3 relax_full.py "$@" > "logs/${tag}.log" 2>&1 &
}

run T100_baseline      --T-raw 100 --rho-min 0.001 --sigma-min 0.001 --out results/full_n21_domrestrict_T100_baseline.npz
run T100_prior         --T-raw 100 --rho-min 0.001 --sigma-min 0.001 --coupling-prior-sign-weight 0.015 --coupling-prior-mag-weight 0.006 --out results/full_n21_domrestrict_T100_prior.npz
run T100_central4      --T-raw 100 --rho-min 0.001 --sigma-min 0.001 --spatial-scheme central4 --out results/full_n21_domrestrict_T100_central4.npz
run T100_central4_prior --T-raw 100 --rho-min 0.001 --sigma-min 0.001 --spatial-scheme central4 --coupling-prior-sign-weight 0.015 --coupling-prior-mag-weight 0.006 --out results/full_n21_domrestrict_T100_central4_prior.npz
run T200_baseline      --T-raw 200 --rho-min 0.001 --sigma-min 0.001 --out results/full_n21_domrestrict_T200_baseline.npz
run T200_prior         --T-raw 200 --rho-min 0.001 --sigma-min 0.001 --coupling-prior-sign-weight 0.015 --coupling-prior-mag-weight 0.006 --out results/full_n21_domrestrict_T200_prior.npz
run T200_central4      --T-raw 200 --rho-min 0.001 --sigma-min 0.001 --spatial-scheme central4 --out results/full_n21_domrestrict_T200_central4.npz
run T200_central4_prior --T-raw 200 --rho-min 0.001 --sigma-min 0.001 --spatial-scheme central4 --coupling-prior-sign-weight 0.015 --coupling-prior-mag-weight 0.006 --out results/full_n21_domrestrict_T200_central4_prior.npz

wait
echo "ALL_DONE"
