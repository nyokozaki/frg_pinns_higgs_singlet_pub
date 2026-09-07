import numpy as np

# 必要な関数をモジュールから呼び出す
# ※ get_running_couplings 等はチェック自体には不要なため除外しています
from thermal_np import scalar_masses_AE

def check_mass_positivity(t_min=-2.0, t_max=0.0, num_t=50,
                          rho_min=0.0, rho_max=2.0, num_rho=100,
                          sigma_min=0.0, sigma_max=2.0, num_sigma=100):
    """
    指定した t の範囲内で 1 + m2^2 (小さい方の固有値) が負にならないかをチェックする関数。

    Parameters:
    -----------
    t_min, t_max : float
        t の探索範囲 (デフォルト: -2.0 から 0.0)
    num_t : int
        t の分割数 (精度を上げたい場合は増やす)
    rho_min, rho_max, num_rho : float, float, int
        rho のグリッド設定
    sigma_min, sigma_max, num_sigma : float, float, int
        sigma のグリッド設定

    Returns:
    --------
    dict
        - 'is_all_positive': すべてのグリッド・t で 1 + m2^2 >= 0 なら True
        - 'global_min': 探索範囲全体での 1 + m2^2 の最小値
        - 'negative_points': 負になった場合の t とその時の最小値のリスト
    """
    
    # rho, sigma のグリッド作成 (tに依存しないためループ外で作成して使い回す)
    rho_vals = np.linspace(rho_min, rho_max, num_rho)
    sigma_vals = np.linspace(sigma_min, sigma_max, num_sigma)
    RHO, SIGMA = np.meshgrid(rho_vals, sigma_vals, indexing='xy')

    # t の配列
    t_vals = np.linspace(t_min, t_max, num_t)

    global_min = np.inf
    negative_points = []

    for t in t_vals:
        T_VAL = np.full_like(RHO, t)

        # 質量固有値の計算
        (mG2, m1, m2), _ = scalar_masses_AE(RHO, SIGMA, T_VAL)

        # 小さい方の固有値を抽出 (元コードの m_small に相当)
        m_small = np.minimum(m1, m2)

        # 1 + m_small の最小値を計算
        current_min = np.min(1.0 + m_small)

        if current_min < global_min:
            global_min = current_min

        # 負になった場合、該当する t とその時の最小値を記録
        if current_min < 0:
            negative_points.append({'t': t, 'min_val': current_min})

    is_all_positive = (global_min >= 0)

    return {
        'is_all_positive': is_all_positive,
        'global_min': global_min,
        'negative_points': negative_points
    }


# ============================================================
# モジュールとしてインポートされた時ではなく、直接実行された場合のテスト用コード
# ============================================================
if __name__ == "__main__":
    print("Starting positivity check for 1 + m^2 in t = [-2.0, 0.0]...")
    
    # 関数の実行 (分割数を変えたい場合は num_t, num_rho, num_sigma を調整してください)
    results = check_mass_positivity(
        t_min=-2.0, t_max=0.0, num_t=100, 
        rho_min=0.0, rho_max=2.0, num_rho=100,
        sigma_min=0.0, sigma_max=2.0, num_sigma=100
    )

    print("-" * 50)
    if results['is_all_positive']:
        print(f"✅ 問題ありません。すべての t において 1 + m^2 >= 0 です。")
        print(f"   (探索範囲内での全体最小値: {results['global_min']:.6f})")
    else:
        print(f"⚠️ 警告: 1 + m^2 が負になる領域が見つかりました。")
        print(f"   (探索範囲内での全体最小値: {results['global_min']:.6f})")
        print("【負になった t の値とその最小値】:")
        for pt in results['negative_points']:
            print(f"  t = {pt['t']:>7.4f}  |  min(1+m^2) = {pt['min_val']:.6f}")