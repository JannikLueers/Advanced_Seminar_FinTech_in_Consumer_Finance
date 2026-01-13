import pandas as pd
import numpy as np

# 1. Deine extrahierten Koeffizienten-Tabellen
coef_data = {
    'X': {
        'Month': list(range(-3, 13)),
        'Discretionary Spending': [0.00, 0.02, 0.01, -0.08, -0.11, -0.11, -0.17, -0.11, -0.11, -0.13, -0.13, -0.11,
                                   -0.12, -0.12, -0.12, -0.10],
        'Clothing': [0.00, -0.08, 0.08, -0.15, -0.12, -0.18, -0.22, -0.25, -0.21, -0.25, -0.28, -0.22, -0.20, -0.25,
                     -0.18, -0.15],
        'Entertainment': [0.00, 0.02, -0.01, -0.18, -0.14, -0.13, -0.12, -0.10, -0.18, -0.17, -0.20, -0.15, -0.14,
                          -0.16, -0.19, -0.13],
        'Restaurants': [0.01, 0.04, 0.01, -0.10, -0.11, -0.13, -0.10, -0.07, -0.15, -0.18, -0.14, -0.08, -0.09, -0.10,
                        -0.15, -0.12],
        'Travel': [0.00, 0.13, -0.05, -0.20, -0.21, -0.23, -0.20, -0.26, -0.29, -0.26, -0.24, -0.20, -0.24, -0.28,
                   -0.22, -0.18],
        'Cash Withdrawals': [0.00, -0.10, -0.05, -0.35, -0.25, -0.28, -0.30, -0.15, -0.42, -0.38, -0.30, -0.22, -0.18,
                             -0.32, -0.34, -0.31]
    },
    'N': {
        'Month': list(range(-3, 13)),
        'Discretionary Spending': [0.00, -0.01, -0.03, -0.08, -0.10, -0.09, -0.10, -0.11, -0.11, -0.13, -0.13, -0.14,
                                   -0.13, -0.12, -0.13, -0.14],
        'Clothing': [-0.01, -0.03, 0.03, -0.11, -0.11, -0.12, -0.14, -0.15, -0.12, -0.08, -0.14, -0.17, -0.19, -0.16,
                     -0.14, -0.16],
        'Entertainment': [0.00, 0.02, 0.01, -0.07, -0.09, -0.07, -0.09, -0.09, -0.15, -0.10, -0.09, -0.10, -0.09, -0.10,
                          -0.10, -0.10],
        'Restaurants': [0.00, -0.01, 0.01, -0.08, -0.11, -0.09, -0.11, -0.10, -0.11, -0.13, -0.13, -0.15, -0.16, -0.15,
                        -0.14, -0.15],
        'Travel': [0.00, -0.03, -0.01, -0.14, -0.16, -0.14, -0.17, -0.19, -0.20, -0.18, -0.15, -0.17, -0.16, -0.17,
                   -0.20, -0.18],
        'Cash Withdrawals': [0.00, -0.02, 0.02, -0.15, -0.22, -0.25, -0.23, -0.26, -0.29, -0.32, -0.28, -0.30, -0.33,
                             -0.29, -0.27, -0.26]
    }
}

# 2. Simulations-Parameter
categories = ["Discretionary Spending", "Clothing", "Entertainment", "Restaurants", "Travel", "Cash Withdrawals"]
n_users_per_sample = 500  # 500 Nutzer pro Gruppe für statistische Signifikanz
noise_sd = 0.05  # Standardabweichung des Rauschens

data_list = []
np.random.seed(42)

for sample_label in ['X', 'N']:
    for i in range(n_users_per_sample):
        user_id = f"{sample_label}_{i}"
        # Individueller Basiswert pro Nutzer (Fixed Effect)
        user_base_log = {cat: np.random.normal(5.0, 0.5) for cat in categories}

        for m_idx, month in enumerate(coef_data[sample_label]['Month']):
            row = {
                'user_id': user_id,
                'sample': sample_label,
                'event_month': month,
                'post': 1 if month >= 0 else 0
            }
            for cat in categories:
                target_coef = coef_data[sample_label][cat][m_idx]
                # Log-Spending = Basis + Koeffizient + Rauschen
                row[cat] = user_base_log[cat] + target_coef + np.random.normal(0, noise_sd)
            data_list.append(row)

# 3. Speichern
df_sim = pd.DataFrame(data_list)
df_sim.to_csv('mind_the_app_category_spending.csv', index=False)

###


# Verifikation für Sample X - Restaurants
sample_x = df_sim[df_sim['sample'] == 'X']
means = sample_x.groupby('event_month')['Restaurants'].mean()
relative_effects = means - means.loc[-1]
print(relative_effects.round(2))