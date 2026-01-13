import pandas as pd
import numpy as np

# 1. HARDGECODETE KOEFFIZIENTEN (Basierend auf den extrahierten Werten aus dem Paper)
coefficients = {
    'X': {
        'Months': list(range(-3, 13)),
        # Logins (Absoluter Zuwachs in Logins)
        'All_Logins': [0.0, -0.2, -0.1, 11.5, 12.0, 12.2, 12.3, 13.0, 12.4, 11.2, 12.2, 12.1, 12.0, 11.9, 13.2, 13.2],
        'PC_Logins': [0.0, -0.2, -0.1, -0.1, 0.0, 0.6, 0.9, 1.1, 0.5, -0.2, -0.2, -0.3, -0.1, -0.3, -0.1, -0.1],
        # Spending (Log-Veränderung)
        'Discretionary': [0.00, 0.02, 0.01, -0.08, -0.11, -0.11, -0.17, -0.11, -0.11, -0.13, -0.13, -0.11, -0.12, -0.12,
                          -0.12, -0.10],
        'Clothing': [0.00, -0.08, 0.08, -0.15, -0.12, -0.18, -0.22, -0.25, -0.21, -0.25, -0.28, -0.22, -0.20, -0.25,
                     -0.18, -0.15],
        'Entertainment': [0.00, 0.02, -0.01, -0.18, -0.14, -0.13, -0.12, -0.10, -0.18, -0.17, -0.20, -0.15, -0.14,
                          -0.16, -0.19, -0.13],
        'Restaurants': [0.00, 0.04, 0.01, -0.10, -0.11, -0.13, -0.10, -0.07, -0.15, -0.18, -0.14, -0.08, -0.09, -0.10,
                        -0.15, -0.12],
        'Travel': [0.00, 0.13, -0.05, -0.20, -0.21, -0.23, -0.20, -0.26, -0.29, -0.26, -0.24, -0.20, -0.24, -0.28,
                   -0.22, -0.18],
        'Cash_Withdrawals': [0.00, -0.10, -0.05, -0.35, -0.25, -0.28, -0.30, -0.15, -0.42, -0.38, -0.30, -0.22, -0.18,
                             -0.32, -0.34, -0.31]
    },
    'N': {
        'Months': list(range(-3, 13)),
        # Logins (Absoluter Zuwachs in Logins)
        'All_Logins': [0.0, 0.2, 0.9, 8.0, 7.8, 6.5, 6.4, 6.2, 6.0, 5.9, 6.0, 5.7, 5.6, 5.5, 5.6, 5.8],
        'PC_Logins': [0.0, 0.2, 0.9, 2.5, 1.8, 1.0, 0.9, 0.8, 0.7, 0.4, 0.4, 0.2, 0.1, 0.0, 0.0, 0.0],
        # Spending (Log-Veränderung)
        'Discretionary': [0.00, -0.01, -0.03, -0.08, -0.10, -0.09, -0.10, -0.11, -0.11, -0.13, -0.13, -0.14, -0.13,
                          -0.12, -0.13, -0.14],
        'Clothing': [0.00, -0.03, 0.03, -0.11, -0.11, -0.12, -0.14, -0.15, -0.12, -0.08, -0.14, -0.17, -0.19, -0.16,
                     -0.14, -0.16],
        'Entertainment': [0.00, 0.02, 0.01, -0.07, -0.09, -0.07, -0.09, -0.09, -0.15, -0.10, -0.09, -0.10, -0.09, -0.10,
                          -0.10, -0.10],
        'Restaurants': [0.00, -0.01, 0.01, -0.08, -0.11, -0.09, -0.11, -0.10, -0.11, -0.13, -0.13, -0.15, -0.16, -0.15,
                        -0.14, -0.15],
        'Travel': [0.00, -0.03, -0.01, -0.14, -0.16, -0.14, -0.17, -0.19, -0.20, -0.18, -0.15, -0.17, -0.16, -0.17,
                   -0.20, -0.18],
        'Cash_Withdrawals': [0.00, -0.02, 0.02, -0.15, -0.22, -0.25, -0.23, -0.26, -0.29, -0.32, -0.28, -0.30, -0.33,
                             -0.29, -0.27, -0.26]
    }
}


# 2. SIMULATIONS-LOGIK
def generate_final_dataset(n_users_per_sample=1000):
    np.random.seed(42)
    data_rows = []

    spending_categories = ["Discretionary", "Clothing", "Entertainment", "Restaurants", "Travel", "Cash_Withdrawals"]
    login_categories = ["All_Logins", "PC_Logins"]

    for sample in ['X', 'N']:
        for i in range(n_users_per_sample):
            user_id = f"{sample}_{i}"

            # Individuelle Basis-Levels (Fixed Effects)
            user_fe_spending = {cat: np.random.normal(5.0, 0.3) for cat in spending_categories}
            #user_fe_spending = {cat: np.random.normal(1.0, 100.0) for cat in spending_categories}

            user_fe_logins = {l: np.random.normal(5.0, 1.0) for l in login_categories}
            #user_fe_logins = {l: np.random.normal(5.0, 1.0) for l in login_categories} #

            for m_idx, m in enumerate(range(-3, 13)):
                post = 1 if m >= 0 else 0
                row = {
                    'user_id': user_id,
                    'sample': sample,
                    'event_month': m,
                    'post': post
                }

                # Logins hinzufügen (Lineare Skala)
                for l in login_categories:
                    coef = coefficients[sample][l][m_idx]
                    row[l] = user_fe_logins[l] + coef + np.random.normal(0, 0.1)

                # Spending hinzufügen (Log-Skala)
                for cat in spending_categories:
                    coef = coefficients[sample][cat][m_idx]
                    row[cat] = user_fe_spending[cat] + coef + np.random.normal(0, 0.05)

                data_rows.append(row)

    return pd.DataFrame(data_rows)


# 3. DATENSATZ ERSTELLEN UND SPEICHERN
df_final = generate_final_dataset()

# Spalten sortieren für bessere Lesbarkeit
cols_order = ['user_id', 'sample', 'event_month', 'post', 'All_Logins', 'PC_Logins'] + \
             ["Discretionary", "Clothing", "Entertainment", "Restaurants", "Travel", "Cash_Withdrawals"]
df_final = df_final[cols_order]

df_final.to_csv('mind_the_app_mock_data_login_a_category_spending.csv', index=False)
print("Vollständiger Datensatz mit hardgecodeten Werten für Logins und Spendings erstellt.")
print(df_final.head())