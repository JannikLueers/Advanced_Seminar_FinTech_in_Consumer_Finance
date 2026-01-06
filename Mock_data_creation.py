import pandas as pd
import numpy as np

# ==========================================
# KONFIGURATION (Die "Knobs" zum Anpassen)
# ==========================================

# 1. Stichprobengröße (Anzahl der Nutzer)
# Paper: Sample X (729), Sample N (6873) [cite: 181, 190]
# Wir reduzieren Sample N hier etwas für die Übersichtlichkeit,
# behalten aber das Verhältnis grob bei.
N_USERS_X = 750
N_USERS_N = 1500

# 2. Zeitraum (Relativ zur Installation)
# Paper: t = -3 bis t = +12 [cite: 59]
TIME_RANGE = range(-3, 13)

# 3. Parameter für LOGINS (Login-Häufigkeit pro Monat)
# Siehe Figure 2 und Table 3 im Paper [cite: 224, 227, 228]
# Sample X: Startet bei ~4, steigt um ~10.7 auf ~14.7
BASE_LOGINS_X = 4.0
EFFECT_LOGINS_X = 10.7

# Sample N: Startet bei ~4, steigt auf ~10 (Effekt ~6.0), hat aber Pre-Trends
BASE_LOGINS_N = 4.0
EFFECT_LOGINS_N = 6.0

# 4. Parameter für SPENDING (Diskretionäre Ausgaben in $)
# Siehe Table 2 und Table 4 [cite: 188, 192, 253, 256]
# Sample X: ~3700$ Basis, -11.6% Effekt
BASE_SPEND_X = 3700
EFFECT_SPEND_PCT_X = -0.116

# Sample N: ~2500$ Basis, -7.6% Effekt
BASE_SPEND_N = 2500
EFFECT_SPEND_PCT_N = -0.076

# 5. Rauschen (Noise)
# Setze dies auf 0.0 für absolut "perfekte" deterministische Daten.
# Setze es höher (z.B. 0.5 oder 1.0), um realistische Schwankungen zu simulieren.
NOISE_LEVEL = 0.05


def generate_mock_data():
    """
    Erstellt einen Mock-Datensatz basierend auf 'Mind the App'.
    """
    np.random.seed(42)  # Für reproduzierbare Ergebnisse

    # -------------------------------------------------------
    # SCHRITT 1: Nutzer-Basis erstellen (Cross-Sectional)
    # -------------------------------------------------------

    # Liste der User IDs für beide Gruppen
    users_x = pd.DataFrame({'user_id': [f'X_{i}' for i in range(N_USERS_X)], 'group': 'Sample X'})
    users_n = pd.DataFrame({'user_id': [f'N_{i}' for i in range(N_USERS_N)], 'group': 'Sample N'})

    all_users = pd.concat([users_x, users_n], ignore_index=True)

    # Zuweisung von statischen Eigenschaften (Einkommen, Basis-Ausgaben-Level)
    # Wir nutzen Normalverteilungen um die Mittelwerte aus Table 2 [cite: 184, 191]
    # Sample X ist etwas reicher als Sample N.

    # Einkommen (Income)
    all_users['base_income'] = np.where(
        all_users['group'] == 'Sample X',
        np.random.normal(16900, 2000 * NOISE_LEVEL, len(all_users)),  # Mean ~16.9k
        np.random.normal(14300, 2000 * NOISE_LEVEL, len(all_users))  # Mean ~14.3k
    )

    # Individueller "Fixed Effect" für Ausgaben (Manche geben immer mehr aus als andere)
    all_users['user_spend_fe'] = np.random.normal(1.0, 0.1 * NOISE_LEVEL, len(all_users))

    # -------------------------------------------------------
    # SCHRITT 2: Panel-Struktur erstellen (Long Format)
    # -------------------------------------------------------

    # Jeden Nutzer für jeden Monat im Zeitraum replizieren
    # Das nennt man "Cartesian Product" oder "Cross Join"
    time_df = pd.DataFrame({'event_month': list(TIME_RANGE)})

    df = all_users.merge(time_df, how='cross')

    # Sortieren für Übersichtlichkeit
    df = df.sort_values(['group', 'user_id', 'event_month']).reset_index(drop=True)

    # -------------------------------------------------------
    # SCHRITT 3: Variablen simulieren (Time-Series Logic)
    # -------------------------------------------------------

    # Variable: POST (Dummy: 1 wenn nach Installation, sonst 0) [cite: 210]
    df['post'] = (df['event_month'] >= 0).astype(int)

    # --- A. SIMULATION DER LOGINS ---

    # Basis-Rauschen pro Monat
    random_noise_login = np.random.normal(0, 1.0 * NOISE_LEVEL, len(df))

    # Logik für Sample X:
    # Konstanter PC Login (~4) + Starker Anstieg Mobile ab t=0
    # Siehe Figure 2 (c) [cite: 227]

    # Logik für Sample N:
    # Endogener Anstieg VOR t=0 (Pre-Trend), dann Anstieg auf neues Level
    # Siehe Figure 2 (d) [cite: 228]

    conditions = [
        (df['group'] == 'Sample X'),
        (df['group'] == 'Sample N')
    ]

    # 1. Basis-Level (Intercept)
    base_logins = np.select(conditions, [BASE_LOGINS_X, BASE_LOGINS_N])

    # 2. Treatment Effekt (Nur wenn Post == 1)
    treatment_effect = np.select(conditions, [EFFECT_LOGINS_X, EFFECT_LOGINS_N]) * df['post']

    # 3. Sample N spezifischer Pre-Trend (Der "Endogenitäts-Peak")
    # Wir simulieren einen kleinen Anstieg bei t=-1 und t=0 für Sample N
    pre_trend_n = np.where(
        (df['group'] == 'Sample N') & (df['event_month'].isin([-1, 0])),
        2.5,  # Künstlicher Peak von ca. 2.5 Logins extra
        0
    )

    df['logins_total'] = base_logins + treatment_effect + pre_trend_n + random_noise_login
    # Runden, da es keine halben Logins gibt, und sicherstellen, dass es >= 0 ist
    df['logins_total'] = np.maximum(df['logins_total'], 0).round().astype(int)

    # --- B. SIMULATION DER AUSGABEN (Spending) ---

    # Basis-Ausgaben je nach Gruppe [cite: 188, 192]
    base_spend_amount = np.select(
        conditions,
        [BASE_SPEND_X, BASE_SPEND_N]
    )

    # Prozentualer Schock (Treatment)
    # Sample X: -11.6%, Sample N: -7.6% [cite: 253, 256]
    spend_shock = np.select(
        conditions,
        [EFFECT_SPEND_PCT_X, EFFECT_SPEND_PCT_N]
    ) * df['post']

    # Berechnung:
    # Spending = (Basis * User_Fixed_Effect) * (1 + Schock) * Zufallsrauschen
    random_noise_spend = np.random.normal(1, 0.05 * NOISE_LEVEL, len(df))

    df['discretionary_spending'] = (
            base_spend_amount * df['user_spend_fe'] * (1 + spend_shock) * random_noise_spend
    )

    # Log Spending (für die Regression oft genutzt, siehe Table 4 )
    df['log_discretionary_spending'] = np.log(df['discretionary_spending'])

    return df


# ==========================================
# DATEN GENERIEREN UND PRÜFEN
# ==========================================

# Datensatz erstellen
df_mock = generate_mock_data()

# Kurzer Check der Ergebnisse (Mittelwerte vor und nach t=0)
print("Check: Durchschnittliche Logins pro Monat (Vorher vs. Nachher)")
summary = df_mock.groupby(['group', 'post'])[['logins_total', 'discretionary_spending']].mean().round(2)
print(summary)

print("\nCheck: Vorschau der ersten 5 Zeilen für einen User aus Sample X:")
print(df_mock[df_mock['user_id'] == 'X_0'].head(5)[
          ['user_id', 'event_month', 'post', 'logins_total', 'discretionary_spending']])

# ==========================================
# EXPORT TO EXCEL
# ==========================================

# This saves the dataframe 'df_mock' to an Excel file named 'mock_data.xlsx'.
# index=False ensures that the pandas row numbers (0, 1, 2...) are not saved as a separate column.

# df_mock.to_excel("mind_the_app_mock_data.xlsx", index=False)

# print("Success! The file 'mind_the_app_mock_data.xlsx' has been saved.")