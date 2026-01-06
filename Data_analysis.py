import pandas as pd
from linearmodels import PanelOLS
import statsmodels.api as sm

# ==========================================
# 1. DATEN LADEN
# ==========================================
# Wir laden den Datensatz, den wir im vorherigen Schritt erstellt haben.
file_path = "mind_the_app_mock_data.xlsx"
# Falls du CSV genutzt hast, ändere dies zu .csv und pd.read_csv
df = pd.read_excel(file_path)

print(f"Daten geladen: {df.shape[0]} Zeilen.")


# ==========================================
# 2. DEFINITION DER REGRESSIONS-FUNKTION
# ==========================================
def run_fixed_effects_model(data, sample_name, dependent_var):
    """
    Führt die Regression nach Gleichung (1) aus dem Paper durch.
    y_it = beta * Post_it + Fe_i + Fe_t + e_it
    """
    print(f"\n--- Analyse für {sample_name}: Abhängige Variable '{dependent_var}' ---")

    # SCHRITT A: Daten filtern
    # Wir nehmen nur die User der entsprechenden Gruppe (Sample X oder Sample N)
    df_sample = data[data['group'] == sample_name].copy()

    # SCHRITT B: MultiIndex setzen
    # linearmodels benötigt einen Index aus [Entity, Time]
    # Entity = user_id (für den Term delta_i)
    # Time = event_month (für den Term gamma_j)
    df_sample = df_sample.set_index(['user_id', 'event_month'])

    # SCHRITT C: Variablen definieren
    # Y = Abhängige Variable (z.B. log spending)
    y = df_sample[dependent_var]

    # X = Unabhängige Variable (Post Dummy)
    # Wir fügen eine Konstante hinzu (Intercept), das ist Standard in statsmodels/linearmodels
    x = df_sample[['post']]
    x = sm.add_constant(x)

    # SCHRITT D: Das Modell aufsetzen (PanelOLS)
    # entity_effects=True  -> Aktiviert User Fixed Effects (delta_i)
    # time_effects=True    -> Aktiviert Time Fixed Effects (gamma_j)
    # drop_absorbed=True   -> Entfernt Variablen, die durch FE erklärt werden (hier nicht der Fall, aber gute Praxis)
    mod = PanelOLS(y, x, entity_effects=True, time_effects=True, drop_absorbed=True)

    # SCHRITT E: Modell fitten (Das eigentliche Rechnen)
    # cov_type='clustered' -> Wir nutzen "Clustered Standard Errors" auf User-Level.
    # Das ist wichtig! Im Paper steht: "cluster the standard errors at the individual level".
    # Das macht die p-Werte robuster.
    res = mod.fit(cov_type='clustered', cluster_entity=True)

    # Ergebnis ausgeben
    print(res.summary)

    return res


# ==========================================
# 3. ANALYSE DURCHFÜHREN
# ==========================================

# --- Analyse 1: Ausgaben (Spending) ---
# Wir nutzen den Logarithmus der Ausgaben, wie in Table 4[cite: 791].
# Interpretation: Der Koeffizient ist die prozentuale Änderung.

# Für Sample X (Die "Guten" Daten - Kausaler Effekt)
res_spend_x = run_fixed_effects_model(df, 'Sample X', 'log_discretionary_spending')

# Für Sample N (Die "Verrauschten" Daten - Endogen)
res_spend_n = run_fixed_effects_model(df, 'Sample N', 'log_discretionary_spending')

# --- Analyse 2: Logins (Attention) ---
# Analyse der Login-Häufigkeit wie in Table 3[cite: 781].
# Hier nehmen wir die absoluten Zahlen ('logins_total'), keinen Logarithmus.

# Für Sample X
res_login_x = run_fixed_effects_model(df, 'Sample X', 'logins_total')

# ==========================================
# 4. KURZINTERPRETATION DER ERGEBNISSE
# ==========================================
print("\n=== ZUSAMMENFASSUNG & VERGLEICH MIT MOCK-PARAMETERN ===")
print("Erinnerung an unsere Mock-Parameter:")
print("  - Sample X Spending Effekt: -11.6% (-0.116)")
print("  - Sample X Login Effekt:    +10.7")

print("\nGemessene Ergebnisse (Koeffizient von 'post'):")
print(f"  - Sample X Spending (beta): {res_spend_x.params['post']:.4f}")
print(f"  - Sample X Logins (beta):   {res_login_x.params['post']:.4f}")

if abs(res_spend_x.params['post'] - (-0.116)) < 0.01:
    print("\n✅ Erfolg! Das Modell hat die wahren Parameter der Simulation fast exakt wiedergefunden.")
else:
    print("\n⚠️ Abweichung! Prüfe, ob 'Noise' im Datensatz-Generator zu hoch eingestellt war.")