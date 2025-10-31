import numpy as np
import skfuzzy as fuzz
from skfuzzy import control as ctrl
import matplotlib.pyplot as plt

# ----------------------------
# 1️⃣ Definisi variabel fuzzy
# ----------------------------
permintaan = ctrl.Antecedent(np.arange(0, 5001, 1), 'permintaan')
persediaan = ctrl.Antecedent(np.arange(0, 1001, 1), 'persediaan')
produksi   = ctrl.Consequent(np.arange(0, 8001, 1), 'produksi')

# ----------------------------
# 2️⃣ Membership Functions
# ----------------------------
permintaan['turun'] = fuzz.trimf(permintaan.universe, [0, 1000, 3000])
permintaan['naik']  = fuzz.trimf(permintaan.universe, [1000, 3000, 5000])

persediaan['sedikit'] = fuzz.trimf(persediaan.universe, [0, 200, 430])
persediaan['sedang']  = fuzz.trimf(persediaan.universe, [200, 430, 800])
persediaan['banyak']  = fuzz.trimf(persediaan.universe, [430, 800, 1000])

produksi['berkurang'] = fuzz.trimf(produksi.universe, [0, 2000, 7000])
produksi['bertambah'] = fuzz.trimf(produksi.universe, [2000, 7000, 8000])

# ----------------------------
# 3️⃣ Aturan fuzzy (sesuai soal)
# ----------------------------
rule1 = ctrl.Rule(permintaan['turun'] & persediaan['banyak'], produksi['berkurang'])
rule2 = ctrl.Rule(permintaan['turun'] & persediaan['sedang'], produksi['berkurang'])
rule3 = ctrl.Rule(permintaan['turun'] & persediaan['sedikit'], produksi['bertambah'])
rule4 = ctrl.Rule(permintaan['naik']  & persediaan['banyak'], produksi['berkurang'])
rule5 = ctrl.Rule(permintaan['naik']  & persediaan['sedang'], produksi['bertambah'])
rule6 = ctrl.Rule(permintaan['naik']  & persediaan['sedikit'], produksi['bertambah'])

# ----------------------------
# 4️⃣ Sistem kontrol dan simulasi
# ----------------------------
produksi_ctrl = ctrl.ControlSystem([rule1, rule2, rule3, rule4, rule5, rule6])
produksi_simulasi = ctrl.ControlSystemSimulation(produksi_ctrl)

# Input crisp (bisa ubah sesuai soal)
produksi_simulasi.input['permintaan'] = 2000  # antara 1000–3000
produksi_simulasi.input['persediaan'] = 450   # antara 200–700

# Jalankan sistem fuzzy
produksi_simulasi.compute()

# ----------------------------
# 5️⃣ Tampilkan hasil
# ----------------------------
print("=== HASIL FUZZY MAMDANI ===")
print("Permintaan :", 2000)
print("Persediaan :", 450)
print(f"Produksi   : {produksi_simulasi.output['produksi']:.2f} kemasan")

# Visualisasi hasil fuzzy
produksi.view(sim=produksi_simulasi)
plt.show()