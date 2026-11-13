# Práctica 6 - AHPSort II: clasificación de provincias por nivel de seguridad
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd

CRIT = ["Crímenes", "Robos", "Vehículos", "Orden público", "Drogas"]
CLASES = ["Alto", "Medio", "Bajo"]          # nivel de seguridad: CP1, CP2, CP3
COLOR = {"Alto": "#2e9e5b", "Medio": "#f2c14e", "Bajo": "#d1495b"}
ANDALUCIA = ["Almería", "Cádiz", "Córdoba", "Granada", "Huelva", "Jaén", "Málaga", "Sevilla"]

# Prioridades locales de los 6 RP (Tabla 13), una fila por criterio
PRIOR_RP = [[.336, .132, .086, .032, .026, .023],    # Crímenes
            [.320, .139, .074, .047, .031, .022],    # Robos
            [.459, .124, .069, .048, .034, .026],    # Vehículos
            [.308, .149, .071, .060, .032, .023],    # Orden público
            [.385, .112, .062, .033, .023, .018]]    # Drogas

# ---------------------------------------------------------------- 1. Pesos AHP (Tabla 4)
A = np.array([[1, 5, 3, 7, 3], [1/5, 1, 3, 1/2, 1/4], [1/3, 1/3, 1, 1/5, 1/5],
              [1/7, 2, 5, 1, 1/5], [1/3, 4, 5, 5, 1]])
valores, vectores = np.linalg.eig(A)             # el peso es el autovector del mayor autovalor
k = valores.real.argmax()
pesos = np.abs(vectores[:, k].real)
pesos = pesos / pesos.sum()                      # normalizado para que sumen 1
CI = (valores[k].real - 5) / 4                   # índice de consistencia (n = 5)
CR = CI / 1.12                                   # razón de consistencia (RI = 1,12 para n = 5)
print("Pesos:", dict(zip(CRIT, pesos.round(4).tolist())))
print(f"lambda_max = {valores[k].real:.4f}  CI = {CI:.4f}  CR = {CR:.4f} ->",
      "válido" if CR < 0.1 else "NO válido (CR > 0,10)")

# ---------------------------------------------------------------- funciones AHPSort
def puntos_referencia(maximos):
    """6 puntos equiespaciados entre 0 y el máximo de cada criterio (filas RP1..RP6)."""
    rp = []
    for i in range(6):
        rp.append([i / 5 * m for m in maximos])
    return np.array(rp)

def perfiles_centrales(tasas):
    """CP1, CP2, CP3 = percentiles 20, 50 y 80 de cada criterio."""
    cp = []
    for p in [20, 50, 80]:
        cp.append([np.percentile(tasas[c], p) for c in CRIT])
    return np.array(cp)

def curvas_desde_rp(rp):
    """Función de cada criterio: puntos (x = valor del RP, y = su prioridad)."""
    return [(rp[:, j], PRIOR_RP[j]) for j in range(5)]

def prioridades_locales(valores, curvas):
    """Prioridad local de cada valor = interpolación lineal sobre la curva de su criterio.
    np.interp(x, xs, ys) busca x entre los puntos xs y calcula su y en la recta que los une;
    si x se sale del rango, devuelve la prioridad del extremo (truncamiento)."""
    resultado = np.zeros(valores.shape)
    for j in range(5):
        xs, ys = curvas[j]
        resultado[:, j] = np.interp(valores[:, j], xs, ys)
    return resultado

def clasificar(tasas, cp, curvas):
    """Prioridad global = suma(peso * prioridad local). Cada alternativa va al CP más cercano."""
    g_alt = prioridades_locales(tasas[CRIT].values, curvas) @ pesos     # score de cada provincia
    loc_cp = prioridades_locales(cp, curvas)
    g_cp = loc_cp @ pesos                                              # score de CP1, CP2, CP3
    clase = []
    for g in g_alt:                               # para cada provincia
        distancias = np.abs(g - g_cp)             # distancia a CP1, CP2, CP3
        clase.append(CLASES[distancias.argmin()])  # el más cercano: Alto, Medio o Bajo
    res = pd.DataFrame({"Prioridad global": g_alt, "Nivel de seguridad": clase}, index=tasas.index)
    tabla_cp = pd.DataFrame(loc_cp, index=CLASES, columns=CRIT)
    tabla_cp["Global"] = g_cp
    return res, tabla_cp

# ---------------------------------------------------------------- datos
# Tabla 1 (Andalucía 2017, hechos conocidos)
abs_and = pd.DataFrame({"Crímenes": [23, 61, 20, 32, 8, 17, 58, 86],
                        "Robos": [17284, 26571, 14508, 19139, 11222, 7548, 46137, 52849],
                        "Vehículos": [2101, 3945, 1691, 2299, 1409, 718, 6377, 9662],
                        "Orden público": [387, 722, 219, 368, 156, 175, 830, 716],
                        "Drogas": [267, 818, 148, 622, 139, 129, 934, 415]}, index=ANDALUCIA)

# Los ficheros escriben algunos nombres distinto que el mapa; se unifican con el del mapa
ALIAS = {"Alicante/Alacant": "Alacant/Alicante", "Balears (Illes)": "Illes Balears",
         "Balears, Illes": "Illes Balears", "Castellón/Castelló": "Castelló/Castellón",
         "Coruña (A)": "A Coruña", "Coruña, A": "A Coruña", "Rioja (La)": "La Rioja",
         "Rioja, La": "La Rioja", "Valencia/València": "València/Valencia",
         "Palmas (Las)": "Las Palmas", "Palmas, Las": "Las Palmas", "Álava": "Araba/Álava",
         "Alicante": "Alacant/Alicante", "Baleares": "Illes Balears", "Castellón": "Castelló/Castellón",
         "Coruña": "A Coruña", "Rioja": "La Rioja", "Valencia": "València/Valencia",
         "Palmas": "Las Palmas", "Santa Cruz": "Santa Cruz de Tenerife"}

def cargar_tasas(fichero, hoja=0):
    d = pd.read_excel("data/" + fichero, sheet_name=hoja)
    d["Provincia"] = d["Provincia"].replace(ALIAS)
    return d.set_index("Provincia")[CRIT]

hab = pd.read_excel("data/habitantes.xlsx", header=None, skiprows=2, usecols=[1, 2])
hab.columns = ["Provincia", "Habitantes"]
hab["Provincia"] = hab["Provincia"].replace(ALIAS)
hab = hab.set_index("Provincia")["Habitantes"].astype(float)

# ---------------------------------------------------------------- mapas (geopandas)
def pintar_mapa(ax, ruta, campo, res, titulo, con_valores=False):
    """Dibuja el mapa coloreando cada provincia según su nivel de seguridad."""
    mapa = gpd.read_file(ruta)
    clases = res["Nivel de seguridad"].to_dict()
    mapa["color"] = [COLOR.get(clases.get(nombre), "#dddddd") for nombre in mapa[campo]]
    mapa.plot(ax=ax, color=mapa["color"], edgecolor="white")
    if con_valores:                                  # escribe nombre y prioridad global
        for _, fila in mapa.iterrows():
            punto = fila.geometry.centroid
            valor = res.loc[fila[campo], "Prioridad global"]
            ax.text(punto.x, punto.y, f"{fila[campo]}\n{valor:.3f}", ha="center", fontsize=7)
    ax.set_title(titulo)
    ax.axis("off")
    ax.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=COLOR[c]) for c in CLASES],
              labels=["Alto (criminalidad baja)", "Medio", "Bajo (criminalidad alta)"],
              title="Nivel de seguridad", loc="lower right", fontsize=7, frameon=False)

RUTA_AND = "data/Mapas/Andalucia/da03_provincia.shp"
RUTA_ESP = "data/Mapas/Espana/recintos_provinciales_inspire_peninbal_etrs89.shp"

def mapa_andalucia(res, titulo, fichero):
    fig, ax = plt.subplots(figsize=(8, 5.5))
    pintar_mapa(ax, RUTA_AND, "PROVINCIA", res, titulo, con_valores=True)
    fig.savefig("resultados/" + fichero, dpi=150, bbox_inches="tight")

def mapas_espana(lista, fichero):
    """lista = [(resultado, titulo), ...]; todos los mapas llevan la misma leyenda."""
    fig, ejes = plt.subplots(1, len(lista), figsize=(7.5 * len(lista), 6.5), squeeze=False)
    for ax, (res, titulo) in zip(ejes[0], lista):
        pintar_mapa(ax, RUTA_ESP, "NAMEUNIT", res, titulo)
    fig.savefig("resultados/" + fichero, dpi=170, bbox_inches="tight")

def mostrar(titulo):
    print("\n" + "=" * 70 + "\n" + titulo + "\n" + "=" * 70)

pd.set_option("display.width", 200, "display.float_format", "{:.4f}".format)
NOMBRES_RP = ["RP1", "RP2", "RP3", "RP4", "RP5", "RP6"]

# ---------------------------------------------------------------- Ejercicio 1 (datos absolutos)
mostrar("EJERCICIO 1: Andalucía 2017, datos absolutos")
rp1 = puntos_referencia([206, 254572, 35127, 3824, 1608])                  # Tabla 5
cp1 = np.array([[20, 5000, 5000, 300, 300], [100, 95000, 19000, 1800, 600],
                [160, 180000, 27000, 2500, 900]], float)                   # Tabla 3
prior_cp1 = [[.277, .261, .138, .259, .262], [.058, .075, .063, .066, .070],
             [.029, .032, .040, .034, .035]]                               # Tabla 6
# La función de cada criterio pasa por los 6 RP y por los 3 CP (se ordenan de menor a mayor x)
curvas1 = []
for j in range(5):
    puntos = []
    for i in range(6):
        puntos.append((rp1[i, j], PRIOR_RP[j][i]))
    for i in range(3):
        puntos.append((cp1[i, j], prior_cp1[i][j]))
    puntos.sort()
    curvas1.append(([p[0] for p in puntos], [p[1] for p in puntos]))
res1, tabla7 = clasificar(abs_and, cp1, curvas1)
print("Tabla 5 (RP):\n", pd.DataFrame(rp1, index=NOMBRES_RP, columns=CRIT))
print("\nTabla 7:\n", tabla7)
print("\nClasificación:\n", res1)
tabla7.to_csv("resultados/ej1_tabla7.csv"); res1.to_csv("resultados/ej1_clasificacion.csv")
mapa_andalucia(res1, "Ejercicio 1: nivel de seguridad (datos absolutos)", "ej1_mapa.png")

# ---------------------------------------------------------------- Ejercicio 2 (por 10.000 hab.)
mostrar("EJERCICIO 2: Andalucía 2017 por 10.000 habitantes")
tasas_and = abs_and.astype(float)
for prov in ANDALUCIA:
    tasas_and.loc[prov] = abs_and.loc[prov] / hab[prov] * 10000
rp2 = puntos_referencia(tasas_and.max())
cp2 = perfiles_centrales(tasas_and)
res2, tabla_cp2 = clasificar(tasas_and, cp2, curvas_desde_rp(rp2))
print("Tasas por 10.000 hab.:\n", tasas_and)
print("\nTabla 8 (CP):\n", pd.DataFrame(cp2, index=CLASES, columns=CRIT))
print("\nTabla 9 (RP):\n", pd.DataFrame(rp2, index=NOMBRES_RP, columns=CRIT))
print("\nPrioridades de los CP:\n", tabla_cp2)
print("\nClasificación:\n", res2)
res2.to_csv("resultados/ej2_clasificacion.csv")
mapa_andalucia(res2, "Ejercicio 2: nivel de seguridad (por 10.000 hab.)", "ej2_mapa.png")

# ---------------------------------------------------------------- Ejercicio 3 (España 2017)
mostrar("EJERCICIO 3: España 2017, 50 provincias")
tasas17 = cargar_tasas("Datos_nacionales_2017_por_10000_ES.xlsx")
# (a) verificación: hechos absolutos / tasa = misma población para los 5 criterios
poblacion = abs_and / tasas17.loc[ANDALUCIA] * 10000
print("(a) Población implícita por criterio (debe ser igual en las 5 columnas):\n", poblacion.round(0))
rp3 = puntos_referencia(tasas17.max())
cp3 = perfiles_centrales(tasas17)
curvas3 = curvas_desde_rp(rp3)
res3, tabla14 = clasificar(tasas17, cp3, curvas3)
print("\nTabla 11 (CP):\n", pd.DataFrame(cp3, index=CLASES, columns=CRIT))
print("\nTabla 12 (RP):\n", pd.DataFrame(rp3, index=NOMBRES_RP, columns=CRIT))
print("\nTabla 14:\n", tabla14)
print("\nClasificación:\n", res3.sort_values("Prioridad global", ascending=False))
print("\nReparto:", res3["Nivel de seguridad"].value_counts().to_dict())
res3.to_csv("resultados/ej3_clasificacion.csv")
mapas_espana([(res3, "Ejercicio 3: nivel de seguridad 2017")], "ej3_mapa_2017.png")

# ---------------------------------------------------------------- Ejercicio 4 (España 2023)
mostrar("EJERCICIO 4: España 2023 con la calibración de 2017")
tasas23 = cargar_tasas("Datos_nacionales_2023_por_10000_ES.xlsx", hoja="Datos 2023")
# Las tasas por encima de RP6 se truncan: np.interp les da la prioridad de RP6
print("Tasas 2023 por encima de RP6:", (tasas23 > rp3[-1]).sum().to_dict())
res4, _ = clasificar(tasas23, cp3, curvas3)

# Tabla de transición (provincias ordenadas de más a menos seguras en 2017)
transicion = pd.DataFrame(index=res3.sort_values("Prioridad global", ascending=False).index)
transicion["Clase 2017"] = res3["Nivel de seguridad"]
transicion["Clase 2023"] = res4["Nivel de seguridad"]
cambio = []
for prov in transicion.index:
    antes = CLASES.index(transicion.loc[prov, "Clase 2017"])   # 0 = Alto, 1 = Medio, 2 = Bajo
    despues = CLASES.index(transicion.loc[prov, "Clase 2023"])
    if despues < antes:
        cambio.append("Mejora")
    elif despues > antes:
        cambio.append("Empeora")
    else:
        cambio.append("Sin cambio")
transicion["Cambio"] = cambio

# Matriz de cambios: filas = clase en 2017, columnas = clase en 2023
matriz = pd.DataFrame(0, index=CLASES, columns=CLASES)
for prov in transicion.index:
    matriz.loc[transicion.loc[prov, "Clase 2017"], transicion.loc[prov, "Clase 2023"]] += 1

print("\nTabla de transición:\n", transicion)
print("\nMatriz de cambios (filas 2017, columnas 2023):\n", matriz)
res4.to_csv("resultados/ej4_clasificacion_2023.csv")
transicion.to_csv("resultados/ej4_transicion.csv")
matriz.to_csv("resultados/ej4_matriz_cambios.csv")
mapas_espana([(res3, "2017"), (res4, "2023")], "ej4_mapas_2017_vs_2023.png")
print("\nListo: resultados en la carpeta resultados/")
