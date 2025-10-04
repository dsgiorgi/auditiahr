
def kpi_msgs(di, dp, eo, th):
    msgs=[]
    if di is None: msgs.append("No se pudo calcular la 'regla del 80%' (faltan datos).")
    else: msgs.append("✅ Cumple la 'regla del 80%'." if di>=th["DI"] else "❌ No cumple la 'regla del 80%'.")
    if dp is None: msgs.append("Faltan datos para comparar tasas entre grupos.")
    else:
        if dp<=th["DP_GREEN"]: msgs.append("✅ Las tasas de avance son similares entre grupos.")
        elif dp<=th["DP_YELLOW"]: msgs.append("🟡 Hay alguna diferencia entre grupos.")
        else: msgs.append("🔴 Hay una brecha grande entre grupos.")
    if eo is None: msgs.append("Para 'Igualdad de oportunidades' necesitás una columna de '¿realmente calificado?'.")
    else:
        if eo<=th["EO_GREEN"]: msgs.append("✅ Entre calificados, la oportunidad es parecida entre grupos.")
        elif eo<=th["EO_YELLOW"]: msgs.append("🟡 Entre calificados, hay pequeña diferencia.")
        else: msgs.append("🔴 Entre calificados, hay una brecha importante.")
    return msgs
