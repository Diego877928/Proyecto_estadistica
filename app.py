import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# =========================================================
# CONFIGURACION GENERAL
# =========================================================

plt.style.use("ggplot")

st.set_page_config(
    page_title="Análisis Estadístico",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================================================
# FUNCION PARA MOSTRAR GRAFICOS
# =========================================================

def mostrar_grafico(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

# =========================================================
# SIDEBAR - CARGA Y NAVEGACION
# =========================================================

st.sidebar.image(
    "https://cdn-icons-png.flaticon.com/512/1822/1822094.png",
    width=90
)
st.sidebar.title("📊 Panel de Control")
st.sidebar.markdown("---")

st.sidebar.header("📥 Cargar Datos")
archivo = st.sidebar.file_uploader(
    "Sube tu archivo CSV",
    type=["csv"],
    help="Sube cualquier CSV con tus propios datos. Los gráficos se adaptarán automáticamente."
)

# Mensaje si no hay archivo
if archivo is None:
    st.sidebar.info("💡 Usando el CSV de ejemplo (estudiantes). Puedes subir tu propio CSV arriba.")

st.sidebar.markdown("---")
st.sidebar.header("🗂️ Menú de Navegación")

opcion_menu = st.sidebar.radio(
    "Selecciona la sección:",
    (
        "📋 Vista General",
        "📊 Variables Cualitativas",
        "🔢 Datos No Agrupados",
        "📈 Datos Agrupados",
        "🔗 Relaciones entre Variables"
    )
)

# =========================================================
# CARGA DE DATOS
# =========================================================

@st.cache_data
def cargar_datos_ejemplo():
    import os
    ruta = os.path.join(os.path.dirname(__file__), "estudiantes.csv")
    return pd.read_csv(ruta)

if archivo is not None:
    try:
        df = pd.read_csv(archivo)
        fuente = f"📂 Archivo cargado: **{archivo.name}**"
    except Exception as e:
        st.error(f"Error al leer el archivo: {e}")
        st.stop()
else:
    df = cargar_datos_ejemplo()
    fuente = "📂 Datos de ejemplo: **estudiantes.csv** (45 registros)"

# Detectar tipos de columnas
cols_cuantitativas = df.select_dtypes(include=[np.number]).columns.tolist()
cols_cualitativas = df.select_dtypes(exclude=[np.number]).columns.tolist()

# =========================================================
# TITULO PRINCIPAL
# =========================================================

st.title("📊 PROCESAMIENTO ESTADÍSTICO CON PYTHON")
st.markdown(f"**Fuente de datos:** {fuente}")
st.markdown("---")

# =========================================================
# SECCION 1: VISTA GENERAL
# =========================================================

if opcion_menu == "📋 Vista General":

    st.header("📌 FASE 1: CARGA Y VISTA GENERAL DE DATOS")

    # Métricas rápidas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Registros", len(df))
    with col2:
        st.metric("Total de Columnas", len(df.columns))
    with col3:
        st.metric("Variables Cuantitativas", len(cols_cuantitativas))
    with col4:
        st.metric("Variables Cualitativas", len(cols_cualitativas))

    st.success(f"✅ Datos cargados correctamente — {len(df)} registros encontrados.")

    # Base de datos completa
    st.subheader("📋 BASE DE DATOS COMPLETA")
    st.dataframe(df, use_container_width=True)

    # Tipos de columnas
    st.subheader("📋 Tipos de Datos por Columna")
    tipos_df = pd.DataFrame({
        "Columna": df.columns,
        "Tipo": df.dtypes.values.astype(str),
        "Valores únicos": [df[c].nunique() for c in df.columns],
        "Valores nulos": [df[c].isnull().sum() for c in df.columns]
    })
    st.dataframe(tipos_df, use_container_width=True)

    # Estadísticas descriptivas
    if len(cols_cuantitativas) > 0:
        st.subheader("📋 Estadísticas Descriptivas")
        st.dataframe(df[cols_cuantitativas].describe().round(4), use_container_width=True)

# =========================================================
# SECCION 2: VARIABLES CUALITATIVAS
# =========================================================

elif opcion_menu == "📊 Variables Cualitativas":

    st.header("📌 FASE 2: VARIABLES CUALITATIVAS")

    if len(cols_cualitativas) == 0:
        st.warning("⚠️ No se encontraron variables cualitativas (texto/categoría) en el CSV.")
    else:
        var_cuali = st.selectbox(
            "Selecciona una variable cualitativa:",
            cols_cualitativas
        )

        st.write(f"**Variable seleccionada:** `{var_cuali}`")

        # ---- Tabla de Frecuencia Cualitativa ----
        frecuencia = df[var_cuali].value_counts()

        tabla_cualitativa = pd.DataFrame({
            "Categoría": frecuencia.index,
            "fi": frecuencia.values
        })

        tabla_cualitativa["hi"] = tabla_cualitativa["fi"] / tabla_cualitativa["fi"].sum()
        tabla_cualitativa["hi%"] = tabla_cualitativa["hi"] * 100
        tabla_cualitativa["Fi"] = tabla_cualitativa["fi"].cumsum()
        tabla_cualitativa["Hi"] = tabla_cualitativa["hi"].cumsum()
        tabla_cualitativa = tabla_cualitativa.round(4)

        st.subheader("📋 Tabla de Frecuencia Cualitativa")
        st.dataframe(tabla_cualitativa, use_container_width=True)

        st.markdown("---")

        # ---- Gráficos en columnas ----
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Gráfico de Barras")
            fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=110)
            colores = sns.color_palette("viridis", len(tabla_cualitativa))
            barras = ax1.bar(
                tabla_cualitativa["Categoría"],
                tabla_cualitativa["fi"],
                color=colores,
                edgecolor="black"
            )
            ax1.set_title(f"DISTRIBUCIÓN POR {var_cuali.upper()}", fontsize=13, fontweight="bold")
            ax1.set_xlabel(var_cuali)
            ax1.set_ylabel("Frecuencia Absoluta (fi)")
            ax1.grid(True, linestyle="--", alpha=0.4)
            plt.xticks(rotation=30, ha="right")
            for barra in barras:
                altura = barra.get_height()
                ax1.text(
                    barra.get_x() + barra.get_width() / 2,
                    altura + 0.1,
                    str(int(altura)),
                    ha="center", fontsize=9, fontweight="bold"
                )
            mostrar_grafico(fig1)

        with col2:
            st.subheader("📊 Gráfico de Torta")
            fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=110)
            ax2.pie(
                tabla_cualitativa["fi"],
                labels=tabla_cualitativa["Categoría"],
                autopct="%1.1f%%",
                startangle=90,
                colors=sns.color_palette("pastel", len(tabla_cualitativa))
            )
            ax2.set_title(f"PROPORCIÓN POR {var_cuali.upper()}", fontsize=13, fontweight="bold")
            mostrar_grafico(fig2)

        st.markdown("---")

        # ---- Gráfico de Frecuencia Relativa ----
        st.subheader("📊 Gráfico de Frecuencia Relativa (%)")
        fig3, ax3 = plt.subplots(figsize=(9, 4), dpi=110)
        colores2 = sns.color_palette("coolwarm", len(tabla_cualitativa))
        barras2 = ax3.barh(
            tabla_cualitativa["Categoría"],
            tabla_cualitativa["hi%"],
            color=colores2,
            edgecolor="black"
        )
        ax3.set_title(f"FRECUENCIA RELATIVA (%) POR {var_cuali.upper()}", fontsize=13, fontweight="bold")
        ax3.set_xlabel("Porcentaje (%)")
        ax3.grid(True, linestyle="--", alpha=0.4)
        for barra in barras2:
            ancho = barra.get_width()
            ax3.text(
                ancho + 0.3,
                barra.get_y() + barra.get_height() / 2,
                f"{ancho:.1f}%",
                va="center", fontsize=9
            )
        mostrar_grafico(fig3)

# =========================================================
# SECCION 3: DATOS NO AGRUPADOS
# =========================================================

elif opcion_menu == "🔢 Datos No Agrupados":

    st.header("📌 FASE 3: DATOS NO AGRUPADOS (Variable Discreta)")

    if len(cols_cuantitativas) == 0:
        st.warning("⚠️ No se encontraron variables numéricas en el CSV.")
    else:
        var_discreta = st.selectbox(
            "Selecciona una variable numérica discreta:",
            cols_cuantitativas
        )

        datos = df[var_discreta].dropna()
        n_unicos = datos.nunique()

        if n_unicos > 30:
            st.info(f"ℹ️ La variable `{var_discreta}` tiene {n_unicos} valores únicos. Se recomienda ir a **Datos Agrupados** para una mejor visualización.")

        # ---- Medidas Estadísticas ----
        st.subheader("📋 Medidas Estadísticas")

        media = datos.mean()
        mediana = datos.median()
        moda_val = datos.mode()[0]
        varianza = datos.var()
        desviacion = datos.std()
        val_max = datos.max()
        val_min = datos.min()
        rango = val_max - val_min

        tabla_medidas = pd.DataFrame({
            "Medida": ["Media", "Mediana", "Moda", "Varianza", "Desviación Estándar",
                       "Valor Máximo", "Valor Mínimo", "Rango"],
            "Valor": [
                round(media, 4), round(mediana, 4), round(moda_val, 4),
                round(varianza, 4), round(desviacion, 4),
                val_max, val_min, rango
            ]
        })

        col1, col2 = st.columns([1, 1])
        with col1:
            st.dataframe(tabla_medidas, use_container_width=True)
        with col2:
            st.metric("Media", round(media, 2))
            st.metric("Mediana", round(mediana, 2))
            st.metric("Desv. Estándar", round(desviacion, 2))
            st.metric("Rango", round(rango, 2))

        # ---- Tabla de Frecuencia Discreta ----
        st.subheader("📋 Tabla de Frecuencia Discreta")

        frecuencia_disc = datos.value_counts().sort_index()

        tabla_discreta = pd.DataFrame({
            "Valor": frecuencia_disc.index,
            "fi": frecuencia_disc.values
        })
        tabla_discreta["hi"] = tabla_discreta["fi"] / tabla_discreta["fi"].sum()
        tabla_discreta["hi%"] = tabla_discreta["hi"] * 100
        tabla_discreta["Fi"] = tabla_discreta["fi"].cumsum()
        tabla_discreta["Hi"] = tabla_discreta["hi"].cumsum()
        tabla_discreta = tabla_discreta.round(4)

        st.dataframe(tabla_discreta, use_container_width=True)

        st.markdown("---")

        # ---- Gráficos ----
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Gráfico de Bastón")
            fig4, ax4 = plt.subplots(figsize=(7, 4), dpi=110)
            markerline, stemlines, baseline = ax4.stem(
                tabla_discreta["Valor"],
                tabla_discreta["fi"]
            )
            plt.setp(stemlines, linewidth=2, color="steelblue")
            plt.setp(markerline, markersize=7, color="darkblue")
            ax4.set_title(f"DISTRIBUCIÓN DE {var_discreta.upper()}", fontsize=13, fontweight="bold")
            ax4.set_xlabel(var_discreta)
            ax4.set_ylabel("Frecuencia")
            ax4.grid(True, linestyle="--", alpha=0.4)
            mostrar_grafico(fig4)

        with col2:
            st.subheader("📊 Box Plot")
            fig5, ax5 = plt.subplots(figsize=(7, 4), dpi=110)
            sns.boxplot(x=datos, ax=ax5, color="lightcoral")
            ax5.set_title(f"BOX PLOT DE {var_discreta.upper()}", fontsize=13, fontweight="bold")
            ax5.set_xlabel(var_discreta)
            ax5.grid(True, linestyle="--", alpha=0.4)
            mostrar_grafico(fig5)

        st.markdown("---")

        # ---- Histograma con KDE ----
        st.subheader("📊 Histograma con Curva de Densidad")
        fig6, ax6 = plt.subplots(figsize=(10, 4), dpi=110)
        sns.histplot(datos, kde=True, color="teal", ax=ax6, edgecolor="black")
        ax6.set_title(f"DISTRIBUCIÓN DE {var_discreta.upper()}", fontsize=13, fontweight="bold")
        ax6.set_xlabel(var_discreta)
        ax6.set_ylabel("Frecuencia")
        ax6.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig6)

# =========================================================
# SECCION 4: DATOS AGRUPADOS
# =========================================================

elif opcion_menu == "📈 Datos Agrupados":

    st.header("📌 FASE 4: DATOS AGRUPADOS (Variable Continua)")

    if len(cols_cuantitativas) == 0:
        st.warning("⚠️ No se encontraron variables numéricas en el CSV.")
    else:
        var_continua = st.selectbox(
            "Selecciona una variable cuantitativa continua:",
            cols_cuantitativas
        )

        datos = df[var_continua].dropna()
        n = len(datos)
        val_max = datos.max()
        val_min = datos.min()
        rango = val_max - val_min

        # ---- Parámetros de Sturges ----
        k = int(1 + 3.322 * np.log10(n))
        amplitud = int(np.ceil(rango / k))

        st.subheader("📋 Parámetros de Sturges")

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("N° de Datos (n)", n)
            st.metric("Valor Máximo", round(val_max, 2))
        with col2:
            st.metric("Valor Mínimo", round(val_min, 2))
            st.metric("Rango", round(rango, 2))
        with col3:
            st.metric("N° de Clases (k)", k)
            st.metric("Amplitud (c)", amplitud)

        # ---- Tabla de Datos Agrupados ----
        intervalos = pd.interval_range(
            start=val_min,
            end=val_min + k * amplitud,
            freq=amplitud
        )

        df_temp = df[[var_continua]].copy().dropna()
        df_temp["Intervalo"] = pd.cut(df_temp[var_continua], bins=intervalos)

        tabla_agrupada = (
            df_temp.groupby("Intervalo", observed=True)
            .size()
            .reset_index(name="fi")
        )

        tabla_agrupada["Intervalo_Texto"] = tabla_agrupada["Intervalo"].apply(
            lambda x: f"{round(x.left, 2)} — {round(x.right, 2)}"
        )
        tabla_agrupada["Marca de Clase"] = tabla_agrupada["Intervalo"].apply(
            lambda x: round((x.left + x.right) / 2, 2)
        )
        tabla_agrupada["hi"] = tabla_agrupada["fi"] / tabla_agrupada["fi"].sum()
        tabla_agrupada["hi%"] = tabla_agrupada["hi"] * 100
        tabla_agrupada["Fi"] = tabla_agrupada["fi"].cumsum()
        tabla_agrupada["Hi"] = tabla_agrupada["hi"].cumsum()
        tabla_agrupada = tabla_agrupada.round(4)

        tabla_final = tabla_agrupada[[
            "Intervalo_Texto", "Marca de Clase", "fi", "hi", "hi%", "Fi", "Hi"
        ]].rename(columns={"Intervalo_Texto": "Intervalo"})

        st.subheader("📋 Tabla de Datos Agrupados")
        st.dataframe(tabla_final, use_container_width=True)

        st.markdown("---")

        # ---- Gráficos Agrupados ----
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("📊 Histograma")
            fig7, ax7 = plt.subplots(figsize=(7, 4), dpi=110)
            ax7.hist(datos, bins=k, color="skyblue", edgecolor="black")
            ax7.set_title(f"HISTOGRAMA DE {var_continua.upper()}", fontsize=13, fontweight="bold")
            ax7.set_xlabel(var_continua)
            ax7.set_ylabel("Frecuencia")
            ax7.grid(True, linestyle="--", alpha=0.4)
            mostrar_grafico(fig7)

        with col2:
            st.subheader("📊 Polígono de Frecuencia")
            fig8, ax8 = plt.subplots(figsize=(7, 4), dpi=110)
            ax8.plot(
                tabla_agrupada["Marca de Clase"],
                tabla_agrupada["fi"],
                marker="o", linewidth=2.5, color="royalblue", markersize=7
            )
            ax8.fill_between(tabla_agrupada["Marca de Clase"], tabla_agrupada["fi"], alpha=0.2, color="royalblue")
            ax8.set_title("POLÍGONO DE FRECUENCIA", fontsize=13, fontweight="bold")
            ax8.set_xlabel("Marca de Clase")
            ax8.set_ylabel("Frecuencia")
            ax8.grid(True, linestyle="--", alpha=0.4)
            mostrar_grafico(fig8)

        st.markdown("---")

        col3, col4 = st.columns(2)

        with col3:
            st.subheader("📊 Histograma + Polígono")
            fig9, ax9 = plt.subplots(figsize=(7, 4), dpi=110)
            ax9.hist(datos, bins=k, color="lightblue", edgecolor="black", alpha=0.7, label="Histograma")
            ax9.plot(
                tabla_agrupada["Marca de Clase"],
                tabla_agrupada["fi"],
                marker="o", linewidth=2.5, color="crimson", markersize=6, label="Polígono"
            )
            ax9.set_title("HISTOGRAMA Y POLÍGONO", fontsize=13, fontweight="bold")
            ax9.set_xlabel(var_continua)
            ax9.set_ylabel("Frecuencia")
            ax9.legend()
            ax9.grid(True, linestyle="--", alpha=0.4)
            mostrar_grafico(fig9)

        with col4:
            st.subheader("📊 Ojiva (Frec. Acumulada)")
            fig10, ax10 = plt.subplots(figsize=(7, 4), dpi=110)
            ax10.plot(
                tabla_agrupada["Marca de Clase"],
                tabla_agrupada["Fi"],
                marker="o", linewidth=2.5, color="seagreen", markersize=7
            )
            ax10.fill_between(tabla_agrupada["Marca de Clase"], tabla_agrupada["Fi"], alpha=0.15, color="seagreen")
            ax10.set_title("OJIVA (FRECUENCIA ACUMULADA)", fontsize=13, fontweight="bold")
            ax10.set_xlabel("Marca de Clase")
            ax10.set_ylabel("Frecuencia Acumulada (Fi)")
            ax10.grid(True, linestyle="--", alpha=0.4)
            mostrar_grafico(fig10)

# =========================================================
# SECCION 5: RELACIONES ENTRE VARIABLES
# =========================================================

elif opcion_menu == "🔗 Relaciones entre Variables":

    st.header("📌 FASE 5: RELACIONES ENTRE VARIABLES")

    if len(cols_cuantitativas) == 0 or len(cols_cualitativas) == 0:
        st.warning("⚠️ Se necesitan al menos una variable cualitativa y una cuantitativa para esta sección.")
    else:

        st.subheader("📊 Comparativa: Cuantitativa agrupada por Cualitativa")

        col_sel1, col_sel2 = st.columns(2)
        with col_sel1:
            cuali = st.selectbox("Variable de Agrupación (Cualitativa):", cols_cualitativas)
        with col_sel2:
            cuanti = st.selectbox("Variable a Medir (Cuantitativa):", cols_cuantitativas)

        # Boxplot comparativo
        fig11, ax11 = plt.subplots(figsize=(10, 5), dpi=110)
        orden = df.groupby(cuali)[cuanti].median().sort_values(ascending=False).index
        sns.boxplot(data=df, x=cuali, y=cuanti, hue=cuali, palette="Set2", ax=ax11,
                    order=orden, legend=False)
        ax11.set_title(f"DISTRIBUCIÓN DE {cuanti.upper()} AGRUPADO POR {cuali.upper()}", fontsize=13, fontweight="bold")
        ax11.set_xlabel(cuali)
        ax11.set_ylabel(cuanti)
        ax11.grid(True, linestyle="--", alpha=0.3)
        plt.xticks(rotation=30, ha="right")
        mostrar_grafico(fig11)

        # Tabla resumen
        st.subheader(f"📋 Estadísticas de {cuanti} por {cuali}")
        resumen = df.groupby(cuali)[cuanti].agg(
            Media="mean", Mediana="median", Min="min", Max="max",
            Desv_Std="std", Conteo="count"
        ).round(2).sort_values("Media", ascending=False)
        st.dataframe(resumen, use_container_width=True)

        st.markdown("---")

        # Violín si hay suficientes datos
        st.subheader("📊 Gráfico de Violín")
        fig12, ax12 = plt.subplots(figsize=(10, 5), dpi=110)
        try:
            sns.violinplot(data=df, x=cuali, y=cuanti, hue=cuali, palette="muted",
                           ax=ax12, order=orden, legend=False)
            ax12.set_title(f"VIOLÍN: {cuanti.upper()} por {cuali.upper()}", fontsize=13, fontweight="bold")
            ax12.grid(True, linestyle="--", alpha=0.3)
            plt.xticks(rotation=30, ha="right")
            mostrar_grafico(fig12)
        except Exception:
            st.info("No hay suficientes datos para el gráfico de violín con esta combinación.")
            plt.close(fig12)

        st.markdown("---")

        # Correlación entre cuantitativas
        if len(cols_cuantitativas) >= 2:
            st.subheader("📊 Mapa de Correlación entre Variables Numéricas")
            fig13, ax13 = plt.subplots(figsize=(8, 6), dpi=110)
            correlacion = df[cols_cuantitativas].corr()
            sns.heatmap(
                correlacion, annot=True, fmt=".2f", cmap="coolwarm",
                ax=ax13, linewidths=0.5, vmin=-1, vmax=1
            )
            ax13.set_title("MAPA DE CALOR - CORRELACIÓN", fontsize=13, fontweight="bold")
            mostrar_grafico(fig13)

# =========================================================
# FOOTER
# =========================================================

st.markdown("---")
st.success("✅ Análisis generado correctamente con los datos cargados.")
st.caption("📌 Tip: Sube un nuevo CSV desde la barra lateral y todos los gráficos y tablas se actualizarán automáticamente.")
