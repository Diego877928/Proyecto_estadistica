import sys
import os

EN_STREAMLIT = "streamlit" in sys.modules

if not EN_STREAMLIT:
    try:
        import streamlit as _st
        EN_STREAMLIT = _st.runtime.exists()
    except Exception:
        EN_STREAMLIT = False

import matplotlib
if EN_STREAMLIT:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

plt.style.use("ggplot")


EXCLUIR_CUALI  = ["nombre", "name", "id", "id_estudiante", "estudiante"]
MAX_CATEGORIAS = 20


def cargar_csv():
    ruta = os.path.join(os.path.dirname(os.path.abspath(__file__)), "estudiantes.csv")
    return pd.read_csv(ruta)

def cols_cuali_utiles(df):
    resultado = []
    for col in df.select_dtypes(exclude=[np.number]).columns:
        if col.lower() in EXCLUIR_CUALI:
            continue
        if df[col].nunique() <= MAX_CATEGORIAS:
            resultado.append(col)
    return resultado

def mostrar_grafico(fig):
    fig.tight_layout()
    if EN_STREAMLIT:
        import streamlit as st
        st.pyplot(fig)
        plt.close(fig)
    else:
        plt.show()  
        plt.close(fig)


def modo_terminal():
    df          = cargar_csv()
    cols_cuanti = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_cuali  = cols_cuali_utiles(df)

    sep = "=" * 60

    print(f"\n{sep}")
    print("       PROCESAMIENTO ESTADISTICO CON PYTHON")
    print(sep)
    print(f"\n  Archivo  : estudiantes.csv")
    print(f"  Registros: {len(df)}")
    print(f"  Columnas : {list(df.columns)}")

    print(f"\n{sep}")
    print("  FASE 1: BASE DE DATOS COMPLETA")
    print(sep)
    print(df.to_string())

    var_cuali = "carrera" if "carrera" in cols_cuali else (cols_cuali[0] if cols_cuali else None)

    if var_cuali:
        print(f"\n{sep}")
        print(f"  FASE 2: VARIABLE CUALITATIVA — {var_cuali.upper()}")
        print(sep)

        frec = df[var_cuali].value_counts()
        t = pd.DataFrame({"Categoria": frec.index, "fi": frec.values})
        t["hi"]  = t["fi"] / t["fi"].sum()
        t["hi%"] = (t["hi"] * 100).round(2)
        t["Fi"]  = t["fi"].cumsum()
        t["Hi"]  = t["hi"].cumsum().round(4)
        print("\n  Tabla de Frecuencia Cualitativa:")
        print(t.to_string(index=False))

        # Grafico de barras
        print("\n  >>> Mostrando grafico de barras (cierra la ventana para continuar)...")
        fig, ax = plt.subplots(figsize=(8, 4))
        colores = sns.color_palette("viridis", len(t))
        barras  = ax.bar(t["Categoria"], t["fi"], color=colores, edgecolor="black")
        ax.set_title(f"DISTRIBUCION POR {var_cuali.upper()}", fontweight="bold")
        ax.set_xlabel(var_cuali)
        ax.set_ylabel("Frecuencia (fi)")
        ax.grid(True, linestyle="--", alpha=0.4)
        plt.xticks(rotation=25, ha="right", fontsize=9)
        for b in barras:
            h = b.get_height()
            ax.text(b.get_x() + b.get_width()/2, h + 0.05, str(int(h)),
                    ha="center", fontsize=9, fontweight="bold")
        mostrar_grafico(fig)

        if len(t) <= 10:
            print("  >>> Mostrando grafico de torta (cierra la ventana para continuar)...")
            fig2, ax2 = plt.subplots(figsize=(6, 5))
            ax2.pie(t["fi"], labels=t["Categoria"], autopct="%1.1f%%",
                    startangle=90, colors=sns.color_palette("pastel", len(t)))
            ax2.set_title(f"PROPORCION POR {var_cuali.upper()}", fontweight="bold")
            mostrar_grafico(fig2)

    var_disc = "nota" if "nota" in cols_cuanti else (cols_cuanti[0] if cols_cuanti else None)

    if var_disc:
        datos = df[var_disc].dropna()
        print(f"\n{sep}")
        print(f"  FASE 3: DATOS NO AGRUPADOS — {var_disc.upper()}")
        print(sep)
        print(f"  Media              : {datos.mean():.4f}")
        print(f"  Mediana            : {datos.median():.4f}")
        print(f"  Moda               : {datos.mode()[0]:.4f}")
        print(f"  Varianza           : {datos.var():.4f}")
        print(f"  Desviacion estandar: {datos.std():.4f}")
        print(f"  Valor maximo       : {datos.max()}")
        print(f"  Valor minimo       : {datos.min()}")
        print(f"  Rango              : {datos.max() - datos.min()}")

        frec = datos.value_counts().sort_index()
        td = pd.DataFrame({"Valor": frec.index, "fi": frec.values})
        td["hi"]  = td["fi"] / td["fi"].sum()
        td["hi%"] = (td["hi"] * 100).round(2)
        td["Fi"]  = td["fi"].cumsum()
        td["Hi"]  = td["hi"].cumsum().round(4)
        print("\n  Tabla de Frecuencia Discreta:")
        print(td.to_string(index=False))

        # Grafico de baston
        print("\n  >>> Mostrando grafico de baston (cierra la ventana para continuar)...")
        fig3, ax3 = plt.subplots(figsize=(9, 4))
        ml, sl, bl = ax3.stem(td["Valor"], td["fi"])
        plt.setp(sl, linewidth=2, color="steelblue")
        plt.setp(ml, markersize=7, color="darkblue")
        ax3.set_title(f"GRAFICO DE BASTON — {var_disc.upper()}", fontweight="bold")
        ax3.set_xlabel(var_disc)
        ax3.set_ylabel("Frecuencia")
        ax3.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig3)

        # Box plot
        print("  >>> Mostrando box plot (cierra la ventana para continuar)...")
        fig4, ax4 = plt.subplots(figsize=(8, 4))
        sns.boxplot(x=datos, ax=ax4, color="lightcoral")
        ax4.set_title(f"BOX PLOT — {var_disc.upper()}", fontweight="bold")
        ax4.set_xlabel(var_disc)
        ax4.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig4)

        # Histograma con KDE
        print("  >>> Mostrando histograma (cierra la ventana para continuar)...")
        fig5, ax5 = plt.subplots(figsize=(9, 4))
        sns.histplot(datos, kde=True, color="teal", ax=ax5, edgecolor="black")
        ax5.set_title(f"HISTOGRAMA CON DENSIDAD — {var_disc.upper()}", fontweight="bold")
        ax5.set_xlabel(var_disc)
        ax5.set_ylabel("Frecuencia")
        ax5.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig5)

    candidatos_cont = [c for c in cols_cuanti if c != var_disc]
    var_cont = "edad" if "edad" in candidatos_cont else (candidatos_cont[0] if candidatos_cont else None)

    if var_cont:
        datos = df[var_cont].dropna()
        n       = len(datos)
        val_max = datos.max()
        val_min = datos.min()
        rango   = val_max - val_min
        k       = int(1 + 3.322 * np.log10(n))
        amp     = int(np.ceil(rango / k))

        print(f"\n{sep}")
        print(f"  FASE 4: DATOS AGRUPADOS — {var_cont.upper()}")
        print(sep)
        print(f"  Cantidad de datos (n): {n}")
        print(f"  Valor maximo         : {val_max}")
        print(f"  Valor minimo         : {val_min}")
        print(f"  Rango                : {rango}")
        print(f"  Numero de clases (k) : {k}")
        print(f"  Amplitud (c)         : {amp}")

        intervalos = pd.interval_range(start=val_min, end=val_min + k * amp, freq=amp)
        df_temp = df[[var_cont]].copy().dropna()
        df_temp["Intervalo"] = pd.cut(df_temp[var_cont], bins=intervalos)
        tag = df_temp.groupby("Intervalo", observed=True).size().reset_index(name="fi")
        tag["Intervalo_txt"]  = tag["Intervalo"].apply(lambda x: f"{round(x.left,1)} - {round(x.right,1)}")
        tag["Marca de Clase"] = tag["Intervalo"].apply(lambda x: round((x.left + x.right) / 2, 2))
        tag["hi"]  = tag["fi"] / tag["fi"].sum()
        tag["hi%"] = (tag["hi"] * 100).round(2)
        tag["Fi"]  = tag["fi"].cumsum()
        tag["Hi"]  = tag["hi"].cumsum().round(4)
        tf = tag[["Intervalo_txt", "Marca de Clase", "fi", "hi", "hi%", "Fi", "Hi"]]
        print("\n  Tabla de Datos Agrupados:")
        print(tf.to_string(index=False))

        # Histograma
        print("\n  >>> Mostrando histograma (cierra la ventana para continuar)...")
        fig6, ax6 = plt.subplots(figsize=(8, 4))
        ax6.hist(datos, bins=k, color="skyblue", edgecolor="black")
        ax6.set_title(f"HISTOGRAMA — {var_cont.upper()}", fontweight="bold")
        ax6.set_xlabel(var_cont)
        ax6.set_ylabel("Frecuencia")
        ax6.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig6)

        # Poligono de frecuencia
        print("  >>> Mostrando poligono de frecuencia (cierra la ventana para continuar)...")
        fig7, ax7 = plt.subplots(figsize=(8, 4))
        ax7.plot(tag["Marca de Clase"], tag["fi"],
                 marker="o", linewidth=2.5, color="royalblue", markersize=7)
        ax7.fill_between(tag["Marca de Clase"], tag["fi"], alpha=0.2, color="royalblue")
        ax7.set_title("POLIGONO DE FRECUENCIA", fontweight="bold")
        ax7.set_xlabel("Marca de Clase")
        ax7.set_ylabel("Frecuencia")
        ax7.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig7)

        # Histograma + Poligono
        print("  >>> Mostrando histograma + poligono (cierra la ventana para continuar)...")
        fig8, ax8 = plt.subplots(figsize=(8, 4))
        ax8.hist(datos, bins=k, color="lightblue", edgecolor="black", alpha=0.7, label="Histograma")
        ax8.plot(tag["Marca de Clase"], tag["fi"],
                 marker="o", linewidth=2.5, color="crimson", markersize=6, label="Poligono")
        ax8.set_title("HISTOGRAMA Y POLIGONO", fontweight="bold")
        ax8.set_xlabel(var_cont)
        ax8.set_ylabel("Frecuencia")
        ax8.legend()
        ax8.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig8)

        # Ojiva
        print("  >>> Mostrando ojiva (cierra la ventana para continuar)...")
        fig9, ax9 = plt.subplots(figsize=(8, 4))
        ax9.plot(tag["Marca de Clase"], tag["Fi"],
                 marker="o", linewidth=2.5, color="seagreen", markersize=7)
        ax9.fill_between(tag["Marca de Clase"], tag["Fi"], alpha=0.15, color="seagreen")
        ax9.set_title("OJIVA (FRECUENCIA ACUMULADA)", fontweight="bold")
        ax9.set_xlabel("Marca de Clase")
        ax9.set_ylabel("Frecuencia Acumulada (Fi)")
        ax9.grid(True, linestyle="--", alpha=0.4)
        mostrar_grafico(fig9)

    print(f"\n{sep}")
    print("  PROGRAMA EJECUTADO CORRECTAMENTE")
    print(sep + "\n")



if not EN_STREAMLIT:
    modo_terminal()



else:
    import streamlit as st

    st.set_page_config(
        page_title="Análisis Estadístico",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    @st.cache_data
    def cargar():
        return cargar_csv()

    df          = cargar()
    cols_cuanti = df.select_dtypes(include=[np.number]).columns.tolist()
    cols_cuali  = cols_cuali_utiles(df)

    st.sidebar.image("https://cdn-icons-png.flaticon.com/512/1822/1822094.png", width=90)
    st.sidebar.title("📊 Panel de Control")
    st.sidebar.markdown("---")
    st.sidebar.info(f"📂 Archivo: **estudiantes.csv**\n\n👥 Registros: **{len(df)}**")
    st.sidebar.markdown("---")
    st.sidebar.header("🗂️ Secciones")

    opcion = st.sidebar.radio(
        "¿Qué querés ver?",
        (
            "📋 Vista General",
            "📊 Variables Cualitativas",
            "🔢 Datos No Agrupados",
            "📈 Datos Agrupados",
        )
    )

    # ---- Titulo ----
    st.title("📊 PROCESAMIENTO ESTADÍSTICO CON PYTHON")
    st.markdown("**Archivo:** `estudiantes.csv` — 45 registros cargados")
    st.markdown("---")

    if opcion == "📋 Vista General":

        st.header("📌 FASE 1: DATOS GENERALES")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total de registros",    len(df))
        c2.metric("Columnas",              len(df.columns))
        c3.metric("Variables numéricas",   len(cols_cuanti))
        c4.metric("Variables de categoría", len(cols_cuali))

        st.success(f"✅ Archivo leído correctamente — {len(df)} estudiantes encontrados.")

        st.subheader("📋 Tabla de Datos Completa")
        st.dataframe(df, use_container_width=True)

        st.subheader("📋 Información de las Columnas")
        info = pd.DataFrame({
            "Columna":        df.columns,
            "Tipo":           df.dtypes.values.astype(str),
            "Valores únicos": [df[c].nunique() for c in df.columns],
            "Datos nulos":    [df[c].isnull().sum() for c in df.columns],
        })
        st.dataframe(info, use_container_width=True)

        if cols_cuanti:
            st.subheader("📋 Resumen Estadístico")
            st.dataframe(df[cols_cuanti].describe().round(4), use_container_width=True)

    elif opcion == "📊 Variables Cualitativas":

        st.header("📌 FASE 2: VARIABLES CUALITATIVAS")

        if not cols_cuali:
            st.warning("⚠️ No se encontraron variables de categoría útiles en el CSV.")
        else:
            var = st.selectbox("Elegí la variable a analizar:", cols_cuali)

            frec = df[var].value_counts()
            t = pd.DataFrame({"Categoría": frec.index, "fi": frec.values})
            t["hi"]  = t["fi"] / t["fi"].sum()
            t["hi%"] = t["hi"] * 100
            t["Fi"]  = t["fi"].cumsum()
            t["Hi"]  = t["hi"].cumsum()
            t = t.round(4)

            st.subheader("📋 Tabla de Frecuencia Cualitativa")
            st.dataframe(t, use_container_width=True)
            st.markdown("---")

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("📊 Gráfico de Barras")
                fig1, ax1 = plt.subplots(figsize=(7, 4), dpi=100)
                colores = sns.color_palette("viridis", len(t))
                barras  = ax1.bar(t["Categoría"], t["fi"], color=colores, edgecolor="black")
                ax1.set_title(f"DISTRIBUCIÓN POR {var.upper()}", fontsize=12, fontweight="bold")
                ax1.set_xlabel(var)
                ax1.set_ylabel("Frecuencia (fi)")
                ax1.grid(True, linestyle="--", alpha=0.4)
                plt.xticks(rotation=25, ha="right", fontsize=9)
                for b in barras:
                    h = b.get_height()
                    ax1.text(b.get_x() + b.get_width()/2, h + 0.05, str(int(h)),
                             ha="center", fontsize=9, fontweight="bold")
                mostrar_grafico(fig1)

            with col2:
                st.subheader("📊 Gráfico de Torta")
                if len(t) > 10:
                    st.info(f"⚠️ Esta variable tiene {len(t)} categorías — demasiadas para una torta legible.")
                else:
                    fig2, ax2 = plt.subplots(figsize=(6, 5), dpi=100)
                    ax2.pie(t["fi"], labels=t["Categoría"], autopct="%1.1f%%",
                            startangle=90, colors=sns.color_palette("pastel", len(t)))
                    ax2.set_title(f"PROPORCIÓN POR {var.upper()}", fontsize=12, fontweight="bold")
                    mostrar_grafico(fig2)

            st.markdown("---")
            st.subheader("📊 Frecuencia Relativa (%)")
            fig3, ax3 = plt.subplots(figsize=(9, max(3, len(t) * 0.5)), dpi=100)
            colores2 = sns.color_palette("coolwarm", len(t))
            barras2  = ax3.barh(t["Categoría"], t["hi%"], color=colores2, edgecolor="black")
            ax3.set_title(f"FRECUENCIA RELATIVA (%) — {var.upper()}", fontsize=12, fontweight="bold")
            ax3.set_xlabel("Porcentaje (%)")
            ax3.grid(True, linestyle="--", alpha=0.4)
            for b in barras2:
                w = b.get_width()
                ax3.text(w + 0.2, b.get_y() + b.get_height()/2, f"{w:.1f}%", va="center", fontsize=9)
            mostrar_grafico(fig3)

    elif opcion == "🔢 Datos No Agrupados":

        st.header("📌 FASE 3: DATOS NO AGRUPADOS")

        if not cols_cuanti:
            st.warning("⚠️ No se encontraron columnas numéricas en el CSV.")
        else:
            var   = st.selectbox("Elegí la variable numérica discreta:", cols_cuanti)
            datos = df[var].dropna()

            if datos.nunique() > 30:
                st.info(f"ℹ️ `{var}` tiene {datos.nunique()} valores distintos. Para mejor análisis, probá la sección **Datos Agrupados**.")

            st.subheader("📋 Medidas Estadísticas")
            tabla_med = pd.DataFrame({
                "Medida": ["Media", "Mediana", "Moda", "Varianza",
                           "Desviación Estándar", "Valor Máximo", "Valor Mínimo", "Rango"],
                "Valor": [
                    round(datos.mean(), 4), round(datos.median(), 4),
                    round(datos.mode()[0], 4), round(datos.var(), 4),
                    round(datos.std(), 4), datos.max(), datos.min(),
                    datos.max() - datos.min()
                ]
            })
            c1, c2 = st.columns([1, 1])
            with c1:
                st.dataframe(tabla_med, use_container_width=True)
            with c2:
                st.metric("Media",          round(datos.mean(), 2))
                st.metric("Mediana",        round(datos.median(), 2))
                st.metric("Desv. Estándar", round(datos.std(), 2))
                st.metric("Rango",          round(datos.max() - datos.min(), 2))

            frec = datos.value_counts().sort_index()
            td = pd.DataFrame({"Valor": frec.index, "fi": frec.values})
            td["hi"]  = td["fi"] / td["fi"].sum()
            td["hi%"] = td["hi"] * 100
            td["Fi"]  = td["fi"].cumsum()
            td["Hi"]  = td["hi"].cumsum()
            td = td.round(4)

            st.subheader("📋 Tabla de Frecuencia Discreta")
            st.dataframe(td, use_container_width=True)
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Gráfico de Bastón")
                fig4, ax4 = plt.subplots(figsize=(7, 4), dpi=100)
                ml, sl, bl = ax4.stem(td["Valor"], td["fi"])
                plt.setp(sl, linewidth=2, color="steelblue")
                plt.setp(ml, markersize=7, color="darkblue")
                ax4.set_title(f"BASTÓN — {var.upper()}", fontsize=12, fontweight="bold")
                ax4.set_xlabel(var)
                ax4.set_ylabel("Frecuencia")
                ax4.grid(True, linestyle="--", alpha=0.4)
                mostrar_grafico(fig4)

            with c2:
                st.subheader("📊 Box Plot")
                fig5, ax5 = plt.subplots(figsize=(7, 4), dpi=100)
                sns.boxplot(x=datos, ax=ax5, color="lightcoral")
                ax5.set_title(f"BOX PLOT — {var.upper()}", fontsize=12, fontweight="bold")
                ax5.set_xlabel(var)
                ax5.grid(True, linestyle="--", alpha=0.4)
                mostrar_grafico(fig5)

            st.markdown("---")
            st.subheader("📊 Histograma con Curva de Densidad")
            fig6, ax6 = plt.subplots(figsize=(10, 4), dpi=100)
            sns.histplot(datos, kde=True, color="teal", ax=ax6, edgecolor="black")
            ax6.set_title(f"DISTRIBUCIÓN DE {var.upper()}", fontsize=12, fontweight="bold")
            ax6.set_xlabel(var)
            ax6.set_ylabel("Frecuencia")
            ax6.grid(True, linestyle="--", alpha=0.4)
            mostrar_grafico(fig6)

    elif opcion == "📈 Datos Agrupados":

        st.header("📌 FASE 4: DATOS AGRUPADOS")

        if not cols_cuanti:
            st.warning("⚠️ No se encontraron columnas numéricas en el CSV.")
        else:
            var   = st.selectbox("Elegí la variable cuantitativa continua:", cols_cuanti)
            datos = df[var].dropna()
            n       = len(datos)
            val_max = datos.max()
            val_min = datos.min()
            rango   = val_max - val_min
            k       = int(1 + 3.322 * np.log10(n))
            amp     = int(np.ceil(rango / k))

            st.subheader("📋 Parámetros de Sturges")
            c1, c2, c3 = st.columns(3)
            c1.metric("Cantidad de datos (n)", n)
            c1.metric("Valor máximo",          round(val_max, 2))
            c2.metric("Valor mínimo",           round(val_min, 2))
            c2.metric("Rango",                  round(rango, 2))
            c3.metric("Nº de clases (k)",        k)
            c3.metric("Amplitud (c)",            amp)

            intervalos = pd.interval_range(start=val_min, end=val_min + k * amp, freq=amp)
            df_temp = df[[var]].copy().dropna()
            df_temp["Intervalo"] = pd.cut(df_temp[var], bins=intervalos)
            tag = df_temp.groupby("Intervalo", observed=True).size().reset_index(name="fi")
            tag["Intervalo_txt"]  = tag["Intervalo"].apply(lambda x: f"{round(x.left,2)} — {round(x.right,2)}")
            tag["Marca de Clase"] = tag["Intervalo"].apply(lambda x: round((x.left + x.right) / 2, 2))
            tag["hi"]  = tag["fi"] / tag["fi"].sum()
            tag["hi%"] = tag["hi"] * 100
            tag["Fi"]  = tag["fi"].cumsum()
            tag["Hi"]  = tag["hi"].cumsum()
            tag = tag.round(4)
            tf  = tag[["Intervalo_txt", "Marca de Clase", "fi", "hi", "hi%", "Fi", "Hi"]]
            tf  = tf.rename(columns={"Intervalo_txt": "Intervalo"})

            st.subheader("📋 Tabla de Datos Agrupados")
            st.dataframe(tf, use_container_width=True)
            st.markdown("---")

            c1, c2 = st.columns(2)
            with c1:
                st.subheader("📊 Histograma")
                fig7, ax7 = plt.subplots(figsize=(7, 4), dpi=100)
                ax7.hist(datos, bins=k, color="skyblue", edgecolor="black")
                ax7.set_title(f"HISTOGRAMA — {var.upper()}", fontsize=12, fontweight="bold")
                ax7.set_xlabel(var)
                ax7.set_ylabel("Frecuencia")
                ax7.grid(True, linestyle="--", alpha=0.4)
                mostrar_grafico(fig7)

            with c2:
                st.subheader("📊 Polígono de Frecuencia")
                fig8, ax8 = plt.subplots(figsize=(7, 4), dpi=100)
                ax8.plot(tag["Marca de Clase"], tag["fi"],
                         marker="o", linewidth=2.5, color="royalblue", markersize=7)
                ax8.fill_between(tag["Marca de Clase"], tag["fi"], alpha=0.2, color="royalblue")
                ax8.set_title("POLÍGONO DE FRECUENCIA", fontsize=12, fontweight="bold")
                ax8.set_xlabel("Marca de Clase")
                ax8.set_ylabel("Frecuencia")
                ax8.grid(True, linestyle="--", alpha=0.4)
                mostrar_grafico(fig8)

            st.markdown("---")
            c3, c4 = st.columns(2)
            with c3:
                st.subheader("📊 Histograma + Polígono")
                fig9, ax9 = plt.subplots(figsize=(7, 4), dpi=100)
                ax9.hist(datos, bins=k, color="lightblue", edgecolor="black", alpha=0.7, label="Histograma")
                ax9.plot(tag["Marca de Clase"], tag["fi"],
                         marker="o", linewidth=2.5, color="crimson", markersize=6, label="Polígono")
                ax9.set_title("HISTOGRAMA Y POLÍGONO", fontsize=12, fontweight="bold")
                ax9.set_xlabel(var)
                ax9.set_ylabel("Frecuencia")
                ax9.legend()
                ax9.grid(True, linestyle="--", alpha=0.4)
                mostrar_grafico(fig9)

            with c4:
                st.subheader("📊 Ojiva")
                fig10, ax10 = plt.subplots(figsize=(7, 4), dpi=100)
                ax10.plot(tag["Marca de Clase"], tag["Fi"],
                          marker="o", linewidth=2.5, color="seagreen", markersize=7)
                ax10.fill_between(tag["Marca de Clase"], tag["Fi"], alpha=0.15, color="seagreen")
                ax10.set_title("OJIVA (FRECUENCIA ACUMULADA)", fontsize=12, fontweight="bold")
                ax10.set_xlabel("Marca de Clase")
                ax10.set_ylabel("Frecuencia Acumulada (Fi)")
                ax10.grid(True, linestyle="--", alpha=0.4)
                mostrar_grafico(fig10)

    st.markdown("---")
    st.success("✅ Análisis generado correctamente.")
