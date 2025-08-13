import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURACIÓN DE LA PÁGINA ---
st.set_page_config(
    page_title="Dashboard de Fraude",
    page_icon="🛡️",
    layout="wide"
)

# --- ESTILOS CSS PERSONALIZADOS ---
st.markdown("""
<style>
    /* Estilo para las tarjetas de métricas (KPIs) */
    .kpi-card {
        background-color: #FFFFFF;
        padding: 20px;
        border-radius: 10px;
        border: 1px solid #E0E0E0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        text-align: center;
        height: 150px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .kpi-card h3 {
        font-size: 18px;
        color: #5f6368;
        margin-bottom: 5px;
    }
    .kpi-card h2 {
        font-size: 36px;
        font-weight: bold;
        color: #202124;
    }
    .kpi-card .delta {
        font-size: 14px;
    }
    /* Clases de color para los deltas */
    .delta-red { color: #d9534f; }
    .delta-green { color: #5cb85c; }
</style>
""", unsafe_allow_html=True)


# --- CARGA DE MODELO Y DATOS (CON CACHING) ---
@st.cache_resource
def load_model_and_columns():
    try:
        model = joblib.load('fraud_model.joblib')
        model_cols = joblib.load('model_columns.joblib')
        return model, model_cols
    except FileNotFoundError:
        return None, None

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('./data/transactions.csv')
        # Feature engineering básico para análisis
        df['errorBalanceOrig'] = df['oldbalanceOrg'] + df['amount'] - df['newbalanceOrig']
        df['esTipoRelevante'] = df['type'].apply(lambda x: 1 if x in ['TRANSFER', 'CASH_OUT'] else 0)
        # Convertir 'step' a un ciclo de 24 horas para un análisis más realista
        df['hour_of_day'] = df['step'] % 24
        return df
    except FileNotFoundError:
        return None

# Cargar artefactos
model, model_cols = load_model_and_columns()
df = load_data()

# --- VALIDACIÓN DE CARGA ---
if model is None or model_cols is None or df is None:
    st.error("❌ Error Crítico: No se encontraron los archivos necesarios. Ejecuta el script de entrenamiento.")
    st.stop()


# --- BARRA LATERAL CON NAVEGACIÓN ---
st.sidebar.title("fraudApp 🤖")
# st.sidebar.image('./assets/fraudApp.png', width=100)

page = st.sidebar.radio(
    "Seleccione una página:",
    ["Resumen Ejecutivo", "Análisis Interactivo", "Detector de Fraude"]
)


# --- PÁGINA: RESUMEN EJECUTIVO ---
if page == "Resumen Ejecutivo":
    st.title("🛡️ Resumen Ejecutivo y KPIs")
    st.markdown("Una vista de alto nivel sobre la situación del fraude basada en el dataset completo.")

    # --- KPIs GLOBALES (SOBRE TODO EL DATASET) ---
    total_transactions = df.shape[0]
    total_fraud = df['isFraud'].sum()
    total_amount_fraud = df[df['isFraud'] == 1]['amount'].sum()
    fraud_rate = (total_fraud / total_transactions) * 100

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    with kpi1:
        st.markdown(f'<div class="kpi-card"><h3>Total Transacciones</h3><h2>{total_transactions:,}</h2></div>', unsafe_allow_html=True)
    with kpi2:
        st.markdown(f'<div class="kpi-card" style="border-left: 5px solid #d9534f;"><h3>Fraudes Detectados</h3><h2>{total_fraud:,}</h2></div>', unsafe_allow_html=True)
    with kpi3:
        st.markdown(f'<div class="kpi-card"><h3>Monto en Riesgo</h3><h2>${total_amount_fraud:,.0f}</h2></div>', unsafe_allow_html=True)
    with kpi4:
        st.markdown(f'<div class="kpi-card" style="border-left: 5px solid #5cb85c;"><h3>Tasa de Fraude</h3><h2>{fraud_rate:.3f}%</h2></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

    # --- INSIGHTS CLAVE ---
    st.subheader("💡 Insights y Conclusiones Principales")
    
    insight1, insight2 = st.columns(2)
    with insight1:
        st.info("**Insight 1: El Fraude tiene un Patrón Claro**")
        st.write("El 100% de las transacciones fraudulentas ocurren **exclusivamente** en operaciones de **TRANSFER** y **CASH_OUT**. Esto permite enfocar los recursos de monitoreo en estas áreas de alto riesgo.")

    with insight2:
        st.warning("**Insight 2: Comportamiento de 'Vaciado de Cuenta'**")
        st.write("Se identificó un patrón clave: el **intento de vaciar la cuenta**. La característica `intentoVaciado` (cuando `amount == oldbalanceOrg`) fue uno de los predictores más fuertes del modelo.")
    
    st.success("**Insight 3: La Clave está en los Balances Anómalos**")
    st.write("La característica más importante para el modelo fue `errorBalanceOrig`. Una discrepancia en los saldos del originador es un **indicador de alerta roja (red flag)** extremadamente potente de actividad fraudulenta.")

    st.markdown("---")

    # --- EJEMPLOS DE TRANSACCIONES ---
    st.subheader("Visualización de Transacciones Clave")
    st.markdown("Ejemplos para contextualizar los montos y tipos de transacciones.")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Ejemplos de Transacciones Fraudulentas**")
        df_fraud_examples = df[df['isFraud'] == 1][['type', 'amount', 'oldbalanceOrg', 'newbalanceDest']].head()
        st.dataframe(df_fraud_examples.style.format({"amount": "${:,.2f}", "oldbalanceOrg": "${:,.2f}", "newbalanceDest": "${:,.2f}"}), use_container_width=True)

    with col2:
        st.markdown("**Ejemplos de Transacciones Legítimas**")
        # Usamos una muestra para obtener variedad y filtramos por tipos relevantes para una comparación justa
        df_legit_examples = df[(df['isFraud'] == 0) & (df['type'].isin(['TRANSFER', 'CASH_OUT']))][['type', 'amount', 'oldbalanceOrg', 'newbalanceDest']].sample(5, random_state=42)
        st.dataframe(df_legit_examples.style.format({"amount": "${:,.2f}", "oldbalanceOrg": "${:,.2f}", "newbalanceDest": "${:,.2f}"}), use_container_width=True)


# --- PÁGINA: ANÁLISIS INTERACTIVO ---
elif page == "Análisis Interactivo":
    st.title("🔬 Análisis Interactivo de Transacciones")
    st.markdown("Explore los datos aplicando filtros para descubrir patrones.")

    # --- FILTROS EN EL CUERPO PRINCIPAL ---
    st.markdown("---")
    st.markdown("#### Filtros de Datos")
    filter_col1, filter_col2 = st.columns(2)

    with filter_col1:
        transaction_type = st.multiselect(
            "Filtrar por Tipo de Transacción:",
            options=df['type'].unique(),
            default=df['type'].unique()
        )

    with filter_col2:
        fraud_status = st.selectbox(
            "Filtrar por Status de Fraude:",
            options=["Todos", "Solo Fraude", "Solo Legítimas"],
            index=0
        )

    # --- APLICAR FILTROS AL DATAFRAME ---
    df_filtered = df[df['type'].isin(transaction_type)]
    if fraud_status == "Solo Fraude":
        df_filtered = df_filtered[df_filtered['isFraud'] == 1]
    elif fraud_status == "Solo Legítimas":
        df_filtered = df_filtered[df_filtered['isFraud'] == 0]

    st.markdown("---")

    # --- CUADRÍCULA DE VISUALIZACIONES (GRID LAYOUT) ---
    st.subheader("Análisis Visual Dinámico")
    grid1_col1, grid1_col2 = st.columns(2)

    with grid1_col1:
        st.markdown("##### Riesgo por Tipo de Transacción")
        risk_by_type = df_filtered.groupby('type')['isFraud'].sum().sort_values(ascending=False)
        fig_risk = px.bar(risk_by_type, x=risk_by_type.index, y=risk_by_type.values,
                          labels={'y': 'Cantidad de Fraudes', 'x': 'Tipo de Transacción'},
                          color=risk_by_type.values, color_continuous_scale='Reds')
        st.plotly_chart(fig_risk, use_container_width=True)

    with grid1_col2:
        st.markdown("##### Distribución de Montos (Legítimas vs. Fraude)")
        fig_box = px.box(df_filtered, x='isFraud', y='amount',
                         labels={'isFraud': '¿Es Fraude?', 'amount': 'Monto'},
                         color='isFraud', color_discrete_map={0: "#5cb85c", 1: "#d9534f"})
        fig_box.update_yaxes(type="log")
        st.plotly_chart(fig_box, use_container_width=True)


    grid2_col1, grid2_col2 = st.columns(2)

    with grid2_col1:
        st.markdown("##### Características más Importantes (Modelo)")
        importances = pd.DataFrame({
            'feature': model_cols,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        fig_imp = px.bar(importances.head(10), x='importance', y='feature', orientation='h',
                         labels={'feature': 'Característica', 'importance': 'Importancia Relativa'})
        fig_imp.update_layout(yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig_imp, use_container_width=True)

    with grid2_col2:
        st.markdown("##### Patrones de Balance vs. Monto")
        sample_df = df_filtered.sample(n=min(5000, len(df_filtered)))
        fig_scatter = px.scatter(sample_df, x='oldbalanceOrg', y='amount', color='isFraud',
                                 opacity=0.5,
                                 labels={'oldbalanceOrg': 'Saldo Original', 'amount': 'Monto Transacción'},
                                 color_discrete_map={0: "rgba(92, 184, 92, 0.5)", 1: "rgba(217, 83, 79, 1)"})
        st.plotly_chart(fig_scatter, use_container_width=True)
    
    st.markdown("---")
    st.markdown("##### Actividad de Transacciones por Hora del Día")
    # Usamos una muestra para que el gráfico sea más rápido y legible
    sample_df_time = df_filtered.sample(n=min(10000, len(df_filtered)))
    fig_time_scatter = px.scatter(
        sample_df_time,
        x='hour_of_day',
        y='amount',
        color='isFraud',
        log_y=True,
        color_discrete_map={0: "rgba(92, 184, 92, 0.5)", 1: "rgba(217, 83, 79, 1)"},
        labels={
            "hour_of_day": "Hora del Día (0-23h)",
            "amount": "Monto de la Transacción (Escala Log)",
            "isFraud": "Es Fraude"
        },
        title="Horas de Mayor Actividad vs. Monto de Transacción"
    )
    st.plotly_chart(fig_time_scatter, use_container_width=True)


# --- PÁGINA: DETECTOR DE FRAUDE ---
elif page == "Detector de Fraude":
    st.title("Detector de Fraude Manual")
    st.markdown("Ingrese los datos de una transacción para evaluarla con el modelo de IA.")
    st.write("Información relevante: la casilla 'step' representa la hora del día en la que se realizó la transacción formato 0-23 ejemplo 791 % 24 = 23. El mismo es un formato utilizado para el entendimiento a la hora del aprendizaje del modelo.")

    with st.form("prediction_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            step = st.number_input("Paso (Step)", min_value=1, step=1, value=1)
            amount = st.number_input("Monto (Amount)", min_value=0.0, format="%.2f", value=1000.0)
        with c2:
            type_trans = st.selectbox("Tipo de Transacción", options=df['type'].unique())
            oldbalanceOrg = st.number_input("Saldo Origen (Old)", min_value=0.0, format="%.2f", value=5000.0)
        with c3:
            newbalanceOrg = st.number_input("Nuevo Saldo Origen (New)", min_value=0.0, format="%.2f", value=4000.0)
            oldbalanceDest = st.number_input("Saldo Destino (Old)", min_value=0.0, format="%.2f", value=2000.0)
            newbalanceDest = st.number_input("Nuevo Saldo Destino (New)", min_value=0.0, format="%.2f", value=3000.0)

        submit_button = st.form_submit_button("🔍 Evaluar Transacción")

    if submit_button:
        with st.spinner('Analizando con el modelo...'):
            # 1. Crear DataFrame con los datos del formulario
            input_data = {
                'step': step, 'amount': amount, 'oldbalanceOrg': oldbalanceOrg,
                'newbalanceOrig': newbalanceOrg, 'oldbalanceDest': oldbalanceDest,
                'newbalanceDest': newbalanceDest, 'isFlaggedFraud': 0
            }
            input_df = pd.DataFrame([input_data])

            # 2. Aplicar la misma ingeniería de características
            input_df['esTipoRelevante'] = 1 if type_trans in ['TRANSFER', 'CASH_OUT'] else 0
            input_df['intentoVaciado'] = 1 if (oldbalanceOrg > 0 and amount == oldbalanceOrg) else 0
            input_df['errorBalanceOrig'] = oldbalanceOrg + amount - newbalanceOrg

            # 3. Aplicar One-Hot Encoding
            for t in [col for col in model_cols if col.startswith('type_')]:
                type_name = t.split('_')[1]
                input_df[t] = 1 if type_trans == type_name else 0

            # 4. Alinear columnas con las que el modelo fue entrenado
            final_df = input_df.reindex(columns=model_cols, fill_value=0)

            # 5. Realizar la predicción
            prediction_proba = model.predict_proba(final_df)[0]
            prediction = (prediction_proba[1] > 0.5).astype(int)

            # 6. Mostrar el resultado
            prob_fraud = prediction_proba[1]
            if prediction == 1:
                st.error(f"🚨 ALERTA: Transacción Marcada como FRAUDULENTA (Probabilidad: {prob_fraud:.2%})")
            else:
                st.success(f"✅ Transacción Considerada LEGÍTIMA (Probabilidad de fraude: {prob_fraud:.2%})")

            st.progress(prob_fraud)
            st.write("---")
            st.write("Valores de entrada procesados por el modelo:")
            st.dataframe(final_df)
