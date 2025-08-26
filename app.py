import streamlit as st
import pandas as pd
import joblib

# ----------------------------
# Configuração básica da página
# ----------------------------
st.set_page_config(page_title="Predição de Dor Pós-operatória", layout="centered")

# reduzir espaçamento superior para a imagem
st.markdown("""
<style>
    .block-container { padding-top: 0.5rem; }
</style>
""", unsafe_allow_html=True)

# ----------------------------
# Imagem de cabeçalho (TOPO)
# ----------------------------
HEADER_IMAGE = "figura.jpg"  # ajuste o nome se necessário
try:
    st.image(HEADER_IMAGE, use_container_width=True)
except Exception as e:
    st.warning(f"Não foi possível carregar a figura de topo ({HEADER_IMAGE}). Erro: {e}")

st.caption("Uso acadêmico. As predições não substituem o julgamento clínico.")

# ----------------------------
# Carregar modelos
# ----------------------------
@st.cache_resource(show_spinner=False)
def load_models():
    logreg_24 = joblib.load("logreg_24h.pkl")  # modelo 24h
    gb_72 = joblib.load("gb_72h.pkl")          # modelo 72h
    return logreg_24, gb_72

try:
    logreg_24, gb_72 = load_models()
except Exception as e:
    st.error(f"Erro ao carregar os modelos (.pkl): {e}")
    st.stop()

# ----------------------------
# Mapeamentos e ordem de colunas (conforme treino)
# ----------------------------
MAP_BIN = {"Não": 0, "Sim": 1}
MAP_SEX = {"Feminino": 0, "Masculino": 1}

COLS_24 = ["Redução oclusal", "Fotobiomodulação", "AINES", "Sexo", "Idade"]
COLS_72 = ["Redução oclusal", "Fotobiomodulação", "Sexo", "Idade"]

# ----------------------------
# Funções auxiliares
# ----------------------------
def preparar_24h(reducao, fotobio, aine, sexo, idade) -> pd.DataFrame:
    X = pd.DataFrame([{
        "Redução oclusal": MAP_BIN[reducao],
        "Fotobiomodulação": MAP_BIN[fotobio],
        "AINES": MAP_BIN[aine],
        "Sexo": MAP_SEX[sexo],
        "Idade": idade
    }])
    return X[COLS_24]

def preparar_72h(reducao, fotobio, sexo, idade) -> pd.DataFrame:
    X = pd.DataFrame([{
        "Redução oclusal": MAP_BIN[reducao],
        "Fotobiomodulação": MAP_BIN[fotobio],
        "Sexo": MAP_SEX[sexo],
        "Idade": idade
    }])
    return X[COLS_72]

def interpretar_prob(prob: float, horas: int) -> str:
    return f"Existe a probabilidade de **{prob:.0%}** de presença de dor em **{horas} horas**."

# ----------------------------
# UI: formulário de entrada
# ----------------------------
with st.form("form"):
    st.subheader("Entradas clínicas")
    idade = st.number_input("Idade (anos)", min_value=18, max_value=100, value=32)
    sexo = st.selectbox("Sexo", ["Feminino", "Masculino"])
    reducao = st.selectbox("Redução oclusal", ["Não", "Sim"])
    fotobio = st.selectbox("Fotobiomodulação", ["Não", "Sim"])
    aine = st.selectbox("Uso de AINEs", ["Não", "Sim"])
    submit = st.form_submit_button("Calcular")

# ----------------------------
# Predição (apenas probabilidade)
# ----------------------------
if submit:
    try:
        # ---- 24h (Logistic Regression) ----
        X24 = preparar_24h(reducao, fotobio, aine, sexo, idade)
        p24 = float(logreg_24.predict_proba(X24)[0][1])

        st.subheader("Predição — 24 horas")
        st.metric("Probabilidade de dor (24h)", f"{p24:.0%}")
        st.write(interpretar_prob(p24, 24))

        # ---- 72h (Gradient Boosting) ----
        X72 = preparar_72h(reducao, fotobio, sexo, idade)
        p72 = float(gb_72.predict_proba(X72)[0][1])

        st.subheader("Predição — 72 horas")
        st.metric("Probabilidade de dor (72h)", f"{p72:.0%}")
        st.write(interpretar_prob(p72, 72))

    except KeyError as e:
        st.error(f"Valor inválido em alguma entrada: {e}")
    except Exception as e:
        st.error(f"Ocorreu um erro ao calcular as predições: {e}")
