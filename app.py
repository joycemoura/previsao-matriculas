import streamlit as st
import joblib
import pandas as pd

modelo = joblib.load('modelo_matriculas.pkl')

st.title("Previsão de Matrículas — Escolas do Paraná - 2023")
st.write("Este app utiliza um modelo de Regressão Linear treinado com dados do Censo Escolar 2023 (INEP) para estimar o número de matrículas de uma escola.")

st.markdown("---")

st.subheader("Infraestrutura da Escola")
st.write("Selecione os itens de infraestrutura que a escola possui:")

col1, col2 = st.columns(2)

with col1:
    tem_agua_potavel = st.checkbox("Água potável")
    tem_energia = st.checkbox("Energia elétrica")
    tem_esgoto = st.checkbox("Esgoto")
    tem_internet = st.checkbox("Internet")
    tem_banda_larga = st.checkbox("Banda larga")
    tem_computador = st.checkbox("Computadores")
    tem_biblioteca = st.checkbox("Biblioteca")

with col2:
    tem_lab_informatica = st.checkbox("Laboratório de Informática")
    tem_lab_ciencias = st.checkbox("Laboratório de Ciências")
    tem_quadra = st.checkbox("Quadra de Esportes")
    tem_cozinha = st.checkbox("Cozinha")
    tem_refeitorio = st.checkbox("Refeitório")
    tem_banheiro_acessivel = st.checkbox("Banheiro Acessível")

# Calcula o índice automaticamente (soma dos itens marcados)
itens_infra = [
    tem_agua_potavel, tem_energia, tem_esgoto, tem_internet,
    tem_banda_larga, tem_computador, tem_biblioteca,
    tem_lab_informatica, tem_lab_ciencias, tem_quadra,
    tem_cozinha, tem_refeitorio, tem_banheiro_acessivel
]
indice_infraestrutura = sum(itens_infra)

st.info(f"Índice de Infraestrutura calculado: **{indice_infraestrutura} de 13**")

st.markdown("---")


st.subheader("Porte da Escola")

qtd_salas = st.number_input("Quantidade de salas", min_value=0, value=5)
qtd_professores = st.number_input("Quantidade de professores", min_value=0, value=10)

st.markdown("---")


if st.button("Prever número de matrículas"):
    entrada = pd.DataFrame([{
        'indice_infraestrutura': indice_infraestrutura,
        'qtd_salas': qtd_salas,
        'qtd_professores': qtd_professores
    }])

    pred = modelo.predict(entrada)[0]

    st.success(f"Previsão: **{pred:.0f} matrículas**")

    st.caption("""
    **Sobre o modelo:** Regressão Linear treinada com dados do Censo Escolar 2023 (PR).
    - R² = 0.68 (explica 68% da variação nas matrículas)
    - Erro médio (MAE) = ~85 matrículas
    - O fator com maior peso é a quantidade de professores, seguido pela quantidade de salas.
    """)

st.markdown("---")
st.caption("Fonte dos dados: Censo Escolar 2023 — INEP | Projeto desenvolvido como REA (Recurso Educacional Aberto)")