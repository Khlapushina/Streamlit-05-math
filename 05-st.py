import streamlit as st
import pandas as pd
from sklearn.linear_model import LogisticRegression
import matplotlib.pyplot as plt

# Основной код сервиса
st.title('Регрессия и графики')

# Загрузка CSV-файла
uploaded_file = st.file_uploader("Загрузите файл CSV", type="csv")
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    data = data.drop(columns = ['Unnamed: 0'], axis = 1)

    # Отображение данных
    st.dataframe(data)

    # Выполнение логистической регрессии
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    model = LogisticRegression()
    model.fit(X, y)

    # Вывод результатов логистической регрессии
    st.subheader("Результаты логистической регрессии:")
    coef_dict = dict(zip(X.columns, model.coef_[0]))
    st.write(coef_dict)

    # Окна графиков
    st.subheader("Графики:")
    first_feature = st.selectbox("Выберите первую фичу", X.columns)
    second_feature = st.selectbox("Выберите вторую фичу", X.columns)
    chart_type = st.selectbox("Выберите тип графика", ["scatter", "bar plot", "plot"])

     # Создание выбранного графика
    fig, ax = plt.subplots() 
    if chart_type == "scatter":
        plt.scatter(data[first_feature], data[second_feature])
        plt.xlabel(first_feature)
        plt.ylabel(second_feature)
        ax = st.pyplot(fig)
    elif chart_type == "bar plot":
        data.plot(x=first_feature, y=second_feature, kind="bar")
        ax = st.pyplot(fig)
    else:
        data.plot(x=first_feature, y=second_feature)
        ax = st.pyplot(fig)