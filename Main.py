import streamlit as st

st.set_page_config(
    page_title="Hello",
    page_icon="👋",
)

st.write("# Добро пожаловать на страничку нашего проекта! 👋")


st.markdown(
    """
    Приложение позволяет классифицировать ваши изображения.

    **👈 Выберите что вам необходимо: локализация или детекция объекта?**
    ### Что можно найти в этом сервисе?
    - Страницу, позволяющую определить на пользовательской фотографии кабельную башню или ветрогенератор
    - Страницу, позволяющую классифицировать изображение клеток крови
    - Страницу с информацией о модели:
    - - число эпох обучения
    - - объем выборок
    - - метрики (для детекции mAP, график PR кривой, confusion matrix и т.д.)

    ### Над проектом трудились:
    - [Даша](https://github.com/Dasha0203)
    - [Андрей](https://github.com/Andriano2323)
"""
)
