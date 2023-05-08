import streamlit as st
from generation import *
st.markdown("""
<style>
.css-nqowgj edgvbvh3
{
    visibility: hidden;
}
.css-h5rgaw egzxvld1
{
    visibility: hidden;
}
</style>
""", unsafe_allow_html=True)

st.markdown('## Генерирую текстовое описание изображений ')
st.markdown('made by Kolesnikov Dmitry')
#st.markdown('---')
image = st.file_uploader("Загрузите изображение", type=['png', 'jpg','jpeg'])
if image is not None:
    st.image(image)
    img = Image.open(image)
    st.markdown('Результат генерации описания (модель LSTM + ResNet):')
    text = predict(img)
    st.markdown('#### __' + text + '__')
