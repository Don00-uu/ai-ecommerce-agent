import os
import csv
from dotenv import load_dotenv
import streamlit as st
import openai

load_dotenv()

client = openai.OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)

st.title("电商文案批量生成器")
st.write("每行输入一个商品名，点击按钮批量生成卖点。")

text = st.text_area("商品列表", "无线蓝牙耳机\n牛角梳\n檀木按摩梳", height=150)

if st.button("开始生成"):
    products = [line.strip() for line in text.splitlines() if line.strip()]
    results = []
    progress = st.progress(0)

    for i, product in enumerate(products):
        resp = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "你是电商文案助手"},
                {"role": "user", "content": f"给{product}写3个卖点"},
            ],
        )
        results.append((product, resp.choices[0].message.content))
        progress.progress((i + 1) / len(products))

    with open("文案结果.csv", "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["商品名称", "文案"])
        for product, content in results:
            writer.writerow([product, content])

    st.success("全部完成，已保存到 文案结果.csv")
    for product, content in results:
        st.subheader(product)
        st.write(content)