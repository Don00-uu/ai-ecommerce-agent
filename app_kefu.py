import os
import json
import streamlit as st
import openai
from dotenv import load_dotenv

load_dotenv()   # 把 .env 文件里的密钥读进来

chat_client = openai.OpenAI(
    api_key=os.environ["DEEPSEEK_API_KEY"],
    base_url="https://api.deepseek.com",
)
emb_client = openai.OpenAI(
    api_key=os.environ["SILICONFLOW_API_KEY"],
    base_url="https://api.siliconflow.cn/v1",
)

# ===== 知识库 =====
docs = [
    "退换货政策：签收后 7 天内，商品未使用、包装完好，可申请无理由退货；30 天内出现质量问题可换货。",
    "物流时效：默认发中通快递，江浙沪次日达，其他地区 2-4 天，偏远地区 5-7 天。",
    "发票：支持开电子发票，下单时在备注里填写抬头和税号。",
    "檀木保养：避免长时间沾水和暴晒，每月用木蜡油擦拭一次，可保持光泽。",
    "支付方式：支持微信、支付宝、银行卡支付，暂不支持货到付款。",
    "优惠券规则：满 199 减 20，满 299 减 50，优惠券不与其他活动叠加。",
    "发货时间：每天下午 4 点前付款的订单当天发出，之后的顺延到第二天。",
]

# 算好一次就缓存起来，别每次点按钮都重算
@st.cache_data(show_spinner="正在准备知识库…")
def load_vectors():
    vectors = []
    for d in docs:
        vectors.append(
            emb_client.embeddings.create(model="BAAI/bge-m3", input=d).data[0].embedding
        )
    return vectors

doc_vectors = load_vectors()

# ===== 检索 =====
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(y * y for y in b) ** 0.5
    return dot / (na * nb)

def rewrite(question):
    prompt = f"""把下面这句用户的口语问题，改写成一句明确的检索问句。
只输出改写结果本身，不要解释、不要反问、不要加引号。

示例：
用户：这个能便宜点吗 → 优惠券和满减活动规则
用户：我下单了多久能收到 → 物流配送时效

现在改写这一句：
用户：{question} →"""
    resp = chat_client.chat.completions.create(
        model="deepseek-chat",
        messages=[{"role": "user", "content": prompt}],
    )
    return resp.choices[0].message.content.strip().strip('"')

def search(question, top_n=4):
    qv = emb_client.embeddings.create(
        model="BAAI/bge-m3", input=rewrite(question)
    ).data[0].embedding
    scores = []
    for doc, dv in zip(docs, doc_vectors):
        scores.append((cosine(qv, dv), doc))
    scores.sort(reverse=True)
    return [doc for score, doc in scores[:top_n]]

# ===== 工具 =====
inventory = {"无线蓝牙耳机": 128, "牛角梳": 0, "檀木按摩梳": 56}
prices = {"无线蓝牙耳机": 199, "牛角梳": 89, "檀木按摩梳": 129}

def get_stock(product):
    return inventory.get(product, -1)

def get_price(product):
    return prices.get(product, -1)

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_stock",
            "description": "查询某个商品的库存数量",
            "parameters": {
                "type": "object",
                "properties": {"product": {"type": "string", "description": "商品名称"}},
                "required": ["product"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_price",
            "description": "查询某个商品的价格，单位是元",
            "parameters": {
                "type": "object",
                "properties": {"product": {"type": "string", "description": "商品名称"}},
                "required": ["product"],
            },
        },
    },
]

# ===== 主流程：这次改成返回结果，不再 print =====
def ask(question):
    related = search(question)
    context = "\n".join(related)

    messages = [
        {"role": "system", "content": "你是电商客服助手。回答时优先依据下面的资料；需要库存或价格这类具体数据时，调用提供的工具。资料里没有的不要编造。\n" + context},
        {"role": "user", "content": question},
    ]

    resp = chat_client.chat.completions.create(
        model="deepseek-chat", messages=messages, tools=tools
    )
    msg = resp.choices[0].message

    if msg.tool_calls:
        messages.append(msg)
        for call in msg.tool_calls:
            args = json.loads(call.function.arguments)
            name = call.function.name
            if name == "get_stock":
                result = get_stock(args["product"])
            elif name == "get_price":
                result = get_price(args["product"])
            else:
                result = "未知工具"
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": str(result),
            })
        resp = chat_client.chat.completions.create(
            model="deepseek-chat", messages=messages, tools=tools
        )
        msg = resp.choices[0].message

    return msg.content, related

# ===== 界面 =====
st.title("电商智能客服 Agent")
st.caption("可以问我：退换货、物流、发票、保养，也可以问具体商品的库存和价格。")

question = st.text_input("请输入您的问题", "我不想要了")

if st.button("发送"):
    if question.strip():
        with st.spinner("正在思考…"):
            answer, related = ask(question)
        st.markdown("**客服回答：**")
        st.write(answer)
        with st.expander("看看它在后台检索到了哪些资料"):
            for d in related:
                st.write("· " + d)