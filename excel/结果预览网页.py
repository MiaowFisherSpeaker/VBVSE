import streamlit as st
import pandas as pd
st.title("结果预览网页")

root_dir = "../"
result1_df = pd.read_csv("result1.csv")
result2_df = pd.read_csv("result2.csv")

# 测试集 文搜图的 文本测试集
word_test = pd.read_csv(f"{root_dir}data/q2/word_test.csv") # text_id ,caption

# 测试集 图搜文的 图片测试集
image_test = pd.read_csv(f"{root_dir}data/q3/image_test.csv")
image_test_id = image_test["image_id"].tolist()
word_3 = pd.read_csv(f"{root_dir}data/q3/word_data.csv")  # 对应的文本数据text_id,caption两列

# 第二问
st.sidebar.markdown("### 第二问 文搜图")
query_num2 = st.sidebar.text_input("输入数字：0-4999", key="num1")
if query_num2 != "":
    query_num2 = int(query_num2)+1
    num2 = (query_num2-2)*5
    st.markdown(f"### 第二问 文搜图")
    st.markdown(f"第{query_num2}个文本搜索结果")
    # 结果df
    df = result1_df.iloc[num2:num2+5]
    st.dataframe(df)
    st.sidebar.markdown("检索文本:")
    text_id = df['text_id'].iloc[0]
    st.sidebar.markdown(f"**{word_test['caption'].loc[word_test['text_id'] == text_id].iloc[0]}**")
    st.sidebar.markdown(f"对应文本id: {df['text_id'].iloc[0]}")
    # 分3列显示图片
    col1, col2 = st.columns(2)
    for i in range(5):
        if i < 3:
            col = col1
        else:
            col = col2
        col.image(f"{root_dir}data/q2/ImageData/{df['result_image_id'].iloc[i]}",
                  caption=f"{i+1}:{df['result_image_id'].iloc[i]}")



# 第三问
st.sidebar.markdown("### 第三问 图搜文")
query_num3 = st.sidebar.text_input("输入数字：0-4999", key="num2")
if query_num3 != "":
    query_num3 = int(query_num3)
    num3 = query_num3*5
    st.markdown(f"### 第三问 图搜文")
    st.markdown(f"第{query_num3}个图片搜索结果")
    # 结果df
    df = result2_df.iloc[num3:num3+5]
    st.dataframe(df)
    st.sidebar.markdown("检索图片:")
    query_image_id = image_test_id[query_num3]
    st.sidebar.markdown(f"**{query_image_id}**")
    st.sidebar.markdown(f"这里是检查: {df['image_id'].iloc[0]}")

    # 分两列，左边显示图片，右边显示5个文本
    col1, col2 = st.columns(2)
    col1.image(f"{root_dir}data/q3/ImageData/{df['image_id'].iloc[0]}")
    col2.markdown("检索结果:")
    for i in range(5):
        text_id = df['result_text_id'].iloc[i]
        col2.markdown(f"{i+1}:{word_3.loc[word_3['text_id'] == text_id, 'caption'].values[0]}")