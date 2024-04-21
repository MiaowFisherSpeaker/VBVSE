# import re
# from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


# WHITESPACE_HANDLER = lambda k: re.sub('\s+', ' ', re.sub('\n+', ' ', k.strip()))
# def get_summary_model():
#     model_name = "csebuetnlp/mT5_multilingual_XLSum"
#     tokenizer = AutoTokenizer.from_pretrained(model_name)
#     model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
#     return tokenizer,model
# tokenizer,model = get_summary_model() # 这里用到的是不在__main__文件也会调用
# def get_summary_text(article_text:str,tokenizer=tokenizer,model=model):
#     input_ids = tokenizer(
#         [WHITESPACE_HANDLER(article_text)],
#         return_tensors="pt",
#         padding="max_length",
#         truncation=True,
#         max_length=512
#     )["input_ids"]
#
#     output_ids = model.generate(
#         input_ids=input_ids,
#         max_length=84,
#         no_repeat_ngram_size=2,
#         num_beams=4
#     )[0]
#
#     summary = tokenizer.decode(
#         output_ids,
#         skip_special_tokens=True,
#         clean_up_tokenization_spaces=False
#     )
#
#     return summary

# 个人感觉就是文搜图才用得到这个
# 效果只能说用来概括可以，但是如果反映到图片上，细节不能体现。
# 就比如A比B多10分拿下比赛，两个球队起了争执。只会概括前面，没有概括后面。
# 预处理可以做的：
# 1。 删除后面.gif这类文件名格式的

#----------------------------------------------------------------------
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline
model_name = "deepset/roberta-base-squad2"
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)


def get_summary_text(context:str,question="主要内容是什么?请使用原文单词",nlp=nlp):
    """只对单句话处理"""
    QA_input = {
        'question': question,
        'context': context
    }
    res = nlp(QA_input)
    return res['answer']
# ----------------------------------------------------------------------

#----------------------------------------------------------------------
# from transformers import pipeline
#
# nlp = pipeline("question-answering", "timpal0l/mdeberta-v3-base-squad2")
#
#
# # question = "Where do I live?"
# # context = "My name is Tim and I live in Sweden."
#
#
# def get_summary_text(context: str, question="Where do I live?", nlp=nlp):
#     return nlp(question=question, context=context)['answer']


if __name__ == '__main__':
    text = ["胡睿宝再次炮轰恒大 里皮亲信力挺其改回年龄",
            "天津病例流调为何疑团重重?<人名>对话疾控专家<人名>",
            "从指南变迁看 --β 受体阻滞剂的争议与再评价",
            "只看阵容对手就已经吓尿了,巴西队黄金一代神挡杀神,佛挡杀佛!",
            '"凡相,墨象"<人名>艺术作品展 水墨艺术与人文情怀的极致表达',
            "这样手套箱就拆下来了",
            "这样手套箱就拆下来了  ——环球网报道",
            "My name is Tim and I live in Sweden.",
            "官方正版羽毛球教学与训练羽毛球运动教学与训练教程<人名> 高等 ——环球网"]
    # for t in text:
    #     print(get_summary_text(t))
    for t in text:
        print(get_summary_text(t))
