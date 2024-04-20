项目结构
- ./data 数据文件夹
  - 其中ImageData是数据集（.jpg)s
  - ImageWordData.csv是数据集对应的标注,一个图片路径对应一个文字
- ./indexs 向量数据创建的索引文件夹
  - I[日期].index
  - T[日期].index
  - 备注：日期就是泰迪杯2024年的[月(2位)][日(2位)]
  - I.index是图片向量的索引，用于文搜图
  - T.index是文本向量的索引，用于图搜文
- models 我的模型package
  - utils.py 一些工具函数
    - dataset.py 按批次产生数据
    - evaluate.py 评估函数
    - get_model.py 提供的在此处训练的模型的，获取模型的接口函数
    - loss.py 损失函数
    - optimizer.py 优化器
  - config.py 配置文件
  - embedding_models.py 图、文嵌入模型
  - vsepp.py 图文检索模型(魔改)
- pts 保存的模型checkpoit参数文件夹
  - 一般为best_model[日期].ckpt
- calculate_recall_at_k.py 使用向量数据库计算召回率的脚本
- get_test_index.py 生成测试集向量数据库索引的脚本
  - 运行时请修改if __name__ == '__main__':下的main函数的参数
  - 生成的索引文件会保存在./indexs文件夹下
  - 生成过程打印信息保存在根目录下的get_test_index.log文件中
- myDataset.py 数据集类，用于划分得到【train,val,test】，以及在根路径（项目路径）生成对应的【】_data.json
- 许多.ipynb文件，用于训练、测试、评估、调参等
- test.py 测试脚本,也是用于调试，这个不重要
- train.py 训练脚本,一些参数在models里的config.py中修改
  - 注意，训练前，请自行创建./model文件夹，否则报错，这里没修改。


写测试脚本时，导入模型的方式：
```python
from models import get_model, ImageRepExtractor, TextRepExtractor, VSEPP
model = get_model("your_ckpt_path")
```

### 微调(训练步骤，泰迪杯2024B):
首先确保你的数据在./data文件夹下
1. 运行myDataset.py，生成数据集划分文件(生成json)
2. 创建./model文件夹
3. 运行train.py，训练模型得到ckpt文件

### 使用模型(泰迪杯2024B):
1. 建立./pts文件夹
2. 将./model文件夹下的ckpt文件建议重命名移动到./pts文件夹下

3. （可选）运行calculate_recall_at_k.py，计算召回率

### 网页部署(泰迪杯2024B):
1. 运行get_test_index.py，生成索引文件I.index和T.index,并命名为I[日期].index和T[日期].index
2. 命令行到根目录（项目目录）`streamlit run test_page.py`
3. 浏览器打开`http://localhost:8501/`即可看到网页，也会自动弹出