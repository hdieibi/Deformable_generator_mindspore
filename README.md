# Deformable_generator_mindspore
Deformable Generator based on Mindspore

哈工程2023模式识别和数字图像处理的最后一次课程作业

时间有限，复现出来的效果不好，重构和采样生成的图像示例如imgs文件夹所示，作业要求的生成部分代码没有写完

代码和老师给的代码有些地方不一样，是我自己改的。有兴趣的可以自己再改

### 环境配置

路径cd到项目文件夹下

```python
conda create -n mindspore python=3.9
conda activate mindspore

pip install -r requirements.txt
# 这个文件是直接用pip freeze导出来的
# 如果出了问题，按照代码里import的包一个一个装就可以
```

### 数据集

使用老师发在群里的1Kfaces数据，放在data文件夹下就可以

### 运行

```python
# 终端
python main.py --config configs/faces.yaml
# 重构和采样的图像以及模型权重全部保存在log文件夹里
```

### 评价、生成

这一步目前还没有完善，作业里要求的采样某一维度和交换图像表观几何都还没写，有能力的可以自己写一下

```python
# 还没有完成，无法直接执行
python main.py --mode eval --config configs/faces.yaml --checkpoint 模型保存的权重
```

