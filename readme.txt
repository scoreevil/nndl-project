北京邮电大学 神经网络课程设计 2026.1

models文件夹中包含模型架构定义及其训练文件
newdataset为构建的新数据集
NewDatasetCreate为标注数据用的Web应用
results为任选任务一对应的生成句子和分数数据
run为可执行脚本，可快捷复现代码
utils为工具库，包含大量开发过程中的组件，包括数据预处理，评价指标，API调用等

注：由于教学云平台上传大小有限制，这里删除了一些大文件，具体包括：
1. 原数据集及预处理后的词表和npz文件
2. 训练出来的模型checkpoint
3. 用于构建新数据集的照片集
4. 由于github限制缺失newdataset/features/train_feature.npz，请自行复现代码
