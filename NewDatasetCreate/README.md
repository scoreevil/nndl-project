# 服饰图像人机结合标注工具 - Web版本

基于Flask + Vue的Web标注工具，用于服饰图像的半自动标注。

## 功能特性

- ✅ Web界面，无需图形界面环境
- ✅ 图像列表浏览（支持分页，10张/页）
- ✅ 鼠标拖拽绘制矩形选框
- ✅ 自动裁剪和填充（白底，224×224）
- ✅ 自研Transformer模型描述生成
- ✅ LMMs API描述生成（服饰+背景）
- ✅ 人工编辑描述
- ✅ 标注数据保存（JSON格式）
- ✅ 批量导出数据集（NPZ格式）
- ✅ 延迟导入torch，规避NumPy版本冲突

## 技术栈

- **后端**: Flask + Flask-CORS
- **前端**: Vue 3 (CDN) + Axios
- **图像处理**: OpenCV + PIL
- **模型推理**: PyTorch (延迟导入)
- **API调用**: requests + openai

## 安装依赖

```bash
cd NewDatasetCreate
pip install -r requirements.txt
```

## 配置

编辑 `backend/config.py` 文件，设置：

1. **图像目录**：`IMAGE_DIR` - 指向包含1000张图像的目录
2. **模型路径**：`SELF_MODEL_CHECKPOINT` - 自研模型checkpoint路径
3. **API密钥**：设置环境变量或直接在代码中配置
   - `OPENAI_API_KEY` - GPT-4V
   - `DASHSCOPE_API_KEY` 或 `QWEN_API_KEY` - Qwen-VL
   - `MOONSHOT_API_KEY` 或 `KIMI_API_KEY` - Kimi-VL
   - `ARK_API_KEY` 或 `DOUBAO_API_KEY` - Doubao-VL

## 使用方法

### 启动服务器

**方法1：使用启动脚本（推荐）**
```bash
cd NewDatasetCreate
bash run.sh
```

**方法2：直接运行**
```bash
cd NewDatasetCreate
python -m backend.app
```

**方法3：从backend目录运行**
```bash
cd NewDatasetCreate/backend
python app.py
```

服务器将在 `http://localhost:5000` 启动。

### 访问Web界面

在浏览器中打开：
```
http://localhost:5000
```

或如果远程访问：
```
http://your-server-ip:5000
```

## 操作说明

1. **加载图像**：点击左侧列表中的图像文件名
2. **绘制选框**：在图像展示区拖拽鼠标绘制绿色矩形选框
3. **确认选框**：点击"确认选框"按钮，预览区显示裁剪后的图像
4. **生成描述**：
   - 点击"生成自研模型描述"（需要先确认选框）
   - 点击"生成LMMs描述"（基于整图）
5. **编辑描述**：在右侧文本框中编辑LMMs描述
6. **保存标注**：点击"保存标注"按钮
7. **下一张**：点击"下一张"按钮继续
8. **导出数据集**：点击"导出数据集"按钮

## API接口

### 获取图像列表
```
GET /api/images/list?page=1&per_page=10
```

### 获取图像文件
```
GET /api/images/file?path=image.jpg
```

### 裁剪图像
```
POST /api/annotation/crop
Body: {
    "image_path": "...",
    "box_coords": [x1, y1, x2, y2]
}
```

### 生成自研模型描述
```
POST /api/model/self/generate
Body: {
    "cropped_image_path": "..."
}
```

### 生成LMMs描述
```
POST /api/model/lmm/generate
Body: {
    "full_image_path": "...",
    "lmm_name": "qwen"  // 可选
}
```

### 保存标注
```
POST /api/annotation/save
Body: {
    "image_path": "...",
    "box_coords": [x1, y1, x2, y2],
    "self_desc": "...",
    "lmm_clothing_desc": "...",
    "lmm_bg1": "...",
    "lmm_bg2": "..."
}
```

### 获取进度
```
GET /api/annotation/progress
```

### 导出数据集
```
POST /api/annotation/export
Body: {
    "output_path": "..."  // 可选
}
```

## 文件结构

```
NewDatasetCreate/
├── backend/
│   ├── __init__.py
│   ├── app.py              # Flask应用
│   ├── config.py           # 配置文件
│   ├── image_utils.py      # 图像处理
│   ├── model_utils.py      # 模型调用（延迟导入torch）
│   └── data_utils.py       # 数据管理
├── frontend/
│   └── index.html          # Vue前端页面
├── requirements.txt        # 依赖列表
└── README.md              # 说明文档
```

## 优势

1. **无需图形界面**：Web应用，可在任何环境运行
2. **规避NumPy冲突**：延迟导入torch，避免版本冲突
3. **跨平台**：支持Windows/Linux/Mac
4. **易于部署**：可部署到服务器，多人协作标注
5. **响应式设计**：适配不同屏幕尺寸

## 注意事项

1. 确保图像目录包含1000张图像（jpg/png格式）
2. 自研模型需要先训练并保存checkpoint
3. LMMs API需要配置有效的API密钥
4. 标注数据会自动保存到 `annotations.json`
5. 临时裁剪图像保存在 `temp_cropped/` 目录

## 故障排除

### 问题：无法访问Web界面
- 检查Flask服务器是否正常启动
- 检查防火墙设置，确保5000端口开放
- 尝试使用 `0.0.0.0` 作为host（已在代码中设置）

### 问题：图像无法加载
- 检查 `config.py` 中的 `IMAGE_DIR` 路径是否正确
- 检查图像文件是否存在且可读

### 问题：模型推理失败
- 检查模型checkpoint路径是否正确
- 检查词汇表文件是否存在
- 查看后端日志获取详细错误信息

### 问题：LMMs API调用失败
- 检查API密钥是否正确配置
- 检查网络连接
- 查看浏览器控制台和后端日志

