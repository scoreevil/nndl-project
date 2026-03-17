"""
Flask后端API服务器
"""
from flask import Flask, request, jsonify, send_from_directory
from pathlib import Path
import os
import sys

# 尝试导入flask_cors，如果不存在则使用简单的CORS处理
try:
    from flask_cors import CORS
    CORS_AVAILABLE = True
except ImportError:
    CORS_AVAILABLE = False
    print("警告: flask-cors未安装，将使用简单的CORS处理")

# 添加项目根目录到路径
_app_file = Path(__file__).resolve()
BACKEND_DIR = _app_file.parent
NEWDATASET_DIR = BACKEND_DIR.parent
PROJECT_ROOT = NEWDATASET_DIR.parent

# 添加路径到sys.path
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(NEWDATASET_DIR))

# 尝试不同的导入方式
try:
    from backend import config
    from backend import image_utils
    from backend import model_utils
    from backend import data_utils
except ImportError:
    # 如果相对导入失败，尝试直接导入
    import config
    import image_utils
    import model_utils
    import data_utils

# 设置静态文件目录
FRONTEND_DIR = Path(__file__).parent.parent / 'frontend'
app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path='')

# 配置CORS
if CORS_AVAILABLE:
    CORS(app)
else:
    # 简单的CORS处理
    @app.after_request
    def after_request(response):
        response.headers.add('Access-Control-Allow-Origin', '*')
        response.headers.add('Access-Control-Allow-Headers', 'Content-Type,Authorization')
        response.headers.add('Access-Control-Allow-Methods', 'GET,PUT,POST,DELETE,OPTIONS')
        return response


@app.route('/')
def index():
    """前端页面"""
    return send_from_directory(str(FRONTEND_DIR), 'index.html')


@app.route('/api/images/list', methods=['GET'])
def get_image_list():
    """获取图像列表"""
    try:
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', config.ITEMS_PER_PAGE))
        
        image_dir = config.IMAGE_DIR
        if not image_dir.exists():
            return jsonify({"error": f"图像目录不存在: {image_dir}"}), 404
        
        # 获取所有图像文件
        image_files = sorted([
            str(f) for f in list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + 
            list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.PNG"))
        ])
        
        total = len(image_files)
        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        
        images = []
        for img_path in image_files[start_idx:end_idx]:
            img_name = Path(img_path).name
            images.append({
                "name": img_name,
                "path": img_path,
                "url": f"/api/images/file?path={img_name}"
            })
        
        return jsonify({
            "images": images,
            "total": total,
            "page": page,
            "per_page": per_page,
            "total_pages": (total + per_page - 1) // per_page
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/images/file', methods=['GET'])
def get_image_file():
    """获取图像文件"""
    try:
        image_name = request.args.get('path')
        if not image_name:
            return jsonify({"error": "缺少path参数"}), 400
        
        image_dir = config.IMAGE_DIR
        image_path = image_dir / image_name
        
        if not image_path.exists():
            return jsonify({"error": f"图像文件不存在: {image_name}"}), 404
        
        return send_from_directory(str(image_dir), image_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/images/info', methods=['GET'])
def get_image_info():
    """获取图像信息"""
    try:
        image_path = request.args.get('path')
        if not image_path:
            return jsonify({"error": "缺少path参数"}), 400
        
        if not os.path.exists(image_path):
            return jsonify({"error": f"图像文件不存在: {image_path}"}), 404
        
        info = image_utils.get_image_info(image_path)
        return jsonify(info)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/annotation/crop', methods=['POST'])
def crop_image():
    """裁剪图像"""
    try:
        data = request.json
        image_path = data.get('image_path')
        box_coords = data.get('box_coords')  # [x1, y1, x2, y2]
        
        if not image_path or not box_coords:
            return jsonify({"error": "缺少必要参数"}), 400
        
        if len(box_coords) != 4:
            return jsonify({"error": "选框坐标格式错误"}), 400
        
        # 验证坐标
        if not image_utils.validate_box_coords(image_path, tuple(box_coords)):
            return jsonify({"error": "选框坐标无效"}), 400
        
        # 裁剪和填充
        cropped_path, cropped_pil = image_utils.crop_and_pad(
            image_path, tuple(box_coords), config.IMAGE_SIZE
        )
        
        # 返回裁剪后图像的URL
        cropped_name = Path(cropped_path).name
        cropped_url = f"/api/images/cropped?path={cropped_name}"
        
        return jsonify({
            "success": True,
            "cropped_path": cropped_path,
            "cropped_url": cropped_url
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/images/cropped', methods=['GET'])
def get_cropped_image():
    """获取裁剪后的图像"""
    try:
        image_name = request.args.get('path')
        if not image_name:
            return jsonify({"error": "缺少path参数"}), 400
        
        cropped_dir = config.TEMP_CROPPED_DIR
        image_path = cropped_dir / image_name
        
        if not image_path.exists():
            return jsonify({"error": f"图像文件不存在: {image_name}"}), 404
        
        return send_from_directory(str(cropped_dir), image_name)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/self/generate', methods=['POST'])
def generate_self_desc():
    """生成自研模型描述"""
    try:
        data = request.json
        cropped_image_path = data.get('cropped_image_path')
        
        if not cropped_image_path:
            return jsonify({"error": "缺少cropped_image_path参数"}), 400
        
        if not os.path.exists(cropped_image_path):
            return jsonify({"error": f"裁剪图像不存在: {cropped_image_path}"}), 404
        
        # 生成描述
        description = model_utils.self_model_infer(cropped_image_path)
        
        return jsonify({
            "success": True,
            "description": description
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/model/lmm/generate', methods=['POST'])
def generate_lmm_desc():
    """生成LMMs描述"""
    try:
        data = request.json
        full_image_path = data.get('full_image_path')
        lmm_name = data.get('lmm_name')  # 可选，指定使用的LMM
        
        if not full_image_path:
            return jsonify({"error": "缺少full_image_path参数"}), 400
        
        if not os.path.exists(full_image_path):
            return jsonify({"error": f"图像文件不存在: {full_image_path}"}), 404
        
        # 生成描述
        clothing, bg1, bg2 = model_utils.lmm_infer(full_image_path, lmm_name)
        
        return jsonify({
            "success": True,
            "clothing_desc": clothing,
            "bg1_desc": bg1,
            "bg2_desc": bg2
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/annotation/save', methods=['POST'])
def save_annotation():
    """保存标注"""
    try:
        data = request.json
        use_lmm_as_first = data.get('use_lmm_as_first', False)
        
        annotation_data = {
            "image_path": data.get('image_path'),
            "box_coords": data.get('box_coords'),
            "self_desc": data.get('self_desc', ''),
            "lmm_clothing_desc": data.get('lmm_clothing_desc', ''),
            "lmm_bg1": data.get('lmm_bg1', ''),
            "lmm_bg2": data.get('lmm_bg2', ''),
            "use_lmm_as_first": use_lmm_as_first,
            "status": "completed"
        }
        
        # 验证必要字段
        if not annotation_data['image_path'] or not annotation_data['box_coords']:
            return jsonify({"error": "缺少必要字段"}), 400
        
        # 根据use_lmm_as_first验证第一个描述
        first_desc = annotation_data['lmm_clothing_desc'] if use_lmm_as_first else annotation_data['self_desc']
        if not first_desc:
            desc_type = "LMM" if use_lmm_as_first else "自研模型"
            return jsonify({"error": f"{desc_type}描述为空"}), 400
        
        # 保存
        data_utils.save_annotation(annotation_data, str(config.ANNOTATION_JSON))
        
        return jsonify({"success": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/annotation/progress', methods=['GET'])
def get_progress():
    """获取标注进度"""
    try:
        count = data_utils.get_annotation_count(str(config.ANNOTATION_JSON))
        
        # 获取总图像数
        image_dir = config.IMAGE_DIR
        if image_dir.exists():
            total = len(list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + 
                       list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.PNG")))
        else:
            total = 0
        
        return jsonify({
            "annotated": count,
            "total": total,
            "percentage": (count * 100 // total) if total > 0 else 0
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/annotation/export', methods=['POST'])
def export_dataset():
    """导出数据集"""
    try:
        # 允许没有JSON数据或JSON数据为空的情况
        data = request.json if request.is_json else {}
        output_path = data.get('output_path') if data else None
        
        if not output_path:
            output_path = str(config.OUTPUT_NPZ)
        
        data_utils.export_dataset(str(config.ANNOTATION_JSON), output_path)
        
        return jsonify({
            "success": True,
            "output_path": output_path
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/annotation/status', methods=['GET'])
def get_annotation_status():
    """获取标注状态"""
    try:
        status = data_utils.get_annotation_status(str(config.ANNOTATION_JSON))
        return jsonify(status)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/api/batch/generate', methods=['POST'])
def batch_generate_all():
    """批量生成所有图像的描述"""
    try:
        data = request.json
        lmm_name = data.get('lmm_name', 'qwen')
        use_lmm_as_first = data.get('use_lmm_as_first', False)
        
        # 获取所有图像文件
        image_dir = config.IMAGE_DIR
        if not image_dir.exists():
            return jsonify({"error": f"图像目录不存在: {image_dir}"}), 404
        
        image_files = sorted([
            str(f) for f in list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png")) + 
            list(image_dir.glob("*.JPG")) + list(image_dir.glob("*.PNG"))
        ])
        
        if len(image_files) == 0:
            return jsonify({"error": "没有找到图像文件"}), 404
        
        # 加载已有标注
        annotations = data_utils.load_annotations(str(config.ANNOTATION_JSON))
        annotation_dict = {ann.get('image_path'): ann for ann in annotations}
        
        success_count = 0
        fail_count = 0
        
        # 批量处理
        for image_path in image_files:
            try:
                # 检查是否已有标注
                existing = annotation_dict.get(image_path, {})
                
                # 生成LMM描述
                clothing_desc, bg1_desc, bg2_desc = model_utils.lmm_infer(image_path, lmm_name)
                
                # 如果使用LMM作为第一个描述，则不需要生成自研模型描述
                if use_lmm_as_first:
                    # 更新或创建标注
                    annotation_data = {
                        "image_path": image_path,
                        "box_coords": existing.get('box_coords', []),
                        "self_desc": existing.get('self_desc', ''),
                        "lmm_clothing_desc": clothing_desc,
                        "lmm_bg1": bg1_desc,
                        "lmm_bg2": bg2_desc,
                        "use_lmm_as_first": True,
                        "status": "completed" if existing.get('box_coords') else "pending"
                    }
                else:
                    # 需要生成自研模型描述（需要选框）
                    if not existing.get('box_coords'):
                        # 如果没有选框，跳过自研模型描述生成
                        annotation_data = {
                            "image_path": image_path,
                            "box_coords": [],
                            "self_desc": "",
                            "lmm_clothing_desc": clothing_desc,
                            "lmm_bg1": bg1_desc,
                            "lmm_bg2": bg2_desc,
                            "use_lmm_as_first": False,
                            "status": "pending"
                        }
                    else:
                        # 有选框，可以生成自研模型描述
                        # 需要先裁剪图像
                        cropped_path, _ = image_utils.crop_and_pad(
                            image_path, 
                            tuple(existing['box_coords']), 
                            config.IMAGE_SIZE
                        )
                        self_desc = model_utils.self_model_infer(cropped_path)
                        
                        annotation_data = {
                            "image_path": image_path,
                            "box_coords": existing['box_coords'],
                            "self_desc": self_desc,
                            "lmm_clothing_desc": clothing_desc,
                            "lmm_bg1": bg1_desc,
                            "lmm_bg2": bg2_desc,
                            "use_lmm_as_first": False,
                            "status": "completed"
                        }
                
                # 保存标注
                data_utils.save_annotation(annotation_data, str(config.ANNOTATION_JSON))
                success_count += 1
                
            except Exception as e:
                print(f"处理图像 {image_path} 失败: {e}")
                fail_count += 1
                continue
        
        return jsonify({
            "success": True,
            "success_count": success_count,
            "fail_count": fail_count,
            "total": len(image_files)
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)

