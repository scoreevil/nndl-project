"""
图像质量检测器：检测图像是否包含完整背景、人像和服饰
使用PIL和numpy实现，无需OpenCV
"""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
from PIL import Image, ImageStat, ImageFilter
import os


class ImageQualityDetector:
    """图像质量检测器"""
    
    def __init__(self, use_face_detection: bool = False):
        """
        初始化检测器
        
        Args:
            use_face_detection: 是否使用人脸检测（需要OpenCV，默认False）
        """
        self.use_face_detection = use_face_detection
        self.face_cascade = None
        
        if use_face_detection:
            try:
                import cv2
                # 尝试加载OpenCV的人脸检测器
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                if os.path.exists(cascade_path):
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                    self.cv2 = cv2
                else:
                    print("警告: 未找到人脸检测模型，将使用替代方法")
                    self.use_face_detection = False
            except ImportError:
                print("警告: OpenCV未安装，将使用替代方法进行人像检测")
                self.use_face_detection = False
            except Exception as e:
                print(f"警告: 无法加载人脸检测器: {e}")
                self.use_face_detection = False
    
    def detect_faces(self, image_array: np.ndarray) -> int:
        """
        检测图像中的人脸数量
        
        Args:
            image_array: 图像数组 (RGB格式)
            
        Returns:
            检测到的人脸数量
        """
        if not self.use_face_detection or self.face_cascade is None:
            return 0
        
        try:
            gray = self.cv2.cvtColor(image_array, self.cv2.COLOR_RGB2GRAY)
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            return len(faces)
        except Exception as e:
            return 0
    
    def detect_human_body(self, image_array: np.ndarray) -> bool:
        """
        使用简单方法检测是否可能包含人体
        基于图像中的边缘和纹理特征
        
        Args:
            image_array: 图像数组 (RGB格式)
            
        Returns:
            是否可能包含人体
        """
        try:
            # 转换为灰度
            if len(image_array.shape) == 3:
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image_array
            
            # 使用Sobel算子检测边缘
            from scipy import ndimage
            try:
                sobel_x = ndimage.sobel(gray, axis=1)
                sobel_y = ndimage.sobel(gray, axis=0)
                edges = np.hypot(sobel_x, sobel_y)
                
                # 计算边缘密度
                edge_density = np.sum(edges > np.percentile(edges, 90)) / edges.size
                
                # 如果边缘密度适中（0.05-0.3），可能包含人体
                return 0.05 < edge_density < 0.3
            except ImportError:
                # 如果没有scipy，使用简单的梯度方法
                grad_x = np.abs(np.diff(gray, axis=1))
                grad_y = np.abs(np.diff(gray, axis=0))
                edges = np.concatenate([grad_x.flatten(), grad_y.flatten()])
                edge_density = np.sum(edges > np.percentile(edges, 90)) / len(edges)
                return 0.05 < edge_density < 0.3
        except Exception:
            # 如果检测失败，基于图像中心区域的特征判断
            h, w = image_array.shape[:2]
            center_region = image_array[h//4:3*h//4, w//4:3*w//4]
            if len(center_region.shape) == 3:
                center_variance = np.var(center_region)
                # 中心区域有足够的纹理变化，可能包含主体
                return center_variance > 500
            return False
    
    def calculate_sharpness(self, image_array: np.ndarray) -> float:
        """
        计算图像清晰度（使用拉普拉斯算子）
        
        Args:
            image_array: 图像数组 (RGB格式)
            
        Returns:
            清晰度分数（越高越清晰）
        """
        try:
            # 转换为灰度
            if len(image_array.shape) == 3:
                gray = np.dot(image_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                gray = image_array
            
            # 使用拉普拉斯算子
            from scipy import ndimage
            try:
                laplacian = ndimage.laplacian(gray)
                sharpness = laplacian.var()
                return float(sharpness)
            except ImportError:
                # 如果没有scipy，使用简单的二阶差分
                grad_x = np.diff(gray, axis=1)
                grad_y = np.diff(gray, axis=0)
                second_grad_x = np.diff(grad_x, axis=1)
                second_grad_y = np.diff(grad_y, axis=0)
                laplacian_approx = np.abs(second_grad_x).mean() + np.abs(second_grad_y).mean()
                return float(laplacian_approx * 100)  # 缩放以匹配scipy的结果范围
        except Exception:
            return 0.0
    
    def detect_background_completeness(self, image_array: np.ndarray) -> float:
        """
        检测背景完整性
        基于图像边缘区域的颜色分布和纹理
        
        Args:
            image_array: 图像数组 (RGB格式)
            
        Returns:
            背景完整性分数 (0-1)
        """
        try:
            h, w = image_array.shape[:2]
            
            # 提取边缘区域（图像四周10%的区域）
            border_width = int(min(h, w) * 0.1)
            if border_width < 1:
                border_width = 1
            
            # 上边缘
            top_border = image_array[0:border_width, :]
            # 下边缘
            bottom_border = image_array[h-border_width:h, :]
            # 左边缘
            left_border = image_array[:, 0:border_width]
            # 右边缘
            right_border = image_array[:, w-border_width:w]
            
            # 合并所有边缘区域
            borders = []
            for border in [top_border, bottom_border, left_border, right_border]:
                if border.size > 0:
                    if len(border.shape) == 3:
                        borders.append(border.reshape(-1, border.shape[-1]))
                    else:
                        borders.append(border.flatten())
            
            if len(borders) == 0:
                return 0.0
            
            borders = np.vstack(borders) if len(borders) > 0 else np.array([])
            
            if borders.size == 0:
                return 0.0
            
            # 计算边缘区域的颜色方差（背景通常颜色变化较小）
            if len(borders.shape) == 2 and borders.shape[1] >= 3:
                border_variance = np.var(borders[:, :3], axis=0).mean()
            else:
                border_variance = np.var(borders)
            
            # 归一化到0-1范围（经验值：方差小于1000认为背景较完整）
            completeness = 1.0 - min(border_variance / 1000.0, 1.0)
            
            return float(completeness)
        except Exception:
            return 0.0
    
    def detect_clothing_features(self, image_array: np.ndarray) -> float:
        """
        检测服饰特征
        基于颜色分布、纹理和边缘特征
        
        Args:
            image_array: 图像数组 (RGB格式)
            
        Returns:
            服饰特征分数 (0-1)
        """
        try:
            # 计算颜色多样性（服饰通常有丰富的颜色）
            if len(image_array.shape) == 3:
                # 计算RGB各通道的直方图
                hist_r = np.histogram(image_array[:, :, 0].flatten(), bins=32, range=(0, 256))[0]
                hist_g = np.histogram(image_array[:, :, 1].flatten(), bins=32, range=(0, 256))[0]
                hist_b = np.histogram(image_array[:, :, 2].flatten(), bins=32, range=(0, 256))[0]
            else:
                hist_r = np.histogram(image_array.flatten(), bins=32, range=(0, 256))[0]
                hist_g = hist_r
                hist_b = hist_r
            
            # 计算直方图的熵（颜色分布的多样性）
            def entropy(hist):
                hist = hist[hist > 0]
                if len(hist) == 0:
                    return 0
                prob = hist / hist.sum()
                return -np.sum(prob * np.log2(prob + 1e-10))
            
            color_diversity = (entropy(hist_r) + entropy(hist_g) + entropy(hist_b)) / 3.0
            
            # 归一化（经验值：熵大于5认为有较好的颜色多样性）
            clothing_score = min(color_diversity / 5.0, 1.0)
            
            return float(clothing_score)
        except Exception:
            return 0.0
    
    def evaluate_image(self, image_path: str) -> Dict[str, float]:
        """
        评估单张图像的质量
        
        Args:
            image_path: 图像路径
            
        Returns:
            评估结果字典，包含各项分数
        """
        try:
            # 使用PIL读取图像
            img = Image.open(str(image_path))
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # 转换为numpy数组
            image_array = np.array(img)
            
            # 各项检测
            face_count = self.detect_faces(image_array)
            has_face = 1.0 if face_count > 0 else 0.0
            
            has_body = 1.0 if self.detect_human_body(image_array) else 0.0
            
            sharpness = self.calculate_sharpness(image_array)
            # 归一化清晰度（经验值：>100认为清晰，如果没有scipy可能值较小，调整阈值）
            sharpness_score = min(sharpness / 100.0, 1.0) if sharpness > 0 else min(sharpness / 10.0, 1.0)
            
            background_score = self.detect_background_completeness(image_array)
            clothing_score = self.detect_clothing_features(image_array)
            
            # 综合评分（加权平均）
            # 权重：人像0.3，清晰度0.25，背景0.25，服饰0.2
            overall_score = (
                (has_face + has_body) / 2.0 * 0.3 +
                sharpness_score * 0.25 +
                background_score * 0.25 +
                clothing_score * 0.2
            )
            
            # 判断是否有效（综合分数>0.5且至少有人像或人体）
            is_valid = overall_score > 0.5 and (has_face > 0 or has_body > 0)
            
            return {
                'has_face': has_face,
                'has_body': has_body,
                'sharpness': sharpness,
                'sharpness_score': sharpness_score,
                'background_completeness': background_score,
                'clothing_features': clothing_score,
                'overall_score': overall_score,
                'is_valid': is_valid
            }
        except Exception as e:
            print(f"评估图像 {image_path} 时出错: {e}")
            return {
                'has_face': 0.0,
                'has_body': 0.0,
                'sharpness': 0.0,
                'sharpness_score': 0.0,
                'background_completeness': 0.0,
                'clothing_features': 0.0,
                'overall_score': 0.0,
                'is_valid': False
            }
    
    def filter_images(
        self,
        image_dir: str,
        output_dir: str,
        num_images: int = 1000,
        min_score: float = 0.5,
        target_size: tuple = (224, 224),
        prefix: str = "bg_fashion"
    ) -> List[Tuple[str, Dict]]:
        """
        筛选并处理图像
        
        Args:
            image_dir: 源图像目录
            output_dir: 输出目录
            num_images: 需要选择的图像数量
            min_score: 最低综合分数阈值
            target_size: 目标尺寸
            prefix: 输出文件名前缀
            
        Returns:
            选中的图像列表和评估结果
        """
        print("="*60)
        print("图像质量检测与筛选")
        print("="*60)
        
        # 获取所有图像文件
        source_path = Path(image_dir)
        image_files = list(source_path.glob("*.jpg"))
        image_files.extend(list(source_path.glob("*.JPG")))
        image_files.extend(list(source_path.glob("*.png")))
        image_files.extend(list(source_path.glob("*.PNG")))
        
        if len(image_files) == 0:
            print(f"错误: 在 {image_dir} 中未找到图像文件")
            return []
        
        print(f"\n找到 {len(image_files)} 张图像")
        print("开始评估图像质量...")
        
        # 评估所有图像
        evaluated_images = []
        from tqdm import tqdm
        
        for img_file in tqdm(image_files, desc="评估图像"):
            result = self.evaluate_image(str(img_file))
            if result['is_valid'] and result['overall_score'] >= min_score:
                evaluated_images.append((str(img_file), result))
        
        # 按综合分数排序
        evaluated_images.sort(key=lambda x: x[1]['overall_score'], reverse=True)
        
        print(f"\n有效图像数量: {len(evaluated_images)}")
        print(f"筛选阈值: 综合分数 >= {min_score}")
        
        # 选择前N张
        selected_images = evaluated_images[:num_images]
        
        if len(selected_images) < num_images:
            print(f"警告: 只有 {len(selected_images)} 张图像满足条件，少于请求的 {num_images} 张")
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 处理选中的图像
        print(f"\n开始处理选中的 {len(selected_images)} 张图像...")
        from tqdm import tqdm
        
        processed_count = 0
        for idx, (img_path, result) in enumerate(tqdm(selected_images, desc="处理图像")):
            try:
                # 读取并调整大小
                img = Image.open(img_path)
                if img.mode != 'RGB':
                    img = img.convert('RGB')
                img_resized = img.resize(target_size, Image.Resampling.LANCZOS)
                
                # 保存
                output_filename = f"{prefix}_{idx+1:04d}.jpg"
                output_filepath = output_path / output_filename
                img_resized.save(output_filepath, "JPEG", quality=95, optimize=True)
                
                processed_count += 1
            except Exception as e:
                print(f"\n错误: 处理 {img_path} 时出错: {e}")
                continue
        
        print("\n" + "="*60)
        print("处理完成")
        print("="*60)
        print(f"成功处理: {processed_count} 张")
        print(f"输出目录: {output_path.absolute()}")
        
        # 打印统计信息
        if selected_images:
            scores = [r[1]['overall_score'] for r in selected_images]
            print(f"\n质量统计:")
            print(f"  平均综合分数: {np.mean(scores):.3f}")
            print(f"  最高分数: {np.max(scores):.3f}")
            print(f"  最低分数: {np.min(scores):.3f}")
        
        print("="*60)
        
        return selected_images


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="图像质量检测与筛选")
    parser.add_argument("--image_dir", type=str, default="test/image",
                       help="源图像目录")
    parser.add_argument("--output_dir", type=str, default="newdataset/images",
                       help="输出目录")
    parser.add_argument("--num_images", type=int, default=1000,
                       help="需要选择的图像数量")
    parser.add_argument("--min_score", type=float, default=0.5,
                       help="最低综合分数阈值")
    parser.add_argument("--target_size", type=int, nargs=2, default=[224, 224],
                       help="目标尺寸 (width height)")
    parser.add_argument("--prefix", type=str, default="bg_fashion",
                       help="输出文件名前缀")
    parser.add_argument("--no_face_detection", action="store_true",
                       help="禁用人脸检测")
    
    args = parser.parse_args()
    
    detector = ImageQualityDetector(use_face_detection=not args.no_face_detection)
    
    detector.filter_images(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        num_images=args.num_images,
        min_score=args.min_score,
        target_size=tuple(args.target_size),
        prefix=args.prefix
    )


if __name__ == "__main__":
    main()

