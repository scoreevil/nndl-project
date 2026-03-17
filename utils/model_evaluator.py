"""
统一的模型评价器：整合所有评价指标
提供统一的接口来评估模型生成结果的质量
"""
import torch
import sys
import os
from pathlib import Path
from typing import List, Dict, Optional
from .evaluators.meteor_evaluator import METEOREvaluator
from .evaluators.rouge_l_evaluator import ROUGELEvaluator
from .evaluators.cider_d_evaluator import CIDErDEvaluator
from .evaluators.spice_evaluator import SPICEEvaluator


class ModelEvaluator:
    """
    模型评价器：基于METEOR、ROUGE-L、CIDEr-D和SPICE评价指标
    
    提供统一的接口来评估模型生成结果的质量
    """
    
    def __init__(self):
        """
        初始化模型评价器
        
        使用METEOR、ROUGE-L、CIDEr-D和SPICE指标进行评价
        """
        self.meteor_evaluator = METEOREvaluator()
        self.rouge_l_evaluator = ROUGELEvaluator()
        self.cider_d_evaluator = CIDErDEvaluator()
        self.spice_evaluator = SPICEEvaluator()
    
    def evaluate(self, candidates: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        评估一批候选句的质量
        
        Args:
            candidates: 候选句列表（生成的描述列表）
            references: 参考句列表（每个元素是一个参考句列表，可以有多个参考描述）
        
        Returns:
            评价结果字典，包含：
            - 'meteor_mean': 平均METEOR得分
            - 'meteor_scores': 每个样本的METEOR得分列表
            - 'rouge_l_mean': 平均ROUGE-L得分
            - 'rouge_l_scores': 每个样本的ROUGE-L得分列表
            - 'cider_d_mean': 平均CIDEr-D得分
            - 'cider_d_scores': 每个样本的CIDEr-D得分列表
            - 'spice_mean': 平均SPICE得分
            - 'spice_scores': 每个样本的SPICE得分列表
        """
        num_samples = len(candidates)
        print(f"  计算METEOR指标 ({num_samples} 个样本)...")
        meteor_results = self.meteor_evaluator.evaluate_batch(candidates, references)
        print(f"  [OK] METEOR完成: {meteor_results['meteor_mean']:.4f}")
        
        print(f"  计算ROUGE-L指标 ({num_samples} 个样本)...")
        rouge_l_results = self.rouge_l_evaluator.evaluate_batch(candidates, references)
        print(f"  [OK] ROUGE-L完成: {rouge_l_results['rouge_l_mean']:.4f}")
        
        print(f"  计算CIDEr-D指标 ({num_samples} 个样本)...")
        cider_d_results = self.cider_d_evaluator.evaluate_batch(candidates, references)
        print(f"  [OK] CIDEr-D完成: {cider_d_results['cider_d_mean']:.4f}")
        
        print(f"  计算SPICE指标 ({num_samples} 个样本)...")
        spice_results = self.spice_evaluator.evaluate_batch(candidates, references)
        print(f"  [OK] SPICE完成: {spice_results['spice_mean']:.4f}")
        
        return {
            'meteor_mean': meteor_results['meteor_mean'],
            'meteor_scores': meteor_results['meteor_scores'],
            'rouge_l_mean': rouge_l_results['rouge_l_mean'],
            'rouge_l_scores': rouge_l_results['rouge_l_scores'],
            'cider_d_mean': cider_d_results['cider_d_mean'],
            'cider_d_scores': cider_d_results['cider_d_scores'],
            'spice_mean': spice_results['spice_mean'],
            'spice_scores': spice_results['spice_scores']
        }
    
    def evaluate_single(self, candidate: str, reference: str) -> Dict[str, float]:
        """
        评估单个候选句的质量
        
        Args:
            candidate: 候选句（生成的描述）
            reference: 参考句（真实描述，可以是字符串或字符串列表）
        
        Returns:
            评价结果字典
        """
        if isinstance(reference, str):
            reference = [reference]
        
        return self.evaluate([candidate], [reference])
    
    def evaluate_model(self, model_checkpoint_path: str, 
                      val_feature_file: str, val_sequences_file: str,
                      vocab_file: str, model_type: str = "model1b",
                      batch_size: int = 32, max_len: int = 25,
                      device: Optional[str] = None) -> Dict[str, float]:
        """
        评估模型：加载模型checkpoint，生成描述，并使用评价指标评估
        
        Args:
            model_checkpoint_path: 模型checkpoint文件路径（.pt文件）
            val_feature_file: 验证集特征文件路径（.npz文件）
            val_sequences_file: 验证集文本序列文件路径（.pt文件，包含val_sequences）
            vocab_file: 词典文件路径（.json文件）
            model_type: 模型类型，"model1"、"model1_resnet"、"model1b"、"model2"、"model2_enhanced"、"model2_enhanced_2"或"model5"，默认"model1b"
            batch_size: 批大小，默认32
            max_len: 最大生成长度，默认25
            device: 设备（"cuda"或"cpu"），默认None（自动选择）
        
        Returns:
            评价结果字典，包含所有评价指标的平均得分
        """
        # 设置设备
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        device = torch.device(device)
        
        # 加载词典（确保使用绝对路径）
        vocab_path = Path(vocab_file)
        if not vocab_path.is_absolute():
            # 如果是相对路径，尝试从当前工作目录或项目根目录解析
            if vocab_path.exists():
                vocab_path = vocab_path.resolve()
            else:
                # 尝试从项目根目录查找
                project_root = Path(__file__).parent.parent
                vocab_path = (project_root / vocab_file).resolve()
        
        import json
        with open(str(vocab_path), 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        vocab_size = vocab_data['vocab_size']
        
        # 加载模型
        print(f"\n加载模型: {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=device)
        
        # 根据模型类型加载相应的模型类
        if model_type == "model5":
            from models.model5_full_transformer import FashionCaptionModelTransformer
            model = FashionCaptionModelTransformer(vocab_size=vocab_size)
        elif model_type == "model1_resnet":
            from models.model1_resnet import FashionCaptionModelResNet
            # 从checkpoint中获取模型参数（如果存在，支持新旧版本）
            resnet_type = checkpoint.get('resnet_type', 'resnet50')
            pretrained = checkpoint.get('pretrained', True)
            # 新版本参数（超大容量版 - 目标80%+）
            embed_dim = checkpoint.get('embed_dim', 768)  # 默认768（超大容量版）
            hidden_dim = checkpoint.get('hidden_dim', 1024)  # 默认1024（超大容量版）
            num_layers = checkpoint.get('num_layers', 5)  # 默认5（5层LSTM）
            dropout = checkpoint.get('dropout', 0.4)  # 默认0.4（防止过拟合）
            # 如果checkpoint中没有这些参数，可能是旧版本，使用旧默认值
            if 'embed_dim' not in checkpoint:
                embed_dim = 256  # 旧版本默认值
                hidden_dim = 512  # 旧版本默认值
                num_layers = 6  # 旧版本是6层RNN（但新版本是5层LSTM）
            model = FashionCaptionModelResNet(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                resnet_type=resnet_type,
                pretrained=pretrained,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type == "model2_enhanced_2":
            from models.model2_enhanced_2 import FashionCaptionModelEnhanced
            # 从checkpoint中获取模型参数（如果存在）
            embed_dim = checkpoint.get('embed_dim', 512)
            hidden_dim = checkpoint.get('hidden_dim', 768)
            num_layers = checkpoint.get('num_layers', 3)
            dropout = checkpoint.get('dropout', 0.3)
            model = FashionCaptionModelEnhanced(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type == "model2_enhanced":
            from models.model2_enhanced import FashionCaptionModelEnhanced
            # 从checkpoint中获取模型参数（如果存在）
            embed_dim = checkpoint.get('embed_dim', 512)
            hidden_dim = checkpoint.get('hidden_dim', 768)
            num_layers = checkpoint.get('num_layers', 3)
            dropout = checkpoint.get('dropout', 0.3)
            model = FashionCaptionModelEnhanced(
                vocab_size=vocab_size,
                embed_dim=embed_dim,
                hidden_dim=hidden_dim,
                num_layers=num_layers,
                dropout=dropout
            )
        elif model_type == "model2":
            from models.model2_local_selfattn_attention_rnn import FashionCaptionModelAttention
            model = FashionCaptionModelAttention(vocab_size=vocab_size)
        elif model_type == "model1b":
            from models.model1b_cnn_2layer_lstm import FashionCaptionModelLSTM
            model = FashionCaptionModelLSTM(vocab_size=vocab_size)
        elif model_type == "model1":
            from models.model1_regular_cnn_6layer_rnn import FashionCaptionModel
            model = FashionCaptionModel(vocab_size=vocab_size)
        else:
            raise ValueError(f"未知的模型类型: {model_type}，支持的类型: 'model1', 'model1_resnet', 'model1b', 'model2', 'model2_enhanced', 'model2_enhanced_2', 'model5'")
        
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(device)
        model.eval()
        print(f"[OK] 模型已加载到设备: {device}")
        
        # 加载验证集数据
        print(f"\n加载验证集数据...")
        val_sequences_data = torch.load(val_sequences_file, map_location='cpu')
        val_sequences = val_sequences_data['val_sequences']
        
        from utils.dataset import FashionCaptionDataset, get_dataloader
        val_dataset = FashionCaptionDataset(val_feature_file, val_sequences)
        val_loader = get_dataloader(
            dataset=val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        print(f"[OK] 验证集样本数: {len(val_dataset)}")
        
        # 加载原始数据以获取参考描述
        print(f"\n加载参考描述...")
        from utils.data_loader import load_and_validate_dataset
        
        # 构建captions.json和images目录的路径
        # vocab_path已经在上面转换为绝对路径了
        vocab_path = Path(vocab_file)
        if not vocab_path.is_absolute():
            if vocab_path.exists():
                vocab_path = vocab_path.resolve()
            else:
                project_root = Path(__file__).parent.parent
                vocab_path = (project_root / vocab_file).resolve()
        
        # 如果vocab_file是dataset/vocab.json，则captions.json应该在dataset/captions.json
        if vocab_path.name == "vocab.json" and vocab_path.parent.name == "dataset":
            # vocab_file是dataset/vocab.json
            captions_file = vocab_path.parent / "captions.json"
            images_dir = vocab_path.parent / "images"
        else:
            # 尝试其他路径
            captions_file = vocab_path.parent / "captions.json"
            images_dir = vocab_path.parent / "images"
            
            # 如果不存在，尝试从项目根目录查找
            if not captions_file.exists():
                project_root = Path(__file__).parent.parent
                captions_file = project_root / "dataset" / "captions.json"
                images_dir = project_root / "dataset" / "images"
        
        # 转换为绝对路径
        captions_file = captions_file.resolve()
        images_dir = images_dir.resolve()
        
        if not captions_file.exists():
            raise FileNotFoundError(f"找不到captions.json文件，已尝试路径: {captions_file}")
        
        if not images_dir.exists():
            raise FileNotFoundError(f"找不到images目录，已尝试路径: {images_dir}")
        
        print(f"使用captions.json路径: {captions_file}")
        print(f"使用images目录路径: {images_dir}")
        
        _, val_data, _ = load_and_validate_dataset(
            captions_file=str(captions_file),
            images_dir=str(images_dir),
            random_seed=42
        )
        
        # 检查验证集是否扩展（特征数量 > 原始数据数量）
        num_val_images = len(val_data)
        num_val_samples = len(val_dataset)
        is_expanded = num_val_samples > num_val_images
        
        print(f"\n验证集信息:")
        print(f"  原始图像数量: {num_val_images}")
        print(f"  验证集样本数: {num_val_samples}")
        print(f"  是否扩展: {is_expanded}")
        
        # 准备参考描述
        # 策略：如果验证集是扩展的，使用ExpandedFashionCaptionDataset的逻辑重建映射
        # 或者，为每个样本匹配对应的原始图像的所有参考描述
        reference_captions = []
        
        if is_expanded:
            print(f"  注意: 验证集是扩展后的（每个图像可能有多个样本）")
            print(f"  正在为每个样本匹配对应的原始图像参考描述...")
            
            # 构建原始图像的参考描述库（每个图像的所有描述）
            image_refs = []
            for item in val_data:
                captions = item.get('captions', [])
                if len(captions) > 0:
                    # 保留所有描述作为参考（用于多参考评估，如CIDEr）
                    cleaned_captions = [cap for cap in captions if isinstance(cap, str) and cap.strip()]
                    image_refs.append(cleaned_captions if cleaned_captions else [''])
                else:
                    image_refs.append([''])
            
            # 为每个验证集样本匹配参考描述
            # 方法：使用val_sequences与原始图像的序列进行匹配
            # 如果无法精确匹配，使用特征索引的推断（假设扩展是按顺序的）
            from utils.text_processor import TextProcessor
            processor = TextProcessor(min_freq=3)
            processor.load_vocab(vocab_file)
            
            # 为原始图像生成序列（用于匹配）
            original_val_sequences = []
            for item in val_data:
                captions = item.get('captions', [])
                if len(captions) > 0:
                    # 使用第一个描述生成序列
                    seq = processor.text_to_sequence(captions[0], max_len=30)
                    original_val_sequences.append(seq)
                else:
                    original_val_sequences.append([2, 3] + [0] * 28)  # <START><END><PAD>...
            
            # 为每个验证集样本匹配参考描述
            # 优化：使用简单的索引推断，避免O(n²)的双重循环
            print(f"  使用快速索引推断匹配（避免O(n²)复杂度）...")
            samples_per_image = num_val_samples / num_val_images if num_val_images > 0 else 1
            
            for sample_idx in range(num_val_samples):
                # 使用简单的模运算推断（假设扩展是按顺序的）
                img_idx = int(sample_idx / samples_per_image) if samples_per_image > 0 else sample_idx % num_val_images
                img_idx = min(img_idx, num_val_images - 1)
                
                # 使用匹配到的原始图像的所有参考描述
                reference_captions.append(image_refs[img_idx] if img_idx < len(image_refs) else [''])
                
                # 每100个样本打印一次进度
                if (sample_idx + 1) % 100 == 0:
                    print(f"    已匹配 {(sample_idx + 1)} / {num_val_samples} 个样本 ({100*(sample_idx+1)/num_val_samples:.1f}%)")
            
            print(f"  [OK] 完成匹配 {num_val_samples} 个样本")
        else:
            # 未扩展的验证集：直接对应
            print(f"  验证集未扩展，直接对应参考描述")
            for item in val_data:
                captions = item.get('captions', [])
                if len(captions) > 0:
                    # 保留所有描述作为参考（用于多参考评估）
                    cleaned_captions = [cap for cap in captions if isinstance(cap, str) and cap.strip()]
                    reference_captions.append(cleaned_captions if cleaned_captions else [''])
                else:
                    reference_captions.append([''])
            
            # 确保数量匹配
            if len(reference_captions) != len(val_dataset):
                print(f"警告: 参考描述数量({len(reference_captions)})与验证集样本数({len(val_dataset)})不一致")
                if len(reference_captions) > len(val_dataset):
                    reference_captions = reference_captions[:len(val_dataset)]
                else:
                    reference_captions.extend([['']] * (len(val_dataset) - len(reference_captions)))
        
        # 生成描述并收集结果
        print(f"\n开始生成描述...")
        print(f"  总样本数: {len(val_dataset)}, 批次大小: {batch_size}, 预计批次: {(len(val_dataset) + batch_size - 1) // batch_size}")
        generated_captions = []
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(val_loader):
                # 打印每个batch的开始（前5个batch和每10个batch）
                if batch_idx < 5 or (batch_idx + 1) % 10 == 0:
                    print(f"  处理批次 {batch_idx + 1} / {(len(val_dataset) + batch_size - 1) // batch_size}...")
                
                local_feat = batch['local_feat'].to(device)
                
                # 生成描述（根据模型类型调用不同的generate方法）
                if model_type == "model5":
                    generated_sequences = model.generate(local_feat, max_len=max_len, temperature=0.7)
                elif model_type == "model1_resnet":
                    # 优化：使用贪心解码 + 最小长度限制和<END>惩罚，防止过早截断
                    generated_sequences, _ = model.generate(
                        local_feat, 
                        max_len=max_len, 
                        beam_size=1,  # 使用贪心解码（避免卡住，速度更快）
                        length_penalty=0.6,
                        min_length=30,  # 最小长度30词（参考长度40.5词，30词确保不过早截断）
                        end_penalty=3.0  # <END>惩罚因子（在序列长度<min_length时对<END>概率进行惩罚）
                    )
                elif model_type in ["model2_enhanced", "model2_enhanced_2"]:
                    # model2_enhanced和model2_enhanced_2默认使用beam search（beam_size=5），可以显著提升性能
                    # 如果beam_size=1，则使用贪心解码
                    generated_sequences, _ = model.generate(
                        local_feat, 
                        max_len=max_len, 
                        beam_size=5,  # 使用beam search（推荐5）
                        length_penalty=0.6,  # 长度惩罚（0.6对长序列更有利）
                        return_attn=False
                    )
                elif model_type == "model2":
                    generated_sequences, _ = model.generate(local_feat, max_len=max_len, temperature=0.7, return_attn=False)
                elif model_type == "model1b":
                    generated_sequences = model.generate(local_feat, max_len=max_len, temperature=0.7)
                else:
                    generated_sequences = model.generate(local_feat, max_len=max_len)
                
                # 转换为文本
                batch_size_current = local_feat.size(0)
                for i in range(batch_size_current):
                    gen_seq = generated_sequences[i].cpu()
                    
                    # 使用模型的postprocess_caption方法（如果可用）
                    if hasattr(model, 'postprocess_caption'):
                        gen_words = model.postprocess_caption(gen_seq, idx2word)
                    else:
                        # 手动处理
                        gen_words = []
                        for idx in gen_seq.cpu().tolist():
                            if idx == 3:  # <END>
                                break
                            if idx not in [0, 1, 2, 3]:  # 不是<PAD>, <UNK>, <START>, <END>
                                word = idx2word.get(idx, f"<{idx}>")
                                gen_words.append(word)
                    
                    # 确保gen_words是列表
                    if not isinstance(gen_words, list):
                        gen_words = []
                    
                    generated_captions.append(' '.join(gen_words) if gen_words else '')
                
                # 打印进度（每5个batch或每10个batch）
                if (batch_idx + 1) % 5 == 0:
                    processed = min((batch_idx + 1) * batch_size, len(val_dataset))
                    print(f"  [进度] 已处理 {processed} / {len(val_dataset)} 个样本 ({100*processed/len(val_dataset):.1f}%)")
        
        print(f"[OK] 生成了 {len(generated_captions)} 个描述")
        
        # 确保数量匹配
        min_len = min(len(generated_captions), len(reference_captions))
        generated_captions = generated_captions[:min_len]
        reference_captions = reference_captions[:min_len]
        
        # 输出一些生成样本（用于诊断问题）
        print(f"\n{'='*60}")
        print(f"生成的样本示例（前10个）:")
        print(f"{'='*60}")
        num_samples_to_show = min(10, len(generated_captions))
        for i in range(num_samples_to_show):
            gen_text = generated_captions[i] if i < len(generated_captions) else ""
            ref_texts = reference_captions[i] if i < len(reference_captions) else [""]
            ref_text = ref_texts[0] if isinstance(ref_texts, list) and ref_texts else (ref_texts if isinstance(ref_texts, str) else "")
            
            print(f"\n样本 {i+1}:")
            print(f"  生成: {gen_text if gen_text else '(空或无效)'}")
            print(f"  参考: {ref_text if ref_text else '(无参考)'}")
            print(f"  生成长度: {len(gen_text.split()) if gen_text else 0} 词")
            print(f"  参考长度: {len(ref_text.split()) if ref_text else 0} 词")
        
        print(f"\n{'='*60}")
        print(f"样本统计:")
        print(f"{'='*60}")
        # 统计生成结果的质量
        empty_count = sum(1 for cap in generated_captions if not cap or cap.strip() == "")
        avg_gen_len = sum(len(cap.split()) for cap in generated_captions) / len(generated_captions) if generated_captions else 0
        avg_ref_len = sum(len(ref[0].split() if isinstance(ref, list) and ref else (ref.split() if isinstance(ref, str) else [])) for ref in reference_captions) / len(reference_captions) if reference_captions else 0
        
        print(f"  空生成数量: {empty_count} / {len(generated_captions)} ({100*empty_count/len(generated_captions):.1f}%)")
        print(f"  平均生成长度: {avg_gen_len:.1f} 词")
        print(f"  平均参考长度: {avg_ref_len:.1f} 词")
        
        # 使用评价器评估
        print(f"\n{'='*60}")
        print(f"开始评估生成结果...")
        print(f"  样本数: {len(generated_captions)}")
        print(f"  正在计算METEOR、ROUGE-L、CIDEr-D、SPICE指标（可能需要一些时间）...")
        print(f"{'='*60}")
        results = self.evaluate(generated_captions, reference_captions)
        print(f"[OK] 评估完成")
        
        return results

