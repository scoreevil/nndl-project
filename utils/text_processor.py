"""
DeepFashion-MultiModal数据集英文文本预处理模块
"""
import re
from collections import Counter
from typing import List, Dict, Tuple, Optional
import torch
import json
from pathlib import Path


class TextProcessor:
    """英文文本预处理类"""
    
    # 特殊标记
    PAD_TOKEN = "<PAD>"
    UNK_TOKEN = "<UNK>"
    START_TOKEN = "<START>"
    END_TOKEN = "<END>"
    
    # 特殊标记索引
    PAD_IDX = 0
    UNK_IDX = 1
    START_IDX = 2
    END_IDX = 3
    
    def __init__(self, min_freq: int = 3):
        """
        初始化文本处理器
        
        Args:
            min_freq: 词的最小出现频率，低于此频率的词将被过滤
        """
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.vocab_size = 0
    
    def clean_text(self, text: str) -> Optional[str]:
        """
        文本清洗
        
        Args:
            text: 原始文本
        
        Returns:
            清洗后的文本，如果无效则返回None
        """
        # 处理None或非字符串类型
        if text is None:
            return None
        
        if not isinstance(text, str):
            text = str(text)
        
        # 去除首尾空格
        text = text.strip()
        
        # 过滤空字符串或纯空格
        if not text or text.isspace():
            return None
        
        # 检查是否为纯数字
        if text.replace('.', '').replace('-', '').isdigit():
            return None
        
        # 转为小写
        text = text.lower()
        
        # 去除标点符号（保留空格和字母数字）
        # 使用正则表达式：保留字母、数字和空格，其他都替换为空格
        text = re.sub(r'[^a-zA-Z0-9\s]', ' ', text)
        
        # 将多个空格替换为单个空格
        text = re.sub(r'\s+', ' ', text)
        
        # 再次去除首尾空格
        text = text.strip()
        
        # 如果清洗后为空，返回None
        if not text:
            return None
        
        return text
    
    def tokenize(self, text: str) -> List[str]:
        """
        英文分词（按空格分割）
        
        Args:
            text: 清洗后的文本
        
        Returns:
            单词列表
        """
        # 按空格分词
        words = text.split()
        # 过滤空字符串
        words = [w for w in words if w.strip()]
        return words
        
    def build_vocab(self, train_captions: List[str]) -> Tuple[Dict[str, int], Dict[int, str]]:
        """
        构建词典（仅使用训练集的描述）
        
        Args:
            train_captions: 训练集的所有描述列表
        
        Returns:
            word2idx: 正向词典（word→idx）
            idx2word: 反向词典（idx→word）
        """
        print("开始构建词典...")
        
        # 统计词频
        word_counter = Counter()
        filtered_count = 0
        
        for caption in train_captions:
            # 文本清洗
            cleaned_text = self.clean_text(caption)
            
            # 如果清洗后无效，跳过并计数
            if cleaned_text is None:
                filtered_count += 1
                continue
            
            # 分词
            words = self.tokenize(cleaned_text)
            
            # 如果分词后为空，跳过
            if not words:
                filtered_count += 1
                continue
            
            # 统计词频
            word_counter.update(words)
        
        if filtered_count > 0:
            print(f"警告: 过滤了 {filtered_count} 条无效描述（None、纯数字或空字符串）")
        
        print(f"训练集总词数: {sum(word_counter.values())}")
        print(f"训练集唯一词数: {len(word_counter)}")
        
        # 初始化特殊标记
        self.word2idx = {
            self.PAD_TOKEN: self.PAD_IDX,
            self.UNK_TOKEN: self.UNK_IDX,
            self.START_TOKEN: self.START_IDX,
            self.END_TOKEN: self.END_IDX
        }
        
        # 添加满足最小频率的词
        idx = len(self.word2idx)  # 从4开始（0-3是特殊标记）
        for word, count in word_counter.items():
            if count >= self.min_freq:
                self.word2idx[word] = idx
                idx += 1
        
        # 构建反向词典
        self.idx2word = {idx: word for word, idx in self.word2idx.items()}
        self.vocab_size = len(self.word2idx)
        
        print(f"过滤后词典大小: {self.vocab_size}个单词（包含4个特殊标记）")
        print(f"  特殊标记: {self.PAD_TOKEN}({self.PAD_IDX}), {self.UNK_TOKEN}({self.UNK_IDX}), "
              f"{self.START_TOKEN}({self.START_IDX}), {self.END_TOKEN}({self.END_IDX})")
        
        return self.word2idx, self.idx2word
    
    def save_vocab(self, vocab_file: str):
        """
        保存词典到JSON文件
        
        Args:
            vocab_file: 保存路径
        """
        vocab_data = {
            'word2idx': self.word2idx,
            'idx2word': {str(k): v for k, v in self.idx2word.items()},  # JSON key必须是字符串
            'vocab_size': self.vocab_size,
            'min_freq': self.min_freq,
            'special_tokens': {
                'PAD': {'token': self.PAD_TOKEN, 'idx': self.PAD_IDX},
                'UNK': {'token': self.UNK_TOKEN, 'idx': self.UNK_IDX},
                'START': {'token': self.START_TOKEN, 'idx': self.START_IDX},
                'END': {'token': self.END_TOKEN, 'idx': self.END_IDX}
            }
        }
        
        vocab_path = Path(vocab_file)
        vocab_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(vocab_path, 'w', encoding='utf-8') as f:
            json.dump(vocab_data, f, ensure_ascii=False, indent=2)
        
        print(f"词典已保存到: {vocab_file}")
    
    def load_vocab(self, vocab_file: str):
        """
        从JSON文件加载词典
        
        Args:
            vocab_file: 词典文件路径
        """
        with open(vocab_file, 'r', encoding='utf-8') as f:
            vocab_data = json.load(f)
        
        self.word2idx = vocab_data['word2idx']
        # 将字符串key转换回整数
        self.idx2word = {int(k): v for k, v in vocab_data['idx2word'].items()}
        self.vocab_size = vocab_data['vocab_size']
        self.min_freq = vocab_data.get('min_freq', 3)
        
        print(f"词典已从 {vocab_file} 加载")
        print(f"词典大小: {self.vocab_size}个词")
    
    def text_to_sequence(
        self,
        text: str,
        max_len: int = 20
    ) -> List[int]:
        """
        将文本转换为序列
        
        Args:
            text: 输入的英文描述
            max_len: 最大序列长度
        
        Returns:
            序列列表，长度为max_len
        """
        # 文本清洗
        cleaned_text = self.clean_text(text)
        
        # 如果清洗后无效，返回全PAD序列（除了START和END）
        if cleaned_text is None:
            sequence = [self.START_IDX, self.END_IDX] + [self.PAD_IDX] * (max_len - 2)
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
            return sequence
        
        # 分词
        words = self.tokenize(cleaned_text)
        
        # 如果分词后为空，返回全PAD序列
        if not words:
            sequence = [self.START_IDX, self.END_IDX] + [self.PAD_IDX] * (max_len - 2)
            if len(sequence) > max_len:
                sequence = sequence[:max_len]
            return sequence
        
        # 构建序列：<START> + 词序列 + <END>
        sequence = [self.START_IDX]
        
        # 添加词索引，未知词用<UNK>
        for word in words:
            idx = self.word2idx.get(word, self.UNK_IDX)
            sequence.append(idx)
        
        # 添加<END>
        sequence.append(self.END_IDX)
        
        # 用<PAD>补全到max_len
        if len(sequence) < max_len:
            sequence.extend([self.PAD_IDX] * (max_len - len(sequence)))
        elif len(sequence) > max_len:
            # 如果超过max_len，截断（保留<START>和前面的词，最后添加<END>）
            sequence = sequence[:max_len-1] + [self.END_IDX]
        
        return sequence
    
    def batch_process(
        self,
        data_list: List[Dict],
        max_len: int = 20
    ) -> torch.Tensor:
        """
        批量处理数据集
        
        Args:
            data_list: 数据列表，每个元素包含'captions'字段
            max_len: 最大序列长度
        
        Returns:
            文本序列张量，形状为(N, max_len)，N为样本数
        """
        sequences = []
        warning_count = 0
        
        for idx, item in enumerate(data_list):
            # 取第一个描述（中期简化）
            captions = item.get('captions', [])
            if len(captions) == 0:
                # 如果没有描述，创建空序列
                sequence = [self.START_IDX, self.END_IDX] + [self.PAD_IDX] * (max_len - 2)
                if warning_count < 5:  # 只打印前5个警告
                    print(f"警告: 样本{idx}的captions为空列表")
                    warning_count += 1
            else:
                # 取第一个描述
                caption = captions[0] if isinstance(captions[0], str) else str(captions[0])
                
                # 检查是否为None或无效
                cleaned = self.clean_text(caption)
                if cleaned is None:
                    if warning_count < 5:
                        print(f"警告: 样本{idx}的描述无效（None、纯数字或空字符串）: {caption}")
                        warning_count += 1
                
                sequence = self.text_to_sequence(caption, max_len)
            
            sequences.append(sequence)
        
        if warning_count >= 5:
            print(f"警告: 还有更多无效描述未显示...")
        
        # 转换为torch.Tensor
        tensor = torch.tensor(sequences, dtype=torch.long)
        
        return tensor
    
    def print_example(
        self,
        text: str,
        max_len: int = 20
    ):
        """
        打印序列化示例
        
        Args:
            text: 原始描述
            max_len: 最大序列长度
        """
        # 文本清洗
        cleaned_text = self.clean_text(text)
        
        if cleaned_text is None:
            print(f"原始描述: {text}")
            print(f"警告: 描述无效（None、纯数字或空字符串）")
            sequence = self.text_to_sequence(text, max_len)
            print(f"序列: {sequence}")
            return
        
        # 分词
        words = self.tokenize(cleaned_text)
        sequence = self.text_to_sequence(text, max_len)
        
        print(f"原始描述: {text}")
        print(f"清洗后: {cleaned_text}")
        print(f"分词后: {words}")
        print(f"序列: {sequence}")
        print(f"序列长度: {len(sequence)}")


def build_vocab_and_process(
    train_data: List[Dict],
    val_data: List[Dict],
    test_data: List[Dict],
    max_len: int = 20,
    min_freq: int = 3,
    vocab_file: Optional[str] = "dataset/vocab.json"
) -> Tuple[TextProcessor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    构建词典并处理所有数据集
    
    Args:
        train_data: 训练集数据列表
        val_data: 验证集数据列表
        test_data: 测试集数据列表
        max_len: 最大序列长度
        min_freq: 词的最小出现频率
    
    Returns:
        processor: 文本处理器对象
        train_sequences: 训练集文本序列张量 (N_train, max_len)
        val_sequences: 验证集文本序列张量 (N_val, max_len)
        test_sequences: 测试集文本序列张量 (N_test, max_len)
    """
    # 创建文本处理器
    processor = TextProcessor(min_freq=min_freq)
    
    # 提取训练集的所有描述（用于构建词典）
    print("\n" + "="*50)
    print("步骤1: 提取训练集描述")
    print("="*50)
    train_captions = []
    for item in train_data:
        captions = item.get('captions', [])
        if len(captions) > 0:
            # 取第一个描述
            caption = captions[0] if isinstance(captions[0], str) else str(captions[0])
            train_captions.append(caption)
    
    print(f"训练集描述数量: {len(train_captions)}")
    
    # 构建词典
    print("\n" + "="*50)
    print("步骤2: 构建词典")
    print("="*50)
    processor.build_vocab(train_captions)
    
    # 保存词典到JSON文件
    if vocab_file:
        processor.save_vocab(vocab_file)
    
    # 打印一个序列化示例
    print("\n" + "="*50)
    print("步骤3: 序列化示例")
    print("="*50)
    if len(train_captions) > 0:
        processor.print_example(train_captions[0], max_len=max_len)
    
    # 批量处理各数据集
    print("\n" + "="*50)
    print("步骤4: 批量处理数据集")
    print("="*50)
    
    print("处理训练集...")
    train_sequences = processor.batch_process(train_data, max_len=max_len)
    print(f"训练集序列形状: {train_sequences.shape}")
    
    print("处理验证集...")
    val_sequences = processor.batch_process(val_data, max_len=max_len)
    print(f"验证集序列形状: {val_sequences.shape}")
    
    print("处理测试集...")
    test_sequences = processor.batch_process(test_data, max_len=max_len)
    print(f"测试集序列形状: {test_sequences.shape}")
    
    return processor, train_sequences, val_sequences, test_sequences


if __name__ == "__main__":
    # 测试代码
    from data_loader import load_and_validate_dataset
    
    # 加载数据集
    print("加载数据集...")
    train_data, val_data, test_data = load_and_validate_dataset(
        captions_file="dataset/captions.json",
        images_dir="dataset/images",
        random_seed=42
    )
    
    # 构建词典并处理文本
    processor, train_seq, val_seq, test_seq = build_vocab_and_process(
        train_data=train_data,
        val_data=val_data,
        test_data=test_data,
        max_len=20,
        min_freq=3
    )
    
    print("\n" + "="*50)
    print("文本预处理完成！")
    print("="*50)