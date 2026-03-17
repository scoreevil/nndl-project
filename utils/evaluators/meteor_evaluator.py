"""
METEOR评价指标实现
METEOR (Metric for Evaluation of Translation with Explicit ORdering)
"""
import re
from typing import List, Dict
from collections import defaultdict
import numpy as np


class METEOREvaluator:
    """
    METEOR (Metric for Evaluation of Translation with Explicit ORdering) 评价指标
    
    METEOR旨在解决传统BLEU指标中缺乏召回率考量和同义词匹配的问题。
    它基于词汇对齐，并引入惩罚项来考虑语序。
    
    核心步骤：
    1. 词汇对齐（完全匹配→词干匹配→同义词匹配）
    2. 计算精确率(P)和召回率(R)
    3. 计算F-mean（调和平均值，更侧重召回率）
    4. 计算惩罚项（Penalty，用于惩罚语序不佳的句子）
    5. 最终得分：METEOR = F_mean * (1 - Penalty)
    """
    
    def __init__(self, alpha: float = 0.9, gamma: float = 0.5, theta: float = 3.0):
        """
        初始化METEOR评价器
        
        Args:
            alpha: F-mean计算中的超参数，控制精确率和召回率的权重，默认0.9（更侧重召回率）
            gamma: 惩罚项计算中的超参数，默认0.5
            theta: 惩罚项计算中的超参数，默认3.0
        """
        self.alpha = alpha
        self.gamma = gamma
        self.theta = theta
        
        # 尝试导入nltk用于词干提取和同义词匹配
        self._init_nltk()
    
    def _init_nltk(self):
        """初始化NLTK相关工具（如果可用）"""
        self.use_nltk = False
        self.stemmer = None
        self.wordnet = None
        
        try:
            import nltk
            from nltk.stem import PorterStemmer
            from nltk.corpus import wordnet
            
            # 尝试下载必要的NLTK数据
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    pass
            
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                try:
                    nltk.download('wordnet', quiet=True)
                except:
                    pass
            
            self.stemmer = PorterStemmer()
            self.wordnet = wordnet
            self.use_nltk = True
        except ImportError:
            # 如果没有安装nltk，使用简单的词干提取
            self.use_nltk = False
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词：将文本转换为单词列表（小写）
        
        Args:
            text: 输入文本
        
        Returns:
            单词列表
        """
        # 转换为小写，去除标点符号，按空格分词
        text = text.lower()
        # 保留字母和数字，去除标点
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _get_stem(self, word: str) -> str:
        """
        获取词干（stem）
        
        Args:
            word: 单词
        
        Returns:
            词干
        """
        if self.use_nltk and self.stemmer:
            return self.stemmer.stem(word)
        else:
            # 简单的词干提取：去除常见后缀
            word = word.lower()
            # 简单的后缀去除规则
            if word.endswith('ing'):
                return word[:-3]
            elif word.endswith('ed'):
                return word[:-2]
            elif word.endswith('s'):
                return word[:-1]
            elif word.endswith('es'):
                return word[:-2]
            elif word.endswith('ly'):
                return word[:-2]
            else:
                return word
    
    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """
        判断两个词是否为同义词
        
        Args:
            word1: 单词1
            word2: 单词2
        
        Returns:
            是否为同义词
        """
        if not self.use_nltk or not self.wordnet:
            return False
        
        try:
            synsets1 = self.wordnet.synsets(word1)
            synsets2 = self.wordnet.synsets(word2)
            
            # 检查是否有共同的同义词集
            for syn1 in synsets1:
                for syn2 in synsets2:
                    if syn1 == syn2:
                        return True
            
            # 检查是否有相似的同义词集
            for syn1 in synsets1:
                for syn2 in synsets2:
                    if syn1.wup_similarity(syn2) and syn1.wup_similarity(syn2) > 0.8:
                        return True
            
            return False
        except:
            return False
    
    def _align_words(self, candidate: List[str], reference: List[str]) -> Dict[int, int]:
        """
        词汇对齐：将候选句和参考句中的词进行对齐
        
        对齐的优先级为：完全匹配 → 词干匹配 → 同义词匹配
        
        Args:
            candidate: 候选句单词列表
            reference: 参考句单词列表
        
        Returns:
            对齐字典：{候选句索引: 参考句索引}
        """
        alignment = {}
        used_ref_indices = set()
        
        # 步骤1: 完全匹配
        for i, c_word in enumerate(candidate):
            if i in alignment:
                continue
            for j, r_word in enumerate(reference):
                if j in used_ref_indices:
                    continue
                if c_word == r_word:
                    alignment[i] = j
                    used_ref_indices.add(j)
                    break
        
        # 步骤2: 词干匹配
        for i, c_word in enumerate(candidate):
            if i in alignment:
                continue
            c_stem = self._get_stem(c_word)
            for j, r_word in enumerate(reference):
                if j in used_ref_indices:
                    continue
                r_stem = self._get_stem(r_word)
                if c_stem == r_stem:
                    alignment[i] = j
                    used_ref_indices.add(j)
                    break
        
        # 步骤3: 同义词匹配
        for i, c_word in enumerate(candidate):
            if i in alignment:
                continue
            for j, r_word in enumerate(reference):
                if j in used_ref_indices:
                    continue
                if self._are_synonyms(c_word, r_word):
                    alignment[i] = j
                    used_ref_indices.add(j)
                    break
        
        return alignment
    
    def _count_chunks(self, candidate: List[str], alignment: Dict[int, int]) -> int:
        """
        计算匹配的词块（chunks）数量
        
        词块：候选句中相邻且都匹配的连续词序列
        
        Args:
            candidate: 候选句单词列表
            alignment: 对齐字典
        
        Returns:
            词块数量
        """
        if len(alignment) == 0:
            return 0
        
        # 获取所有对齐的候选句索引（排序）
        aligned_indices = sorted(alignment.keys())
        
        if len(aligned_indices) == 0:
            return 0
        
        # 计算连续词块
        chunks = 1
        for i in range(1, len(aligned_indices)):
            # 如果当前索引和上一个索引不连续，则开始新的词块
            if aligned_indices[i] != aligned_indices[i-1] + 1:
                chunks += 1
        
        return chunks
    
    def compute_meteor(self, candidate: str, reference: str) -> float:
        """
        计算单个候选句相对于单个参考句的METEOR得分
        
        Args:
            candidate: 候选句（生成的描述）
            reference: 参考句（真实描述）
        
        Returns:
            METEOR得分（0-1之间，越高越好）
        """
        # 分词
        candidate_words = self._tokenize(candidate)
        reference_words = self._tokenize(reference)
        
        # 如果候选句或参考句为空，返回0
        if len(candidate_words) == 0 or len(reference_words) == 0:
            return 0.0
        
        # 步骤1: 词汇对齐
        alignment = self._align_words(candidate_words, reference_words)
        matched_unigrams = len(alignment)
        
        # 如果没有任何匹配，返回0
        if matched_unigrams == 0:
            return 0.0
        
        # 步骤2: 计算精确率(P)和召回率(R)
        # P = 在C中匹配的unigram数量 / C中的unigram总数
        precision = matched_unigrams / len(candidate_words)
        
        # R = 在C中匹配的unigram数量 / s_i中的unigram总数
        recall = matched_unigrams / len(reference_words)
        
        # 步骤3: 计算F-mean（调和平均值，更侧重召回率）
        # F_mean = (P * R) / (α * P + (1-α) * R)
        # 其中α=0.9，更侧重召回率
        if precision == 0 and recall == 0:
            f_mean = 0.0
        else:
            f_mean = (precision * recall) / (self.alpha * precision + (1 - self.alpha) * recall)
        
        # 步骤4: 计算惩罚项（Penalty）
        # 用于惩罚语序不佳的句子
        # 匹配的词块(chunks)越少，语序越好，惩罚越小
        chunks = self._count_chunks(candidate_words, alignment)
        
        # Penalty = γ * (chunks / matched_unigrams)^θ
        # 其中γ=0.5，θ=3.0
        if matched_unigrams > 0:
            penalty = self.gamma * ((chunks / matched_unigrams) ** self.theta)
        else:
            penalty = 1.0
        
        # 步骤5: 最终得分
        # METEOR = F_mean * (1 - Penalty)
        meteor_score = f_mean * (1 - penalty)
        
        return max(0.0, min(1.0, meteor_score))  # 确保得分在[0, 1]范围内
    
    def compute_meteor_multiple_references(self, candidate: str, references: List[str]) -> float:
        """
        计算单个候选句相对于多个参考句的METEOR得分
        
        最终得分会选择所有参考句计算出的最高分
        
        Args:
            candidate: 候选句（生成的描述）
            references: 参考句列表（多个真实描述）
        
        Returns:
            METEOR得分（0-1之间，越高越好）
        """
        scores = []
        for ref in references:
            score = self.compute_meteor(candidate, ref)
            scores.append(score)
        
        # 返回最高分
        return max(scores) if scores else 0.0
    
    def evaluate_batch(self, candidates: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        批量评估：计算一批候选句的METEOR得分
        
        Args:
            candidates: 候选句列表（生成的描述列表）
            references: 参考句列表（每个元素是一个参考句列表，可以有多个参考描述）
        
        Returns:
            评价结果字典，包含：
            - 'meteor_mean': 平均METEOR得分
            - 'meteor_scores': 每个样本的METEOR得分列表
        """
        if len(candidates) != len(references):
            raise ValueError(f"候选句数量({len(candidates)})与参考句数量({len(references)})不匹配")
        
        scores = []
        total = len(candidates)
        for idx, (candidate, ref_list) in enumerate(zip(candidates, references)):
            if isinstance(ref_list, str):
                # 如果参考句是单个字符串，转换为列表
                ref_list = [ref_list]
            
            score = self.compute_meteor_multiple_references(candidate, ref_list)
            scores.append(score)
            
            # 每100个样本或每10%打印一次进度
            if (idx + 1) % 100 == 0 or (idx + 1) == total or (idx + 1) % max(1, total // 10) == 0:
                print(f"    METEOR进度: {idx + 1} / {total} ({100*(idx+1)/total:.1f}%)")
        
        return {
            'meteor_mean': np.mean(scores),
            'meteor_scores': scores
        }

