"""
CIDEr-D评价指标实现
CIDEr-D (Consensus-based Image Description Evaluation)
"""
import re
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import numpy as np


class CIDErDEvaluator:
    """
    CIDEr-D (Consensus-based Image Description Evaluation) 评价指标
    
    CIDEr认为一个好的描述应该与大多数描述相似（即与参考句集合中的描述相似）。
    它通过计算TF-IDF向量的余弦相似度来衡量这种"共识"。
    
    核心步骤：
    1. 构建词向量：每个句子（候选句和参考句）表示为加权的TF-IDF向量
       - 对于句子中的每个n-gram（通常n=1到4），其权重：g_k(s_{ij}) = TF(w_k, s_{ij}) * IDF(w_k)
       - TF(w_k, s_{ij})是n-gram w_k在句子s_{ij}中出现的频率
       - IDF(w_k)衡量n-gram w_k在整个数据集中的稀有程度，越稀有权重越高
    
    2. 计算余弦相似度：对于每个n-gram长度，计算候选句C_i与所有参考句s_{ij}的平均余弦相似度
       - CIDEr_n(C_i, S_i) = (1/m) * sum_{j=1}^{m} [ (g_n(C_i) * g_n(s_{ij})) / (||g_n(C_i)|| * ||g_n(s_{ij})||) ]
       - m是参考句的数量
       - g_n(C_i)和g_n(s_{ij})分别表示候选句和特定参考句的n-gram向量
       - *表示点积，||.||表示向量的L2范数（模长）
    
    3. 最终得分：将不同n-gram长度的得分进行加权平均得到最终的CIDEr得分
       - CIDER(C_i, S_i) = sum_{n=1}^{N} w_n * CIDEr_n(C_i, S_i)
       - N通常为4（意味着考虑1到4的n-gram）
       - w_n通常为1/4，表示每个n-gram长度等权重
    
    4. CIDEr-D中的"D"表示引入了句子长度惩罚，用于防止模型生成过短的句子以获得高分
    """
    
    def __init__(self, max_n: int = 4, weights: Optional[List[float]] = None):
        """
        初始化CIDEr-D评价器
        
        Args:
            max_n: 最大n-gram长度，默认4（考虑1到4的n-gram）
            weights: n-gram长度的权重列表，默认None（等权重1/4）
        """
        self.max_n = max_n
        if weights is None:
            self.weights = [1.0 / max_n] * max_n  # 等权重
        else:
            if len(weights) != max_n:
                raise ValueError(f"权重数量({len(weights)})必须等于max_n({max_n})")
            self.weights = weights
    
    def _tokenize(self, text: str) -> List[str]:
        """
        分词：将文本转换为单词列表（小写）
        
        Args:
            text: 输入文本
        
        Returns:
            单词列表
        """
        text = text.lower()
        words = re.findall(r'\b\w+\b', text)
        return words
    
    def _get_ngrams(self, words: List[str], n: int) -> List[Tuple[str, ...]]:
        """
        获取n-gram列表
        
        Args:
            words: 单词列表
            n: n-gram长度
        
        Returns:
            n-gram元组列表
        """
        if len(words) < n:
            return []
        return [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    
    def _compute_idf(self, all_references: List[List[str]], max_n: int) -> Dict[Tuple[str, ...], float]:
        """
        计算所有参考句集合中n-gram的IDF值
        
        Args:
            all_references: 所有参考句列表（每个元素是一个参考句的单词列表）
            max_n: 最大n-gram长度
        
        Returns:
            n-gram到IDF值的字典
        """
        # 统计每个n-gram出现在多少个不同的句子中
        ngram_doc_count = defaultdict(int)
        total_docs = len(all_references)
        
        # 如果参考句为空，返回空字典
        if total_docs == 0:
            return {}
        
        for n in range(1, max_n + 1):
            for ref_words in all_references:
                ngrams = set(self._get_ngrams(ref_words, n))
                for ngram in ngrams:
                    ngram_doc_count[ngram] += 1
        
        # 计算IDF: IDF(w_k) = log((total_docs + 1) / (doc_count(w_k) + 1))
        # 使用+1平滑避免log(0)的问题
        # 注意：当只有一个参考句时，log(2/2)=0会导致IDF全为0
        # 为了在这种情况下也能工作，我们使用特殊处理
        idf_dict = {}
        for ngram, doc_count in ngram_doc_count.items():
            if total_docs == 1:
                # 单参考句情况：使用固定的IDF值避免全为0
                # 使用log(2)作为基础IDF值，这样TF-IDF不会全为0
                idf_dict[ngram] = np.log(2.0)
            else:
                # 多参考句情况：使用标准的平滑IDF计算
                idf_dict[ngram] = np.log((total_docs + 1) / (doc_count + 1))
        
        return idf_dict
    
    def _compute_tf_idf_vector(self, words: List[str], n: int, idf_dict: Dict[Tuple[str, ...], float]) -> Dict[Tuple[str, ...], float]:
        """
        计算句子的TF-IDF向量（对于特定n-gram长度）
        
        Args:
            words: 句子的单词列表
            n: n-gram长度
            idf_dict: n-gram到IDF值的字典
        
        Returns:
            n-gram到TF-IDF权重的字典
        """
        ngrams = self._get_ngrams(words, n)
        if len(ngrams) == 0:
            return {}
        
        # 计算TF（词频）
        tf_dict = defaultdict(int)
        for ngram in ngrams:
            tf_dict[ngram] += 1
        
        # 计算TF-IDF: g_k(s_{ij}) = TF(w_k, s_{ij}) * IDF(w_k)
        tf_idf_dict = {}
        for ngram, tf in tf_dict.items():
            idf = idf_dict.get(ngram, 0.0)
            tf_idf_dict[ngram] = tf * idf
        
        return tf_idf_dict
    
    def _compute_cosine_similarity(self, vec1: Dict[Tuple[str, ...], float], 
                                   vec2: Dict[Tuple[str, ...], float]) -> float:
        """
        计算两个TF-IDF向量的余弦相似度
        
        Args:
            vec1: 向量1（n-gram到TF-IDF权重的字典）
            vec2: 向量2（n-gram到TF-IDF权重的字典）
        
        Returns:
            余弦相似度（0-1之间）
        """
        # 获取所有n-gram的并集
        all_ngrams = set(vec1.keys()) | set(vec2.keys())
        
        if len(all_ngrams) == 0:
            return 0.0
        
        # 计算点积
        dot_product = sum(vec1.get(ngram, 0.0) * vec2.get(ngram, 0.0) for ngram in all_ngrams)
        
        # 计算L2范数
        norm1 = np.sqrt(sum(v ** 2 for v in vec1.values()))
        norm2 = np.sqrt(sum(v ** 2 for v in vec2.values()))
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        # 余弦相似度
        cosine_sim = dot_product / (norm1 * norm2)
        
        return max(0.0, min(1.0, cosine_sim))  # 确保在[0, 1]范围内
    
    def _compute_length_penalty(self, candidate_length: int, reference_lengths: List[int]) -> float:
        """
        计算长度惩罚（CIDEr-D中的"D"）
        
        用于防止模型生成过短的句子以获得高分
        
        Args:
            candidate_length: 候选句长度
            reference_lengths: 参考句长度列表
        
        Returns:
            长度惩罚系数（0-1之间）
        """
        if len(reference_lengths) == 0:
            return 1.0
        
        avg_ref_length = np.mean(reference_lengths)
        
        if avg_ref_length == 0:
            return 1.0
        
        # 长度惩罚：如果候选句长度小于平均参考句长度，应用惩罚
        if candidate_length < avg_ref_length:
            penalty = np.exp(-(avg_ref_length - candidate_length) / avg_ref_length)
        else:
            penalty = 1.0
        
        return penalty
    
    def compute_cider_d(self, candidate: str, references: List[str]) -> float:
        """
        计算单个候选句相对于多个参考句的CIDEr-D得分
        
        Args:
            candidate: 候选句（生成的描述）
            references: 参考句列表（多个真实描述）
        
        Returns:
            CIDEr-D得分（0-1之间，越高越好）
        """
        if len(references) == 0:
            return 0.0
        
        # 分词
        candidate_words = self._tokenize(candidate)
        reference_words_list = [self._tokenize(ref) for ref in references]
        
        # 如果候选句为空，返回0
        if len(candidate_words) == 0:
            return 0.0
        
        # 计算IDF（基于所有参考句）
        idf_dict = self._compute_idf(reference_words_list, self.max_n)
        
        # 计算每个n-gram长度的CIDEr得分
        cider_scores = []
        
        for n in range(1, self.max_n + 1):
            # 计算候选句的TF-IDF向量
            candidate_vec = self._compute_tf_idf_vector(candidate_words, n, idf_dict)
            
            # 如果候选句向量为空，跳过
            if len(candidate_vec) == 0:
                cider_scores.append(0.0)
                continue
            
            # 计算候选句与每个参考句的余弦相似度
            similarities = []
            for ref_words in reference_words_list:
                ref_vec = self._compute_tf_idf_vector(ref_words, n, idf_dict)
                if len(ref_vec) > 0:
                    sim = self._compute_cosine_similarity(candidate_vec, ref_vec)
                    similarities.append(sim)
            
            # 计算平均余弦相似度
            # CIDEr_n(C_i, S_i) = (1/m) * sum_{j=1}^{m} cosine_sim(g_n(C_i), g_n(s_{ij}))
            if len(similarities) > 0:
                cider_n = np.mean(similarities)
            else:
                cider_n = 0.0
            
            cider_scores.append(cider_n)
        
        # 加权平均得到最终的CIDEr得分
        # CIDER(C_i, S_i) = sum_{n=1}^{N} w_n * CIDEr_n(C_i, S_i)
        cider_score = sum(w * score for w, score in zip(self.weights, cider_scores))
        
        # 应用长度惩罚（CIDEr-D中的"D"）
        reference_lengths = [len(ref_words) for ref_words in reference_words_list]
        penalty = self._compute_length_penalty(len(candidate_words), reference_lengths)
        
        cider_d_score = cider_score * penalty
        
        return max(0.0, min(1.0, cider_d_score))  # 确保得分在[0, 1]范围内
    
    def evaluate_batch(self, candidates: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        批量评估：计算一批候选句的CIDEr-D得分
        
        Args:
            candidates: 候选句列表（生成的描述列表）
            references: 参考句列表（每个元素是一个参考句列表，可以有多个参考描述）
        
        Returns:
            评价结果字典，包含：
            - 'cider_d_mean': 平均CIDEr-D得分
            - 'cider_d_scores': 每个样本的CIDEr-D得分列表
        """
        if len(candidates) != len(references):
            raise ValueError(f"候选句数量({len(candidates)})与参考句数量({len(references)})不匹配")
        
        scores = []
        total = len(candidates)
        for idx, (candidate, ref_list) in enumerate(zip(candidates, references)):
            if isinstance(ref_list, str):
                # 如果参考句是单个字符串，转换为列表
                ref_list = [ref_list]
            
            score = self.compute_cider_d(candidate, ref_list)
            scores.append(score)
            
            # 每100个样本或每10%打印一次进度
            if (idx + 1) % 100 == 0 or (idx + 1) == total or (idx + 1) % max(1, total // 10) == 0:
                print(f"    CIDEr-D进度: {idx + 1} / {total} ({100*(idx+1)/total:.1f}%)")
        
        return {
            'cider_d_mean': np.mean(scores),
            'cider_d_scores': scores
        }

