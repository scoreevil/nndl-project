"""
ROUGE-L评价指标实现
ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence)
"""
import re
from typing import List, Dict
import numpy as np


class ROUGELEvaluator:
    """
    ROUGE-L (Recall-Oriented Understudy for Gisting Evaluation - Longest Common Subsequence) 评价指标
    
    ROUGE-L使用"最长公共子序列（LCS）"来衡量句子之间的相似度。
    它不要求单词连续出现，因此可以更好地评估词序的相似性。
    
    核心步骤：
    1. 计算召回率(Recall)和精确率(Precision)
       - R_lcs = LCS(C, s_i) / length(s_i)
       - P_lcs = LCS(C, s_i) / length(C)
    2. 计算F-score
       - ROUGE-L = F_lcs = ((1 + β²) * R_lcs * P_lcs) / (R_lcs + β² * P_lcs)
       - 其中β用于调整P和R的权重，当β→∞时，F-score只取决于召回率R_lcs
    """
    
    def __init__(self, beta: float = 1.0):
        """
        初始化ROUGE-L评价器
        
        Args:
            beta: F-score计算中的超参数，用于调整精确率和召回率的权重，默认1.0
                 当beta→∞时，F-score只取决于召回率（Recall-Oriented的由来）
        """
        self.beta = beta
    
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
    
    def _compute_lcs(self, seq1: List[str], seq2: List[str]) -> int:
        """
        计算两个序列的最长公共子序列（LCS）的长度
        
        使用动态规划算法
        
        Args:
            seq1: 序列1（单词列表）
            seq2: 序列2（单词列表）
        
        Returns:
            LCS的长度
        """
        m, n = len(seq1), len(seq2)
        
        # 如果任一序列为空，返回0
        if m == 0 or n == 0:
            return 0
        
        # 创建DP表：dp[i][j]表示seq1[0:i]和seq2[0:j]的LCS长度
        dp = [[0] * (n + 1) for _ in range(m + 1)]
        
        # 填充DP表
        for i in range(1, m + 1):
            for j in range(1, n + 1):
                if seq1[i-1] == seq2[j-1]:
                    # 如果当前字符匹配，LCS长度+1
                    dp[i][j] = dp[i-1][j-1] + 1
                else:
                    # 如果当前字符不匹配，取之前最大的LCS长度
                    dp[i][j] = max(dp[i-1][j], dp[i][j-1])
        
        return dp[m][n]
    
    def compute_rouge_l(self, candidate: str, reference: str) -> float:
        """
        计算单个候选句相对于单个参考句的ROUGE-L得分
        
        Args:
            candidate: 候选句（生成的描述）
            reference: 参考句（真实描述）
        
        Returns:
            ROUGE-L得分（0-1之间，越高越好）
        """
        # 分词
        candidate_words = self._tokenize(candidate)
        reference_words = self._tokenize(reference)
        
        # 如果候选句或参考句为空，返回0
        if len(candidate_words) == 0 or len(reference_words) == 0:
            return 0.0
        
        # 计算最长公共子序列的长度
        lcs_length = self._compute_lcs(candidate_words, reference_words)
        
        # 如果LCS长度为0，返回0
        if lcs_length == 0:
            return 0.0
        
        # 步骤1: 计算召回率(Recall)和精确率(Precision)
        # R_lcs = LCS(C, s_i) / length(s_i)
        recall = lcs_length / len(reference_words)
        
        # P_lcs = LCS(C, s_i) / length(C)
        precision = lcs_length / len(candidate_words)
        
        # 步骤2: 计算F-score
        # ROUGE-L = F_lcs = ((1 + β²) * R_lcs * P_lcs) / (R_lcs + β² * P_lcs)
        if recall == 0 and precision == 0:
            f_score = 0.0
        else:
            beta_squared = self.beta ** 2
            numerator = (1 + beta_squared) * recall * precision
            denominator = recall + beta_squared * precision
            f_score = numerator / denominator if denominator > 0 else 0.0
        
        return max(0.0, min(1.0, f_score))  # 确保得分在[0, 1]范围内
    
    def compute_rouge_l_multiple_references(self, candidate: str, references: List[str]) -> float:
        """
        计算单个候选句相对于多个参考句的ROUGE-L得分
        
        最终得分会选择所有参考句计算出的最高分
        
        Args:
            candidate: 候选句（生成的描述）
            references: 参考句列表（多个真实描述）
        
        Returns:
            ROUGE-L得分（0-1之间，越高越好）
        """
        scores = []
        for ref in references:
            score = self.compute_rouge_l(candidate, ref)
            scores.append(score)
        
        # 返回最高分
        return max(scores) if scores else 0.0
    
    def evaluate_batch(self, candidates: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        批量评估：计算一批候选句的ROUGE-L得分
        
        Args:
            candidates: 候选句列表（生成的描述列表）
            references: 参考句列表（每个元素是一个参考句列表，可以有多个参考描述）
        
        Returns:
            评价结果字典，包含：
            - 'rouge_l_mean': 平均ROUGE-L得分
            - 'rouge_l_scores': 每个样本的ROUGE-L得分列表
        """
        if len(candidates) != len(references):
            raise ValueError(f"候选句数量({len(candidates)})与参考句数量({len(references)})不匹配")
        
        scores = []
        total = len(candidates)
        for idx, (candidate, ref_list) in enumerate(zip(candidates, references)):
            if isinstance(ref_list, str):
                # 如果参考句是单个字符串，转换为列表
                ref_list = [ref_list]
            
            score = self.compute_rouge_l_multiple_references(candidate, ref_list)
            scores.append(score)
            
            # 每100个样本或每10%打印一次进度
            if (idx + 1) % 100 == 0 or (idx + 1) == total or (idx + 1) % max(1, total // 10) == 0:
                print(f"    ROUGE-L进度: {idx + 1} / {total} ({100*(idx+1)/total:.1f}%)")
        
        return {
            'rouge_l_mean': np.mean(scores),
            'rouge_l_scores': scores
        }

