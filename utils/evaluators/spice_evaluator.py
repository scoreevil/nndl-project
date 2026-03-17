"""
SPICE评价指标实现
SPICE (Semantic Propositional Image Caption Evaluation)
"""
import re
from typing import List, Dict, Tuple, Set
import numpy as np


class SPICEEvaluator:
    """
    SPICE (Semantic Propositional Image Caption Evaluation) 评价指标
    
    SPICE是一个更高级的评价指标，它超越了表面层面的词汇重叠，深入到语义层面。
    SPICE将句子解析为"场景图（Scene Graph）"，然后比较从这些图中提取的语义元组。
    
    核心步骤：
    1. 解析场景图：使用依赖解析器将候选句C和参考句S解析为场景图
       - 场景图包含"对象（Objects）"、"属性（Attributes）"和"关系（Relations）"
    
    2. 提取语义元组（Tuples）：从场景图中提取语义元组
       - 对象元组：(对象) - 例如：(男人)
       - 属性元组：(对象, 属性) - 例如：(男人, 高), (衬衫, 蓝色)
       - 关系元组：(对象, 关系, 对象) - 例如：(男人, 穿着, 衬衫)
    
    3. 计算F1-score：
       - Pspice = |T(C) ∩ T(S)| / |T(C)|
       - Rspice = |T(C) ∩ T(S)| / |T(S)|
       - SPICE = F₁ = (2 * Pspice * Rspice) / (Pspice + Rspice)
    
    4. 使用WordNet同义词词典进行智能匹配
    """
    
    def __init__(self):
        """
        初始化SPICE评价器
        
        尝试初始化NLTK和spaCy用于依赖解析和场景图构建
        """
        self.use_nltk = False
        self.use_spacy = False
        self.wordnet = None
        self.nlp = None
        
        self._init_nlp_tools()
    
    def _init_nlp_tools(self):
        """初始化NLP工具（NLTK和spaCy）"""
        # 尝试导入NLTK
        try:
            import nltk
            from nltk.corpus import wordnet
            
            # 尝试下载必要的NLTK数据
            try:
                nltk.data.find('corpora/wordnet')
            except LookupError:
                try:
                    nltk.download('wordnet', quiet=True)
                except:
                    pass
            
            try:
                nltk.data.find('taggers/averaged_perceptron_tagger')
            except LookupError:
                try:
                    nltk.download('averaged_perceptron_tagger', quiet=True)
                except:
                    pass
            
            try:
                nltk.data.find('tokenizers/punkt')
            except LookupError:
                try:
                    nltk.download('punkt', quiet=True)
                except:
                    pass
            
            self.wordnet = wordnet
            self.use_nltk = True
        except ImportError:
            self.use_nltk = False
        
        # 尝试导入spaCy（用于更准确的依赖解析）
        try:
            import spacy
            try:
                # 尝试加载英文模型
                self.nlp = spacy.load("en_core_web_sm")
                self.use_spacy = True
            except OSError:
                # 如果模型不存在，使用基础模型或跳过
                self.use_spacy = False
        except ImportError:
            self.use_spacy = False
    
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
    
    def _are_synonyms(self, word1: str, word2: str) -> bool:
        """
        判断两个词是否为同义词（使用WordNet）
        
        Args:
            word1: 单词1
            word2: 单词2
        
        Returns:
            是否为同义词
        """
        if not self.use_nltk or not self.wordnet:
            return word1 == word2
        
        try:
            synsets1 = self.wordnet.synsets(word1)
            synsets2 = self.wordnet.synsets(word2)
            
            # 检查是否有共同的同义词集
            for syn1 in synsets1:
                for syn2 in synsets2:
                    if syn1 == syn2:
                        return True
            
            return False
        except:
            return word1 == word2
    
    def _extract_tuples_simple(self, words: List[str]) -> Set[Tuple]:
        """
        简化版本的语义元组提取（不使用完整的场景图解析）
        
        基于简单的规则提取：
        1. 对象元组：(对象)
        2. 属性元组：(对象, 属性) - 基于形容词+名词的组合
        3. 关系元组：(对象, 关系, 对象) - 基于动词+名词的组合
        
        Args:
            words: 单词列表
        
        Returns:
            语义元组集合
        """
        tuples = set()
        
        if len(words) == 0:
            return tuples
        
        # 简单的词性标注（基于常见模式）
        # 这里使用简化的规则，实际应该使用NLTK或spaCy的POS tagging
        
        # 1. 提取对象元组（名词）
        # 假设名词是对象
        nouns = []
        adjectives = []
        verbs = []
        
        # 简单的词性判断（基于常见模式）
        for word in words:
            # 常见名词
            if word in ['dress', 'shirt', 'pants', 'skirt', 'jacket', 'coat', 'man', 'woman', 
                       'person', 'shoes', 'bag', 'hat', 'top', 'bottom', 'outfit', 'clothing']:
                nouns.append(word)
            # 常见形容词
            elif word in ['red', 'blue', 'green', 'black', 'white', 'tall', 'short', 'long', 
                         'big', 'small', 'beautiful', 'elegant', 'casual', 'formal']:
                adjectives.append(word)
            # 常见动词
            elif word in ['wearing', 'wears', 'wear', 'has', 'have', 'is', 'are', 'shows', 'show']:
                verbs.append(word)
        
        # 如果没有识别到名词，将所有词都视为潜在对象
        if len(nouns) == 0:
            nouns = words
        
        # 2. 提取对象元组
        for noun in nouns:
            tuples.add(('object', noun))
        
        # 3. 提取属性元组（形容词+名词的组合）
        # 简化：假设相邻的形容词和名词形成属性关系
        for i in range(len(words) - 1):
            if words[i] in adjectives and words[i+1] in nouns:
                tuples.add(('attribute', words[i+1], words[i]))
        
        # 4. 提取关系元组（动词+名词的组合）
        # 简化：假设动词后跟名词形成关系
        for i in range(len(words) - 1):
            if words[i] in verbs and words[i+1] in nouns:
                tuples.add(('relation', words[i], words[i+1]))
        
        return tuples
    
    def _extract_tuples_with_spacy(self, text: str) -> Set[Tuple]:
        """
        使用spaCy进行更准确的场景图解析和语义元组提取
        
        Args:
            text: 输入文本
        
        Returns:
            语义元组集合
        """
        if not self.use_spacy or not self.nlp:
            # 如果spaCy不可用，回退到简化版本
            words = self._tokenize(text)
            return self._extract_tuples_simple(words)
        
        try:
            doc = self.nlp(text.lower())
            tuples = set()
            
            # 提取对象（名词）
            for token in doc:
                if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':
                    tuples.add(('object', token.lemma_))
            
            # 提取属性（形容词+名词）
            for token in doc:
                if token.pos_ == 'ADJ':
                    # 查找修饰的名词
                    for child in token.children:
                        if child.pos_ == 'NOUN':
                            tuples.add(('attribute', child.lemma_, token.lemma_))
            
            # 提取关系（动词+宾语）
            for token in doc:
                if token.pos_ == 'VERB':
                    verb_lemma = token.lemma_
                    # 查找直接宾语
                    for child in token.children:
                        if child.dep_ == 'dobj' and (child.pos_ == 'NOUN' or child.pos_ == 'PROPN'):
                            tuples.add(('relation', verb_lemma, child.lemma_))
                    # 查找主语（用于完整的关系元组）
                    for child in token.children:
                        if child.dep_ == 'nsubj' and (child.pos_ == 'NOUN' or child.pos_ == 'PROPN'):
                            # 查找宾语
                            for obj_child in token.children:
                                if obj_child.dep_ == 'dobj' and (obj_child.pos_ == 'NOUN' or obj_child.pos_ == 'PROPN'):
                                    tuples.add(('relation', child.lemma_, verb_lemma, obj_child.lemma_))
            
            return tuples
        except:
            # 如果spaCy解析失败，回退到简化版本
            words = self._tokenize(text)
            return self._extract_tuples_simple(words)
    
    def _match_tuples(self, tuple1: Tuple, tuple2: Tuple) -> bool:
        """
        判断两个语义元组是否匹配（使用WordNet同义词）
        
        Args:
            tuple1: 元组1
            tuple2: 元组2
        
        Returns:
            是否匹配
        """
        if len(tuple1) != len(tuple2):
            return False
        
        # 比较元组的每个元素
        for i in range(len(tuple1)):
            elem1 = tuple1[i]
            elem2 = tuple2[i]
            
            # 如果元素相同，匹配
            if elem1 == elem2:
                continue
            
            # 如果是字符串，检查是否为同义词
            if isinstance(elem1, str) and isinstance(elem2, str):
                if self._are_synonyms(elem1, elem2):
                    continue
            
            # 如果不匹配，返回False
            return False
        
        return True
    
    def compute_spice(self, candidate: str, reference: str) -> float:
        """
        计算单个候选句相对于单个参考句的SPICE得分
        
        Args:
            candidate: 候选句（生成的描述）
            reference: 参考句（真实描述）
        
        Returns:
            SPICE得分（0-1之间，越高越好）
        """
        # 提取语义元组
        candidate_tuples = self._extract_tuples_with_spacy(candidate)
        reference_tuples = self._extract_tuples_with_spacy(reference)
        
        if len(candidate_tuples) == 0 and len(reference_tuples) == 0:
            return 1.0  # 如果两者都没有元组，认为完全匹配
        
        if len(candidate_tuples) == 0:
            return 0.0  # 如果候选句没有元组，返回0
        
        if len(reference_tuples) == 0:
            return 0.0  # 如果参考句没有元组，返回0
        
        # 计算匹配的元组数量
        matched_tuples = 0
        for cand_tuple in candidate_tuples:
            for ref_tuple in reference_tuples:
                if self._match_tuples(cand_tuple, ref_tuple):
                    matched_tuples += 1
                    break  # 每个候选元组只匹配一次
        
        # 计算精确率和召回率
        # Pspice = |T(C) ∩ T(S)| / |T(C)|
        precision = matched_tuples / len(candidate_tuples) if len(candidate_tuples) > 0 else 0.0
        
        # Rspice = |T(C) ∩ T(S)| / |T(S)|
        recall = matched_tuples / len(reference_tuples) if len(reference_tuples) > 0 else 0.0
        
        # 计算F1-score
        # SPICE = F₁ = (2 * Pspice * Rspice) / (Pspice + Rspice)
        if precision == 0 and recall == 0:
            f1_score = 0.0
        else:
            f1_score = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return max(0.0, min(1.0, f1_score))  # 确保得分在[0, 1]范围内
    
    def compute_spice_multiple_references(self, candidate: str, references: List[str]) -> float:
        """
        计算单个候选句相对于多个参考句的SPICE得分
        
        最终得分会选择所有参考句计算出的最高分
        
        Args:
            candidate: 候选句（生成的描述）
            references: 参考句列表（多个真实描述）
        
        Returns:
            SPICE得分（0-1之间，越高越好）
        """
        scores = []
        for ref in references:
            score = self.compute_spice(candidate, ref)
            scores.append(score)
        
        # 返回最高分
        return max(scores) if scores else 0.0
    
    def evaluate_batch(self, candidates: List[str], references: List[List[str]]) -> Dict[str, float]:
        """
        批量评估：计算一批候选句的SPICE得分
        
        Args:
            candidates: 候选句列表（生成的描述列表）
            references: 参考句列表（每个元素是一个参考句列表，可以有多个参考描述）
        
        Returns:
            评价结果字典，包含：
            - 'spice_mean': 平均SPICE得分
            - 'spice_scores': 每个样本的SPICE得分列表
        """
        if len(candidates) != len(references):
            raise ValueError(f"候选句数量({len(candidates)})与参考句数量({len(references)})不匹配")
        
        scores = []
        total = len(candidates)
        for idx, (candidate, ref_list) in enumerate(zip(candidates, references)):
            if isinstance(ref_list, str):
                # 如果参考句是单个字符串，转换为列表
                ref_list = [ref_list]
            
            score = self.compute_spice_multiple_references(candidate, ref_list)
            scores.append(score)
            
            # 每100个样本或每10%打印一次进度
            if (idx + 1) % 100 == 0 or (idx + 1) == total or (idx + 1) % max(1, total // 10) == 0:
                print(f"    SPICE进度: {idx + 1} / {total} ({100*(idx+1)/total:.1f}%)")
        
        return {
            'spice_mean': np.mean(scores),
            'spice_scores': scores
        }

