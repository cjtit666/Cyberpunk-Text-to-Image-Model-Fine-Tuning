from math import inf
from typing import List, Tuple
from emoji import demojize # 用于将 😂 这样的图形 Emoji 转换成文本 :face_with_tears_of_joy:


# 分词器类（自然语言 --> 数字索引序列），采用BPE算法
class TextTokenizer:
    def __init__(self, vocab: dict, merges: List[str]): # vocab：词表，用于将token转换成数字id
                                                        # merges：BPE合并规则表

        # 保存词表
        self.vocab = vocab

        # 将merges里的每一行转换成元组（["p a", "n d"] --> [("p", "a"), ("n", "d")]
        pairs = [tuple(pair.split()) for pair in merges]

        # 把pairs中的各元组加上优先级后，转换成字典类型
        self.rank_from_pair = dict(zip(pairs, range(len(pairs)))) # len(pairs): pairs中的元组个数
                                                                  # range(...): [0, 1, 2, ..., len(pairs)-1]
                                                                  # zip: 把 pairs中的元组 和 range(...)生成的列表中的各元素 一一配对
                                                                  # dict：转换成字典类型
        # self.rank_from_pair的形式：{
        #                               ('p','a'): 0,
        #                               ('n','d'): 1,
        #                               ('s','l'): 2,
        #                               ('sl','o'): 3
        #                            }


    # 该函数作用：将输入文本进行分词，获得数字索引（token id）列表
    def tokenize(self, text): # text：输入的文本字符串
        # 获取句子结束符</s>的token id
        sep_token = self.vocab['</s>']

        # 获取句子起始符<s>的token id
        cls_token = self.vocab['<s>']

        # 获取未知词<unk>的token id
        unk_token = self.vocab['<unk>']

        
        # 如果文本中有emoji，将其转换为英文文本（如 smile），避免编码错误
        text = demojize(text, delimiters=['', '']) # delimiters：设置转换后要在文本两边加什么字符

        # 将文本统一转换为小写，并删除ASCII码中不包含的特殊非英文符号
        text = text.lower().encode("ascii", errors="ignore").decode() # .lower(): 转换成小写
                                                                      # .encode(...): 编码成ASCII码，并忽略无法编码的字符
                                                                      # .decode(): 重新转换回字符形式
                                                                      # 举例："Helloé你好" --> "hello"
        
        # tokens用于存储text分词后转换成的数字索引
        tokens = []

        # 把text按照空格拆分，转换成列表形式，遍历拆出来的每个词
        for word in text.split(" "):
            # 如果遍历到的word为空，直接跳过
            if len(word) == 0:
                continue

            # 对word按照BPE算法进行分词，然后遍历分词得到的各子词subword
            for subword in self.get_byte_pair_encoding(word):
                # 将各分词索引存到tokens列表中
                tokens.append(self.vocab.get(subword, unk_token)) # 去vocab中查找subword对应的索引，如果找不到的话就返回<unk>对应的索引
                                                                  # tokens.append(...)：把索引添加到tokens列表中
        
        # 在生成的数字索引列表的最前面加上起始符索引，最后面加上结束符索引，然后返回
        return [cls_token] + tokens + [sep_token]


    # 获取pair对应的在merges里的优先级
    def get_pair_rank(self, pair): # pair：字符对元组
        # 去self.rank_from_pair中查找pair对应的优先级，如果不存在的话（说明pair中的字符对无法合并），就返回inf
        return self.rank_from_pair.get(pair, inf)


    # 该函数作用：对传入的word进行分词
    def get_byte_pair_encoding(self, word): # word：待BPE分词的输入单词（文本形式）

        # 把word拆成一个字符列表，并在最开头加上一个特殊字符，表示空格（因为BPE是从字符开始逐步合并的）
        subwords = ['Ġ'] + list(word) # list(word)：把word拆成字符列表（"cat" --> ['c', 'a', 't']
                                      # subwords: ['Ġ', 'c', 'a', 't']
        
        # 只要subwords中的元素个数>1，就一直尝试合并
        while len(subwords) > 1:
            # 获取subwords中所有相邻的字符对（['Ġ', 'c', 'a', 't'] --> [('Ġ', 'c'), ('c', 'a'), ('a', 't')]
            pairs = list(zip(subwords[:-1], subwords[1:]))
            # 注意：pairs[i]就是(subwords[i], subwords[i+1])
            
            # 找到优先级最高的字符对
            pair_to_merge = min(pairs, key = self.get_pair_rank) # min(列表, key=函数)：取函数返回值最小的元素。
            
            # 如果最高优先级的对都不在merges中（全部返回了inf），说明无法继续合并，退出循环
            if pair_to_merge not in self.rank_from_pair: 
                break
            
            # 获取pair_to_merge在pairs中的下标
            i = pairs.index(pair_to_merge)

            subwords = (
                (subwords[:i] if i > 0 else []) + # 把subwords中下标i之前的字符取出来
                [subwords[i] + subwords[i + 1]] + # 将下标i和i+1对应的字符合并
                (subwords[i + 2:] if i + 2 < len(subwords) else []) # 把subwords中下标i+2及之后的字符取出来
            )

        return subwords
