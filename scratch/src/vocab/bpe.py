from src.vocab import BaseTokenizer, load_file
from collections import OrderedDict, defaultdict
import json
import os



class BPETokenizer(BaseTokenizer):
    def __init__(self, vocab_size, vocab, merge_rules):
        super().__init__(vocab_size)
        self.merge_rules = merge_rules
        self.vocab = vocab
        self.inverse_vocab = OrderedDict([(v,k) for k,v in self.vocab.items() ])

    @staticmethod
    def find_most_frequent_pair(words_list,words_dict,pair_freq_cnt_dict, pair_word_indx_mapping_dict):
        
        if len(pair_freq_cnt_dict.keys()) == 0:
            print(f"The initial dict is empty initialising it\n")
            for ii, (w,u_lis,_) in enumerate(words_list):
                if w=="":
                    continue
                cnt = words_dict[w]
                for jj, (ch1, ch2) in enumerate(zip(u_lis,u_lis[1:])):
                    pair_freq_cnt_dict[(ch1, ch2)] = pair_freq_cnt_dict.get((ch1,ch2),0) + cnt
                    pair_word_indx_mapping_dict[(ch1,ch2)].append((ii,jj))   # defaultdict triggers factory on []

           
        sorted_cnt = OrderedDict(sorted(pair_freq_cnt_dict.items(), key = lambda x : x[1], reverse=True))
        # we have sorted list and now also prepare for next after removal
        most_frequent_pair,_ = next(iter(sorted_cnt.items()), (None,None))
        #  from pair_word_indx mapping get which all words it is present and adjust accordingly
        if most_frequent_pair is not None:
            pair_freq_cnt_dict[most_frequent_pair] = 0
            a,b = most_frequent_pair
            most_frequent_pair_str = f"{a}{b}"
            for ii, jj in pair_word_indx_mapping_dict[most_frequent_pair]:
                og_word, u_list, s_list = words_list[ii]
                prev_char = next_char = None
                if jj >0 and (jj - 1 - s_list[jj - 1]>=0):
                    # previous bigram
                    prev_char = u_list[jj-1-s_list[jj-1]]
                next_pos = jj + len(most_frequent_pair_str)
                if (next_pos < len(s_list) and next_pos-s_list[next_pos]<len(u_list)):
                    next_char = u_list[next_pos-s_list[next_pos]]

                if prev_char!=None:
                    # update previous merging related logic
                    pair_freq_cnt_dict[(prev_char,a)]-=words_dict[og_word]
                    t_list = pair_word_indx_mapping_dict[(prev_char,a)]
                    toDel = []
                    for t in t_list:
                        a1,b1 = t
                        if a1==ii and b1<=jj:
                            toDel.append(t)
                    for t in toDel:
                        pair_word_indx_mapping_dict[(prev_char,a)].remove(t)
                    pair_freq_cnt_dict[(prev_char,most_frequent_pair_str)] = pair_freq_cnt_dict.get((prev_char,most_frequent_pair_str),0) + words_dict[og_word]
                    pair_word_indx_mapping_dict[(prev_char,most_frequent_pair_str)].append((ii,jj-len(prev_char)))
                if next_char!=None:
                    # update previous merging related logic
                    pair_freq_cnt_dict[(b,next_char)]-=words_dict[og_word]
                    t_list = pair_word_indx_mapping_dict[(b,next_char)]
                    toDel = []
                    for t in t_list:
                        a1,b1 = t
                        if a1==ii and b1<jj+len(most_frequent_pair_str):
                            toDel.append(t)
                    for t in toDel:
                        pair_word_indx_mapping_dict[(b,next_char)].remove(t)
                    pair_freq_cnt_dict[(most_frequent_pair_str,next_char)] = pair_freq_cnt_dict.get((most_frequent_pair_str,next_char),0) + words_dict[og_word]
                    pair_word_indx_mapping_dict[(most_frequent_pair_str,next_char)].append((ii,jj))
                # deleting the elements
                u_list[jj-s_list[jj]] = most_frequent_pair_str
                next_del_pos = jj + len(a)
                del u_list[next_del_pos-s_list[next_del_pos]]
                for kk in range(next_del_pos,len(s_list)):
                    s_list[kk]+=1



        return most_frequent_pair

    @staticmethod
    def replace_most_frequent_pair(pair_list=[],freq_pair=()):
        a,b=freq_pair
        new_pair_list=[]
        i=0
        while i < len(pair_list)-1:
            ch1=pair_list[i]
            ch2=pair_list[i+1]
            if ch1==a and ch2==b:
                new_pair_list.append(f"{ch1}{ch2}")
                i+=2
            else:
                new_pair_list.append(ch1)
                i+=1
        if i == len(pair_list) - 1:
            new_pair_list.append(pair_list[-1])
        return new_pair_list
            


    @staticmethod
    def train_vocab(vocab_size, file_path=""):
        # return super().train_vocab(file_path)
        assert vocab_size>=256, f"The vocab_size : {vocab_size} should be more than equal to 256 size"
        lines = load_file(file_path=file_path)
        words_dict = OrderedDict()
        words_list = list()
        for l in lines:
            if l=="":
                continue
            tSplit = l.split()
            tSplit = [f"{t} " for t in tSplit]
            tSplit[-1] = tSplit[-1].strip()
            for t in tSplit:
                words_dict[t] = words_dict.get(t,0) + 1
                if words_dict[t]==1:
                    u_list = [chr(b) for b in t.encode("utf-8")]
                    sub_list = [0] * len(u_list)
                    words_list.append([t, u_list,sub_list])
        initial_vocab=OrderedDict()
        merge_rules = OrderedDict()
        consumed_pos=0
        for i in range(0,256,1):
            initial_vocab[chr(i)] = i
        consumed_pos=255
        initial_unicode_list = [chr(b) for s in words_dict.keys() for b in s.encode("utf-8") ]
        unique_bytes = set(initial_unicode_list)
        for a in unique_bytes:
            if a not in initial_vocab.keys():
                assert consumed_pos+1<vocab_size, f"The unique chars in training data is more than max_vocab size... Some issue will happen while tokenizing"
                initial_vocab[a]=consumed_pos+1
                consumed_pos+=1
        
        is_merge_ongoing = True
        pair_freq_cnt_dict = OrderedDict()
        pair_word_indx_mapping_dict = defaultdict(list)


        while consumed_pos<vocab_size-1 and is_merge_ongoing:
            most_frequent_pair = BPETokenizer.find_most_frequent_pair(words_list,words_dict,pair_freq_cnt_dict, pair_word_indx_mapping_dict)
            if most_frequent_pair is None:
                break
            merge_rules[most_frequent_pair] = consumed_pos + 1
            initial_vocab[f"{most_frequent_pair[0]}{most_frequent_pair[1]}"] = consumed_pos + 1
            consumed_pos+=1
            # initial_unicode_list = BPETokenizer.replace_most_frequent_pair(initial_unicode_list,most_frequent_pair)

        return BPETokenizer(vocab_size,vocab=initial_vocab,merge_rules=merge_rules) 
    
    def encode(self,input_str):
        unique_list = [chr(i) for i in input_str.encode("utf-8")]
        def get_pair_list(ulist):
            ret_list=[]
            for ch1,ch2 in zip(ulist,ulist[1:]):
                ret_list.append((ch1,ch2))
            return ret_list
        
        def get_minimal_merge_idx(pair_list):
            minIdx = float('inf')
            ret_val = None
            for p in pair_list:
                if p in self.merge_rules.keys():
                    t_val = self.merge_rules[p]
                    if t_val <= minIdx:
                        minIdx = t_val
                        ret_val = p
            return minIdx, ret_val

        
        is_merging=True
        token_ids = []
        # for u in unique_list:
        #     token_ids.append(self.vocab[u])
        while is_merging and len(unique_list)>=2:
            pair_list = get_pair_list(unique_list)
            min_idx, pair_repl = get_minimal_merge_idx(pair_list=pair_list)
            if min_idx==float('inf'):
                is_merging = False
                break
            unique_list = BPETokenizer.replace_most_frequent_pair(pair_list=unique_list,freq_pair=pair_repl)

        for p in unique_list:
            token_ids.append(self.vocab[f"{p}"])
        return token_ids

    def decode(self, token_ids_list=[]):
        tokens = []
        for t in token_ids_list:
            tokens.append(self.inverse_vocab[t])
        raw = "".join(tokens)
        decoded = bytes([ord(c) for c in raw]).decode("utf-8")
        print(f"The decoded tokens are {decoded!r}")
        return decoded




    def save_vocab(self, vocab_dir):
        os.makedirs(vocab_dir, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": dict(self.vocab),
            "merge_rules": [[k[0], k[1], v] for k, v in self.merge_rules.items()]
        }
        with open(os.path.join(vocab_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_vocab(vocab_dir):
        with open(os.path.join(vocab_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = OrderedDict(data["vocab"])
        merge_rules = OrderedDict(((row[0], row[1]), row[2]) for row in data["merge_rules"])
        return BPETokenizer(data["vocab_size"], vocab=vocab, merge_rules=merge_rules)



class BPETokenizer_A(BaseTokenizer):
    def __init__(self, vocab_size, vocab, merge_rules):
        super().__init__(vocab_size)
        self.merge_rules = merge_rules
        self.vocab = vocab
        self.inverse_vocab = OrderedDict([(v,k) for k,v in self.vocab.items() ])

    @staticmethod
    def find_most_frequent_pair(pair_list=[]):
        ord_cnt= OrderedDict()
        for ch1, ch2 in zip(pair_list,pair_list[1:]):
            ord_cnt[(ch1,ch2)] = ord_cnt.get((ch1,ch2),0) + 1
        
        sorted_cnt = OrderedDict(sorted(ord_cnt.items(), key = lambda x : x[1], reverse=True))
        return sorted_cnt

    @staticmethod
    def replace_most_frequent_pair(pair_list=[],freq_pair=()):
        a,b=freq_pair
        new_pair_list=[]
        i=0
        while i < len(pair_list)-1:
            ch1=pair_list[i]
            ch2=pair_list[i+1]
            if ch1==a and ch2==b:
                new_pair_list.append(f"{ch1}{ch2}")
                i+=2
            else:
                new_pair_list.append(ch1)
                i+=1
        if i == len(pair_list) - 1:
            new_pair_list.append(pair_list[-1])
        return new_pair_list
            


    @staticmethod
    def train_vocab(vocab_size, file_path=""):
        # return super().train_vocab(file_path)
        assert vocab_size>=256, f"The vocab_size : {vocab_size} should be more than equal to 256 size"
        lines = load_file(file_path=file_path)
        new_lines=list()
        for l in lines:
            tSplit = l.split()
            tSplit = [f"{t} " for t in tSplit]
            tSplit[-1] = tSplit[-1].strip()
            new_lines.extend(tSplit)
        lines = new_lines
        initial_vocab=OrderedDict()
        initial_inverse_vocab=OrderedDict()
        merge_rules = OrderedDict()
        consumed_pos=0
        for i in range(0,256,1):
            initial_vocab[chr(i)] = i
        consumed_pos=255
        initial_unicode_list = [chr(b) for s in lines for b in s.encode("utf-8")]
        unique_bytes = set(initial_unicode_list)
        for a in unique_bytes:
            if a not in initial_vocab.keys():
                assert consumed_pos+1<vocab_size, f"The unique chars in training data is more than max_vocab size... Some issue will happen while tokenizing"
                initial_vocab[a]=consumed_pos+1
                consumed_pos+=1
        
        while consumed_pos<vocab_size-1 and len(initial_unicode_list)>=2:
            sorted_bigrams = BPETokenizer.find_most_frequent_pair(initial_unicode_list)
            most_frequent_pair,_ = next(iter(sorted_bigrams.items()), (None,None))
            if most_frequent_pair is None:
                break
            merge_rules[most_frequent_pair] = consumed_pos + 1
            initial_vocab[f"{most_frequent_pair[0]}{most_frequent_pair[1]}"] = consumed_pos + 1
            consumed_pos+=1
            initial_unicode_list = BPETokenizer.replace_most_frequent_pair(initial_unicode_list,most_frequent_pair)

        return BPETokenizer(vocab_size,vocab=initial_vocab,merge_rules=merge_rules) 
    
    def encode(self,input_str):
        unique_list = [chr(i) for i in input_str.encode("utf-8")]
        def get_pair_list(ulist):
            ret_list=[]
            for ch1,ch2 in zip(ulist,ulist[1:]):
                ret_list.append((ch1,ch2))
            return ret_list
        
        def get_minimal_merge_idx(pair_list):
            minIdx = float('inf')
            ret_val = None
            for p in pair_list:
                if p in self.merge_rules.keys():
                    t_val = self.merge_rules[p]
                    if t_val <= minIdx:
                        minIdx = t_val
                        ret_val = p
            return minIdx, ret_val

        
        is_merging=True
        token_ids = []
        # for u in unique_list:
        #     token_ids.append(self.vocab[u])
        while is_merging and len(unique_list)>=2:
            pair_list = get_pair_list(unique_list)
            min_idx, pair_repl = get_minimal_merge_idx(pair_list=pair_list)
            if min_idx==float('inf'):
                is_merging = False
                break
            unique_list = BPETokenizer.replace_most_frequent_pair(pair_list=unique_list,freq_pair=pair_repl)

        for p in unique_list:
            token_ids.append(self.vocab[f"{p}"])
        return token_ids






    def save_vocab(self, vocab_dir):
        os.makedirs(vocab_dir, exist_ok=True)
        data = {
            "vocab_size": self.vocab_size,
            "vocab": dict(self.vocab),
            "merge_rules": [[k[0], k[1], v] for k, v in self.merge_rules.items()]
        }
        with open(os.path.join(vocab_dir, "tokenizer.json"), "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @staticmethod
    def load_vocab(vocab_dir):
        with open(os.path.join(vocab_dir, "tokenizer.json"), "r", encoding="utf-8") as f:
            data = json.load(f)
        vocab = OrderedDict(data["vocab"])
        merge_rules = OrderedDict(((row[0], row[1]), row[2]) for row in data["merge_rules"])
        return BPETokenizer(data["vocab_size"], vocab=vocab, merge_rules=merge_rules)
