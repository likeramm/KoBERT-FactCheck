import os
from zipfile import ZipFile
import requests
from transformers import BertTokenizer

class KoBERTTokenizer(BertTokenizer):
    def __init__(self, vocab_file, spm_model_file, **kwargs):
        # [수정] do_lower_case 설정을 직접 건드리지 않고, 
        # 부모 클래스(BertTokenizer) 생성자에 전달합니다.
        if "do_lower_case" not in kwargs:
            kwargs["do_lower_case"] = False
            
        super().__init__(vocab_file, **kwargs)
        
        self.spm_model_file = spm_model_file

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *init_inputs, **kwargs):
        if "skt/kobert-base-v1" in pretrained_model_name_or_path:
            from transformers.utils import cached_file
            
            vocab_file = cached_file(pretrained_model_name_or_path, "vocab.txt", **kwargs)
            
            try:
                spm_model_file = cached_file(pretrained_model_name_or_path, "spiece.model", **kwargs)
            except:
                spm_model_file = cached_file(pretrained_model_name_or_path, "tokenizer.model", **kwargs)

            kwargs.pop("vocab_file", None)
            kwargs.pop("spm_model_file", None)
            
            tokenizer = cls(vocab_file, spm_model_file, **kwargs)
            return tokenizer
            
        return super().from_pretrained(pretrained_model_name_or_path, *init_inputs, **kwargs)

    def __getstate__(self):
        state = self.__dict__.copy()
        state["spm_model_file"] = None
        return state

    def __setstate__(self, d):
        self.__dict__ = d
        self.spm_model_file = None