class Translator:
    def __init__(self, name="Helsinki-NLP/opus-mt-ru-en"):
        from transformers import MarianMTModel, MarianTokenizer
        self.tok = MarianTokenizer.from_pretrained(name)
        self.model = MarianMTModel.from_pretrained(name).eval()
        self.cache = {}

    def __call__(self, text):
        if text in self.cache:
            return self.cache[text]
        import torch
        with torch.no_grad():
            b = self.tok([text], return_tensors="pt", truncation=True, max_length=128)
            ids = self.model.generate(**b, num_beams=4, max_new_tokens=64)
        r = self.tok.decode(ids[0], skip_special_tokens=True).strip() or text
        if len(self.cache) > 1000:
            self.cache.clear()
        self.cache[text] = r
        return r
