from googletrans import Translator
from core.mapper import map


class TextHelper:
    def __init__(self):
        self.translator = Translator()

    def translate(self, text, dst_lang):
        output = self.translator.translate(
            text, dest=map(dst_lang))
        return output.text
