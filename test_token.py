import sentencepiece as spm
if __name__ == '__main__':
    s2 = spm.SentencePieceProcessor(model_file="../model/mt5-large-simplify/spiece_cn.model")
    result = s2.Encode("本院认为，被告人刘连春以非法占有为目的，秘密窃取他人财物，数额较大，其行为已构成盗窃罪，公诉机关指控成立，本院依法予以支持")
    print(result)
    for i in result:
        print(s2.id_to_piece(i))