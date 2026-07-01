import marimo

__generated_with = "0.18.4"
app = marimo.App(width="medium", layout_file="layouts/NaiveBayes.slides.json")


@app.cell
def _():
    import marimo as mo
    import numpy as np
    import random
    from string import ascii_lowercase
    import pandas as pd 
    return ascii_lowercase, mo, np, random


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 0. 模拟贝叶斯
    """)
    return


@app.cell
def _(np):
    def loadDataset(setens = None, labels = None):
        """手写数据集"""
        # 词向量
        if setens is None:
            setens = [
                "my dog has flea problem help please",
                "maybe not take him to dog park stupid",
                "my dalmation is so cute i love him",
                "stop posting stupid worthless garbage",
                "mr licks ate my steak how to stop him",
                "quit buying worthless dog food stupid"
            ]
        seten_vectors = [s.split(" ") for s in setens]

        # 标签
        if labels is None:
            labels = [0,1,0,1,0,1]
        return seten_vectors, labels

    def createVocabList(dataset):
        """根据数据集中出现的所有单词创建词汇表"""
        vocabs = set()
        for words in dataset:
            vocabs = vocabs | set(words)
        return list(vocabs)

    def words2Vectors(
        vocab_list, # 词汇表
        sentence, # 输入句子
    ):
        """根据词汇表, 将当前句子转换成词向量"""
        stc_vectors = np.zeros(len(vocab_list))
        for word in sentence:
            if word in vocab_list:
                stc_vectors[vocab_list.index(word)] += 1
        return stc_vectors


    return loadDataset, words2Vectors


@app.cell
def _(np):
    def TrainNaiveBayes(
        trn_dataset, # 词向量化的训练集
        trn_labels,
        vocabList
    ):
        """训练贝叶斯分类器"""
        # 样本量
        num_sentens = len(trn_dataset) # 所有 sentens 都是词向量
        num_words = len(trn_dataset[0])
        # P(c=1|W)的概率, 即训练集中句子属于 1 类的概率
        p_sent_ilegal = sum(trn_labels) / num_sentens
        # P(c=0|W)的概率, 即训练集中句子属于 0 类的概率
        p_sent_legal = 1 - p_sent_ilegal
        # P(c|wi)的概率, 即训练集中每个词出现在 1/0 类句子中的概率
        p_word_ilegal = np.zeros(num_words)
        p_word_legal = np.zeros(num_words)
        for sidx, sent in enumerate(trn_dataset):
            if trn_labels[sidx] == 1:
                p_word_ilegal += np.array(sent)
            else:
                p_word_legal += np.array(sent)
        p_word_ilegal = (p_word_ilegal + 1) / (p_word_ilegal.sum() + len(vocabList))
        p_word_legal = (p_word_legal + 1) / (p_word_legal.sum() + len(vocabList))
        return p_sent_ilegal, p_sent_legal, p_word_ilegal, p_word_legal

    def TestNaiveBayes(
        tst_dataset, # 词向量化测试集
        predictor: dict
    ):
        # 预测结果
        predLabel = np.zeros(len(tst_dataset))
        # 预测
        for sidx, sent in enumerate(tst_dataset):
            p_legal_word = predictor['pWordLegal'] * np.array(sent) # 计算概率可能会下溢出, 改成 log(a*b) = log(a)+log(b)
            p_legal_word = np.prod(p_legal_word[p_legal_word != 0])
            p_legal = p_legal_word * predictor['pSentLegal'] 

            p_ilegal_word = predictor['pWordIlegal'] * np.array(sent)
            p_ilegal_word = np.prod(p_ilegal_word[p_ilegal_word != 0])
            p_ilegal = p_ilegal_word * predictor['pSentIlegal']

            if p_ilegal > p_legal:
                predLabel[sidx] = 1
        return predLabel
    return TestNaiveBayes, TrainNaiveBayes


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 随机生成句子作为训练集和测试集
    """)
    return


@app.cell
def _(ascii_lowercase, loadDataset, random, words2Vectors):
    def RandomVocabList():
        def RandomWord(leng):
            # 指定长度的单词
            selectLetters = [random.choice(ascii_lowercase) for _ in range(leng)]
            return "".join(selectLetters)
        # 随机 497 个好词
        wrdLeng = [random.randint(3,7) for _ in range(497)]
        lglWord = [RandomWord(l) for l in wrdLeng]
    
        # 三个差词
        ilgWord = ["fuck", "stupid", "bitch"]

        return lglWord + ilgWord

    def RandomSentenceList(vocabList):
        def RandomSent(leng, vocabList):
            # 以 2:8 概率生成非法/合法句子
            sent = " ".join([random.choice(vocabList) for _ in range(leng)])
            if random.choice([1,0,0,0,0]):
                sent = "fuck " + sent
            return sent
            
        # 生成随机句子, 10000 句
        stcLeng = [random.randint(10,15) for _ in range(10000)]
        randSent = [RandomSent(l, vocabList) for l in stcLeng]
        # label
        stcLabel = []
        for s in randSent:
            if "fuck" in s:
                stcLabel.append(1)
            elif "stupid" in s:
                stcLabel.append(1)
            elif "bitch" in s:
                stcLabel.append(1)
            else:
                stcLabel.append(0)
        return randSent, stcLabel

    # 约 2600 的非法句子
    trnVocabList = RandomVocabList()
    trnSentList, trnSentLabel = loadDataset(*RandomSentenceList(trnVocabList))
    tstSentList, tstSentLabel = loadDataset(*RandomSentenceList(trnVocabList))

    ## 词向量化数据集
    trnSentList = [words2Vectors(vocab_list=trnVocabList, sentence=sent) for sent in trnSentList]
    tstSentList = [words2Vectors(vocab_list=trnVocabList, sentence=sent) for sent in tstSentList]
    return trnSentLabel, trnSentList, trnVocabList, tstSentLabel, tstSentList


@app.cell
def _(TrainNaiveBayes, trnSentLabel, trnSentList, trnVocabList):
    pSentIlegal, pSentLegal, pWordIlegal, pWordLegal = TrainNaiveBayes(trnSentList, trnSentLabel, trnVocabList)
    return pSentIlegal, pSentLegal, pWordIlegal, pWordLegal


@app.cell
def _(
    TestNaiveBayes,
    pSentIlegal,
    pSentLegal,
    pWordIlegal,
    pWordLegal,
    trnSentLabel,
    trnSentList,
    tstSentLabel,
    tstSentList,
):
    from sklearn.naive_bayes import BernoulliNB
    from sklearn.metrics import accuracy_score
    BNB = BernoulliNB(alpha=1.0, binarize=None)
    BNB.fit(trnSentList, trnSentLabel)
    BNB_pred = BNB.predict(tstSentList)
    BNBacc = accuracy_score(tstSentLabel, BNB_pred)
    print("BNB acc: ", BNBacc)

    myPred = TestNaiveBayes(
        tst_dataset=tstSentList,
        predictor={
            "pSentLegal": pSentLegal, 
            "pSentIlegal": pSentIlegal,
            "pWordIlegal": pWordIlegal,
            "pWordLegal": pWordLegal
        }
    )
    myAcc = accuracy_score(tstSentLabel, myPred)
    print("my acc: ", myAcc)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## 邮件分类
    """)
    return


if __name__ == "__main__":
    app.run()
