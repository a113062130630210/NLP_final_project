import openai
import numpy
import pandas as pd
import random

def ask_chatGPT(prompt):
    return openai.Completion.create(
        model="text-davinci-003",
        prompt=prompt,
        max_tokens=128,
        temperature=0.5,
    )["choices"][0]["text"]
    

def say_target():
    response = ask_chatGPT("你是一名廣告內容執法人員，接下來我會給你一些廣告詞，如果你認為內容看起來可能並未經過政府核准就擅自廣告，請回答「1」；如果你認為過度誇大，或是含有猥褻內容，請回答「0」。我們的行動會分為兩階段，「訓練階段」和「測試階段」，在「訓練階段」，我會告訴你「廣告詞的內容」和「應該回答1還是0」讓你熟悉任務內容。在「測試階段」，我只會給你「廣告詞的內容」，讓你自行去做判斷。現在開始「訓練階段」")
    
def say_target2():
    response = ask_chatGPT("現在開始「測試階段」，我只會告訴你廣告內容，不再告訴你正確答案。請依你在「訓練階段」的所學，判斷要回答「0」，還是要回答「1」。")
    
def training(sentence, ans):
    response = ask_chatGPT("測驗規則：如果你認為廣告內容看起來可能並未經過政府核准就擅自廣告，你應該回答「1」；如果你認為過度誇大，或是含有猥褻內容，你應該回答「0」" + "\n\n廣告內容：\n" + sentence + "\n\n對於這個廣告，你應該要回答：" + str(ans) + "\n\n請由這個廣告內容和答案，分析為什麼這則廣告的答案會是" + str(ans))
    
def testing(sentence):
    response = ask_chatGPT("你是一名廣告內容執法人員，接下來我會給你一些廣告詞的片段。如果你覺得該廣告看起來可能並未經過政府機關的核准就擅自發布，請回答「1」；如果你覺得該廣告內容對商品的效果過度誇大，或是含有猥褻內容，請回答「0」。" + "\n\n以下為廣告內容：\n" + sentence)
    print(response)
    if ('0' in response):
        return 0
    return 1

if __name__ == '__main__':
    openai.api_key = "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"
    
    test_data = pd.read_csv("COS_test.csv")
    ans = pd.read_csv("COS_Sample.csv")
    train_data = pd.read_csv("COS_train.csv")
    
    #say_target();
    '''
    for i in range (20):
        if (i % 10 == 0):
            print(i)
            #if (len(train_data.loc[i,"sentence"]) > 180):
            #if (train_data.loc[i,"sentence"].count(".")>= 19):
        index = random.randint(0,train_data.shape[0]-1)
        training(train_data.loc[index,"sentence"], train_data.loc[index,"label_for_kaggle"])
    '''
    #say_target2();
    for i in range (100):
        if (i % 10 == 0):
            print(i)
        ans.loc[i, "label_for_kaggle"] = testing(test_data.loc[i,"sentence"])
    
    print([ans.iloc[i][1] for i in range (100)])
    ans.to_csv("COS_sample.csv", index=False)
