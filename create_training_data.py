import re
'''
# given a text file with boundary marker create y and x

X = ['โดยตลอดแต่', 'ความเป็น', 'ศาสตร์ที่', 'สอนในสาขา', 'นั้นๆไม่', 'สามารถที่', 'จะช่วยให้', 'ผู้เรียน', 'หลุดพ้นไป', 'จากทัศนะ', 'ครอบงำที่', 'มองปัญหา', 'ความยากจน', 'มองคนจนแบบ', 'เดิมๆและ', 'แสดงออกต่อ', 'คนจนเหล่า', 'นั้นใน', 'ลักษณะที่', 'เป็นภาระ']
Y = ['1001000100', '10001000', '100000100', '100101000', '10001100', '100000100', '101000100', '10010000', '100000010', '10010000', '100000100', '10010000', '100010000', '1001010100', '10001100', '1000100100', '101010000', '100010', '100000100', '10001000']
len X  20
len Y 20

'''

# TODO: a function to create a training set given several corpuses
# create training data(directory) ?
# TODO: a function to create a training set given a labeled corpus
# create training data(labeled_corpus) ,  return x , y






# remove symbols from text, keep only Thai and eng characters , with |
'''
โดย|ตลอด|แต่|ความ|เป็น|ศาสตร์|ที่|สอน|ใน|สาขา|นั้น|ๆ|ไม่|สามารถ|ที่|จะ|ช่วย|ให้|ผู้|เรียน|หลุดพ้น|ไป|จาก|ทัศนะ|ครอบงำ|ที่|มอง|
'''
def clean(input):
    pattern = re.compile(r' |<\w\w>|<\/\w\w>|/s|,|:|\(|\)|"|\.|\d|\-|\[|\]')
    pattern2 = re.compile(r'\|{2,}')

    # article_00001
    with open(input + '.txt', 'r') as f:
        lines = f.read()

        # print(line)
    with open(input + '_clean.txt', 'w') as w:
        # for line in lines:
        sub = re.sub(pattern, "", lines)
        sub = re.sub(pattern2, "|", sub)
        # print(sub)
        w.write(sub)

# data= 'article_00001'
# clean(data)



'''
create this
100|1000|100|1000|1000|100000|100|100|10|1000|1000|1|100|100000|
'''
def create_label(input):
    thai_chars = 'กขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ'  # 1..73

    pattern = re.compile(r'[a-zA-Zกขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ]')
    pattern2 = re.compile(r'\|[a-zA-Zกขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ]')
    pattern3 = re.compile(r'\A[a-zA-Zกขฃคฅฆงจฉชซฌญฐฎฏฐฑฒณดตถทธนบปผฝพฟภมยรฤลฦวศษสหฬอฮฯะัาำิีึืุูเแโใไๅๆ็่้๊๋์ํ]')
    pattern4 = re.compile(r"\|$")
    # pattern3 = re.compile(r"/\r?\n|\r/g")
    # pattern4 = re.compile(r"/\r?\n|\r/g")

    # article_00001
    with open(input + '.txt', 'r') as f:
        # lines = f.read()
        for line in f:
            print("1")
            with open(input + '_label.txt', 'a') as w:
                # for line in f:
                # print(line)
                # sub = re.sub(pattern4,"",line)

                line = re.sub(pattern2, "|1", line)

                line = re.sub(pattern3, "1", line)
                line = re.sub(pattern, "0", line)
                line = re.sub(pattern4, "", line)

                # print("sub: ", sub)
                w.write(line)


# create training data x and y
'''
this  function is used to create training data x and y
given a cleaned file
X = ['โดยตลอดแต่', 'ความเป็น', 'ศาสตร์ที่', 'สอนในสาขา', 'นั้นๆไม่', 'สามารถที่',...
Y = ['1001000100', '10001000', '100000100', '100101000', '10001100',....
'''

def create_training(f):
    with open(f + ".txt", 'r') as g:
        line = g.read()
        line = line.replace('\n', '').replace('\r', '')
        splited = line.split("|")
        # print(splited)

    # print("-------------")

    seq_list = []
    seq = []
    len_seq = 10
    for word in splited:
        # current_len = len(seq)+len(word)
        current_len = len("".join(seq))
        # print("current: " ,current_len)
        if current_len + len(word) > len_seq:
            seq = "".join(seq)
            seq_list.append(seq)
            # print("begin new seq ")

            seq = []
            seq += word
            # seq="".join(word)
        else:
            seq += word
            # seq="".join(word)
        # print(seq, " num ", len(seq))

    print(seq_list)
    return seq_list


#TODO: pad each data to max_seq_len
# TODO: create batches https://cs230-stanford.github.io/pytorch-nlp.html
#  in model, do not calculate padding
'''
# tutorial
https://towardsdatascience.com/taming-lstms-variable-sized-mini-batches-and-why-pytorch-is-good-for-your-health-61d35642972e
# create baches tutorial

https://arxiv.org/pdf/1810.03479.pdf
#Sentence Segmentation for Classical Chinese Based on LSTM with Radical Embedding

https://cs230-stanford.github.io/pytorch-nlp.html
#create batch


'''