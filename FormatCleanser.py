import re

def apos_cleanser(match_obj):
    cnt = 0
    string = match_obj.group(0)
    for i in range(5):
        if string[i] == "'" and string[len(string) - i - 1] == "'":
            cnt += 1
        else:
            break
    if cnt == 4:
        cnt = 3
    return string[i : (len(string) - i)]

def format_cleanse(input_file, output_file):
    string = open(input_file, 'r', encoding='utf-8').read()
    string = re.sub("'{2,5}.*?'{2,5}", apos_cleanser, string)
    string = re.sub("={2,6}.*?={2,6}", "", string)
    string = re.sub("\*+ ", "", string)
    string = re.sub("\#+ ", "", string)
    string = re.sub("\#+ ", "", string)
    string = re.sub("  +", " ", string)
    output = open(output_file, "w+", encoding='utf-8')
    output.write(string)