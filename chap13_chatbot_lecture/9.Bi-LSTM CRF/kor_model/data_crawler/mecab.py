from konlpy.tag import Mecab

def tockenizer(from_path, to_path) :
    try :
        print("tockenizing start")
        mecab = Mecab('/usr/local/lib/mecab/dic/mecab-ko-dic')
        with open(from_path, 'r') as in_file :
            with open(to_path, 'w+') as out_file:
                for line in in_file.readlines() :
                    for word, tag in mecab.pos(line):
                        out_file.write('{0} '.format(word))
                        if(tag in ('SF','EOS', 'EF')) :
                            out_file.write('{0} '.format('\n'))
        print("tockenizing done")
    except Exception as e:
        print ("error on tockenizer : {0}".format(e))