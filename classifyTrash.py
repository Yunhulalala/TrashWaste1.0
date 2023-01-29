import json

def classify_trash(trashobj=''):
    classify_rules = {"可回收垃圾":["充电宝","包","洗护用品","塑料玩具","塑料器皿","塑料衣架","玻璃器皿","金属器皿","快递纸袋",
                               "插头电线","旧衣服","易拉罐","枕头","毛绒玩具","鞋","砧板","纸盒纸箱","调料瓶","酒瓶",
                               "金属食品罐","金属厨具","锅","食用油桶","饮料瓶","书籍纸张","垃圾桶"],
                      "厨余垃圾": ["剩饭剩菜","大骨头","果皮果肉","茶叶渣","菜帮菜叶","蛋壳","鱼骨"],
                      "有害垃圾": ["干电池","软膏","过期药物"],
                      "其他垃圾": ["一次性快餐盒","污损塑料","烟蒂","牙签","花盆","陶瓷器皿","筷子","污损用纸"]}
    HazardousWaste = classify_rules['有害垃圾']  # 有害垃圾
    RecyclableWaste = classify_rules['可回收垃圾']  # 可回收垃圾
    HouseholdFoodWaste = classify_rules['厨余垃圾']  # 厨余垃圾。
    ResidualWaste = classify_rules['其他垃圾']  # 其他垃圾
    if trashobj in HazardousWaste:
        return '有害垃圾'
    elif trashobj in RecyclableWaste:
        return '可回收垃圾'
    elif trashobj in HouseholdFoodWaste:
        return '厨余垃圾'
    elif trashobj in ResidualWaste:
        return '其他垃圾'
    else:
        return '暂时还不知道是什么垃圾呢！'

if __name__=='__main__':
    print(classify_trash('玻璃器皿'))
    print(classify_trash('大骨头'))
    print(classify_trash('干电池'))
    print(classify_trash('一次性快餐盒'))





