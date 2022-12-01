import json
import glob
import os
import re

themedir=os.path.join(os.path.dirname(__file__),"themes")
print(themedir)
def get_available_theme_files():
    return glob.glob(os.path.join(themedir,"*.json"))
    
def get_available_themes():
    themes=[]
    for themefn in get_available_theme_files():
        with open(themefn,'r') as f:
            theme_dict=json.load(f)
            print(theme_dict)
            if 'theme_name' in theme_dict.keys():
                themes.append(theme_dict)
    return themes

def apply_theme(stylesheet,theme):
    for key, value in theme['app_color'].items():
        stylesheet = re.sub(r'\b' + key + r'\b', value, stylesheet)
    print(stylesheet)
    return stylesheet
            

