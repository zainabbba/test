import streamlit as st
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold, StratifiedKFold, cross_val_score
import numpy as np
model_summary=pickle.load(open('bert.pkl','rb'))
model=pickle.load(open('best_model.pkl','rb'))
model2=pickle.load(open('best_model2.pkl','rb'))
vectorizer=pickle.load(open("vectorizer.pickle", 'rb'))  
import streamlit as st
import streamlit.components.v1 as components
import streamlit as st
import streamlit as st
from re import I
import re
import pyshorteners as sh
import requests
from bs4 import BeautifulSoup as bs
import pandas as pd
import numpy as np
import csv
import urllib.request as urllib
from urllib.request import urlopen
from pprint import pprint
from hijri_converter import Hijri, Gregorian

import re


import streamlit as st

with open('style.css') as f:
    st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)


url2='https://www.saudiexchange.sa/wps/portal/saudiexchange/newsandreports/issuer-news/issuer-announcements/issuer-announcements-details/!ut/p/z1/lU_dboIwFH4WL3ZpesBR2CUbG7KgguBUbkwtnZIIJaW4uacfBbNEp4sjJek5_X5RghYoKcg-2xCZ8YLsmnmZ4JVhY9CHFkxgGD8BhsjFs9jSHB2j-SnAGrkYwrEdTnTTAPcNUPIvPkSBAeFLMBr4MAUX8G18uPLZN_gnp5CJ_WA2CV4fXc-3BxY2zwEXKp6Z_O7QAv4IGRHRNjGvRvU0NBes4rWgDE03TNpFweuCspwV0s8q6RBJUEgJ3TKf7dkuIBuGImVN0n1WcVEpFFqqTcWo5MIpvfQ4H_I136Glpmv3ai4FT2sq40PJOoDkjXxzB-MO1K-DrrXAxiTKvpoXDTphIui24_0AxnW-ZsJZH1fvgueX1YiqRFWho8Kq3-4l-5RRq9ylSRt2wETGVfwyn80WkHl96-N5kFunx-71vgHQvrvV/dz/d5/L0lHSklna0tDbEVKSUtJS1VRb2dwUkEhIS9vSHdRQUVNSUFBQ0VFaGdDS000emxHWUVLVWxTVUtXdEcwWVdnQSEhLzRKQ2lqc1lwTWhUalVFNWxFbXQyVXR0TlF6VzdLVzFtbzVBIS9aN181QTYwMkg4ME8wSFRDMDYwU0c2VVQ4MURRNi9aNl81QTYwMkg4ME8wSFRDMDYwU0c2VVQ4MURJMy9jb21wYW55U3ltYm9sLzEyMTQvZ2xvYmFsL2h0dHA6JTAlMHRhZGF3dWwlMC9hbm5DYXQvMS9hbklkLzYzMjU0/'


url='https://www.saudiexchange.sa/wps/portal/saudiexchange/newsandreports/issuer-news/issuer-announcements/issuer-announcements-details/!ut/p/z1/lY9fb4IwFMU_ix_A9AKxsEc2NmRBBZEpvJhaOm0ilJTi5j79-OOy6Obimj70Nr9z7jkoRSuUFuTAt0RxUZB9MycpXo9sDPrYghmMFw-AIXJxvLA0R8doeQ5YExdDOLXDmW6OwH0BlP5LD1EwgvApmBg-zMEFfJserhz7hv3pOTKz78wmwfO96_m2YWHzEvil4sWSnx064I-QEZFdE_NqVE9DS8kqUUvK0HzLlF0Uoi4oy1mhfF4phyiCQkrojvnswPYB2TIUtatJduCVkFVLoaT9qRhVQjqll53mY74Re5ToYHRtSimymqrFsWQ9oERjf3qXjXHEP5pJg96MSLrrWe0LmNb5hklng5Ku_asU-bcDaaPTNvhJtR52QsXeVdS59VzWKAImuWhjlnkcr4B7Q-vt0cit82sPBp_yQkJB/dz/d5/L0lHSklna0tDbEVKSUtJS1VRb2dwUkEhIS9vSHdRQUVNSUFBQ0VFaGdDS000emxHWUVLVWxTVUtXdEcwWVdnQSEhLzRKQ2lqc1lwTWhUalVFNWxFbXQyVXR0TlF6VzdLVzFtbzVBIS9aN181QTYwMkg4ME8wSFRDMDYwU0c2VVQ4MURRNi9aNl81QTYwMkg4ME8wSFRDMDYwU0c2VVQ4MURJMy9jb21wYW55U3ltYm9sLzIwMzAvZ2xvYmFsL2h0dHA6JTAlMHRhZGF3dWwlMC9hbm5DYXQvMS9hbklkLzYzMjY1/'

# urls = [url, 'https://www.saudiexchange.sa/wps/portal/saudiexchange/newsandreports/issuer-news/issuer-announcements/issuer-announcements-details/!ut/p/z1/lY_dcoIwEIWfxQdwsugY6GVaWqSDCuIvN04MqWZGCBOCrX36BrHtaGvHZnKxe-bbs2dRghYoyelebKgWMqc70y8TvOoRDJ2-AyPoTx4AQ-zh6cSx3A5G83PAGXgYoiGJRh27B94MUPKveYjDHkRP4aAbwBg8wLfNw5VHbtifnCMjcmebBM_3nh-QroPtS-CXEy-W_LzhCPwRMqbqeIl9NapvobnipawU42i84ZrkuaxyxjOe60CU2qWaoohRtuUB3_NdSDccxfVqmu5FKVVZU2jZtmqt5ExL5RZ--qUcsrXcoWVdF0qmFdOTQ8EbQUtjf6oLYxyLd9NZ0FhRxbYNa30Cwypbc-WuT9KLktm3A62jszr4aWrVRND8TcdHt4ZLzUTIlZAmJCqy6XQBwm87r4_dzDn_pNX6ANBF7g4!/dz/d5/L0lHSklna0tDbEVKSUtJS1VRb2dwUkEhIS9vSHdRQUVNSUFBQ0VFaGdDS000emxHWUVLVWxTVUtXdEcwWVdnQSEhLzRKQ2lqc1lwTWhUalVFNWxFbXQyVXR0TlF6VzdLVzFtbzVBIS9aN181QTYwMkg4ME8wSFRDMDYwU0c2VVQ4MURRNi9aNl81QTYwMkg4ME8wSFRDMDYwU0c2VVQ4MURJMy9jb21wYW55U3ltYm9sLzk1MzEvZ2xvYmFsL2h0dHA6JTAlMHRhZGF3dWwlMC9hbm5DYXQvMS9hbklkLzc3NzEz/']

# for url in urls:
#     processed_url = url
  

def process_data(soup):

    with open('1comp.csv', 'a', encoding='utf-8-sig') as file:
        writer = csv.writer(file)
        writer.writerow(['Date', 'numUp', 'numDown', 'TITLE', 'table1', 'table2', 'main_Article'])

        tables = soup.find_all('table', {'class': 'stacktable large-only'})

        if len(tables) in (1, 8):
            main_Article = extract_mainArticle(soup)
            writer.writerow(['', '', '', '', '', '', main_Article])  # Fill other fields if needed

        elif len(tables) == 4:
            table1, table2, table3, table4 = [table.get_text().strip() for table in tables]
            main_Article = extract_mainArticle(soup)

            if 'الخسائر المتراكمة' in table3:
                if 'توضيح' in table4:
                    main_Article = extract_mainArticle(soup)  # Update if needed

                if 'ربحية (خسارة) السهم' in table2:
                    extract_table2(soup)

                if 'ربحية (خسارة) السهم' not in table1 and table3 and table4:
                    extract_table1(soup)

            elif 'إجمالي حقوق المساهمين (بعد استبعاد حقوق الأقلية)' in table2:
                if 'ربحية (خسارة) السهم' in table2:
                    extract_table2(soup)

                if 'ربحية (خسارة) السهم' not in table1 and table3:
                    extract_table1(soup)


        elif len(soup.find_all('table', {'class': 'stacktable large-only'}))==3:
           
            table1=soup.findAll('table')[0].get_text().strip()
            table2=soup.findAll('table')[1].get_text().strip()
            table3=soup.findAll('table')[2].get_text().strip()
            main_Article=extract_mainArticle(soup)
            if 'إجمالي حقوق المساهمين (بعد استبعاد حقوق الأقلية)' in table2:
                if 'ربحية (خسارة) السهم' in table2 :
                    extract_table2(soup)
                if 'ربحية (خسارة) السهم' not in table1 and  table3:
                    extract_table1(soup)


        elif len(soup.find_all('table', {'class': 'stacktable large-only'}))==2:
            table1=soup.findAll('table')[0].get_text().strip()
            table2=soup.findAll('table')[1].get_text().strip()
            main_Article=extract_mainArticle(soup)
            if 'السنة الحالية' in table1:
                if 'ربحية (خسارة) السهم' in table1 :
                    for row in soup.findAll('table')[0].tbody.findAll('tr'):
                        extract_table2_currentYear(soup)

            elif  'ربحية (خسارة) السهم' in table2 :
                for row in soup.findAll('table')[1].tbody.findAll('tr'):
                    extract_table2_currentYear(soup)


    def joinText_table1():
        if  'السنة الحالية' in soup.findAll('table')[0].get_text().strip()and len(column1)>3:
            main1={'السنة الحالية':column1,'السنة الماضية ':column2}

        elif len(column1_table1)> 3 and len(column2_table1)> 3 and len(column3_table1)> 3 and len(column4_table1)> 3 and len(column5_table1)> 3:
            main1={'الربع الحالي':column1_table1,'الربع المماثل من العام السابق':column2_table1,'التغير%':column3_table1,'الربع السابق':column4_table1,
                    '	التغير %':column5_table1}
        else:
            main1='Null'
        return main1


       
    def joinText_table2():
        if  'السنة الحالية' in soup.findAll('table')[0].get_text().strip():
            main2='Null'
        elif len(column1)>=2:
            main2={'الفترة الحالية':column1,'الفترة المماثلة من العام السابق':column2}
        else:
            main2='Null'
        return main2 
    
    def extract_title():
        title=soup.findAll('h3', {})[1].text
        result_title = ''.join([i for i in removespace(title)])
        return result_title
          
    def extract_date():
        time_stamp = soup.findAll('div', {'class': 'date'})[1].get_text().strip()
        year=time_stamp[0:4]
        month=time_stamp[5:7]
        day=time_stamp[8:10]
        Date = str(Hijri(int(year), int(month), int(day)).to_gregorian())
        return Date
        
    return extract_date(),extract_title(),joinText_table1(),joinText_table2(),str(main_Article)


def is_what_percent_of(currentـquarter,same_previousـquarter):
  if numbers_processing(currentـquarter) and numbers_processing(same_previousـquarter) and numbers_processing(currentـquarter) !=0:
    return str(round(((numbers_processing(currentـquarter) - numbers_processing(same_previousـquarter)) / numbers_processing(same_previousـquarter))*100))+"٪"


def numbers_processing(text):

  text = text.rstrip(text[-1])
  text = text.replace("'", '')
  text = text.replace("-", '')
  if '-'  in text:
    text = text.replace("-", '')
  if ',' not in text:
    return round(float(((text))))

  if text[0]=='"':
    text=text.replace('"','')
  if text[0]=="[":
    text=text.replace('[','')
  if text[1]==',':
    return round(float(((text[0:1]))))
  if text[2]==',':
    return round(float(((text[0:2]))))
  elif text[3]==',':
    text=text.replace(',','')
    return round(float(((text[0:3]))))


  elif text[2]=='.':
    text = text.rstrip(text[0:1])
    text = text.rstrip(text[-2:-1])

    return round(float(((text[0:2]))))
  elif text[1]=='.':
    text = text.rstrip(text[0:1])
    text = text.rstrip(text[-2:-1])

    return round(float(((text[0:2]))))

  else:
    print(text)
    return round(float(((text))))

def isNaN(string):
    return string != string
def is_what_percent_of2(currentـquarter,same_previousـquarter):
  if numbers_processing_table2(same_previousـquarter)!=0:
    return str(round((numbers_processing_table2(currentـquarter) - numbers_processing_table2(same_previousـquarter)) / numbers_processing_table2(same_previousـquarter)*100))+'٪'


def textClean(text):
  if "\'" in text:
    text=text.replace('\'','')
  if '٪, ' in text:
    text=text.replace('٪,','٪')
  if '"' in text:
    text=text.replace('"','')
  if '",' in text:
    text=text.replace('",','')
  if '}' and '{' in text:
    text=text.replace('}','')
    text=text.replace('{','')
  if 'بنسبة,' in text:
    text=text.replace('بنسبة,','بنسبة')
  if ',ريال' in text:
    text=text.replace(',ريال','ريال')
  if '--' in text:
    text=text.replace('--','')
  if '1213 -1.04 %' in text:
    text=text.replace('1213 -1.04 %','')
  if ')' and '(' in text:
    text=text.replace(')','')
    text=text.replace('(','')
  return text

# worldlist_title=textClean(text).split()
# title_months=''.join(worldlist_title[-2:])

def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)

def numbers_processing_table2(text):
  text = text.rstrip(text[-1])
  text = text.replace("'", '')
  text = text.replace("[", '')
  text = text.replace("]", '')
  text = text.replace("}", '')
  # text = text.replace("-", '')

  text=text.replace('"','')
  if len(text)<=2:
    text = text.replace(",", '')
    return (float(((text))))
  if text[1]=='.':

    # text = text.rstrip(text[-1])
    return (float(((text))))
  if text[0]=='"':
    text=text.replace('"','')
    return (float(((text))))

  if ',' in text and len(text)>6:
    text = text.replace(",", '.')
    return (float(((text[0:5]))))
  if ',' in text and len(text)<5:
    text = text.replace(",", '.')
    return (float(((text[0:3]))))
  else:
    text = text.replace(",", '.')
    return (float(((text))))


def table_table2(label2,title,table1,table2,main_Article1):


    table1withdate=''
    table3 = ' '
    table6=' '
    table1=str(table1)
    table2=str(table2)
    main_Article1=str(main_Article1)

    if (label2=='اعلان خسائر'):
      if table1=='0' and table2=='0':
        table1withdate = ' '
      elif table1!='0'and len(table1)<376:
        senten=table1
        words = senten.split()
      if words[0]=="{'السنة":
        if (len(words)>20):
          currentـquarter=words[6]
          same_previousـquarter=words[21]
        else:
          currentـquarter=words[5]
          same_previousـquarter=words[15]
        replaceTo_negative='  ربح  '

        if currentـquarter[1]=='-':
          replaceTo_negative=' خسارة '
        change=is_what_percent_of2((currentـquarter), (same_previousـquarter))
        firstsen_po=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة ) مبلغ {currentـquarter} ريال ومقارنة للسنة الماضية  قد زاد صافي الربح بنسبة {(change)} ')

        firstsen=(f'بلغ صافي  الخسارة ( بعد الزكاة والضريبة) مبلغ {currentـquarter} ريال ومقارنة للسنة الماضية  قد زاد صافي الخسارة بنسبة{(change)}')
        seconif=(f'بلغ صافي  الخسارة ( بعد الزكاة والضريبة) مبلغ {currentـquarter}ريال ومقارنة للسنة الماضية  قد نقص صافي الخسارة بنسبة{(change)}')

        if((same_previousـquarter)<(currentـquarter)):
          table1withdate=firstsen
        elif ((same_previousـquarter)>(currentـquarter)):
          table1withdate=seconif
      elif words[0]=="{'الربع":

        currentـquarter=words[5]
        change2=words[21][0:4]
        previousـquarter=words[28]
        same_previousـquarter=words[15]
        change=words[-2][0:4]
        change_previousـquarter=words[-2][0:4]

        if change2 =="'-',":
          change2=' خسارة بمقدار'+same_previousـquarter
        elif change2[1]=='-':
          change2="تغيير بنسبة"+words[21][0:4]+"٪"
        else:
          change2="تغيير بنسبة"+words[21][0:4]+"٪"

        if change =="'-',":
          change=' خسارة بمقدار'+previousـquarter
        elif change2[1]=='-':
          change2="تغيير بنسبة"+words[-2][0:4]+"٪"
        else:
          change="تغيير بنسبة"+words[-2][0:4]+"٪"

        replaceTo_negative=' ربح '

        if currentـquarter[1]=='-':
          replaceTo_negative='خسارة'
        firstsen=(f'بلغ صافي الخسارة ( بعد الزكاة والضريبة) مبلغ {currentـquarter} ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')
        seconif=(f'بلغ صافي الخسارة ( بعد الزكاة والضريبة) مبلغ {currentـquarter}ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')
        firstsen_po=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة) مبلغ {currentـquarter}  ريال ومقارنة بالربع المماثل من العام السابق    قد حققت الشركة {(change2)}')
        seconif_po=(f'بلغ صافي الربح ( بعد الزكاة والضريبة) مبلغ {currentـquarter}ريال ومقارنة بالربع المماثل من العام السابق   قد حققت الشركة   {(change2)}')
        secondsen=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة   {(change)}')
        secondsen_po=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة  {(change)}')
        thirdsen=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة   {(change)}')
        thirdsen_po=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة   {(change)}')
        seconif_firstsen=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة   {(change)}')
        secondif_secondsen=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة   {(change)}')
        seconif_firstsen=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة   {(change)}')
        secondif_secondsen=(f'أيضا،ومقارنة بالربع السابق  قد حققت الشركة   {(change)}')

        if (currentـquarter)=='0':

          if((same_previousـquarter)<(currentـquarter)):
            table1withdate=firstsen
            if ((previousـquarter)< (currentـquarter)):
              table1withdate=firstsen+secondsen
            elif ((previousـquarter)>(currentـquarter)):
              table1withdate=firstsen+thirdsen
          elif ((same_previousـquarter)>(currentـquarter)):
            table1withdate=seconif
            if ((previousـquarter)< (currentـquarter)):
              table1withdate=seconif+seconif_firstsen
            elif ((previousـquarter)>(currentـquarter)):
              table1withdate=seconif+secondif_secondsen

        elif (currentـquarter)!='0':

          if((same_previousـquarter)<(currentـquarter)):
            table1withdate=firstsen_po
            if ((previousـquarter)< (currentـquarter)):
              table1withdate=firstsen_po+secondsen_po
            elif ((previousـquarter)>(currentـquarter)):
              table1withdate=firstsen_po+thirdsen
          elif ((same_previousـquarter)>(currentـquarter)):
            table1withdate=seconif_po
            if ((previousـquarter)< (currentـquarter)):
              table1withdate=seconif_po+seconif_firstsen
            elif ((previousـquarter)>(currentـquarter)):
              table1withdate=seconif_po+secondif_secondsen

    elif (label2=='اعلان أرباح '):
      if table1=='Null' and table2=='Null':
        table1withdate = ' '
      elif table1!='Null'and len(table1)<376:
        senten=str(table1)
        words = senten.split()
        if words[0]=="{'السنة":
            if (len(words)>20):
              currentـquarter=words[6]
              same_previousـquarter=words[21]
              change=is_what_percent_of2((currentـquarter), (same_previousـquarter))

            else:
              currentـquarter=words[5]
              same_previousـquarter=words[15]
              change=is_what_percent_of2((currentـquarter), (same_previousـquarter))
            
            firstsen=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة )   مبلغ {currentـquarter} ريال ومقارنة للسنة الماضية  قد زاد صافي الربح بنسبة {(change)} ')
            seconif=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة  )   مبلغ {currentـquarter}ريال ومقارنة للسنة الماضية  قد نقص صافي الربح بنسبة {(change)}')
            equalsen=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة  )   مبلغ {numbers_processing(currentـquarter)}   ريال ومقارنة بالربع المماثل من العام السابق  قد كان نفسه')
            if((same_previousـquarter)<(currentـquarter)):
              table1withdate=firstsen
            elif ((same_previousـquarter)>(currentـquarter)):
              table1withdate=seconif
            elif((same_previousـquarter)==(currentـquarter)):
              table1withdate=equalsen
        elif words[0]=="{'الربع":
          currentـquarter=words[5]
          same_previousـquarter=words[15]
          previousـquarter=words[28]
          change2=words[21][0:4]
          change=words[-2][0:4]
          if change2 =="'-'," :
            change2=' خسارة بمقدار'+same_previousـquarter
          elif change2[1]=='-':
            change2=" تغيير  بنسبة"+words[21][0:4]+"٪"
          else:
            change2="تغيير بنسبة"+words[21][0:4]+"٪"
          #  print(index)

          if change =="'-',":
            change=' خسارة بمقدار'+previousـquarter
          elif change[1]=='-':
            change="تغيير بنسبة"+words[-2][0:4]+"٪"

          else:
            change="تغيير بنسبة"+words[-2][0:4]+"٪"
          # print(index,change)

          replaceTo_negative=' ربح '
          if currentـquarter[1]=='-':
            replaceTo_negative='خسارة'
          firstsen=(f'بلغ صافي {replaceTo_negative}(بعد الزكاة والضريبة  ) مبلغ {currentـquarter} ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')

          secondsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة  {(change)}')
          thirdsen=(f'ومقارنة بالربع السابق   قد حققت الشركة  {(change)}')
          seconif=(f'بلغ صافي {replaceTo_negative}(بعد الزكاة والضريبة  ) مبلغ {(currentـquarter)}  ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')
          seconif_firstsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة  {(change)}')
          secondif_secondsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة  {(change)}')
          equalsen=(f'بلغ صافي الربح ( بعد الزكاة والضريبة) مبلغ  {numbers_processing(currentـquarter)}   ريال ومقارنة بالربع المماثل من العام السابق  قد كان نفسه')
          
          if((same_previousـquarter)<(currentـquarter)):
            table1withdate=firstsen
            if ((previousـquarter)< (currentـquarter)):
              table1withdate=firstsen+secondsen
            elif ((previousـquarter)>(currentـquarter)):
              table1withdate=firstsen+thirdsen
          elif((same_previousـquarter)==(currentـquarter)):
            table1withdate=equalsen

            if ((previousـquarter)< (currentـquarter)):
              table1withdate=firstsen+secondsen
            elif ((previousـquarter)>(currentـquarter)):
              table1withdate=firstsen+thirdsen
          elif ((same_previousـquarter)>(currentـquarter)):
            table1withdate=seconif
            if ((previousـquarter)< (currentـquarter)):
              table1withdate=seconif+seconif_firstsen

            elif ((previousـquarter)>(currentـquarter)):
              table1withdate=seconif+secondif_secondsen



    if (label2=='اعلان خسائر'):
        if table1=='0' and table2=='0':
          table1withdate = ' '
        elif table1!='0' and len(table1)>=376:
          senten=table1
          words = senten.split()
          if words[0]=="{'السنة":
            currentـquarter=words[6]
            change2=words[23][0:4]
            previousـquarter=words[29]
            same_previousـquarter=words[17]
            change=words[-3][0:4]
            replaceTo_negative=' ربح '
            if currentـquarter[1]=='-':
              replaceTo_negative=' خسارة '
            firstsen_po=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة) مبلغ {currentـquarter} ريال ومقارنة بالربع المماثل من العام السابق  قد زاد صافي الربح بنسبة {(change)}')
            firstsen=(f'بلغ صافي الخسارة (بعد الزكاة والضريبة) مبلغ {currentـquarter} ريال ومقارنة للسنة الماضية  قد زاد صافي الخسارة بنسبة{(change)}')
            seconif=(f'بلغ صافي الخسارة (بعد الزكاة والضريبة) مبلغ {currentـquarter}ريال ومقارنة للسنة الماضية  قد نقص صافي الخسارة بنسبة{(change)}')
            if(numbers_processing(same_previousـquarter)<numbers_processing(currentـquarter)):
              table1withdate=firstsen
            elif (numbers_processing(same_previousـquarter)>numbers_processing(currentـquarter)):
              table1withdate=seconif
        elif words[0]=="{'الربع":
            if len(table1)==399:
              change2=words[33][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change2[1]=='-':
                change2=" تغيير بنسبة"+words[33][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[33][0:4]+"٪"
            else:
              change2=words[23][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change2[1]=='-':
                change2="تغيير بنسبة"+words[23][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[23][0:4]+"٪"

            if words[8][1:7]=="الربع":
              currentـquarter=words[6]
              same_previousـquarter=words[17]
              previousـquarter=words[32]
              change2=words[24][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change2[1]=='-':
                change2="تغيير بنسبة"+words[24][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[24][0:4]+"٪"
              if change =="'-',":
                change=' خسارة بمقدار'+previousـquarter
              elif change[1]=='-':
                change="تغيير بنسبة"+words[-2][0:4]+"٪"
              else:
                change="تغيير بنسبة"+words[-2][0:4]+"٪"
            else:
              currentـquarter=words[5]
              change2=words[21][0:4]
              previousـquarter=words[28]
              same_previousـquarter=words[15]
              change=words[-2][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change2[1]=='-':
                change2="تغيير  بنسبة"+words[21][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[21][0:4]+"٪"
              if change =="'-',":
                change=' خسارة بمقدار'+previousـquarter
              elif change[1]=='-':
                change=" تغيير بنسبة"+words[-2][0:4]+"٪"
              else:
                change="تغيير بنسبة"+words[-2][0:4]+"٪"
              replaceTo_negative=' ربح '

            if currentـquarter[1]=='-':
              replaceTo_negative=' خسارة '


            firstsen=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة) مبلغ {currentـquarter} ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')
            seconif=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة) مبلغ {currentـquarter}ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')
            firstsen_po=(f'بلغ صافي الربح (بعد الزكاة والضريبة) مبلغ {currentـquarter}  ريال ومقارنة بالربع المماثل من العام السابق    قد حققت الشركة {(change2)}')
            seconif_po=(f'بلغ صافي الربح (بعد الزكاة والضريبة) مبلغ {currentـquarter}ريال ومقارنة بالربع المماثل من العام السابق   قد حققت الشركة   {(change2)}')
            secondsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            secondsen_po=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            thirdsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            thirdsen_po=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            seconif_firstsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            secondif_secondsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            seconif_firstsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            secondif_secondsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')

            if numbers_processing(currentـquarter)=='0':
              if(numbers_processing(same_previousـquarter)<numbers_processing(currentـquarter)):
                table1withdate=firstsen
              if (numbers_processing(previousـquarter)< numbers_processing(currentـquarter)):
                table1withdate=firstsen+secondsen
              elif (numbers_processing(previousـquarter)>numbers_processing(currentـquarter)):
                table1withdate=firstsen+thirdsen
            elif (numbers_processing(same_previousـquarter)>numbers_processing(currentـquarter)):
              table1withdate=seconif
              if (numbers_processing(previousـquarter)< numbers_processing(currentـquarter)):
                table1withdate=seconif+seconif_firstsen
              elif (numbers_processing(previousـquarter)>numbers_processing(currentـquarter)):
                table1withdate=seconif+secondif_secondsen

            elif (currentـquarter)!='0':
              if((same_previousـquarter)<(currentـquarter)):
                table1withdate=firstsen_po
                if ((previousـquarter)< (currentـquarter)):
                  table1withdate=firstsen_po+secondsen_po
                elif ((previousـquarter)>(currentـquarter)):
                  table1withdate=firstsen_po+thirdsen
              elif ((same_previousـquarter)>(currentـquarter)):
                table1withdate=seconif_po
                if ((previousـquarter)< (currentـquarter)):
                  table1withdate=seconif_po+seconif_firstsen
                elif ((previousـquarter)>(currentـquarter)):
                  table1withdate=seconif_po+secondif_secondsen

    elif (label2=='اعلان أرباح '):
        if table1=='0' and table2=='0':
          table1withdate = ' '
        elif table1!='0'and len(table1)>=376:
          senten=table1
          words = senten.split()
          if words[0]=="{'السنة":
            currentـquarter=words[6]
            previousـquarter=words[29]
            same_previousـquarter=words[17]
            change=words[-3][0:4]
            replaceTo_negative=' ربح '
            if currentـquarter[1]=='-':
              replaceTo_negative=' خسارة '
              firstsen=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة) مبلغ {currentـquarter} ريال ومقارنة للسنة الماضية  قد زاد صافي الربح بنسبة {(change)} ')
              seconif=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة) مبلغ {currentـquarter}ريال ومقارنة للسنة الماضية  قد نقص صافي الربح بنسبة {(change)}')
              equalsen=(f'بلغ صافي {replaceTo_negative} (بعد الزكاة والضريبة) مبلغ  {numbers_processing(currentـquarter)}   ريال ومقارنة بالربع المماثل من العام السابق  قد كان نفسه')

            if((same_previousـquarter)<(currentـquarter)):
              table1withdate=firstsen
            elif ((same_previousـquarter)>(currentـquarter)):
              table1withdate=seconif
            elif((same_previousـquarter)==(currentـquarter)):
              table1withdate=equalsen
          elif words[0]=="{'الربع":
            if len(table1)==399:
              change2=words[33][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change[1]=='-':
                change=" تغيير بنسبة"+words[33][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[33][0:4]+"٪"
            else:
              change2=words[23][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change2[1]=='-':
                change2="تغيير التغيير بنسبة"+words[23][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[23][0:4]+"٪"
            if words[8][1:7]=="الربع":
              currentـquarter=words[6]
              same_previousـquarter=words[17]
              previousـquarter=words[32]
              change2=words[24][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change2[1]=='-':
                change2=" تغيير  بنسبة"+words[24][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[24][0:4]+"٪"
              if change =="'-',":
                change=' خسارة بمقدار'+previousـquarter
              elif change2[1]=='-':
                change2=" تغيير بنسبة"+words[-2][0:4]+"٪"
              else:
                change="تغيير بنسبة"+words[-2][0:4]+"٪"
            else:
              currentـquarter=words[5]
              change2=words[21][0:4]
              previousـquarter=words[28]
              same_previousـquarter=words[15]
              change=words[-2][0:4]
              if change2 =="'-',":
                change2=' خسارة بمقدار'+same_previousـquarter
              elif change2[1]=='-':
                change2="تغيير بنسبة"+words[21][0:4]+"٪"
              else:
                change2="تغيير بنسبة"+words[21][0:4]+"٪"
              if change =="'-',":
                change=' خسارة بمقدار'+previousـquarter
              elif change2[1]=='-':
                change2="تغيير بنسبة"+words[-2][0:4]+"٪"
              else:
                change="تغيير بنسبة"+words[-2][0:4]+"٪"
            replaceTo_negative=' ربح '
            if currentـquarter[1]=='-':
              replaceTo_negative=' خسارة '
            firstsen=(f'بلغ صافي {replaceTo_negative}(بعد الزكاة والضريبة) مبلغ {currentـquarter} ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')
            secondsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            thirdsen=(f'ومقارنة بالربع السابق   قد حققت الشركة  {(change)}')
            seconif=(f'بلغ صافي {replaceTo_negative}(بعد الزكاة والضريبة) مبلغ {(currentـquarter)}  ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة  {(change2)}')
            seconif_firstsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            secondif_secondsen=(f'أيضا، ومقارنة بالربع السابق قد حققت الشركة {(change)}')
            equalsen=(f'بلغ صافي الربح(بعد الزكاة والضريبة) مبلغ  {(currentـquarter)}   ريال ومقارنة بالربع المماثل من العام السابق  قد كان نفسه')
            if((same_previousـquarter)<(currentـquarter)):
              table1withdate=firstsen
              if ((previousـquarter)< (currentـquarter)):
                table1withdate=firstsen+secondsen
              elif ((previousـquarter)>(currentـquarter)):
                table1withdate=firstsen+thirdsen
            elif((same_previousـquarter)==(currentـquarter)):
              table1withdate=equalsen
              if ((previousـquarter)< (currentـquarter)):
                table1withdate=firstsen+secondsen
              elif ((previousـquarter)>(currentـquarter)):
                table1withdate=firstsen+thirdsen
            elif ((same_previousـquarter)>(currentـquarter)):
              table1withdate=seconif
              if ((previousـquarter)< (currentـquarter)):
                table1withdate=seconif+seconif_firstsen
              elif ((previousـquarter)>(currentـquarter)):
                table1withdate=seconif+secondif_secondsen
    def is_what_percent_of2(currentـquarter,same_previousـquarter):
      if numbers_processing_table2(same_previousـquarter)!=0:
        return str(round((numbers_processing_table2(currentـquarter) - numbers_processing_table2(same_previousـquarter)) / numbers_processing_table2(same_previousـquarter)*100))+'٪'
      
    if (label2=='اعلان أرباح '):
      if table2=='0' :
        table3 = ' '
      else:
        senten=table2
        words = senten.split()
        if len(words)==19:
          currentـquarter=words[7]
          same_previousـquarter=words[18]
          if same_previousـquarter[0]=="-":
            print(same_previousـquarter)
        elif len(words)==31:
          currentـquarter=words[13]
          same_previousـquarter=words[30]
        elif len(words)==11:
          currentـquarter=words[3]
          same_previousـquarter=words[10]
        else:
          currentـquarter=words[8]
          same_previousـquarter=words[20]

        if same_previousـquarter=="-":
          change=str(numbers_processing_table2(same_previousـquarter))+"٪"
        elif same_previousـquarter[1]=='-':
          change=str(numbers_processing_table2(same_previousـquarter))+"٪"
        else:
          change="ربح "+is_what_percent_of2((currentـquarter), (same_previousـquarter))
        
        firstsen=(f'ربحية السهم بلغت {numbers_processing_table2(currentـquarter)}ريال ومقارنة ب{numbers_processing_table2(same_previousـquarter)}ريال للفترة المماثلة من العام السابق قد حققت نسبة {change}')
        seconif=(f'ربحية السهم بلغت {numbers_processing_table2(currentـquarter)}ريال ومقارنة ب{numbers_processing_table2(same_previousـquarter)}ريال للفترة المماثلة من العام السابق قد حققت نسبة {change}')

        equalsen=(f'ربحية السهم بلغت {numbers_processing_table2(currentـquarter)}ريال ومقارنة ب{numbers_processing_table2(same_previousـquarter)}ريال للفترة المماثلة من العام السابق  قد كانت نفسها')
        
        if((same_previousـquarter)<(currentـquarter)):
          table3=firstsen
        
        elif((same_previousـquarter)==(currentـquarter)):
          table3=firstsen
        elif ((same_previousـquarter)>(currentـquarter)):
          table3=seconif



    elif (label2=='اعلان خسائر'):
      if table2=='0':
        table3 = ' '
      else:
        senten=table2
        words = senten.split()
        if len(words)==19:
          currentـquarter=words[7]
          same_previousـquarter=words[18]
        elif len(words)==31:
          currentـquarter=words[13]
          same_previousـquarter=words[30]

        elif len(words)==11:
          currentـquarter=words[3]
          same_previousـquarter=words[10]
        else:
          currentـquarter=words[8]
          same_previousـquarter=words[20]


        if same_previousـquarter=="-":
          change=str(numbers_processing_table2(same_previousـquarter))+"٪"
        elif same_previousـquarter[1]=='-':
          change=str(numbers_processing_table2(same_previousـquarter))+"٪"
        else:
          change=is_what_percent_of2((currentـquarter), (same_previousـquarter))

        firstsen=(f'خسارة السهم بلغت {numbers_processing_table2(currentـquarter)}ريال مقارنة ب{numbers_processing_table2(same_previousـquarter)}ريال للفترة المماثلة من العام السابق  قد حققت نسبة  {change}')
        seconif=(f'خسارة السهم بلغت {numbers_processing_table2(currentـquarter)}ريال مقارنة ب{numbers_processing_table2(same_previousـquarter)}ريال للفترة المماثلة من العام السابق قد حققت نسبة {change}')
        equalsen=(f'خسارة السهم بلغت {numbers_processing_table2(currentـquarter)} ريال مقارنة ب{numbers_processing_table2(same_previousـquarter)}ريال للفترة المماثلة من العام السابق  قد كانت نفسها ')
        
        if(numbers_processing_table2(same_previousـquarter)<numbers_processing_table2(currentـquarter)):
          table3=firstsen
          
        elif(numbers_processing_table2(same_previousـquarter)==numbers_processing_table2(currentـquarter)):
          table3=equalsen
        elif (numbers_processing_table2(same_previousـquarter)>numbers_processing_table2(currentـquarter)):
          table3=seconif
          
    if (label2=='اعلان أرباح '):
      if table2=='0':
        table6 = ' '
      elif  table2.split()[5]!='المماثلة':
        senten=table2
        words = senten.split()
        if len(words)==31:
          currentـquarter=words[6]
          same_previousـquarter=words[23]
        if str(numbers_processing_table2(same_previousـquarter))[0]=='-':
          change='خسارة بمقدار'+is_what_percent_of2((currentـquarter), (same_previousـquarter))
        else:
          change=is_what_percent_of2((currentـquarter), (same_previousـquarter))
      else:

        currentـquarter=words[5]
        same_previousـquarter=words[17]

        if str(numbers_processing_table2(same_previousـquarter))[0]=='-':
          change='خسارة بمقدار'+same_previousـquarter
        else:
          change=is_what_percent_of2((currentـquarter), (same_previousـquarter))

        # if str(numbers_processing_table2(same_previousـquarter))[0]=='-':
        #   currentـquarter='خسارة بمقدار'+str(numbers_processing_table2(same_previousـquarter))
        # else:
        #   currentـquarter='ربح بنسبة'+same_previousـquarter

      equalsen=(f'بلغ صافي الربح(بعد الزكاة والضريبة) مبلغ{(currentـquarter)}ريال ومقارنة بالربع المماثل من العام السابق  قد كان نفسه')
      firstsen=(f'بلغ صافي الربح (بعد الزكاة والضريبة) مبلغ{(currentـquarter)}ريال ومقارنة بالربع المماثل من العام السابق  قد  حققت الشركة { (change)}')
      seconif=(f'بلغ صافي الربح (بعد الزكاة والضريبة) مبلغ{(currentـquarter)}ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة{ (change)}')

      if((same_previousـquarter)<(currentـquarter)):
        table6=firstsen

      elif ((same_previousـquarter)>(currentـquarter)):
        table6=seconif
      elif((same_previousـquarter)==(currentـquarter)):
        table6=equalsen


    elif (label2=='اعلان خسائر'):
      if table2=='0':
        table6 = ' '
      elif table2.split()[5]!='المماثلة':
        senten=table2
        words = senten.split()
        if len(words)==31:
          currentـquarter=words[6]
          same_previousـquarter=words[23]
          if str((same_previousـquarter))[0]=='-':
            change='خسارة بمقدار'+same_previousـquarter
          else:
            change=is_what_percent_of2((currentـquarter), (same_previousـquarter))
        else:
          currentـquarter=words[5]
          same_previousـquarter=words[17]
          if str(numbers_processing_table2(same_previousـquarter))[0]=='-':
            change='خسارة بمقدار'+same_previousـquarter
          else:
            change=is_what_percent_of2((currentـquarter), (same_previousـquarter))

        firstsen=(f'بلغ صافي الخسارة (بعد الزكاة والضريبة) مبلغ{(currentـquarter)}ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة   { (change)}')
        seconif=(f'بلغ صافي الخسارة (بعد الزكاة والضريبة) مبلغ{(currentـquarter)}ريال ومقارنة بالربع المماثل من العام السابق  قد حققت الشركة { (change)}')
        equalsen=(f'بلغ صافي الخسارة(بعد الزكاة والضريبة) مبلغ{(currentـquarter)}ريال ومقارنة بالربع المماثل من العام السابق  قد كان نفسه')
        
        if(numbers_processing(same_previousـquarter)<numbers_processing(currentـquarter)):
          table6=firstsen
        elif (numbers_processing(same_previousـquarter)>numbers_processing(currentـquarter)):
          table6=seconif
        
        elif(numbers_processing(same_previousـquarter)==numbers_processing(currentـquarter)):
          table6=equalsen

    table1_result=table1withdate
    table2_result=table3
    table2_result2=table6
    
    if type(main_Article1) !=str:
      articlesummry=main_Article1
    else:
      articlesummry=main_Article1

    title=str(title)
    Title=textClean("\n"+str(title))
    Title=str(Title)
    worldlist_title=textClean(title).split()
    title_months="("+' '.join(worldlist_title[-2:])
    words = Title.split()
    date=' '.join(words[-4:-2])
    result=''
    if words[-2] =='في' :
      date=' '.join(words[-2:])
    elif words[-2]=='العام':
      date=' '.join(words[-2:])
      
    if table2_result2==' ' or isNaN(table2_result2):
      result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي "+str(str(date))+":"+"   "+str(table1_result)+"                       \n \n \n الفترة الحالية "+title_months+":"   +str(table2_result)+"."+"                       \n \n \n"+'ملاحظات إضافية:'+str(articlesummry))
    else:
      result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي "+str(str(date))+":"+"   "+str(table1_result)+"                       \n \n \n الفترة الحالية "+title_months+":"   +str(table2_result2)+".أيضاً "+str(table2_result)+"."+"                       \n \n \n"+'ملاحظات إضافية:'+str(articlesummry))

    if table1==' ' and table2==' ' and label2=='اعلان أرباح ':
      result=textClean("ملخص:"+Title+"                                                 \n \n \n"+'ملاحظات إضافية:'+str(articlesummry))
    elif label2=='اعلان أرباح ' and table1!=' ' and table2!=' '  :
      result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي في:"+str(table1_result)+"                       \n \n \n الفترة الحالية " +title_months+":"   +str(table2_result2)+".أيضاً "+str(table2_result)+"."+"                       \n \n \n"+'ملاحظات إضافية:'+articlesummry)

    #   if has_numbers(Title)==True :
    #     result
    #   elif has_numbers(Title)==False:
    #     result
        # result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي في:"+str(table1_result)+"                       \n \n \n الفترة الحالية " +title_months+":"   +str(table2_result2)+".أيضاً "+str(table2_result)+"."+"                       \n \n \n"+'ملاحظات إضافية:'+articlesummry)
    elif label2=='اعلان أرباح ' and table1!='Null' and table2=='Null'  :
      if has_numbers(Title)==True :
        result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي "+str(str(date))+":"+"   "+str(table1_result)+"                       \n \n \n"+"                                                 \n \n \n"+'ملاحظات إضافية:'+str(articlesummry))
      elif has_numbers(Title)==False:
        result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي في:"+str(table1_result)+"                       \n \n \n" +"                       \n \n \n"+'ملاحظات إضافية:'+articlesummry)


    elif label2=='اعلان خسائر' and table1!='0' and table2=='0':
      if has_numbers(Title)==True:
        result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي "+str(str(date))+"):"+"   "+str(table1_result)+"                        \n \n \n"+"                                                 \n \n \n"+'ملاحظات إضافية:'+str(articlesummry))
      elif has_numbers(Title)==False:
        result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي في:"+str(table1_result)+"\n \n \n                                                                                     "+"                                                 \n \n \n"+'ملاحظات إضافية:'+articlesummry)

    elif label2=='اعلان خسائر' and table1!='0' and table2!='0':
      if has_numbers(Title)==True:
        result
      elif has_numbers(Title)==False:
        result
      # result=textClean("ملخص:"+Title +"                       \n \n \n"+ " الربع الحالي المنتهي في:"+str(table1_result)+"                       \n \n \n الفترة الحالية "+title_months+":"   +str(table2_result2)+".أيضاً "+str(table2_result)+"."+"                       \n \n \n"+'ملاحظات إضافية:'+articlesummry)

    else:
      result=textClean("ملخص:"+Title+"                                                 \n \n \n"+'ملاحظات إضافية:'+str(articlesummry))
    return result










def removespace(text):
    result = " ".join(re.split("\s+", text, flags=re.UNICODE))
    return result
req = requests.get(url)
soup = bs(req.text, 'html.parser')


req = requests.get(url2)
soup2 = bs(req.text, 'html.parser')



column1 = []
column2=[]
column1_table1=[]
column2_table1=[]
column3_table1=[]
column4_table1=[]
column5_table1=[]
column1_new=[]
column2_new=[]
main=' '
main1=''
main2=''
column3=[]


def extract_mainArticle(soup):
    article = soup.find('div', {'class': 'article_body'}).get_text()
    articletable = soup.find('table', {'class': 'stacktable large-only'}).text
    article_main=''
    mainArticle=[]
    words = article.split()
    for i in range(len(words)):
        if(words[i]=='بندتوضيح' or words[i]== 'بندتوضيحيعود' or words[i]=='بندتوضيحمقدمة' or words[i]==' الملفات الملحقة 'or words[i]=='المتحفظملاحظة'):
            article_main=words[i+2:]
            mainArticle="يعود "+'  '.join(article_main)
    return mainArticle


def extract_table1(soup):
    for row in soup.findAll('table')[0].tbody.findAll('tr'):
        first_column=row.find_all('td')[1].get_text().strip() 
        second_column=row.find_all('td')[2].get_text().strip() 
        third_column=row.find_all('td')[3].get_text().strip() 
        fourth_column=row.find_all('td')[4].get_text().strip() 
        fifith_column=row.find_all('td')[5].get_text().strip() 
        column1_table1.append(first_column) 
        column2_table1.append(second_column)
        column3_table1.append(third_column)
        column4_table1.append(fourth_column)
        column5_table1.append(fifith_column)
    return column1_table1,column2_table1,column3_table1,column4_table1,column5_table1

def extract_table2(soup):
    for row in soup.findAll('table')[1].tbody.findAll('tr'):
        first_column=row.find_all('td')[1].get_text().strip()
        second_column=row.find_all('td')[2].get_text().strip()
        column1.append(first_column) 
        column2.append(second_column)
    return column1,column2


def extract_table2_currentYear(soup):
    for row in soup.findAll('table')[1].tbody.findAll('tr'):
        first_column=row.find_all('td')[1].get_text().strip()
        second_column=row.find_all('td')[2].get_text().strip()
        column1.append(first_column)
        column2.append(second_column)
    return column1,column2


import csv


date1, _, _, _, _ = process_data(soup)
_, title1, _, _, _ = process_data(soup)
_, _, table1, _, _ = process_data(soup)
_, _, _, table2, _ = process_data(soup)
_, _, _, _, main_Article1 = process_data(soup)


date2, _, _, _, _ = process_data(soup2)
_, title2, _, _, _ = process_data(soup2)
_, _, table1_2, _, _ = process_data(soup2)
_, _, _, table2_2, _ = process_data(soup2)
_, _, _, _, main_Article2 = process_data(soup2)
table22=str(table1)
table222=str(table2)






#     elif  len(soup.find_all('table', {'class': 'stacktable large-only'}))==4 :
#         table1=soup.findAll('table')[0].get_text().strip()
#         table2=soup.findAll('table')[1].get_text().strip()
#         table3=soup.findAll('table')[2].get_text().strip()
#         table4=soup.findAll('table')[3].get_text().strip()
        
#         main_Article=extract_mainArticle()

#         if  'الخسائر المتراكمة' in table3:
#             if 'توضيح' in  table4:
#                 main_Article=extract_mainArticle()

#             # extract table2
#             if 'ربحية (خسارة) السهم' in table2 :
#                 extract_table2()
#             # extract table1
#             if 'ربحية (خسارة) السهم' not in table1 and  table3  and table4:
#                 extract_table1()
   
#         elif 'إجمالي حقوق المساهمين (بعد استبعاد حقوق الأقلية)' in table2 :

#             # extract table2
#             if 'ربحية (خسارة) السهم' in table2 :
#                 extract_table2()
                          

    
    
    
    


#     def joinText_table1():
#         if  'السنة الحالية' in soup.findAll('table')[0].get_text().strip()and len(column1)>3:
#             main1={'السنة الحالية':column1,'السنة الماضية ':column2}

#         elif len(column1_table1)> 3 and len(column2_table1)> 3 and len(column3_table1)> 3 and len(column4_table1)> 3 and len(column5_table1)> 3:
#             main1={'الربع الحالي':column1_table1,'الربع المماثل من العام السابق':column2_table1,'التغير%':column3_table1,'الربع السابق':column4_table1,
#                     '	التغير %':column5_table1}

#         else:
#             main1='Null'

#         return main1

#     def joinText_table2():
#         if  'السنة الحالية' in soup.findAll('table')[0].get_text().strip():
#             main2='Null'
#         elif len(column1)>=2:
#             main2={'الفترة الحالية':column1,'الفترة المماثلة من العام السابق':column2}
#         else:
#             main2='Null'


#         return main2
#     def extract_title():
#         title=soup.findAll('h3', {})[1].text
#         result_title = ''.join([i for i in removespace(title)])
#         return result_title
#     def extract_date():
#         time_stamp = soup.findAll('div', {'class': 'date'})[1].get_text().strip()
#         year=time_stamp[0:4]
#         month=time_stamp[5:7]
#         day=time_stamp[8:10]
#         Date = str(Hijri(int(year), int(month), int(day)).to_gregorian())
#         return Date
#     def numDown():
#         if soup.find_all('span', {'class': 'down'}):
#             numDown = len(soup.find_all('span', {'class': 'down'}))
#         else:
#             numDown = 0

#         return numDown
#     def numUp():
#         if soup.find_all('span', {'class': 'up'}):
#             numUp = len(soup.find_all('span', {'class': 'up'}))
#         else:
#             numUp = 0
#         return numUp

        

#     writer.writerow([extract_date(),numUp(),numDown(),extract_title(),joinText_table1(),joinText_table2(),str(main_Article)])
  
st.markdown(
    '''
    <style>
    .expander {
        background-color: red;
        color: black; # Adjust this for expander header color
    }
    .streamlit-expanderContent {
        background-color: yellow;
        color: red; # Expander content color
    }
    </style>
    ''',
    unsafe_allow_html=True
)



# ..................
def predict_forest(title):
    processed_features = vectorizer.transform([title]).toarray()
  
   

    prob = model.predict(processed_features)



    return prob



def catogrize(title):


    processed_features = vectorizer.transform([title]).toarray()
  


    prob = model2.predict(processed_features)



    return prob


def join_sentences(sentence1, sentence2):
  """
  Joins two sentences into one, adding a space in between.

  Args:
      sentence1: The first sentence.
      sentence2: The second sentence.

  Returns:
      The combined sentence.
  """
  return f"{sentence1} {sentence2}"

def simple_split(sentence):
  words = sentence.split()
  midpoint = len(words) // 2
  return " ".join(words[:midpoint]), " ".join(words[midpoint:])



def summrize(title):


    from nltk.tokenize import sent_tokenize
    senten=title

    
    sentences_tokens = sent_tokenize(senten)
    if len(sentences_tokens) == 1:

        sentence1, sentence2 = simple_split(senten)
        joined_sentence = sentence1 +" . "+ sentence2
        text = joined_sentence
        if "\n" in text:
            text=text.replace('\n',' ')
            prob = model_summary(text,num_sentences=2)

        else:
            prob = model_summary(text,num_sentences=2)
    else:
        prob = model_summary(title,num_sentences=2)
    return str(prob)



# class_output=predict_forest(str(" شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "))
# class_output2=predict_forest(str(" شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "))
# output_catogrize=predict_forest(str(" شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "))
# output_summary=summrize(str(" شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "))
# new_text = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"
# new_text2 = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"
# new_text = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"
# new_text2 = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"

#     # st.title("Text Analytics Tool")
#     html_temp = """
#     <div style="background-color:#025246 ;padding:5px;margin-bottom:40px;margin-right: 0px;margin-left: 0px">
#     <h2 style="color:white;text-align:center;">Text Analytics Tool </h2>
#     </div>
#     """
#     st.markdown(html_temp, unsafe_allow_html=True)

#     # x= st.text_input("Enter Your name", " اعلان شركة المصافي العربية السعودية عن النتائج المالية الأولية للفترة المنتهية في 2021-09-30 ( تسعة أشهر ) ")
#     # x=st.text("    تعلن شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية  ")
#     html_html="""  
#       <div style="background-color:red;margin-left:80%;font-size:2px;">
#        <h5 style="color:white;text-align:right;">  تعلن شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية </h5>
#        </div>
#     """


#     danger_html = """
#       <div style="background-color:#F08080;padding:10px">
#        <h2 style="color:black;text-align:center;"> خبر خطير</h2>
#        </div>
#     """

#     # st.markdown(html_html, unsafe_allow_html=True)
#     title1="تعلن شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "
#     title2=" عن تغيرات ادارية في الادارة التنفيذية "


# st.write('here', catogrize(str(" شركة المصافي العربية السعودت")))
#     class_output2=predict_forest(str(" شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "))
#     output_catogrize=predict_forest(str(" شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "))
#     output_summary=summrize(str(" شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية "))
#     new_text = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"
#     new_text2 = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"
#     new_text = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"
#     new_text2 = "شركة المصافي العربية السعودية (ساركو) عن تغيرات جديدة في الادارة التنفيذية"
#     summary, catogry, image_col, text_col = st.columns(4)
#     with image_col:
#         st.button("تصنيف الخبر", on_click=lambda: st.success(class_output))
#         st.button("تصنيف  الخبر", on_click=lambda: st.success(class_output2))

#     with catogry:
#         st.button("فئة الخبر", on_click=lambda: st.success(output_catogrize))

#     with summary:
#         st.button("تلخيص الخبر", on_click=lambda: st.success(output_summary))
#     with text_col:
#         st.markdown(""" <div style='width: 400%;padding:0px;margin:10px;font-size:15px;word-wrap: break-word;'> <a  href="https://www.w3schools.com">{}</a> </div>""".format(new_text), unsafe_allow_html=True)
#         st.markdown(""" <div style='width: 400%;padding:0px;margin:10px;font-size:15px;word-wrap: break-word;'> <a  href="https://www.w3schools.com">{}</a> </div>""".format(new_text2), unsafe_allow_html=True)
def textClean(text):
  if "\'" in text:
    text=text.replace('\'','')
  if '٪, ' in text:
    text=text.replace('٪,','٪')
  if '"' in text:
    text=text.replace('"','')
  if '",' in text:
    text=text.replace('",','')
  if '}' and '{' in text:
    text=text.replace('}','')
    text=text.replace('{','')
  if 'بنسبة,' in text:
    text=text.replace('بنسبة,','بنسبة')
  if ',ريال' in text:
    text=text.replace(',ريال','ريال')
  if '--' in text:
    text=text.replace('--','')
  if '1213 -1.04 %' in text:
    text=text.replace('1213 -1.04 %','')
  if ')' and '(' in text:
    text=text.replace(')','')
    text=text.replace('(','')
  return text


worldlist_title=textClean(title1).split()
title_months="("+' '.join(worldlist_title[-2:])
words = title1.split()
date=' '.join(words[-4:-2])
if words[-2] =='في' :
    date=' '.join(words[-2:])
elif words[-2]=='العام':
    date=' '.join(words[-2:])



st.markdown("<div style='width: 300%;text-align:center;font-size:15px;word-wrap: break-word;margin:5% 50%;", unsafe_allow_html=True)
def has_numbers(inputString):
    return any(char.isdigit() for char in inputString)
def display_news_block(urls,titles,news_data,dates,articles,tables1,tables2, index):

    """Displays a block of news items with buttons and text."""

    analyze,summary, catogry, image_col, text_col,_ = st.columns(6)
    

    result=''
    with image_col:
        st.button("تصنيف الخبر ", key=f"classify_button1_{index}", on_click=lambda: st.success(predict_forest(str('شركة المصافي العربية السعودية (ساركو) عن تغيرات ادارية في الادارة التنفيذية '))))
      
    with catogry:
        st.button("فئة الخبر", key=f"category_button_{index}", on_click=lambda: st.success(catogrize(str(news_data[index]))))

    with summary:
        show_text1 = True
        st.button("تلخيص الخبر", key=f"summarize_button_{index}", on_click=lambda: st.markdown(f"""<div style="position: fixed; top: 30%; left: 50%; transform: translateX(-50%); background-color: lightblue; padding: 20px; border: 1px solid transparent; z-index: 999;">{summrize((str(articles[index])))}    </div>""", unsafe_allow_html=True))
  
        # st.button("تلخيص الخبر", key=f"summarize_button_{index}", on_click=lambda: st.markdown(f"""<div style="position: fixed; bottom: 10px; left: 50%; transform: translateX(-50%); background-color: white; padding: 20px; border: 1px solid transparent; z-index: 999;">{summrize((str(articles[index])))}    </div>""", unsafe_allow_html=True))
    
    with analyze:
        # if tables1[index]!='Null' and tables2[index]!='Null':
        #     result=textClean("ملخص:"+str(news_data[index]) +"                       \n \n \n"+ " الربع الحالي المنتهي "+str(str(dates[index]))+":"+"   "+str(tables1[index])+"                       \n \n \n الفترة الحالية "+title_months+":"   +str(tables2[index]))+"."+"                       \n \n \n"+'ملاحظات إضافية:'+(summrize(str(articles[index])))
        # else:
  
        result=table_table2('اعلان أرباح ',(titles[index]),tables1[index],tables2[index],articles[index])
            # result=textClean("ملخص:"+news_data[index]+"                                                 \n \n \n"+'ملاحظات إضافية:'+(summrize(str(articles[index]))))
        st.button("حلل الخبر", key=f"analyze_button_{index}", on_click=lambda: st.markdown(f"""<div  style="text-align:right;position: fixed;font-size:5px;width:80%; height:90vh; top: 30px; left: 50%; transform: translateX(-50%); background-color: lightblue; padding: 20px; border: 1px solid transparent; z-index: 999;"> \n \n \n {result}   </div>""", unsafe_allow_html=True ))


        # else:
        #     result=textClean("ملخص:"+extract_title() +"                       \n \n \n"+ " الربع الحالي المنتهي "+str(str(date))+":"+"   "+str(joinText_table1())+"                       \n \n \n الفترة الحالية "+title_months+":"   +str(table2_result2)+".أيضاً "+str(table2_result)+"."+"                       \n \n \n"+'ملاحظات إضافية:'+str(articlesummry))

        # st.button("حلل الخبر", key=f"summarize_button_{index}", on_click=lambda: st.markdown(f"""<div style='width: 100%;height:10%;padding:5px;margin-right:0%;text-align:center;'> \n \n \n {extract_title()} \n \n \n{joinText_table1()}  \n \n \n{joinText_table2()}  \n \n \n 'ملاحظات إضافية:'{summrize(str(main_Article))}""", unsafe_allow_html=True ))

        # st.button("حلل الخبر", key=f"analyze_button_{index}", on_click=lambda: st.markdown(f"""<div style='width: 100%;height:10%;padding:5px;margin-right:0%;text-align:center;'> \n \n \n {result}""", unsafe_allow_html=True ))


 
  
    with text_col:
        for text in news_data[index]:
            st.markdown(f"""
            <div style='width: 300%;text-align:center;font-size:15px;word-wrap: break-word;margin:5% 50%;'>
                <a href="{urls[index]}">{text}</a>
            </div>
            """, unsafe_allow_html=True)
            
        # x=st.expander('click')
        # with x:
           

        #     st.markdown(f"""<div style='width: 100%;height:10%;padding:5px;margin-right:0%;text-align:center;'> \n \n \n {news_data[index]} \n \n \n{tables[index]}  \n \n \n{str(news_data2[index])}  \n \n \n {str(main_Article)}""", unsafe_allow_html=True )
        #                 # st.write(f"""<div style='width: 200%;height:30%;padding:5px;background-color: red;margin-right:0%;text-align:center;'>{extract_title()},{str(main_Article)}""", unsafe_allow_html=True)                    


# Sample news data (replace with your actual data)


news_data = [
    [title1],
    [title2],

]
titles = [
    title1,
    title2,

]
articles=[
   main_Article1,
     main_Article2,
]
tables1 = [
    [ table1],
    [ table1_2],

]

tables2 = [
    [ table2],
    [ table2_2],

]


dates = [
    [ date1],
    [ date2],

]
urls=[
  url,url2
]



html_temp = """
    <div style="background-color:#025246 ;text-align:center;margin-bottom:20px;text-align:center;margin:5% 5%;width:80%;font-size:3px">
    <h4 style="color:white;">أداة تحليل النصوص للاعلانات الشركات في سوق الاسهم</h4>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
show_text1 = False
for i in range(2):  # Repeat 10 times
    
    display_news_block(urls,titles,news_data,dates,articles,tables1,tables2, i)
    st.markdown("---")  # Add a separator between blocks





  