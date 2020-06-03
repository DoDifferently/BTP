# -*- coding: utf-8 -*-
"""
Created on Thu Apr 26 01:35:54 2018

@author: Shubh
"""

from selenium import webdriver
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.by import By
import time
import pandas as pd 


driver = webdriver.Chrome('chromedriver')
 
driver.get("http://www.repository.voxforge1.org/downloads/SpeechCorpus/Trunk/Audio/Main/16kHz_16bit/")

wait = WebDriverWait(driver, 600)



data = pd.read_csv("Audiofiles.csv")
audiofiles = data['Audiofiles']


for i in audiofiles:
    Menu = '//a[\''+i+'\')]'
    Menu_button = wait.until(EC.presence_of_element_located((
        By.XPATH, Menu)))
    Menu_button.click()
    time.sleep(2)