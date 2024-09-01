

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")




data=pd.read_csv('youtube.csv',encoding='Latin-1')
data.head()




data=data.drop(columns=['link'])
data.head(2)






# ## Data Preprocessing steps
#  - lower casing
#  - \n
#  - email removal
#  - twitter handlers (@)
#  - hashtags
#  - url removals  
#  - punctuation Removal
#  - share,like,comment
#  - numbers
#  - tokenization
#  - emojis
#  - stopwords removal
#  - stemming
#  - lemmatization

# In[11]:


import string
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
import re


#  - **Lower casing**
#  - **\n replace with whitespace**
#  - **email removals**  
#  - **twitter handlers (@)**
#  - **hashtags**

# In[12]:


def clean_txt_func1(text):
    ## Lower Casing
    text=text.lower()
    # \n with whitespace
    text=text.replace('\n',' ')
    ## email removal
    pattern=re.compile(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,6}\b')
    found_urls = pattern.findall(text)
    for url in found_urls:
        text = text.replace(url, '')
    ## hastag removal
    hashtag_pattern = r"#\w+"
    text=re.sub(hashtag_pattern, '', text)
    ## twitter handlers
    handle_pattern = r'@\w+'
    text=re.sub(handle_pattern, '', text)
    return text
    


# In[13]:


data['description_clean']=data['description'].apply(lambda x : clean_txt_func1(x))
data['title_clean']=data['title'].apply(lambda x : clean_txt_func1(x))


# In[14]:


data.head(2)


#  - URL Removal
#  - punctuation Removal
#  - share,like,comment
#  - numbers
#  - tokenization

# ## URL Removals
# some urls in our data
#   - http://seek-discomfort.com/yes-theory
#   - https://www.youtube.com/channel/ucl5d
# 

# 
# * - (?:https?://|ftp://|www\.)  # Match various protocols or www. prefix
#   - [^\s@]+                        # Match URL content (not whitespace or @)
#   - (?:                             # Optional trailing characters
#   - [:/?#+&;\w-]*                # Common URL components
#   -  |                           # or
#   -  \(                              # Opening parenthesis
#   -  [^)]*                       # Anything but closing parenthesis
#   -   \)                          # Closing parenthesis
#   - )?
#   - (?:\s|\Z)                       # Match whitespace or end of string

# In[15]:


import string
from string import punctuation


# In[16]:


def clean_txt_func2(text):
    ## URL Removal
    reg_pattern = re.compile(r'(?:https?://|ftp://|www\.)[^\s@]+(?:[:/?#+&;\w-]*|\([^)]*\))?(?:\s|\Z)')
    found_urls = reg_pattern.findall(text)
    for url in found_urls:
        text = text.replace(url, '')
    ## Punctuation Removal
    text = ''.join([word for word in text if word not in punctuation])

    ## Specific words Removal
    words_to_remove = ["subscribe","subscribers", "like", "comment", "share","join","disclaimer","’"]
    pattern = r'\b(' + '|'.join(words_to_remove) + r')\b|\s*\.\s*'
    text=re.sub(pattern, '', text)

    ## Number Removal
    pattern = r'\b\d+[a-zA-Z]?\b'
    text= re.sub(pattern, '', text)

    ## Tokenization
    text=word_tokenize(text)
    return text


# In[17]:


data['description_clean']=data['description_clean'].apply(lambda x : clean_txt_func2(x))
data['title_clean']=data['title_clean'].apply(lambda x : clean_txt_func2(x))


# In[18]:


data.head(4)


# ## Emojis Removal

# In[19]:


def remove_emojis(text):
    # Define a pattern that matches emojis
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & Pictographs
        "\U0001F680-\U0001F6FF"  # Transport & Map Symbols
        "\U0001F1E0-\U0001F1FF"  # Flags (iOS)
        "\U00002700-\U000027BF"  # Dingbats
        "\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
        "\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
        "\U00002600-\U000026FF"  # Miscellaneous Symbols
        "\U0001F700-\U0001F77F"  # Alchemical Symbols
        "\U00002300-\U000023FF"  # Miscellaneous Technical
        "\U00002000-\U000020FF"  # General Punctuation
        "\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
        "\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
        "\U0001FA00-\U0001FA6F"  # Chess Symbols
        "\U00002B05-\U00002B07"  # Arrows
        "\U00002934-\U00002935"  # Arrows
        "\U00002190-\U000021AA"  # Arrows
        "]+", flags=re.UNICODE
    )

    # Substitute emojis with an empty string
    return emoji_pattern.sub(r'', text)

def clean_txt_func3(word_list):
    return [remove_emojis(word) for word in word_list if remove_emojis(word).strip()]


# In[20]:


data['description_clean']=data['description_clean'].apply(lambda x: clean_txt_func3(x))
data['title_clean']=data['title_clean'].apply(lambda x : clean_txt_func3(x))


# ## Stopwords Removal

# In[21]:


from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem import RegexpStemmer


# - stopwords removal
# - stemming
# - lemmatization

# In[22]:


def clean_txt_func4(text):
    ## Stopwords Removal
    stopwords=nltk.corpus.stopwords.words('english')
    text=[word for word in text if word not in stopwords]
    
    ## Stemming
    # Create a Regexp Stemmer with a custom rule
    custom_rule = r'ing$'
    regexp_stemmer = RegexpStemmer(custom_rule)
    text = [regexp_stemmer.stem(word) for word in text]
    
    ## Lemaatize
    lm=WordNetLemmatizer()
    text = [lm.lemmatize(word) for word in text ]

    return text    


# In[23]:


data['description_clean']=data['description_clean'].apply(lambda x: clean_txt_func4(x))
data['title_clean']=data['title_clean'].apply(lambda x : clean_txt_func4(x))


# In[24]:


data.head(3)


# In[25]:


data.drop(columns=['title','description'],inplace=True)


# In[26]:


data.head(3)


# ## Label Encoding on Category Column

# In[27]:


from sklearn.preprocessing import OrdinalEncoder


# In[28]:


categories_order = [['travel', 'food', 'art_music','history']]
# Initialize the OrdinalEncoder with the specified order
ordinal_encoder = OrdinalEncoder(categories=categories_order)

# Fit and transform the categorical data
data['category_encoded'] = ordinal_encoder.fit_transform(data[['category']])


# In[29]:


data.head(2)


# In[30]:


data['combined']=data['title_clean']+data['description_clean']


# ## Data Splitting

# In[31]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data['combined'],data['category'], test_size=0.2, random_state=42)


# In[32]:


y_train.value_counts()


# ## Vectorization

# In[33]:


from sklearn.feature_extraction.text import TfidfVectorizer



# In[34]:
tfidf = TfidfVectorizer()

def tfidf_vectorization(X_train, X_test):
    # Join tokens into a single string for each document
    X_train_joined = [' '.join(tokens) for tokens in X_train]
    X_test_joined = [' '.join(tokens) for tokens in X_test]
    # Perform TF-IDF vectorization
    X_train_tfidf = tfidf.fit_transform(X_train_joined)
    X_test_tfidf = tfidf.transform(X_test_joined)
    
    return X_train_tfidf, X_test_tfidf


# In[35]:


X_train_vec_tfidf, X_test_vec_tfidf=tfidf_vectorization (X_train, X_test)
X_train_vec_tfidf=X_train_vec_tfidf.toarray()
X_test_vec_tfidf=X_test_vec_tfidf.toarray()


# ## Class Balancing Using SMOTE

# In[36]:


from imblearn.over_sampling import SMOTE

def apply_smote(X_train, y_train):
    smote = SMOTE(random_state=42)
    X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
    return X_train_res, y_train_res


# In[ ]:


X_train_tfidf_res, y_train_res = apply_smote(X_train_vec_tfidf, y_train)


# In[ ]:


y_train_res.value_counts()


# ## Model

# In[ ]:


from sklearn.linear_model import LogisticRegression

models = {
        "Logistic Regression": LogisticRegression(C=10, random_state=42)
    }


def evaluate_svm(X_train, X_test, y_train, y_test):
   
    for model_name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

       





evaluate_svm(X_train_tfidf_res, X_test_vec_tfidf, y_train_res, y_test)


import joblib

# Save the trained model
joblib.dump(models['Logistic Regression'], 'logistic_regression_model.pkl')

# Save the TF-IDF vectorizer
joblib.dump(tfidf, 'tfidf_vectorizer.pkl')





