import re
import pandas as pd
from datasets import Dataset

LABEL_MAPPING = {
    "Support": 0,
    "Feature Request": 1,
    "Billing and Payments": 2
}
ID2LABEL = {v: k for k, v in LABEL_MAPPING.items()}

# Expressions régulières compilées (Arthur + Arslan)
PLACEHOLDER_RE = re.compile(r'\[.*?\]')
GREETINGS_RE   = re.compile(r'^(hello|hi|dear customer|hey)[,\s]+', flags=re.IGNORECASE)
SIGNATURE_RE   = re.compile(r'(sincerely|best regards|warm regards|thank you for your time).*', flags=re.IGNORECASE | re.DOTALL)
VERSION_RE  = re.compile(r'\d+\.\d+(\.\d+)?')
SOFTWARE_RE = re.compile(r'\b(microsoft dynamics 365|microsoft dynamics|laravel 8|laravel|node\.js|elasticsearch|avid pro tools|microsoft office 2021|office 2021|hadoop|apache|cyberlink powerdirector|davinci resolve|asana)\b', flags=re.IGNORECASE)
SPACES_RE = re.compile(r'\s+')

def clean_text_bert(text: str) -> str:
    """Pipeline de nettoyage fusionné (HTML, retours chariot, regex métiers)."""
    if not isinstance(text, str):
        return ""
    # 1. Nettoyage de base (Arthur)
    text = re.sub(r'<.*?>', ' ', text)
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    # 2. Nettoyage NLP/Bert (Arslan)
    text = text.lower()
    text = GREETINGS_RE.sub('', text)
    text = SIGNATURE_RE.sub('', text)
    text = PLACEHOLDER_RE.sub('', text)
    text = VERSION_RE.sub('VERSION', text)
    text = SOFTWARE_RE.sub('SOFTWARE_PRODUCT', text)
    
    return SPACES_RE.sub(' ', text).strip()

def apply_cleaning(df: pd.DataFrame) -> pd.DataFrame:
    """Applique le nettoyage et le mapping des labels au DataFrame."""
    df = df.copy()
    df['text'] = df['text'].apply(clean_text_bert)
    df['text'].replace('', pd.NA, inplace=True)
    
    col = 'queue' if 'queue' in df.columns else 'label'
    df['label'] = df[col].map(LABEL_MAPPING)
    
    df.dropna(subset=['text', 'label'], inplace=True)
    df['label'] = df['label'].astype(int)
    df = df[df['text'].str.split().str.len() > 2]
    return df.reset_index(drop=True)

def make_hf_dataset(df: pd.DataFrame, tokenizer, max_len: int = 128) -> Dataset:
    """Convertit un DataFrame Pandas en Dataset HuggingFace."""
    def tokenize(batch):
        return tokenizer(batch['text'], padding='max_length', truncation=True, max_length=max_len)
    
    ds = Dataset.from_pandas(df[['text', 'label']], preserve_index=False)
    ds = ds.map(tokenize, batched=True)
    ds = ds.rename_column('label', 'labels')
    ds.set_format('torch', columns=['input_ids', 'attention_mask', 'labels'])
    return ds