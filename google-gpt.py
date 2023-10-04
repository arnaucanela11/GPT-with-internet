from dotenv import load_dotenv
import requests
# from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.vectorstores import Chroma
import os
from summarizer import summarize
from bs4 import BeautifulSoup
from langchain import PromptTemplate
load_dotenv()

# START GETTING DATA


def google_search(query, api_key, cx, num_results=5, date_restrict='m1', hl='es'):
    """
    Realiza una búsqueda en Google utilizando la API de Google Search JSON.

    Args:
        query (str): La consulta de búsqueda.
        api_key (str): Tu clave de API de Google.
        cx (str): Tu ID de búsqueda personalizado (cx).
        num_results (int): El número de resultados que deseas (por defecto 10).
        dateRestrict (str): Tiempo de límite de los resultados.
    """
    url = "https://www.googleapis.com/customsearch/v1"

    params = {
        "key": api_key,
        "cx": cx,
        "q": query,
        "num": num_results,
        "hl": hl,
        "dateRestrict": date_restrict
    }

    response = requests.get(url, params=params)
    if response.status_code != 200:
        raise Exception('Google API request failed!')

    data = response.json()

    return data


query = input('Hacme una consulta: ')
results = google_search(query, os.getenv(
    'GOOGLE_API_KEY'), os.getenv('ENGINE_ID'))

# text_splitter = RecursiveCharacterTextSplitter(
#     # Set a really small chunk size, just to show.
#     chunk_size=150,
#     chunk_overlap=10,
#     length_function=len,
#     is_separator_regex=False,
# )
# print(results.get("items", [])[0].get('snippet', ''))
fragments = []

for index, item in enumerate(results.get("items", []), start=1):
    # print(f'INDEX: {index}')
    # print('__________________________________________________________________________\n')

    page_url = item.get('link')
    response = requests.get(page_url)

    if response.status_code == 200:
        page_content = response.text
        soup = BeautifulSoup(page_content, 'html.parser')
        page_text = soup.get_text()
        
        # TEXT SUMMARIZE:
        summarized_text = summarize(item.get('title'), page_text, count=15)
        
        for text in summarized_text:
            cleaned_texts = text.replace('\n', '')
            fragments.append(cleaned_texts)
        
    else:
        print(f"Failed to retrieve content from {page_url}")


os.environ["HNSWLIB_NO_NATIVE"] = "1"


db = Chroma.from_texts(fragments, OpenAIEmbeddings(openai_api_key=os.getenv('OPENAI_API_KEY'), model="text-embedding-ada-002"))

# # embedding_vector = OpenAIEmbeddings().embed_query(query)
# # docs = db.similarity_search_by_vector(embedding_vector)
# # for doc in docs:
# #     print(doc.page_content)

docs = db.similarity_search(query)
sources = ""
for index, doc in enumerate(docs[:8], start=1):
    sources += doc.page_content
    sources += "\n"

prompt_template = PromptTemplate.from_template(
    "Respondeme esta cuestion: {search}, teniendo en cuenta estas fuentes sacadas de internet: {sources} y otras a las que tengas acceso."
)
llm = OpenAI(openai_api_key=os.getenv('OPENAI_API_KEY'))
print(llm(prompt_template.format(search=query, sources=sources)))
   

