import gradio as gr
import uvicorn
from fastapi import FastAPI
from sentence_transformers import SentenceTransformer, util
import pandas as pd

EMB_MODEL = "BAAI/bge-m3"
CSV_PATH = 'desc2024_v2.csv'


class FindTerm:
    def __init__(self, emb_model_path, batch_size=64):
        self.embedder = SentenceTransformer(emb_model_path)
        self.batch_size = batch_size

    def embed(self, vals):
        embs = self.embedder.encode(vals, convert_to_tensor=True, batch_size=self.batch_size)
        embs = util.normalize_embeddings(embs)
        return embs

    def fit(self, CSV_PATH):
        df = pd.read_csv(CSV_PATH)
        df = df[df.Term.str.isalpha()]
        df['Term'] = df.Term.apply(lambda x: x.upper())
        self.dictionary = df['Term'].values
        self.term_embs = self.embed(self.dictionary)

    def masked_edit_distance(self, masked_term, candidate_term):
        dist = 0

        for m_char, c_char in zip(masked_term, candidate_term):
            if m_char == '_' and m_char != c_char:
                dist += 1
            elif m_char != '_' and m_char != c_char:
                return float('inf')

        return dist

    def find_best_match(self, masked_term, candidate_terms):
        best_term = None
        min_distance = float('inf')

        for term in candidate_terms:
            if len(masked_term) != len(term):
                continue

            dist = self.masked_edit_distance(masked_term, term)

            if dist < min_distance:
                min_distance = dist
                best_term = term

        return best_term

    def predict(self, masked_term, term_desc):
        maked_term_emb = self.embed([masked_term])
        term_desc_emb = self.embed([term_desc])
        term_hits = util.semantic_search(maked_term_emb, self.term_embs, score_function=util.dot_score, top_k=20)
        desc_hits = util.semantic_search(term_desc_emb, self.term_embs, score_function=util.dot_score, top_k=20)

        possibilities = list(
            set([self.dictionary[i['corpus_id']] for i in term_hits[0]] + [self.dictionary[i['corpus_id']] for i in
                                                                           desc_hits[0]]))
        final_term = self.find_best_match(masked_term, possibilities)
        if final_term != None:
            return final_term
        else:
            return "Unable to process the input, try adding more data to dictionary"


ft = FindTerm(EMB_MODEL)
ft.fit(CSV_PATH)

with gr.Blocks() as demo:
    masked_text = gr.Textbox(label="Masked Term")
    description = gr.Textbox(label="Description")
    fill_mask = gr.Button()
    unmasked_text = gr.Textbox(label="Unmasked Term", max_lines=2, interactive=False)
    fill_mask.click(ft.predict, inputs=[masked_text, description], outputs=[unmasked_text])
    clear = gr.ClearButton([masked_text, description, unmasked_text])

demo.queue()

CUSTOM_PATH = "/gradio"

app = FastAPI()
@app.get("/")
def read_main():
    return {"message": "This is your main app"}

app = gr.mount_gradio_app(app, demo, path=CUSTOM_PATH)

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)