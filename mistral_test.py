from langchain_community.llms import LlamaCpp
llm = LlamaCpp(
    model_path="/Users/mtis/Local/Code/GitRepos/llama.cpp/llama.cpp/models/mistral/mistral-7b-instruct-v0.1.Q4_K_M.gguf",  # Ton modèle ici
    temperature=0.5,
    n_threads=8,  # Pour M1 Pro
    verbose=True

)
prompt = "Qui es-tu ?"

# Appel du modèle
output = llm.invoke(prompt, max_tokens=100, stop=["</s>"])
# Affichage de la réponse
print("Réponse :")
print(output)
