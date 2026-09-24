# RAG em Python

Pequeno projeto de RAG usando Python

## Como usar

Clone o projeto e, dentro da pasta `docs`, coloque os arquivos `.txt` que serão usados como base de conhecimento. Depois, crie um arquivo `.env` e adicione a API key da sua conta OpenAI.

Exemplo: OPENAI_API_KEY=

O projeto foi desenvolvido utilizando os embeddings da OpenAI.

## Fluxo da main

Na `main`, o projeto começa verificando se os arquivos da base de conhecimento sofreram alguma alteração, comparando o hash atual com o hash salvo anteriormente.

Se houver alguma alteração, ou se ainda não existir um hash salvo, o programa:

- carrega os documentos;
- divide os documentos em chunks;
- gera os embeddings desses chunks;
- armazena os embeddings no ChromaDB.

Depois disso, o usuário pode digitar uma pergunta.

O programa gera o embedding da pergunta e utiliza o ChromaDB para fazer uma busca por similaridade vetorial, recuperando os chunks mais relevantes da base de conhecimento
Esses resultados são separados e usados como contexto para a IA gerar a resposta final.
