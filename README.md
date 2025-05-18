# pdf_parse_vector_db
Parse a PDF and upload it to vector database

<img width="1281" alt="image" src="https://github.com/user-attachments/assets/6742fee2-a619-45bb-8a6d-5faf88e097bd" />


## cURL request
```
curl --location 'http://127.0.0.1:5001/api/v1/ingest-legal-document' \
--form 'file=@"sample.pdf"'
```
