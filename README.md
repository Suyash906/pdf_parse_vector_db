# pdf_parse_vector_db
Parse a PDF and upload it to vector database

<img width="1298" alt="image" src="https://github.com/user-attachments/assets/bbac5b76-4fcf-4306-84f0-91bb39a0a2fe" />



## cURL request
```
curl --location 'http://127.0.0.1:5001/api/v1/search-similar-cases' \
--form 'court_level="2"' \
--form 'case_file=@"AHMEDABAD_C-5-2012_30-01-2023.pdf"'
```
