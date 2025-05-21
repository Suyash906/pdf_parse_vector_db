# pdf_parse_vector_db
Parse a PDF and upload it to vector database

<img width="1298" alt="image" src="https://github.com/user-attachments/assets/bbac5b76-4fcf-4306-84f0-91bb39a0a2fe" />



## cURL request
```
curl --location 'http://127.0.0.1:5001/api/v1/search-similar-cases' \
--form 'court_level="2"' \
--form 'case_file=@"AHMEDABAD_C-5-2012_30-01-2023.pdf"'
```

## Sample Response

```json
{
    "appellant_statistics": {
        "invalid_decisions": 0,
        "total_valid_decisions": 5,
        "win_count": 4,
        "win_percentage": 80.0
    },
    "query": {
        "file_name": "AHMEDABAD_C-5-2012_30-01-2023.pdf",
        "input_court_level": 2,
        "target_court_level": 3
    },
    "result_count": 5,
    "results": [
        {
            "case_decision": "appellant_won",
            "file_id": "e8e841a3e5058c671f0962babec6b400481547c5fa717e05d8e977d520a523c8",
            "file_name": "2022_12_01_Case_Law___1988__33__E_L_T__768__Tri__Del___29_12_1987_.pdf",
            "score": 0.8259715437889099
        },
        {
            "case_decision": "appellant_lost",
            "file_id": "a72b679c2590f9e25ad522f646a7a3b36d348815483448f3630ee6b67659d44e",
            "file_name": "2022_12_01_Case_Law___1988__37__E_L_T__225__Tri__Del___12_07_1988_.pdf",
            "score": 0.8266448974609375
        },
        {
            "case_decision": "appellant_won",
            "file_id": "4b018b87c989c001975d79b9454d6ec203702860d471fb72dbb8d51a4deeb7a3",
            "file_name": "2022_12_01_Case_Law___1988__34__E_L_T__452__Cal____29_09_1980_.pdf",
            "score": 0.8331994414329529
        },
        {
            "case_decision": "appellant_won",
            "file_id": "cf916656c72e82d997f98297cf6c10021dbee0241ad1032b4a51989e1946b2bb",
            "file_name": "2022_12_01_Case_Law___1989__42__E_L_T__267__Tri__Del___29_10_1987_.pdf",
            "score": 0.8413318395614624
        },
        {
            "case_decision": "appellant_won",
            "file_id": "d52372e6e49a02a763bace648be3002f35fc6004bcd101f63f5a504904e2829e",
            "file_name": "2022_12_01_Case_Law___1988__33__E_L_T__534__Tri__Del___13_05_1983_.pdf",
            "score": 0.843200147151947
        }
    ],
    "status": "success"
}
```
