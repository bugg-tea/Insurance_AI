from backend.app.retrieval.final import build_retrieval_pipeline


def mock_chunks():
    return [
        {
            "chunk_id": "1",
            "text": "This policy covers hospitalization expenses",
            "chunk_type": "clause",
            "table_id": "T1"
        },
        {
            "chunk_id": "2",
            "text": "Exclusions: pre-existing diseases are not covered",
            "chunk_type": "exclusion",
            "table_id": "T1"
        },
        {
            "chunk_id": "3",
            "text": "Waiting period for maternity is 2 years",
            "chunk_type": "clause",
            "table_id": "T2"
        }
    ]


def test_pipeline():
    pipeline = build_retrieval_pipeline(mock_chunks())

    queries = [
        "what is not covered in this policy?",
        "what is waiting period?",
        "does it cover hospitalization?"
    ]

    for q in queries:
        print("\n====================")
        print("QUERY:", q)

        results = pipeline.search(q, top_k=3)

        print("RESULT COUNT:", len(results))

        for r in results:
            print("\n--- RESULT ---")
            print("ID:", r.get("chunk_id"))
            print("SCORE:", r.get("final_score"))
        
            print("SCORE:", r.get("retrieval_score"))
            print("TYPE:", r.get("chunk_type"))
            print("TEXT:", r.get("text"))
            print("HINT:", r.get("context_hint"))
            
            
            
            
if __name__ == "__main__":
    test_pipeline()